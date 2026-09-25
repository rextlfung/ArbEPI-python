"""Convergence sanity checks for recon/sense.py's pogm_restart, on a
trivial least-squares problem (A=I) with a known closed-form solution,
across all three momentum variants."""

import pytest

torch = pytest.importorskip("torch")

from recon.mri_operator import build_sense  # noqa: E402
from recon.solvers import cg, pogm_restart  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("mom", ["pgm", "fpgm", "pogm"])
def test_pogm_restart_converges_on_identity_least_squares(mom):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    n = 50
    real = torch.randn(n, generator=g, device=DEVICE)
    imag = torch.randn(n, generator=g, device=DEVICE)
    b = (real + 1j * imag).to(torch.complex64)
    x0 = torch.zeros(n, dtype=torch.complex64, device=DEVICE)

    x, out = pogm_restart(
        x0,
        lambda x: 0.5 * (x - b).norm().item() ** 2,
        lambda x: x - b,
        1.0,
        mom=mom,
        niter=200,
        conv_tol=1e-8,
        fun=lambda it, xk, yk, is_restart, fcostnew, rel_change: fcostnew,
    )
    rel_err = (x - b).norm().item() / b.norm().item()
    assert rel_err < 1e-4
    assert out[-1] < out[0]


@pytest.mark.parametrize("mom", ["fpgm", "pogm"])
@pytest.mark.parametrize("inplace_prox", [False, True])
def test_pogm_restart_matches_closed_form_lasso_regardless_of_prox_style(mom, inplace_prox):
    """A=I LASSO (0.5||x-b||^2 + lam||x||_1) has a closed-form solution,
    the soft-threshold of b -- and its prox step (soft-thresholding) is a
    realistic stand-in for recon/sense.py's g_prox, which mutates
    its argument in place and returns it (a small transient-memory win)
    rather than returning a fresh tensor. Regression test for pogm_restart
    aliasing xnew with znew when g_prox does that: fpgm's prox argument is
    a temporary never read again so it was never affected, but pogm's is
    reused for the next iteration's momentum terms, silently zeroing them.
    Both momentum variants must reach the same optimum regardless of
    whether g_prox mutates in place or returns a fresh tensor."""
    g = torch.Generator(device=DEVICE).manual_seed(1)
    n = 30
    real = torch.randn(n, generator=g, device=DEVICE)
    imag = torch.randn(n, generator=g, device=DEVICE)
    b = (real + 1j * imag).to(torch.complex64)
    lam = 0.3
    x0 = torch.zeros(n, dtype=torch.complex64, device=DEVICE)

    def _soft_threshold(z: torch.Tensor, thresh: float) -> torch.Tensor:
        mag = z.abs()
        scale = torch.clamp(mag - thresh, min=0.0) / (mag + 1e-12)
        return scale

    def fcost(x):
        return 0.5 * (x - b).norm().item() ** 2 + lam * x.abs().sum().item()

    def fgrad(x):
        return x - b

    def g_prox(z, c):
        scale = _soft_threshold(z, c * lam)
        if inplace_prox:
            z.mul_(scale)
            return z
        return z * scale

    x, out = pogm_restart(
        x0, fcost, fgrad, 1.0, mom=mom, niter=300, conv_tol=0.0, g_prox=g_prox,
    )

    x_star = b * _soft_threshold(b, lam)
    rel_err = (x - x_star).norm().item() / x_star.norm().item()
    assert rel_err < 1e-3


def test_pogm_restart_niter_zero_returns_initial_point():
    x0 = torch.zeros(5, dtype=torch.complex64, device=DEVICE)
    x, out = pogm_restart(x0, lambda x: 0.0, lambda x: x, 1.0, niter=0)
    assert torch.equal(x, x0)
    assert len(out) == 1


def test_pogm_restart_rejects_invalid_mom():
    x0 = torch.zeros(5, dtype=torch.complex64, device=DEVICE)
    with pytest.raises(ValueError):
        pogm_restart(x0, lambda x: 0.0, lambda x: x, 1.0, mom="bogus")


def _complex_randn(*shape, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    real = torch.randn(*shape, generator=g, device=DEVICE)
    imag = torch.randn(*shape, generator=g, device=DEVICE)
    return (real + 1j * imag).to(torch.complex64)


def test_cg_sense_solve_recovers_exact_image_when_fully_sampled():
    Nx, Ny, Nz, Nc, Nt = 6, 6, 4, 3, 2

    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=0)
    smaps = smaps / smaps.abs().pow(2).sum(dim=0, keepdim=True).sqrt().clamp_min(1e-6)

    omega = torch.ones(Nx, Ny, Nz, Nt, dtype=torch.bool, device=DEVICE)  # fully sampled
    A = build_sense(smaps, omega)

    x_true = _complex_randn(Nx, Ny, Nz, Nt, seed=1)
    ksp0 = torch.zeros(Nx, Ny, Nz, Nc, Nt, dtype=torch.complex64, device=DEVICE)
    for it in range(Nt):
        y = A.A[it].apply(x_true[..., it])  # (K,Nc)
        flat = torch.zeros(Nx * Ny * Nz, Nc, dtype=torch.complex64, device=DEVICE)
        flat[A.A[it].idx, :] = y
        ksp0[..., it] = flat.reshape(Nx, Ny, Nz, Nc)

    ksp = torch.stack([ksp0[..., it].reshape(-1, Nc)[A.A[it].idx] for it in range(Nt)], dim=-1)
    X, residuals = cg(A, ksp, (Nx, Ny, Nz, Nt), num_iter=50, tol=1e-10)

    assert torch.allclose(X, x_true, atol=1e-3, rtol=1e-3)
    assert residuals[-1] < 1e-5
    # Monotonically non-increasing residual (CG's own guarantee on a
    # well-posed normal-equations problem).
    assert all(residuals[i + 1] <= residuals[i] + 1e-9 for i in range(len(residuals) - 1))


def test_cg_sense_solve_handles_all_zero_kspace():
    Nx, Ny, Nz, Nc, Nt = 4, 4, 4, 2, 1
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=2)
    omega = torch.ones(Nx, Ny, Nz, Nt, dtype=torch.bool, device=DEVICE)
    A = build_sense(smaps, omega)
    K = A.A[0].idx.numel()
    ksp = torch.zeros(K, Nc, Nt, dtype=torch.complex64, device=DEVICE)

    X, residuals = cg(A, ksp, (Nx, Ny, Nz, Nt), num_iter=10)

    assert torch.all(X == 0)
    assert residuals == [1.0]
