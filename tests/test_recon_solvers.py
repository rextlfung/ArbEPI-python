"""Convergence sanity checks for recon/solvers.py's pogm_restart, on a
trivial least-squares problem (A=I) with a known closed-form solution,
across all three momentum variants."""

import pytest

torch = pytest.importorskip("torch")

from recon.solvers import pogm_restart  # noqa: E402

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
    realistic stand-in for recon/reconstruct.py's g_prox, which mutates
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
