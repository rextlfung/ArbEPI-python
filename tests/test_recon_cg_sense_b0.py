"""Correctness check for recon/run_recon.py's cg_sense_solve: CG on a
small, fully-sampled (no acceleration) synthetic SENSE problem should
recover the true image to high precision in a handful of iterations,
independent of which operator (plain or B0-corrected) it's handed -- the
solver itself doesn't know or care which encoding operator A is."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from recon.operators import build_encoding_operator, gather_ksp  # noqa: E402
from recon.run_recon import cg_sense_solve  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    A = build_encoding_operator(smaps, omega)

    x_true = _complex_randn(Nx, Ny, Nz, Nt, seed=1)
    ksp0 = torch.zeros(Nx, Ny, Nz, Nc, Nt, dtype=torch.complex64, device=DEVICE)
    for it in range(Nt):
        y = A.A[it].apply(x_true[..., it])  # (K,Nc)
        flat = torch.zeros(Nx * Ny * Nz, Nc, dtype=torch.complex64, device=DEVICE)
        flat[A.A[it].idx, :] = y
        ksp0[..., it] = flat.reshape(Nx, Ny, Nz, Nc)

    ksp = gather_ksp(ksp0, A)
    X, residuals = cg_sense_solve(A, ksp, (Nx, Ny, Nz, Nt), num_iter=50, tol=1e-10)

    assert torch.allclose(X, x_true, atol=1e-3, rtol=1e-3)
    assert residuals[-1] < 1e-5
    # Monotonically non-increasing residual (CG's own guarantee on a
    # well-posed normal-equations problem).
    assert all(residuals[i + 1] <= residuals[i] + 1e-9 for i in range(len(residuals) - 1))


def test_cg_sense_solve_handles_all_zero_kspace():
    Nx, Ny, Nz, Nc, Nt = 4, 4, 4, 2, 1
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=2)
    omega = torch.ones(Nx, Ny, Nz, Nt, dtype=torch.bool, device=DEVICE)
    A = build_encoding_operator(smaps, omega)
    K = A.A[0].idx.numel()
    ksp = torch.zeros(K, Nc, Nt, dtype=torch.complex64, device=DEVICE)

    X, residuals = cg_sense_solve(A, ksp, (Nx, Ny, Nz, Nt), num_iter=10)

    assert torch.all(X == 0)
    assert residuals == [1.0]
