"""Adjoint self-consistency and spectral-norm sanity checks for
recon/operators.py's block-diagonal SENSE operator, mirroring
../mslr-recon/tests/kernel_tests.jl's Asense forward/adjoint checks."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from recon.operators import build_encoding_operator  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _complex_randn(*shape, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    real = torch.randn(*shape, generator=g, device=DEVICE)
    imag = torch.randn(*shape, generator=g, device=DEVICE)
    return (real + 1j * imag).to(torch.complex64)


def test_adjoint_is_self_consistent_on_odd_spatial_dims():
    """Nz odd (this repo's real data has Nz=45) -- regression case for the
    fftshift/ifftshift adjoint bug mslr-recon's sense_gpu.jl once had."""
    Nx, Ny, Nz, Nc, Nt = 6, 7, 5, 3, 4
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=0)
    omega = torch.stack(
        [torch.rand(Nx, Ny, Nz, device=DEVICE) > 0.5 for _ in range(Nt)], dim=-1
    )
    # BlockDiagonal requires every frame operator to share the same K
    # (sample count); this repo's real acquisitions guarantee that, but a
    # per-frame-independent random mask generally won't, so trim to the min.
    counts = omega.sum(dim=(0, 1, 2))
    k = counts.min().item()
    omega = omega & (torch.cumsum(omega.reshape(-1, Nt), dim=0) <= k).reshape(Nx, Ny, Nz, Nt)

    A = build_encoding_operator(smaps, omega)
    K = A.A[0].idx.numel()

    x = _complex_randn(Nx, Ny, Nz, Nt, seed=1)
    y = _complex_randn(K, Nc, Nt, seed=2)

    lhs = torch.vdot(A.apply(x).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), A.adjoint(y).reshape(-1))
    assert abs(lhs - rhs).item() / abs(lhs).item() < 1e-5


def test_spectral_norm_is_near_unity_for_normalized_smaps_full_sampling():
    """Matches sense_gpu.jl's documented convention: unitary + normalized
    smaps + full sampling gives sigma1(A) ~= 1.0."""
    Nx, Ny, Nz, Nc = 8, 8, 6, 4
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=3)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)
    A = build_encoding_operator(smaps, full_mask.unsqueeze(-1))

    x = _complex_randn(Nx, Ny, Nz, 1, seed=4)
    x = x / x.norm()
    for _ in range(30):
        x = A.adjoint(A.apply(x))
        x = x / x.norm()
    sigma1 = A.apply(x).norm().item()
    assert abs(sigma1 - 1.0) < 1e-3
