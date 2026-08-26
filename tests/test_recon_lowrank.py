"""Algorithm-invariant tests for recon/lowrank.py, mirroring the cases in
../mslr-recon/tests/kernel_tests.jl (patch round-trip, SVST shrinkage,
unit-patch fast path). No golden Julia output is compared here -- see
tests/test_recon_reconstruct.py / scratchpad validation for that; these
tests check mathematical invariants that must hold regardless of backend.
"""

import pytest

torch = pytest.importorskip("torch")

from recon.lowrank import SVST, img2patches, patch_nucnorm, patches2img, patchSVST  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _random_img(Nx, Ny, Nz, Nt, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    real = torch.randn(Nx, Ny, Nz, Nt, generator=g, device=DEVICE)
    imag = torch.randn(Nx, Ny, Nz, Nt, generator=g, device=DEVICE)
    return (real + 1j * imag).to(torch.complex64)


def test_nonoverlapping_patch_roundtrip_is_exact():
    img = _random_img(10, 10, 6, 5)
    ps = ss = (2, 2, 3)
    P = img2patches(img, ps, ss)
    img2 = patches2img(P, ps, ss, (10, 10, 6))
    assert torch.allclose(img2, img, atol=1e-5)


def test_overlapping_patch_roundtrip_with_zero_threshold_is_exact():
    """SVST(P, beta=0) is the identity, so patches2img(SVST(img2patches(img)))
    == img even when patches overlap (exercises the overlap-average path)."""
    img = _random_img(10, 10, 6, 5, seed=1)
    ps, ss = (4, 4, 3), (2, 2, 2)
    P = img2patches(img, ps, ss)
    result, _ = SVST(P, 0.0)
    img2 = patches2img(result, ps, ss, (10, 10, 6))
    assert torch.allclose(img2, img, atol=1e-4)


def test_odd_spatial_dims_roundtrip():
    """Regression case mirroring kernel_tests.jl's odd-Nz check (this repo's
    own real data has Nz=45, an odd dimension)."""
    img = _random_img(9, 7, 5, 4, seed=2)
    ps, ss = (3, 3, 3), (2, 2, 2)
    P = img2patches(img, ps, ss)
    result, _ = SVST(P, 0.0)
    img2 = patches2img(result, ps, ss, (9, 7, 5))
    assert torch.allclose(img2, img, atol=1e-4)


def test_svst_shrinks_singular_values_by_beta():
    X = _random_img(1, 1, 1, 6, seed=3).reshape(1, 6) * 5  # single (1,6) "patch"
    beta = 0.5
    s_before = torch.linalg.svdvals(X)
    result, reg = SVST(X, beta)
    s_after = torch.linalg.svdvals(result)
    expected = torch.clamp(s_before - beta, min=0.0)
    assert torch.allclose(s_after, expected, atol=1e-4)
    assert torch.allclose(reg, expected.sum(), atol=1e-4)


def test_svst_zeros_out_low_energy_patches_exactly():
    X = _random_img(1, 1, 1, 4, seed=4).reshape(1, 4) * 1e-4
    beta = 10.0  # far above ||X||_F
    result, reg = SVST(X, beta)
    assert torch.all(result == 0)
    assert reg.item() == 0.0


def test_unit_patch_fast_path_matches_general_svst():
    img = _random_img(6, 6, 4, 5, seed=5)
    beta = 0.3
    img_fast, reg_fast = patchSVST(img, beta, (1, 1, 1), (1, 1, 1))

    P = img2patches(img, (1, 1, 1), (1, 1, 1))
    result, reg = SVST(P, beta)
    img_general = patches2img(result, (1, 1, 1), (1, 1, 1), (6, 6, 4))

    assert torch.allclose(img_fast, img_general, atol=1e-4)
    assert abs(reg_fast.item() - reg.sum().item()) < 1e-3


def test_patch_nucnorm_matches_direct_svd_for_a_single_global_patch():
    Nx, Ny, Nz, Nt = 5, 5, 3, 4
    img = _random_img(Nx, Ny, Nz, Nt, seed=6)
    P = img2patches(img, (Nx, Ny, Nz), (Nx, Ny, Nz))
    nn = patch_nucnorm(P)
    ref = torch.linalg.svdvals(img.reshape(Nx * Ny * Nz, Nt)).sum()
    assert torch.allclose(nn, ref, atol=1e-4)


def test_img2patches_rejects_nonpositive_stride():
    img = _random_img(4, 4, 4, 2, seed=7)
    with pytest.raises(ValueError):
        img2patches(img, (2, 2, 2), (0, 2, 2))
