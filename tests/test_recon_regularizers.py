"""Algorithm-invariant tests for recon/regularizers.py. The low-rank cases
mirror ../mslr-recon/tests/kernel_tests.jl (patch round-trip, SVST shrinkage,
unit-patch fast path); the wavelet/TV cases check Wavelet3D's adjoint and
isometry and the FBPD step-size bound on ||G||^2.
"""

import pytest

torch = pytest.importorskip("torch")

from recon.regularizers import (  # noqa: E402
    SVST,
    MultiScaleLowRank,
    SectionL1,
    SumScales,
    Wavelet3D,
    WaveletTV,
    img2patches,
    patch_nucnorm,
    patches2img,
    patchSVST,
)

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


def test_patchsvst_chunking_matches_unchunked_result():
    """patchSVST never materializes the full (Np, prod(patch_size), Nt)
    patch tensor -- it gathers/SVSTs/scatters memory-budgeted chunks of
    patches directly (2026-09-22, see its docstring for why: a real
    large-grid/fine-resolution dataset can need ~45GB for that tensor
    alone). A tiny max_chunk_bytes here forces many chunks (as opposed to
    the single chunk every other test in this file exercises, since their
    grids are small enough to fit in one chunk under the real default) --
    result should match the unchunked (one big chunk) computation to
    float32 summation-order noise, not just approximately."""
    img = _random_img(10, 10, 6, 5, seed=8)
    beta = 0.3
    patch_size, stride_size = (3, 3, 3), (2, 2, 2)

    out_unchunked, reg_unchunked = patchSVST(img, beta, patch_size, stride_size, max_chunk_bytes=10_000_000_000)
    out_chunked, reg_chunked = patchSVST(img, beta, patch_size, stride_size, max_chunk_bytes=2000)

    assert torch.allclose(out_unchunked, out_chunked, atol=1e-4)
    assert abs(reg_unchunked.item() - reg_chunked.item()) < 1e-2


def test_sum_scales_adjoint_is_self_consistent():
    S = SumScales((4, 3, 2, 5), 3)
    X = torch.randn(4, 3, 2, 5, 3, dtype=torch.complex64, device=DEVICE)
    y = torch.randn(4, 3, 2, 5, dtype=torch.complex64, device=DEVICE)
    lhs = torch.vdot(S.apply(X).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(X.reshape(-1), S.adjoint(y).reshape(-1))
    assert abs(lhs - rhs).item() / abs(lhs).item() < 1e-5


def test_lowrank_prox_cost_matches_its_own_cost_function():
    """prox's free byproduct (nuclear norm of the thresholded result) must
    equal cost() evaluated on that result."""
    Nx, Ny, Nz, Nt = 8, 8, 4, 6
    scales = [(Nx, Ny, Nz), (4, 4, 4)]
    g = MultiScaleLowRank(scales, scales, (Nx, Ny, Nz, Nt), lambda_global=0.1)
    X = torch.stack([_random_img(Nx, Ny, Nz, Nt, seed=s) for s in (1, 2)], dim=-1)
    out = g.prox(X.clone(), 0.5)
    assert abs(g.last_cost - g.cost(out)) / g.cost(out) < 1e-4


@pytest.mark.parametrize("shape", [(16, 16, 8), (12, 10, 45), (4, 4, 4)])
def test_wavelet3d_adjoint_and_isometry(shape):
    """Odd/non-power-of-2 sizes (Nz=45 in real data) exercise the zero-pad path."""
    W = Wavelet3D(shape, "db4", 3)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    x = torch.randn(*shape, generator=g, device=DEVICE, dtype=torch.complex64)
    y = torch.randn(W.size_out[0], generator=g, device=DEVICE, dtype=torch.complex64)
    lhs = torch.vdot(W.apply(x), y)
    rhs = torch.vdot(x.reshape(-1), W.adjoint(y).reshape(-1))
    assert abs(lhs - rhs).item() / abs(lhs).item() < 1e-5
    torch.testing.assert_close(W.adjoint(W.apply(x)), x, atol=1e-5, rtol=1e-5)


def test_wavelet_tv_g_norm_bound_holds():
    """FBPD's step sizes rely on G_norm_squared >= ||G||^2."""
    shape = (12, 10, 9)
    reg = WaveletTV(shape, 0.01, 0.02)
    v = torch.randn(*shape, dtype=torch.complex64, device=DEVICE)
    for _ in range(100):
        v = reg.G.adjoint(reg.G.apply(v))
        norm_sq = v.norm().item()
        v = v / norm_sq
    assert norm_sq <= reg.G_norm_squared


def test_section_l1_thresholds_each_section_with_its_own_lambda():
    prox = SectionL1([1.0, 2.0], [2, 3])
    v = torch.tensor([3.0, -0.5, 3.0, -2.5, 1.0], device=DEVICE)
    out = prox(v, 0.5)  # thresholds 0.5 and 1.0
    expected = torch.tensor([2.5, 0.0, 2.0, -1.5, 0.0], device=DEVICE)
    torch.testing.assert_close(out, expected)
