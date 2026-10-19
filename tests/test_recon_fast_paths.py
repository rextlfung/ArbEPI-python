"""Tests for the performance paths of recon/: the shift-free centered FFT in
SENSE (against a dense fftshift/ifftshift reference, odd and even sizes), the
shared scatter buffer, the pooled/gesvda SVD behind SVST (against float64), the
forward-product reuse in the mslr solver, and the warm-started power iteration.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from recon import operators  # noqa: E402
from recon.operators import SENSE, SENSE_B0, build_sense  # noqa: E402
from recon.regularizers import (  # noqa: E402
    MultiScaleLowRank,
    _svd,
    _svst_batch,
    patch_nucnorm,
)
from recon.sense import _solve_mslr  # noqa: E402
from recon.solvers import pogm_restart  # noqa: E402
from recon.utils import estimate_spectral_norm, poweriter  # noqa: E402
from tests._helpers import DEVICE, complex_randn  # noqa: E402


def _reference_forward(x, smaps, idx):
    """The definition: smaps -> ifftshift -> fftn -> fftshift -> gather, dense."""
    dims = tuple(range(1, smaps.ndim))
    kc = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x * smaps, dim=dims), dim=dims, norm="ortho"), dim=dims
    )
    return kc.reshape(smaps.shape[0], -1).T[idx]


def _reference_adjoint(y, smaps, idx):
    dims = tuple(range(1, smaps.ndim))
    full = torch.zeros(smaps.shape[0], smaps[0].numel(), dtype=y.dtype, device=y.device)
    full[:, idx] = y.T
    full = full.reshape(smaps.shape)
    xc = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(full, dim=dims), dim=dims, norm="ortho"), dim=dims
    )
    return (xc * smaps.conj()).sum(0)


def _rand_mask(shape, frac, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(shape, generator=g) < frac).to(DEVICE)


@pytest.mark.parametrize("shape", [(8, 6, 4), (7, 5, 9), (8, 9, 5), (1, 6, 7)])
def test_sense_matches_dense_fftshift_reference(shape):
    """Even, odd and mixed sizes: the odd axes are where a +-1 shortcut would fail."""
    Nc = 3
    smaps = complex_randn(Nc, *shape, seed=1)
    x = complex_randn(*shape, seed=2)
    op = SENSE(smaps, _rand_mask(shape, 0.3, seed=3))
    assert op.idx.numel() > 0
    y = op.apply(x)
    assert torch.allclose(y, _reference_forward(x, smaps, op.idx), atol=1e-5, rtol=1e-5)
    y_in = complex_randn(op.idx.numel(), Nc, seed=4)
    assert torch.allclose(
        op.adjoint(y_in), _reference_adjoint(y_in, smaps, op.idx), atol=1e-5, rtol=1e-5
    )


def test_sense_b0_matches_segment_sum_of_reference():
    shape, Nc, L, E = (6, 7, 5), 2, 4, 5
    smaps = complex_randn(Nc, *shape, seed=5)
    op_mask = _rand_mask(shape, 0.4, seed=6)
    K = int(op_mask.sum())
    pos = torch.randint(0, E, (K,), device=DEVICE)
    b = complex_randn(E, L, seed=7)
    c = complex_randn(L, *shape, seed=8)
    op = SENSE_B0(smaps, op_mask, pos, b, c)
    x = complex_randn(*shape, seed=9)
    ref = sum(b[pos, i][:, None] * _reference_forward(x * c[i], smaps, op.idx) for i in range(L))
    assert torch.allclose(op.apply(x), ref, atol=1e-4, rtol=1e-4)
    y = complex_randn(K, Nc, seed=10)
    ref_adj = sum(
        c[i].conj() * _reference_adjoint(y * b[pos, i].conj()[:, None], smaps, op.idx)
        for i in range(L)
    )
    assert torch.allclose(op.adjoint(y), ref_adj, atol=1e-4, rtol=1e-4)


def test_scatter_buffer_is_shared_and_left_zero():
    operators.release_scratch()
    shape, Nc = (6, 5, 4), 2
    smaps = complex_randn(Nc, *shape, seed=1)
    ops = [SENSE(smaps, _rand_mask(shape, 0.3, seed=s)) for s in (2, 3)]
    for op in ops:
        op.adjoint(complex_randn(op.idx.numel(), Nc, seed=4))
    assert len(operators._scatter_buffers) == 1
    (buf,) = operators._scatter_buffers.values()
    assert not buf.any()
    operators.release_scratch()
    assert not operators._scatter_buffers


def _svst_reference(X, beta):
    X = X.to(torch.complex128).cpu()
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    st = torch.clamp(S - beta, min=0)
    zero = torch.linalg.matrix_norm(X, ord="fro") <= beta
    rec = U @ (st.to(Vh.dtype).unsqueeze(-1) * Vh)
    rec[zero] = 0
    return rec, torch.where(zero, torch.zeros_like(st[:, 0]), st.sum(-1))


def _rel_err(rec, ref):
    diff = (rec.cpu().to(torch.complex128) - ref).norm(dim=(1, 2))
    return diff / ref.norm(dim=(1, 2)).clamp_min(1)


def _low_rank_patches(B, m, n, rank=3, amp=20.0, seed=0):
    U = complex_randn(B, m, rank, seed=seed)
    V = complex_randn(B, rank, n, seed=seed + 1)
    return (amp * (U @ V) + complex_randn(B, m, n, seed=seed + 2)).to(DEVICE)


@pytest.mark.parametrize("m,n,B", [(27, 12, 20), (50, 40, 12), (40, 50, 12), (6, 20, 3)])
def test_svst_batch_matches_float64_reference(m, n, B):
    """Includes shapes above 32 on a side with B >= 8, which take gesvda on a GPU;
    some patches are far below beta (skipped) and some above."""
    X = _low_rank_patches(B, m, n, seed=m)
    X[::4] *= 1e-3  # below beta: thresholded to exactly zero
    beta = 5.0
    rec, reg = _svst_batch(X, beta)
    ref, ref_reg = _svst_reference(X, beta)
    assert (rec[::4] == 0).all() and (reg[::4] == 0).all()
    err = _rel_err(rec, ref)
    assert err.max() < 1e-5
    assert torch.allclose(reg.cpu().double(), ref_reg, rtol=1e-5)


def test_svst_all_patches_thresholded_to_zero():
    X = _low_rank_patches(10, 40, 40) * 1e-3
    rec, reg = _svst_batch(X, 1e6)
    assert (rec == 0).all() and (reg == 0).all()


def test_svst_handles_rank_deficient_patches_where_gesvda_raises():
    """Exactly low-rank (no noise) and duplicated-column patches make cuSOLVER's
    gesvda raise; _svd must fall back and still give the right answer."""
    B, m, n = 10, 40, 40
    U, V = complex_randn(B, m, 4, seed=1), complex_randn(B, 4, n, seed=2)
    X = (U @ V).to(DEVICE)
    X[3, :, 20:] = X[3, :, :20]
    X[5] = 0  # an all-zero patch whose norm is not below beta
    beta = 1e-3
    rec, _ = _svst_batch(X, beta)
    ref, _ = _svst_reference(X, beta)
    assert torch.isfinite(rec).all()
    err = _rel_err(rec, ref)
    assert err.max() < 1e-4


def test_svst_nan_patch_is_not_silently_zeroed():
    X = _low_rank_patches(8, 40, 40)
    X[2, 0, 0] = float("nan")
    try:
        rec, _ = _svst_batch(X, 5.0)
    except Exception:
        return  # the SVD failed loudly
    assert not torch.isfinite(rec[2]).all()


def test_svd_values_and_nucnorm_agree_with_factors():
    X = _low_rank_patches(9, 40, 36)
    _, S, _ = _svd(X)
    assert torch.allclose(_svd(X, vals_only=True), S, rtol=1e-4, atol=1e-3)
    assert torch.allclose(
        patch_nucnorm(X).cpu(), torch.linalg.svdvals(X.cpu().to(torch.complex128)).sum().float(),
        rtol=1e-5,
    )


def test_cpu_pool_restores_thread_count():
    before = torch.get_num_threads()
    _svd(_low_rank_patches(8, 30, 20).cpu())
    assert torch.get_num_threads() == before


def test_mslr_solver_reuses_the_forward_product_and_matches_the_plain_loop():
    Nx, Ny, Nz, Nc, Nt, niters = 6, 6, 4, 3, 5, 6
    smaps = complex_randn(Nc, Nx, Ny, Nz, seed=1)
    smaps = smaps / smaps.abs().pow(2).sum(0, keepdim=True).sqrt()
    omega = torch.zeros(Nx, Ny, Nz, Nt, dtype=torch.bool, device=DEVICE)
    for it in range(Nt):
        omega[..., it] = _rand_mask((Ny, Nz), 0.5, seed=10 + it).expand(Nx, Ny, Nz)
    k = int(omega.sum(dim=(0, 1, 2)).min())
    for it in range(Nt):  # equal K per frame
        flat = omega[..., it].reshape(-1).nonzero().squeeze(-1)
        omega[..., it].view(-1)[flat[k:]] = False
    A = build_sense(smaps, omega)
    ksp = complex_randn(k, Nc, Nt, seed=2)
    common = dict(omega=omega, R=2.0, sigma1A=1.0)
    shape, patch, stride = (Nx, Ny, Nz, Nt), [(3, 3, 2)], [(3, 3, 2)]

    calls = {"fwd": 0}
    apply_orig = A.apply

    def counting_apply(x):
        calls["fwd"] += 1
        return apply_orig(x)

    A.apply = counting_apply
    res = _solve_mslr(
        A, ksp, shape, 1.0, patch, stride, 1.0, "pogm", niters, 0.0, False, common
    )
    assert calls["fwd"] == niters + 1  # one per iteration's cost, plus the start's

    # The same iteration with no reuse: cost and gradient each apply A.
    g = MultiScaleLowRank(patch, stride, shape, 1.0)
    S = g.synthesis
    A.apply = apply_orig

    def cost(X):
        return 0.5 * (A.apply(S.apply(X)) - ksp).norm().item() ** 2

    def grad(X):
        return S.adjoint(A.adjoint(A.apply(S.apply(X)) - ksp))

    X0 = S.adjoint(A.adjoint(ksp))
    g.last_cost = g.cost(X0)
    X, _ = pogm_restart(X0, cost, grad, 1.0, mom="pogm", niter=niters, g_prox=g.prox, conv_tol=0.0)
    assert torch.allclose(res.X, X, atol=1e-5, rtol=1e-5)


def test_warm_started_power_iteration_matches_cold_and_is_cheaper():
    Nx, Ny, Nz, Nc, Nt = 8, 8, 6, 4, 6
    smaps = complex_randn(Nc, Nx, Ny, Nz, seed=1)
    smaps = smaps / smaps.abs().pow(2).sum(0, keepdim=True).sqrt()
    base = _rand_mask((Ny, Nz), 0.25, seed=3)
    omega = torch.stack(
        [torch.roll(base, shifts=(it, 2 * it), dims=(0, 1)).expand(Nx, Ny, Nz) for it in range(Nt)],
        dim=-1,
    )
    A = build_sense(smaps, omega)
    x0 = complex_randn(Nx, Ny, Nz, seed=5)

    n_fwd = {"warm": 0, "cold": 0}

    originals = [b.apply for b in A.A]

    def counted(fn, key):
        def apply(x):
            n_fwd[key] += 1
            return fn(x)
        return apply

    cold = max(
        poweriter(counted(b.apply, "cold"), b.adjoint, x0, tol=1e-7) for b in A.A
    )
    for b, fn in zip(A.A, originals):
        b.apply = counted(fn, "warm")
    warm = estimate_spectral_norm(A, x0, tol=1e-7)
    # The stopping rule (change of the estimate below tol) leaves the estimate a
    # few 1e-4 short of the norm from any start: six random starts on this operator
    # spread by 1.6-2e-4 at tol 1e-6..1e-7, so that is the agreement to expect.
    assert abs(warm - cold) / cold < 5e-4
    assert n_fwd["warm"] <= n_fwd["cold"]
