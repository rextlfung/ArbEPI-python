"""Temporal regularization in recon/: the temporal operators (TemporalDiff,
PerFrame, TemporalHighPass), the joint wavelet-TV solver, and the quadratic
high-pass penalty in CG / MSLR."""

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from recon.operators import build_sense  # noqa: E402
from recon.regularizers import (  # noqa: E402
    PerFrame,
    TemporalDiff,
    TemporalHighPass,
    Wavelet3D,
    dct_matrix,
)
from recon.sense import run_sense  # noqa: E402
from recon.solvers import cg  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _crandn(*shape, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g, device=DEVICE),
                         torch.randn(*shape, generator=g, device=DEVICE))


def _inner(a, b):
    return torch.vdot(a.reshape(-1), b.reshape(-1))


def _smaps(Nc, shape):
    s = _crandn(Nc, *shape, seed=0)
    return s / s.abs().pow(2).sum(0, keepdim=True).sqrt()


def _write(tmp_path, x_true, smaps, omega, name="ksp"):
    """Noiseless k-space of x_true (Nx,Ny,Nz,Nt) in preprocess/'s layout."""
    Nx, Ny, Nz, Nt = x_true.shape
    Nc = smaps.shape[0]
    A = build_sense(smaps, omega)
    y = A.apply(x_true)
    dense = torch.zeros(Nx * Ny * Nz, Nc, Nt, dtype=torch.complex64, device=DEVICE)
    for t in range(Nt):
        dense[A.A[t].idx, :, t] = y[:, :, t]
    fn = tmp_path / f"{name}.h5"
    with h5py.File(fn, "w") as f:
        f["ksp_epi_zf"] = dense.reshape(Nx, Ny, Nz, Nc, Nt).cpu().numpy()
        f["smaps"] = smaps.permute(1, 2, 3, 0).cpu().numpy()
        f["omegas"] = omega[0].cpu().numpy()
    return str(fn)


def _random_masks(Nx, Ny, Nz, Nt, frac, seed, same=False):
    """(Nx,Ny,Nz,Nt) masks constant along x, the same sample count per frame."""
    rng = np.random.default_rng(seed)
    k = int(frac * Ny * Nz)
    m = np.zeros((Ny * Nz, Nt), bool)
    first = rng.choice(Ny * Nz, k, replace=False)
    for t in range(Nt):
        m[first if same else rng.choice(Ny * Nz, k, replace=False), t] = True
    m = torch.from_numpy(m.reshape(Ny, Nz, Nt)).to(DEVICE)
    return m.unsqueeze(0).expand(Nx, -1, -1, -1).contiguous()


def test_temporal_diff_is_non_periodic_and_adjoint_consistent():
    D = TemporalDiff((3, 4, 2, 5))
    x = _crandn(3, 4, 2, 5, seed=1)
    y = _crandn(D.size_out[0], seed=2)
    torch.testing.assert_close(_inner(D.apply(x), y), _inner(x, D.adjoint(y)), rtol=1e-4, atol=1e-4)
    ramp = torch.arange(5.0, device=DEVICE).expand(3, 4, 2, 5).to(torch.complex64)
    torch.testing.assert_close(D.apply(ramp), torch.ones(D.size_out[0], dtype=torch.complex64,
                                                         device=DEVICE))  # no wrap term


def test_per_frame_applies_the_3d_operator_to_every_frame_and_is_adjoint_consistent():
    W = Wavelet3D((8, 8, 4), wave="db2", levels=1)
    P = PerFrame(W, 3)
    x = _crandn(8, 8, 4, 3, seed=3)
    y = _crandn(P.size_out[0], seed=4)
    torch.testing.assert_close(_inner(P.apply(x), y), _inner(x, P.adjoint(y)), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(P.apply(x)[: W.size_out[0]], W.apply(x[..., 0]))


def test_temporal_highpass_is_an_orthogonal_projector_onto_the_high_band():
    Nt, tr = 40, 0.5
    hp = TemporalHighPass(Nt, tr, 0.15, DEVICE)
    assert hp.keep == 7  # DCT frequencies k / (2 Nt TR) = k / 40 Hz <= 0.15 Hz: k = 0..6
    P = hp.P
    torch.testing.assert_close(P @ P, P, atol=1e-5, rtol=0)
    torch.testing.assert_close(P, P.T)
    C = dct_matrix(Nt, DEVICE)
    torch.testing.assert_close(C @ C.T, torch.eye(Nt, device=DEVICE), atol=1e-5, rtol=0)
    torch.testing.assert_close(C[:7] @ P, torch.zeros(7, Nt, device=DEVICE), atol=1e-5, rtol=0)
    torch.testing.assert_close(C[7:] @ P, C[7:], atol=1e-5, rtol=0)
    # a slow drift stays almost entirely in the kept band (no wrap-around leakage)
    ramp = torch.linspace(-1, 1, Nt, device=DEVICE)
    assert (hp.apply(ramp).norm() / ramp.norm()) ** 2 < 0.01


def test_joint_wavelet_tv_reproduces_the_per_frame_solver_without_temporal_terms(tmp_path):
    """With identical frames (so the per-frame data scales agree) and no temporal
    term, the joint problem separates into the per-frame ones."""
    Nx, Ny, Nz, Nc, Nt = 8, 8, 8, 3, 3
    x0 = _crandn(Nx, Ny, Nz, 1, seed=5).expand(-1, -1, -1, Nt).contiguous()
    fn = _write(tmp_path, x0, _smaps(Nc, (Nx, Ny, Nz)), _random_masks(Nx, Ny, Nz, Nt, 0.5, 0,
                                                                      same=True))
    kw = dict(fn_ksp=fn, fn_smaps=fn, reg="wavelet-tv", niters=30, device=DEVICE,
              lamb_l1=1e-3, lamb_tv=1e-3, wave="db2", levels=1)
    a = run_sense(**kw).X_recon
    b = run_sense(**kw, joint=True).X_recon
    torch.testing.assert_close(b, a, rtol=1e-3, atol=1e-3 * a.abs().max().item())


def test_temporal_tv_reduces_frame_to_frame_variation_of_a_static_object(tmp_path):
    Nx, Ny, Nz, Nc, Nt = 8, 12, 8, 3, 8
    x0 = _crandn(Nx, Ny, Nz, 1, seed=6).expand(-1, -1, -1, Nt).contiguous()
    fn = _write(tmp_path, x0, _smaps(Nc, (Nx, Ny, Nz)), _random_masks(Nx, Ny, Nz, Nt, 0.35, 1))
    kw = dict(fn_ksp=fn, fn_smaps=fn, reg="wavelet-tv", niters=80, device=DEVICE,
              lamb_l1=1e-3, lamb_tv=1e-3, wave="db2", levels=1)
    free = run_sense(**kw, joint=True).X_recon
    ttv = run_sense(**kw, lamb_ttv=0.05).X_recon

    def fluct(x):
        return (x.abs().std(-1) / x.abs().mean(-1).clamp_min(1e-6)).median().item()

    def err(x):
        return ((x - x0).norm() / x0.norm()).item()

    assert fluct(ttv) < 0.5 * fluct(free)
    assert err(ttv) < err(free)


def test_hp_penalty_keeps_the_low_band_and_shrinks_the_high_band_exactly(tmp_path):
    """Fully sampled with unit-RSS smaps, A^H A = I, so CG with the penalty
    solves (I + mu P) x = x_true: components at or below the cutoff come back
    unchanged, components above it are divided by 1 + mu."""
    Nx, Ny, Nz, Nc, Nt, tr, mu = 6, 6, 4, 2, 40, 0.5, 3.0
    t = torch.arange(Nt, device=DEVICE) * tr
    C = dct_matrix(Nt, DEVICE)
    x0 = _crandn(Nx, Ny, Nz, 1, seed=7)
    low, high = C[4], C[30]  # 0.1 Hz and 0.75 Hz DCT components (k / 40 Hz)
    x_true = (x0 * (3 + low + high)).to(torch.complex64)
    fn = _write(tmp_path, x_true, _smaps(Nc, (Nx, Ny, Nz)), _random_masks(Nx, Ny, Nz, Nt, 1.0, 2))
    x = run_sense(fn_ksp=fn, fn_smaps=fn, reg="none", niters=100, device=DEVICE, cg_tol=1e-7,
                  hp_weight=mu, volume_tr_s=tr).X_recon
    coef = torch.einsum("xyzt,kt->xyzk", x / x0, C.to(torch.complex64))  # DCT along t
    torch.testing.assert_close(coef[..., 4].real, torch.ones_like(coef[..., 4].real), atol=2e-3,
                               rtol=0)
    torch.testing.assert_close(coef[..., 30].real, torch.full_like(coef[..., 30].real, 1 / (1 + mu)),
                               atol=2e-3, rtol=0)
    assert t[-1] > 0  # (time axis used only to document the frequencies above)


def test_cg_reg_normal_solves_the_regularized_normal_equations():
    Nx, Ny, Nz, Nc, Nt = 6, 6, 4, 2, 10
    A = build_sense(_smaps(Nc, (Nx, Ny, Nz)), _random_masks(Nx, Ny, Nz, Nt, 0.4, 3))
    y = A.apply(_crandn(Nx, Ny, Nz, Nt, seed=8))
    hp = TemporalHighPass(Nt, 0.5, 0.15, DEVICE)

    def Q(v):
        return 2.0 * hp.apply(v)

    X, res = cg(A, y, (Nx, Ny, Nz, Nt), num_iter=300, tol=1e-7, reg_normal=Q)
    lhs = A.adjoint(A.apply(X)) + Q(X)
    rhs = A.adjoint(y)
    assert ((lhs - rhs).norm() / rhs.norm()).item() < 1e-5


def test_mslr_and_joint_wavelet_tv_run_with_the_hp_penalty(tmp_path):
    Nx, Ny, Nz, Nc, Nt = 8, 8, 8, 2, 12
    x0 = _crandn(Nx, Ny, Nz, 1, seed=9).expand(-1, -1, -1, Nt).contiguous()
    fn = _write(tmp_path, x0, _smaps(Nc, (Nx, Ny, Nz)), _random_masks(Nx, Ny, Nz, Nt, 0.5, 4))
    r = run_sense(fn_ksp=fn, fn_smaps=fn, reg="mslr", niters=10, device=DEVICE,
                  patch_sizes=[(Nx, Ny, Nz), (4, 4, 4)], strides=[(Nx, Ny, Nz), (4, 4, 4)],
                  hp_weight=1.0, volume_tr_s=0.5)
    assert torch.isfinite(r.X_recon).all()
    assert r.dc_costs[-1] + r.reg_costs[-1] < r.dc_costs[0] + r.reg_costs[0]  # objective
    assert r.meta["hp_keep"] == 2  # k / (2 * 12 * 0.5) = k / 12 Hz <= 0.15: k = 0, 1
    r = run_sense(fn_ksp=fn, fn_smaps=fn, reg="wavelet-tv", niters=10, device=DEVICE,
                  lamb_l1=1e-3, lamb_tv=1e-3, wave="db2", levels=1, hp_weight=1.0,
                  lamb_ttv=1e-3, volume_tr_s=0.5)
    assert torch.isfinite(r.X_recon).all() and r.meta["joint"]


def test_run_sense_rejects_temporal_tv_outside_wavelet_tv(tmp_path):
    with pytest.raises(ValueError, match="lamb_ttv"):
        run_sense(fn_ksp="unused", fn_smaps="unused", reg="mslr", lamb_ttv=0.1)


def test_hp_penalty_as_a_dual_block_converges_faster_than_as_a_smooth_term():
    """Review item 262: as a smooth term the penalty's curvature cuts PDHG's
    primal step to 1/(1 + mu); as a dual block (SpatioTemporalWaveletTV's hp)
    it doesn't. Undersampled, mu = 30, 100 iterations: the dual form reaches a
    3x lower objective (323 vs 1097 when written)."""
    from recon.regularizers import SpatioTemporalWaveletTV
    from recon.solvers import pdhg

    Nx, Ny, Nz, Nc, Nt, mu, lam = 12, 12, 8, 2, 40, 30.0, 1e-3
    x_true = _crandn(Nx, Ny, Nz, 1, seed=11).expand(-1, -1, -1, Nt).contiguous()
    A = build_sense(_smaps(Nc, (Nx, Ny, Nz)), _random_masks(Nx, Ny, Nz, Nt, 0.15, 6))
    y = A.apply(x_true)
    y = y + 0.05 * _crandn(*y.shape, seed=12)
    hp = TemporalHighPass(Nt, 0.5, 0.15, DEVICE)
    shape = (Nx, Ny, Nz, Nt)
    g_dual = SpatioTemporalWaveletTV(shape, lam, lam, wave="db2", levels=1, hp=hp, hp_weight=mu)
    g_plain = SpatioTemporalWaveletTV(shape, lam, lam, wave="db2", levels=1)
    z = torch.zeros(shape, dtype=torch.complex64, device=DEVICE)

    def objective(x):
        return 0.5 * (A.apply(x) - y).norm().item() ** 2 + g_dual.cost(x)

    x_smooth = pdhg(lambda v: A.adjoint(A.apply(v) - y) + mu * hp.apply(v), 1 + mu,
                    g_plain.h_prox, g_plain.G, g_plain.G_norm_squared, z, niter=100)
    x_dual = pdhg(lambda v: A.adjoint(A.apply(v) - y), 1.0, g_dual.h_prox, g_dual.G,
                  g_dual.G_norm_squared, z, niter=100)
    assert objective(x_dual) < 0.5 * objective(x_smooth)
