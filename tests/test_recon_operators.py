"""Tests for recon/operators.py's SENSE, SENSE_B0 and SENSE_B0_R2star:
adjoint self-consistency, spectral norms, and the B0 operators' accuracy
against a brute-force, genuinely time-varying ground-truth forward model."""

import math
import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from mirtorch.linear.mri import mri_exp_approx  # noqa: E402

from recon.operators import (  # noqa: E402
    SENSE,
    SENSE_B0,
    SENSE_B0_R2star,
    build_sense,
    build_sense_b0,
    build_sense_b0_r2star,
    clip_b0_outliers,
)
from recon.utils import (  # noqa: E402
    _brute_force_time_varying_ksp,
    _build_operator,
    _setup_real_scale,
    check_operator_unitary,
    estimate_spectral_norm,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _complex_randn(*shape, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    real = torch.randn(*shape, generator=g, device=DEVICE)
    imag = torch.randn(*shape, generator=g, device=DEVICE)
    return (real + 1j * imag).to(torch.complex64)


def _setup(seed_offset=0, b0_max_hz=40.0, dt_echo=5e-5):
    """b0_max_hz/dt_echo default to a deliberately *small* total phase
    excursion across the echo train (max ~0.15 rad here) -- the regime
    static (single-segment) correction is actually designed for, and where
    it should nearly eliminate the forward-model error, making a sign error
    unambiguous. This repo's real acquisitions are nowhere near this gentle
    (B0 maps up to +-300-350 Hz over a ~70ms, ETL=60 echo train -- tens of
    radians of phase drift, not a fraction of one), where static correction
    provides much weaker benefit -- see test_realistic_regime_only_partly_
    corrects below, which uses those real numbers directly and checks the
    (much weaker) partial-correction claim instead."""
    Nx, Ny, Nz, Nc = 8, 12, 6, 3
    img = _complex_randn(Nx, Ny, Nz, seed=10 + seed_offset)
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=11 + seed_offset)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-6)

    # A smooth, non-trivial field map (a ramp along y plus a bit of curvature).
    yy = torch.linspace(-1, 1, Ny, device=DEVICE).reshape(1, Ny, 1)
    zz = torch.linspace(-1, 1, Nz, device=DEVICE).reshape(1, 1, Nz)
    b0_map = (b0_max_hz * yy + 0.4 * b0_max_hz * zz**2).expand(Nx, Ny, Nz).contiguous()

    # Per-echo (per-ky) acquisition time: uniform spacing, matching
    # sequences/ArbEPI.py's echo_times model (one gro-duration step per
    # echo). Centered so the mean is a clean "nominal TE".
    te = 0.030  # s
    t_per_ky = te + (torch.arange(Ny, device=DEVICE, dtype=torch.float32) - (Ny - 1) / 2) * dt_echo

    y_true = _brute_force_time_varying_ksp(img, smaps, b0_map, t_per_ky)
    y_true_flat = y_true.reshape(Nc, -1).T  # (K,Nc), C-order -- matches SENSE's own flatten

    return img, smaps, b0_map, te, y_true_flat


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

    A = build_sense(smaps, omega)
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
    A = build_sense(smaps, full_mask.unsqueeze(-1))

    x = _complex_randn(Nx, Ny, Nz, 1, seed=4)
    x = x / x.norm()
    for _ in range(30):
        x = A.adjoint(A.apply(x))
        x = x / x.norm()
    sigma1 = A.apply(x).norm().item()
    assert abs(sigma1 - 1.0) < 1e-3


def _build_b0_operator(smaps, samp, b0_map, t_frame_s, L, nbins=20, fit_t_ms=None):
    """One frame's SENSE_B0, built by hand. The segmentation is fit on
    fit_t_ms if given (e.g. a set of distinct echo times, as build_sense_b0
    does), else on every sampled location's own time."""
    idx = torch.nonzero(samp.reshape(-1), as_tuple=False).squeeze(-1)
    t_ms = (t_frame_s.reshape(-1)[idx] * 1000).to(torch.float32)
    b0_neg = (-b0_map).to(torch.float32)
    b, c, _tl = mri_exp_approx(b0_neg, nbins, L, t_ms if fit_t_ms is None else fit_t_ms)
    N = tuple(smaps.shape[1:])
    c = c.transpose(0, 1).reshape((L,) + N).to(smaps.dtype)
    if fit_t_ms is None:
        # b is already one row per sampled location -- pos=arange is the
        # identity gather, recovering SENSE_B0's old one-tensor-per-
        # instance behavior exactly (see its docstring).
        pos = torch.arange(b.shape[0], device=b.device)
    else:
        pos = torch.searchsorted(fit_t_ms, t_ms)
    return SENSE_B0(smaps, samp, pos, b.to(smaps.dtype), c)


def test_l1_matches_static_correction():
    """L=1 (a single time segment centered at the mean sample time) should
    match a static single-segment correction (one exp(+i 2 pi df te) phasor
    baked into smaps) closely
    -- both are, in the end, one global per-voxel phase term applied before
    the FFT; this is the connective-tissue check between the two stages."""

    img, smaps, b0_map, te, y_true_flat = _setup(seed_offset=30)
    Nx, Ny, Nz = smaps.shape[1:]
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)
    t_frame = torch.full((Nx, Ny, Nz), te, device=DEVICE)  # only the mean/te matters at L=1

    static_phasor = torch.exp(1j * (2 * math.pi * te) * b0_map.to(torch.float32)).to(smaps.dtype)
    A_static = SENSE(smaps * static_phasor.unsqueeze(0), full_mask)
    y_static = A_static.apply(img)

    A_l1 = _build_b0_operator(smaps, full_mask, b0_map, t_frame, L=1)
    y_l1 = A_l1.apply(img)

    rel_diff = (y_l1 - y_static).norm().item() / y_static.norm().item()
    assert rel_diff < 1e-3, f"L=1 should match static correction closely, got rel_diff={rel_diff}"


def test_more_segments_reduces_error_in_a_toy_grid():
    """NOT the realistic regime, despite the fixture's B0/ETL parameters
    looking real -- this grid has only 12 distinct echo times, so L>=12
    trivially resolves every one exactly (see the L16 assertion below).
    recon/utils.py's own module docstring documents this
    explicitly: its finding "says nothing about whether L=6 ... is
    adequate at the real ETL=60 scale." See
    test_more_segments_reduces_error_at_real_scale below for the actual
    realistic-regime check, which reuses that script's real-scale ground
    truth directly. Kept as a cheap, fast sanity check that more segments
    monotonically help at all -- not a stand-in for the real-scale test."""
    img, smaps, b0_map, te, y_true_flat = _setup(seed_offset=31, b0_max_hz=350.0, dt_echo=0.0012)
    Nx, Ny, Nz = smaps.shape[1:]
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)

    # Reconstruct each ky-row's actual acquisition time onto the full grid,
    # matching _setup()'s own t_per_ky construction.
    Nyv = Ny
    dt_echo = 0.0012
    ky_idx = torch.arange(Nyv, device=DEVICE, dtype=torch.float32)
    t_per_ky = te + (ky_idx - (Nyv - 1) / 2) * dt_echo
    t_frame = t_per_ky.reshape(1, Nyv, 1).expand(Nx, Nyv, Nz).contiguous()

    def err(L):
        A = _build_b0_operator(smaps, full_mask, b0_map, t_frame, L=L)
        y_hat = A.apply(img)
        return (y_hat - y_true_flat).norm().item() / y_true_flat.norm().item()


    err_none = (SENSE(smaps, full_mask).apply(img) - y_true_flat).norm().item() \
        / y_true_flat.norm().item()
    err_l1 = err(1)
    err_l8 = err(8)
    err_l16 = err(16)

    print(f"uncorrected={err_none:.4f} L1={err_l1:.4f} L8={err_l8:.4f} L16={err_l16:.4f}")
    # Empirically (this fixture): uncorrected~1.46, L1~0.96, L8~0.58, L16~0.00
    # (L16 >= the 12 distinct ky times in this small synthetic grid, so it's
    # essentially exact) -- monotonically, substantially decreasing with L,
    # nowhere near static (L1)'s weak ~5% real-data improvement.
    assert err_l1 < err_none, "even L=1 should help at all vs no correction"
    assert err_l8 < 0.7 * err_l1, (
        f"L=8 should meaningfully beat L=1 (err_l1={err_l1}, err_l8={err_l8})"
    )
    assert err_l16 < 0.1 * err_l8, (
        f"L=16 (>= distinct ky times) should be ~exact (err_l16={err_l16})"
    )


def test_more_segments_reduces_error_at_real_scale():
    """The actual realistic-regime check (real ETL=60, real field-map
    range -300 to +70 Hz), regression-guarding the conclusion
    recon/utils.py's real-scale sweep found: L=32 (the
    production default, item 82) keeps relative forward-model error under
    1%, while L=6 (mirtorch's own Gmri default, no longer used here)
    doesn't come close -- a sharp, Nyquist-like phase transition around
    L=27-32 (matching this scale's bandwidth-time product BT ~= 27), not a
    gradual curve. Reuses analysis.py's own real-scale ground
    truth/operator-construction helpers directly, rather than a third copy
    of them."""

    img, smaps, b0_map, t_per_ky, y_true_flat = _setup_real_scale(seed=200)
    Nx, Ny, Nz = smaps.shape[1:]
    t_frame = t_per_ky.reshape(1, Ny, 1).expand(Nx, Ny, Nz).contiguous()

    def err(L, nbins=128):
        A = _build_operator(smaps, b0_map, t_frame, L=L, nbins=nbins)
        y_hat = A.apply(img)
        return (y_hat - y_true_flat).norm().item() / y_true_flat.norm().item()

    err_l6 = err(6)
    err_l32 = err(32)
    print(f"real-scale forward-model error: L=6 {err_l6:.4f}, L=32 {err_l32:.4f}")
    assert err_l32 < 0.01, (
        f"L=32 should keep relative forward-model error under 1% (got {err_l32:.4f})"
    )
    assert err_l6 > 5 * err_l32, (
        f"L=6 should be far worse than L=32 at this scale (err_l6={err_l6:.4f}, "
        f"err_l32={err_l32:.4f}) -- if not, the sweep's L=32 conclusion needs revisiting"
    )


def test_production_nbins_avoids_row_sum_warning(recwarn):
    """operators.py's own docstring identifies nbins=20 (mirtorch's
    Gmri default, used throughout this file's other tests) as the actual
    root cause of a real signal-loss-plus-incoherent-noise failure --
    confirm the production default (nbins=128) doesn't trip
    _check_b_weight_row_sums' ill-conditioning warning, at real scale."""

    _img, smaps, b0_map, t_per_ky, _y_true_flat = _setup_real_scale(seed=201)
    Nx, Ny, Nz = smaps.shape[1:]
    Nt = 1
    omega = torch.ones(Nx, Ny, Nz, Nt, dtype=torch.bool, device=smaps.device)
    echo_times_yz = t_per_ky.reshape(Ny, 1, 1).expand(Ny, Nz, Nt).contiguous()

    build_sense_b0(smaps, omega, b0_map, echo_times_yz, L=32, nbins=128)

    row_sum_warnings = [w for w in recwarn.list if "b_weights row sums" in str(w.message)]
    assert not row_sum_warnings, (
        f"production nbins=128 should not trigger the ill-conditioning warning, got: "
        f"{[str(w.message) for w in row_sum_warnings]}"
    )


def test_adjoint_is_self_consistent():
    """Mirrors tests/test_recon_operators.py's adjoint check for the plain
    SENSE, extended to the time-segmented operator."""
    Nx, Ny, Nz, Nc, L = 6, 7, 5, 3, 4
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=40)
    samp = torch.rand(Nx, Ny, Nz, device=DEVICE) > 0.5
    b0_map = _complex_randn(Nx, Ny, Nz, seed=41).real * 150
    t_frame = _complex_randn(Nx, Ny, Nz, seed=42).real.abs() * 0.05 + 0.01

    A = _build_b0_operator(smaps, samp, b0_map, t_frame, L=L)
    K = A.idx.numel()

    x = _complex_randn(Nx, Ny, Nz, seed=43)
    y = _complex_randn(K, Nc, seed=44)

    lhs = torch.vdot(A.apply(x).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), A.adjoint(y).reshape(-1))
    assert abs(lhs - rhs).item() / abs(lhs).item() < 1e-4


def test_build_encoding_operator_b0_matches_manual_per_frame_construction():
    """build_sense_b0's BlockDiagonal-of-SENSE_B0 should
    apply identically to manually building each frame's operator the way
    _build_b0_operator does above, confirming the gather plumbing
    (echo_times_yz's idx % (Ny*Nz) lookup, per-frame idx) is wired
    correctly end to end. echo_times_s (Nx-expanded) is built here only
    for _build_b0_operator's own per-frame (Nx,Ny,Nz)-shaped contract, not
    passed to build_sense_b0 itself (see its docstring)."""
    Nx, Ny, Nz, Nc, Nt, L = 5, 6, 4, 2, 3, 3
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=50)
    b0_map = _complex_randn(Nx, Ny, Nz, seed=51).real * 100

    g = torch.Generator(device=DEVICE).manual_seed(52)
    omega = torch.stack(
        [torch.rand(Nx, Ny, Nz, generator=g, device=DEVICE) > 0.4 for _ in range(Nt)], dim=-1
    )
    counts = omega.sum(dim=(0, 1, 2))
    k = counts.min().item()
    omega = omega & (torch.cumsum(omega.reshape(-1, Nt), dim=0) <= k).reshape(Nx, Ny, Nz, Nt)

    # build_sense_b0 fits one segmentation on all frames' distinct echo
    # times. Tie each (iy,iz) to a value from a small fixed pool, identical
    # across every frame, so that set is small and every frame samples all
    # of it (as on real acquisitions, where it is one time per echo index).
    n_distinct = 5
    distinct_times = torch.linspace(0.005, 0.055, n_distinct, device=DEVICE)
    yz_idx = (
        torch.arange(Ny, device=DEVICE).reshape(Ny, 1) * Nz
        + torch.arange(Nz, device=DEVICE).reshape(1, Nz)
    ) % n_distinct
    echo_times_yz = distinct_times[yz_idx]  # (Ny,Nz)
    echo_times_2d = echo_times_yz.unsqueeze(-1).expand(Ny, Nz, Nt).contiguous()  # (Ny,Nz,Nt)
    echo_times_s = echo_times_2d.unsqueeze(0).expand(Nx, -1, -1, -1).contiguous()

    A = build_sense_b0(smaps, omega, b0_map, echo_times_2d, L=L, nbins=10)

    x = _complex_randn(Nx, Ny, Nz, Nt, seed=53)
    y_batched = A.apply(x)

    # build_sense_b0 fits the segmentation once, on all frames' distinct echo
    # times (frame 0's here, which samples every one of them); fit the
    # hand-built reference on the same times so this checks
    # the per-frame gather plumbing, not two different fits. (mri_exp_approx
    # places segments at percentiles of the times it's given, so fitting on
    # per-sample times with repeats would give a different, mask-dependent fit.)
    idx0 = torch.nonzero(omega[..., 0].reshape(-1)).squeeze(-1)
    fit_t_ms = torch.unique((echo_times_s[..., 0].reshape(-1)[idx0] * 1000).to(torch.float32))
    for it in range(Nt):
        A_manual = _build_b0_operator(
            smaps, omega[..., it], b0_map, echo_times_s[..., it], L=L, nbins=10,
            fit_t_ms=fit_t_ms,
        )
        y_manual = A_manual.apply(x[..., it])
        torch.testing.assert_close(y_batched[..., it], y_manual, atol=1e-5, rtol=1e-4)


def test_clip_b0_outliers_keeps_small_maps_and_clips_rare_extremes():
    small = torch.tensor([-5.0, 0.0, 5.0], device=DEVICE)
    torch.testing.assert_close(clip_b0_outliers(small), small)  # order statistics: no clip
    g = torch.Generator(device=DEVICE).manual_seed(7)
    b0 = torch.randn(4000, generator=g, device=DEVICE) * 50
    b0[3], b0[99] = -4e6, 1e5
    out = clip_b0_outliers(b0)
    assert -400 < out.min() and out.max() < 400
    assert (out != b0).sum() <= 10  # only the most extreme few move
    assert clip_b0_outliers(b0, 0) is b0


def test_build_sense_b0_is_robust_to_a_few_diverged_voxels():
    """Two diverged voxels (review item 259: -4 MHz in a +-100 Hz map) widen
    mri_exp_approx's histogram so far that the segmentation fails everywhere;
    with the default percentile clip the operator matches the clean map's."""
    Nx, Ny, Nz, Nc, Nt, L = 16, 16, 12, 2, 1, 8
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=70)
    yy = torch.linspace(-1, 1, Ny, device=DEVICE).reshape(1, Ny, 1)
    b0 = (100.0 * yy).expand(Nx, Ny, Nz).contiguous()
    b0_bad = b0.clone()
    b0_bad[0, 0, 0], b0_bad[1, 0, 0] = -4e6, 1e5
    g = torch.Generator(device=DEVICE).manual_seed(71)
    omega = (torch.rand(Nx, Ny, Nz, generator=g, device=DEVICE) > 0.5).unsqueeze(-1)
    t = 0.02 + 0.001 * torch.arange(Ny, device=DEVICE, dtype=torch.float32)
    echo_times = t.reshape(Ny, 1, 1).expand(Ny, Nz, Nt).contiguous()
    x = _complex_randn(Nx, Ny, Nz, Nt, seed=72)
    x[0, 0, 0] = x[1, 0, 0] = 0  # the diverged voxels' own model doesn't matter here

    def fwd(b, pct):
        return build_sense_b0(smaps, omega, b, echo_times, L=L, nbins=64,
                              b0_clip_percentile=pct).apply(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the unclipped fit trips the row-sum check
        y_clean, y_clip, y_raw = fwd(b0, 0), fwd(b0_bad, 0.1), fwd(b0_bad, 0)
    rel = lambda a: ((a - y_clean).norm() / y_clean.norm()).item()  # noqa: E731
    assert rel(y_clip) < 0.05
    assert rel(y_raw) > 0.3


def test_r2star_zero_map_matches_phase_only_operator():
    """build_sense_b0_r2star with an all-zero R2* map should reproduce
    build_sense_b0's c_phasors exactly (up to float32 rounding). The two are
    built by genuinely different code paths -- None takes c_phasors
    straight from mri_exp_approx's own spatial output, r2star_map=zeros
    goes through this module's manual `torch.exp(tl * psi)` construction
    -- so this is a real cross-check of that construction against the
    library's own convention, not a tautology. Locks in operators.py's
    stated contract that r2star_map=None is a strict special case of
    r2star_map=0."""
    Nx, Ny, Nz, Nc, Nt, L = 5, 6, 4, 2, 3, 3
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=70)
    b0_map = _complex_randn(Nx, Ny, Nz, seed=71).real * 100

    g = torch.Generator(device=DEVICE).manual_seed(52)
    omega = torch.stack(
        [torch.rand(Nx, Ny, Nz, generator=g, device=DEVICE) > 0.4 for _ in range(Nt)], dim=-1
    )
    counts = omega.sum(dim=(0, 1, 2))
    k = counts.min().item()
    omega = omega & (torch.cumsum(omega.reshape(-1, Nt), dim=0) <= k).reshape(Nx, Ny, Nz, Nt)

    n_distinct = 5
    distinct_times = torch.linspace(0.005, 0.055, n_distinct, device=DEVICE)
    yz_idx = (
        torch.arange(Ny, device=DEVICE).reshape(Ny, 1) * Nz
        + torch.arange(Nz, device=DEVICE).reshape(1, Nz)
    ) % n_distinct
    echo_times_yz = distinct_times[yz_idx]  # (Ny,Nz)
    echo_times_2d = echo_times_yz.unsqueeze(-1).expand(Ny, Nz, Nt).contiguous()

    A_none = build_sense_b0(smaps, omega, b0_map, echo_times_2d, L=L, nbins=10)
    r2star_zero = torch.zeros(Nx, Ny, Nz, device=DEVICE)
    A_zero = build_sense_b0_r2star(
        smaps, omega, b0_map, echo_times_2d, r2star_zero, 0.0, L=L, nbins=10,
    )
    assert isinstance(A_zero.A[0], SENSE_B0_R2star)

    x = _complex_randn(Nx, Ny, Nz, Nt, seed=73)
    torch.testing.assert_close(A_none.apply(x), A_zero.apply(x), atol=1e-5, rtol=1e-4)


def test_r2star_generalization_adjoint_is_self_consistent():
    """The check that discriminates this module's PHYSICAL sign
    (psi = i*2*pi*Δf(r) - R2*(r), decaying in the forward direction) from
    recon/lowres_calib.py's flipped sign on the
    (unmerged) worktree-lowres-calib-recon branch (psi_recon =
    i*2*pi*Δf(r) + R2*(r)): the true adjoint identity <Ax,y> == <x,A^H y>
    holds for ANY complex c_phasors under SENSE_B0's `.conj()`
    adjoint -- including this module's decaying psi -- but fails for the
    branch's flipped-sign convenience (which is deliberately not a true
    adjoint; it's a matched-filter decay-compensation trick valid only in
    an adjoint-only, non-iterative reconstruction). Mirrors
    test_adjoint_is_self_consistent above, generalized to a nonzero R2*
    map, so a future change that accidentally reintroduces the flipped
    sign here fails loudly."""
    Nx, Ny, Nz, Nc, L = 6, 7, 5, 3, 4
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=80)
    samp = torch.rand(Nx, Ny, Nz, device=DEVICE) > 0.5
    b0_map = _complex_randn(Nx, Ny, Nz, seed=81).real * 150
    r2star_hz = _complex_randn(Nx, Ny, Nz, seed=82).real.abs() * 40  # 1/s, physically plausible
    t_frame = _complex_randn(Nx, Ny, Nz, seed=83).real.abs() * 0.05 + 0.01

    idx = torch.nonzero(samp.reshape(-1), as_tuple=False).squeeze(-1)
    t_ms = (t_frame.reshape(-1)[idx] * 1000).to(torch.float32)
    b0_neg = (-b0_map).to(torch.float32)
    b_by_echo, _c_phase_only, tl = mri_exp_approx(b0_neg, 20, L, t_ms)

    N = (Nx, Ny, Nz)
    psi = 1j * 2 * math.pi * b0_map.to(torch.complex64) - r2star_hz.to(torch.complex64)
    tl_c = tl.to(torch.complex64)
    c_phasors = torch.exp(tl_c.reshape((L,) + (1,) * len(N)) * psi[None, ...]).to(smaps.dtype)
    pos = torch.arange(b_by_echo.shape[0], device=DEVICE)
    A = SENSE_B0(smaps, samp, pos, b_by_echo.to(smaps.dtype), c_phasors)
    K = A.idx.numel()

    x = _complex_randn(Nx, Ny, Nz, seed=84)
    y = _complex_randn(K, Nc, seed=85)

    lhs = torch.vdot(A.apply(x).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), A.adjoint(y).reshape(-1))
    assert abs(lhs - rhs).item() / abs(lhs).item() < 1e-4


def test_r2star_forward_model_decays_away_from_reference_time():
    """Physical sanity check on the sign, independent of the adjoint
    identity above: c_phasors' MAGNITUDE (the amplitude term any given
    time-segment applies) must be a strictly DECREASING function of that
    segment's time tl[l] -- exactly exp(-R2*(r)*(tl[l] - t_ref_s)) for a
    uniform R2*(r) -- not bounded above by 1: this test's echo times span
    both sides of t_ref_s (as a real acquisition's do -- some echoes fall
    before the nominal-TE echo, some after), and a segment with tl[l] <
    t_ref_s legitimately has |c_phasors|>1 (LESS decay than the reference
    echo, not growth -- an earlier version of this test wrongly asserted
    |c_phasors|<=1 everywhere and failed on exactly this). What the sign
    must never do is flip that direction: a flipped sign
    (recon/lowres_calib.py's branch convention) would make
    |c_phasors| INCREASE with tl[l] instead of decrease (see the module
    docstring's measured tSNR-gets-worse regression).

    Checked directly on |c_phasors| via build_sense_b0's own
    construction, rather than round-tripping through a forward FFT+gather
    simulation and comparing k-space-subset energies: a first attempt at
    that end-to-end approach gave a wildly wrong ratio, traced to a real
    confound, not a bug here -- with a spatially-varying Δf(r), the
    per-voxel PHASE term (which is exactly unit-magnitude and therefore
    energy-preserving on its own) still redistributes energy across
    k-space bins via the FFT, so a biased k-space *subset*'s energy ratio
    reflects that redistribution, not just the R2* attenuation. Checking
    |c_phasors| directly sidesteps that confound entirely."""
    Nx, Ny, Nz, Nc, L = 6, 7, 5, 3, 8
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=90)
    b0_map = _complex_randn(Nx, Ny, Nz, seed=92).real * 150  # realistic-scale Δf, Hz
    r2_uniform = 50.0  # 1/s
    r2star_hz = torch.full((Nx, Ny, Nz), r2_uniform, device=DEVICE)

    t_ref_s = 0.030
    Nt = 1
    omega = torch.ones(Nx, Ny, Nz, Nt, dtype=torch.bool, device=DEVICE)
    # A handful of distinct echo times spanning +-20ms around t_ref_s.
    n_distinct = 6
    distinct_times = t_ref_s + torch.linspace(-0.020, 0.020, n_distinct, device=DEVICE)
    yz_idx = (
        torch.arange(Ny, device=DEVICE).reshape(Ny, 1) * Nz
        + torch.arange(Nz, device=DEVICE).reshape(1, Nz)
    ) % n_distinct
    echo_times_yz = distinct_times[yz_idx].unsqueeze(-1)  # (Ny,Nz,Nt)

    A = build_sense_b0_r2star(
        smaps, omega, b0_map, echo_times_yz, r2star_hz, t_ref_s, L=L, nbins=20,
    )
    c_phasors = A.A[0].c_phasors  # (L,Nx,Ny,Nz)

    # |c_phasors[l]| should be spatially uniform (r2star_hz is) and equal
    # exp(-r2_uniform * |tl[l] - t_ref_s|) -- checked via the spread/bound
    # below rather than re-deriving tl (build_sense_b0 doesn't
    # return it).
    mag = c_phasors.abs()
    per_segment_mag = mag.reshape(L, -1)
    uniform_ref = per_segment_mag[:, :1].expand_as(per_segment_mag)
    assert torch.allclose(per_segment_mag, uniform_ref, atol=1e-5), (
        "c_phasors magnitude should be spatially uniform when r2star_map is spatially uniform"
    )
    # mri_exp_approx places tl via non-decreasing percentiles of the fit
    # times (see its own source), so segment l's magnitude -- returned in
    # that same l-order -- should be non-increasing across l.
    segment_decay = per_segment_mag[:, 0].cpu()
    assert torch.all(segment_decay[:-1] >= segment_decay[1:] - 1e-5), (
        f"|c_phasors| should be non-increasing as segment time increases: {segment_decay.tolist()}"
    )
    # tl's first/last segments land exactly at the min/max fit time (the
    # percentile fractions span [0,1] inclusive), so the ratio should match
    # exp(R2* * full time span) exactly -- here +-20ms around t_ref_s, a 40ms span.
    expected_ratio = math.exp(r2_uniform * 0.040)
    ratio = (segment_decay.max() / segment_decay.min()).item()
    assert abs(ratio - expected_ratio) / expected_ratio < 1e-3, (
        f"max/min decay ratio {ratio:.4f} should match exp(R2*span)={expected_ratio:.4f}"
    )


def test_estimate_spectral_norm_matches_full_sampling_unity_case():
    """Regression check for the refactor into a standalone helper: L=1,
    unit b_weights/c_phasors, full sampling should reduce to
    tests/test_recon_operators.py's own near-unity check for plain
    SENSE with normalized smaps."""
    Nx, Ny, Nz, Nc = 8, 8, 6, 4
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=60)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)

    b_weights = torch.ones(Nx * Ny * Nz, 1, dtype=torch.complex64, device=DEVICE)
    pos = torch.arange(Nx * Ny * Nz, device=DEVICE)
    c_phasors = torch.ones(1, Nx, Ny, Nz, dtype=torch.complex64, device=DEVICE)
    A = SENSE_B0(smaps, full_mask, pos, b_weights, c_phasors)

    x0 = _complex_randn(Nx, Ny, Nz, seed=61)
    sigma1 = estimate_spectral_norm(A, x0, niter=30)
    assert abs(sigma1 - 1.0) < 1e-3


def test_check_operator_unitary_silent_for_trivial_l1_case():
    """The L=1/unit-weights case above is genuinely unitary -- no warning."""
    Nx, Ny, Nz, Nc = 8, 8, 6, 4
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=62)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)
    b_weights = torch.ones(Nx * Ny * Nz, 1, dtype=torch.complex64, device=DEVICE)
    pos = torch.arange(Nx * Ny * Nz, device=DEVICE)
    c_phasors = torch.ones(1, Nx, Ny, Nz, dtype=torch.complex64, device=DEVICE)
    A = SENSE_B0(smaps, full_mask, pos, b_weights, c_phasors)

    x0 = _complex_randn(Nx, Ny, Nz, seed=63)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sigma1 = check_operator_unitary(A, x0, niter=30)
    assert abs(sigma1 - 1.0) < 1e-3


def test_check_operator_unitary_warns_for_real_b0_correction():
    """A genuine (L>1, real field map) B0-corrected operator is not
    guaranteed unitary -- regression guard for the 2026-09-18 finding that
    real SENSE_B0 operators measure sigma1 ~= 1.3, not ~1.0 (see
    operators.py's check_operator_unitary docstring for the real-data
    debugging session this documents). Doesn't hardcode that exact value
    (a different seed/field map would give a different number), just that
    it's measurably away from 1.0 and that check_operator_unitary catches
    it -- a future change that made this operator closer to unitary would
    be fine; one that made it silently *worse* without a warning would not."""
    Nx, Ny, Nz, Nc, L = 8, 8, 6, 4, 6
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=70)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)
    b0_map = _complex_randn(Nx, Ny, Nz, seed=71).real * 200  # real-scale-ish field map, Hz
    t_frame = _complex_randn(Nx, Ny, Nz, seed=72).real.abs() * 0.05 + 0.01  # seconds

    A = _build_b0_operator(smaps, full_mask, b0_map, t_frame, L=L, nbins=40)
    x0 = _complex_randn(Nx, Ny, Nz, seed=73)

    with pytest.warns(UserWarning, match="not unitary"):
        sigma1 = check_operator_unitary(A, x0, niter=100)
    assert abs(sigma1 - 1.0) > 0.05


def test_segment_fit_covers_frames_that_sample_different_echo_times():
    """A frame may sample echo times another frame doesn't (recon/testbed.py
    applies one run's masks to another run's timing): the shared fit covers
    the union, and every sample maps to its own time."""
    from recon.operators import _segment_fit

    Nx, Ny, Nz, Nt = 4, 6, 5, 3
    times = 0.005 + 0.002 * torch.arange(Ny * Nz, device=DEVICE, dtype=torch.float32)
    et = times.reshape(Ny, Nz, 1).expand(Ny, Nz, Nt).contiguous()  # every (ky,kz) distinct
    omega = torch.zeros(Nx, Ny, Nz, Nt, dtype=torch.bool, device=DEVICE)
    omega[:, :3, :, 0] = True  # frame 0: ky 0-2 only
    omega[:, 3:, :, 1] = True  # frame 1: ky 3-5, none of frame 0's times
    omega[:, ::2, :, 2] = True
    b0 = torch.linspace(-30, 30, Nx * Ny * Nz, device=DEVICE).reshape(Nx, Ny, Nz)
    _, _, _, unique_t_ms, pos = _segment_fit(omega, b0, et, L=4, nbins=16)
    assert unique_t_ms.numel() == Ny * Nz
    for it in range(Nt):
        idx = torch.nonzero(omega[..., it].reshape(-1)).squeeze(-1) % (Ny * Nz)
        torch.testing.assert_close(unique_t_ms[pos[it]], times[idx] * 1000, atol=1e-3, rtol=0)


def test_build_sense_b0_shares_segment_tensors_across_frames():
    """Item 163: an earlier version built an independent (L,*N) c_phasors per
    frame, enough redundancy to OOM a real reconstruction. Every frame must
    hold the *same* tensors (shared storage), not per-frame copies."""
    Nx, Ny, Nz, Nc, Nt, L = 5, 6, 4, 2, 3, 3
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=60)
    b0_map = _complex_randn(Nx, Ny, Nz, seed=61).real * 100
    omega = torch.ones(Nx, Ny, Nz, Nt, dtype=torch.bool, device=DEVICE)
    t_y = torch.linspace(0.005, 0.03, Ny, device=DEVICE).reshape(Ny, 1, 1)
    times = t_y.expand(Ny, Nz, Nt).contiguous()

    A = build_sense_b0(smaps, omega, b0_map, times, L=L, nbins=10)
    assert len(A.A) == Nt
    for f in A.A[1:]:
        assert f.c_phasors.data_ptr() == A.A[0].c_phasors.data_ptr()
        assert f.b_by_echo.data_ptr() == A.A[0].b_by_echo.data_ptr()

    A2 = build_sense_b0_r2star(
        smaps, omega, b0_map, times, torch.full((Nx, Ny, Nz), 20.0, device=DEVICE),
        t_ref_s=0.015, L=L, nbins=10,
    )
    for f in A2.A[1:]:
        assert f.c_phasors.data_ptr() == A2.A[0].c_phasors.data_ptr()
        assert f.b_by_echo.data_ptr() == A2.A[0].b_by_echo.data_ptr()


def test_coarse_nbins_triggers_row_sum_warning():
    """Item 164: the positive case of _check_b_weight_row_sums. A toy field
    map with a wide range and few histogram bins gives an ill-conditioned
    segmentation fit (row sums far from 1). (The negative case at production
    nbins is test_production_nbins_avoids_row_sum_warning; the
    _setup_real_scale map does not reproduce the failure even at nbins=20.)"""
    Nx, Ny, Nz, Nc, L = 5, 6, 4, 2, 3
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=60)
    b0_map = _complex_randn(Nx, Ny, Nz, seed=61).real * 100
    omega = torch.ones(Nx, Ny, Nz, 1, dtype=torch.bool, device=DEVICE)
    t_y = torch.linspace(0.005, 0.03, Ny, device=DEVICE).reshape(Ny, 1, 1)
    times = t_y.expand(Ny, Nz, 1).contiguous()
    with pytest.warns(UserWarning, match="b_weights row sums"):
        build_sense_b0(smaps, omega, b0_map, times, L=L, nbins=10)
