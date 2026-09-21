"""Validates recon/operators.py's GatheredSenseB0/build_encoding_operator_b0
against the same brute-force, genuinely time-varying ground-truth forward
model tests/test_recon_b0_correction.py uses for the static (single-segment)
stage -- reused directly here (not reimplemented) so both stages are held to
the exact same ground truth."""

import math
import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from mirtorch.linear.mri import mri_exp_approx  # noqa: E402

from recon.operators import (  # noqa: E402
    GatheredSenseB0,
    build_encoding_operator_b0,
    check_operator_unitary,
    estimate_spectral_norm,
)
from tests.test_recon_b0_correction import DEVICE, _complex_randn, _setup  # noqa: E402


def _build_b0_operator(smaps, samp, b0map_hz, t_frame_s, L, nbins=20):
    idx = torch.nonzero(samp.reshape(-1), as_tuple=False).squeeze(-1)
    t_ms = (t_frame_s.reshape(-1)[idx] * 1000).to(torch.float32)
    b0_neg = (-b0map_hz).to(torch.float32)
    b, c, _tl = mri_exp_approx(b0_neg, nbins, L, t_ms)
    N = tuple(smaps.shape[1:])
    c = c.transpose(0, 1).reshape((L,) + N).to(smaps.dtype)
    # b is already one row per sampled location -- pos=arange is the
    # identity gather, recovering GatheredSenseB0's old one-tensor-per-
    # instance behavior exactly (see its docstring).
    pos = torch.arange(b.shape[0], device=b.device)
    return GatheredSenseB0(smaps, samp, pos, b.to(smaps.dtype), c)


def test_l1_matches_static_correction():
    """L=1 (a single time segment centered at the mean sample time) should
    match recon/operators.py's static single-segment correction closely
    -- both are, in the end, one global per-voxel phase term applied before
    the FFT; this is the connective-tissue check between the two stages."""
    from recon.operators import GatheredSense, demodulate_smaps

    img, smaps, b0map_hz, te, y_true_flat = _setup(seed_offset=30)
    Nx, Ny, Nz = smaps.shape[1:]
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)
    t_frame = torch.full((Nx, Ny, Nz), te, device=DEVICE)  # only the mean/te matters at L=1

    A_static = GatheredSense(demodulate_smaps(smaps, b0map_hz, te), full_mask)
    y_static = A_static.apply(img)

    A_l1 = _build_b0_operator(smaps, full_mask, b0map_hz, t_frame, L=1)
    y_l1 = A_l1.apply(img)

    rel_diff = (y_l1 - y_static).norm().item() / y_static.norm().item()
    assert rel_diff < 1e-3, f"L=1 should match static correction closely, got rel_diff={rel_diff}"


def test_more_segments_reduces_error_in_a_toy_grid():
    """NOT the realistic regime, despite the fixture's B0/ETL parameters
    looking real -- this grid has only 12 distinct echo times, so L>=12
    trivially resolves every one exactly (see the L16 assertion below).
    recon/analysis.py's own module docstring documents this
    explicitly: its finding "says nothing about whether L=6 ... is
    adequate at the real ETL=60 scale." See
    test_more_segments_reduces_error_at_real_scale below for the actual
    realistic-regime check, which reuses that script's real-scale ground
    truth directly. Kept as a cheap, fast sanity check that more segments
    monotonically help at all -- not a stand-in for the real-scale test."""
    img, smaps, b0map_hz, te, y_true_flat = _setup(seed_offset=31, b0_max_hz=350.0, dt_echo=0.0012)
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
        A = _build_b0_operator(smaps, full_mask, b0map_hz, t_frame, L=L)
        y_hat = A.apply(img)
        return (y_hat - y_true_flat).norm().item() / y_true_flat.norm().item()

    from recon.operators import GatheredSense

    err_none = (GatheredSense(smaps, full_mask).apply(img) - y_true_flat).norm().item() \
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
    recon/analysis.py's real-scale sweep found: L=32 (the
    production default, item 82) keeps relative forward-model error under
    1%, while L=6 (mirtorch's own Gmri default, no longer used here)
    doesn't come close -- a sharp, Nyquist-like phase transition around
    L=27-32 (matching this scale's bandwidth-time product BT ~= 27), not a
    gradual curve. Reuses analysis.py's own real-scale ground
    truth/operator-construction helpers directly, rather than a third copy
    of them."""
    from recon.analysis import _build_operator, _setup_real_scale

    img, smaps, b0map_hz, t_per_ky, y_true_flat = _setup_real_scale(seed=200)
    Nx, Ny, Nz = smaps.shape[1:]
    t_frame = t_per_ky.reshape(1, Ny, 1).expand(Nx, Ny, Nz).contiguous()

    def err(L, nbins=128):
        A = _build_operator(smaps, b0map_hz, t_frame, L=L, nbins=nbins)
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
    from recon.analysis import _setup_real_scale
    from recon.operators import build_encoding_operator_b0

    _img, smaps, b0map_hz, t_per_ky, _y_true_flat = _setup_real_scale(seed=201)
    Nx, Ny, Nz = smaps.shape[1:]
    Nt = 1
    omega = torch.ones(Nx, Ny, Nz, Nt, dtype=torch.bool, device=smaps.device)
    echo_times_yz = t_per_ky.reshape(Ny, 1, 1).expand(Ny, Nz, Nt).contiguous()

    build_encoding_operator_b0(smaps, omega, b0map_hz, echo_times_yz, L=32, nbins=128)

    row_sum_warnings = [w for w in recwarn.list if "b_weights row sums" in str(w.message)]
    assert not row_sum_warnings, (
        f"production nbins=128 should not trigger the ill-conditioning warning, got: "
        f"{[str(w.message) for w in row_sum_warnings]}"
    )


def test_adjoint_is_self_consistent():
    """Mirrors tests/test_recon_operators.py's adjoint check for the plain
    GatheredSense, extended to the time-segmented operator."""
    Nx, Ny, Nz, Nc, L = 6, 7, 5, 3, 4
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=40)
    samp = torch.rand(Nx, Ny, Nz, device=DEVICE) > 0.5
    b0map_hz = _complex_randn(Nx, Ny, Nz, seed=41).real * 150
    t_frame = _complex_randn(Nx, Ny, Nz, seed=42).real.abs() * 0.05 + 0.01

    A = _build_b0_operator(smaps, samp, b0map_hz, t_frame, L=L)
    K = A.idx.numel()

    x = _complex_randn(Nx, Ny, Nz, seed=43)
    y = _complex_randn(K, Nc, seed=44)

    lhs = torch.vdot(A.apply(x).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), A.adjoint(y).reshape(-1))
    assert abs(lhs - rhs).item() / abs(lhs).item() < 1e-4


def test_build_encoding_operator_b0_matches_manual_per_frame_construction():
    """build_encoding_operator_b0's BlockDiagonal-of-GatheredSenseB0 should
    apply identically to manually building each frame's operator the way
    _build_b0_operator does above, confirming the gather plumbing
    (echo_times_yz's idx % (Ny*Nz) lookup, per-frame idx) is wired
    correctly end to end. echo_times_s (Nx-expanded) is built here only
    for _build_b0_operator's own per-frame (Nx,Ny,Nz)-shaped contract, not
    passed to build_encoding_operator_b0 itself (see its docstring)."""
    Nx, Ny, Nz, Nc, Nt, L = 5, 6, 4, 2, 3, 3
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=50)
    b0map_hz = _complex_randn(Nx, Ny, Nz, seed=51).real * 100

    omega = torch.stack([torch.rand(Nx, Ny, Nz, device=DEVICE) > 0.4 for _ in range(Nt)], dim=-1)
    counts = omega.sum(dim=(0, 1, 2))
    k = counts.min().item()
    omega = omega & (torch.cumsum(omega.reshape(-1, Nt), dim=0) <= k).reshape(Nx, Ny, Nz, Nt)

    # build_encoding_operator_b0 now relies on every frame sampling from the
    # same *set* of distinct echo times (true of this repo's real
    # acquisitions -- see its own docstring), so this fixture must respect
    # that: tie each (iy,iz) to a value from a small fixed pool, identical
    # across every frame (a stronger invariant than physically needed --
    # real timing only fixes it per echo *index*, not per (iy,iz) -- but
    # sufficient to keep every frame's sampled times a subset of frame 0's).
    n_distinct = 5
    distinct_times = torch.linspace(0.005, 0.055, n_distinct, device=DEVICE)
    yz_idx = (
        torch.arange(Ny, device=DEVICE).reshape(Ny, 1) * Nz
        + torch.arange(Nz, device=DEVICE).reshape(1, Nz)
    ) % n_distinct
    echo_times_yz = distinct_times[yz_idx]  # (Ny,Nz)
    echo_times_2d = echo_times_yz.unsqueeze(-1).expand(Ny, Nz, Nt).contiguous()  # (Ny,Nz,Nt)
    echo_times_s = echo_times_2d.unsqueeze(0).expand(Nx, -1, -1, -1).contiguous()

    A = build_encoding_operator_b0(smaps, omega, b0map_hz, echo_times_2d, L=L, nbins=10)

    x = _complex_randn(Nx, Ny, Nz, Nt, seed=53)
    y_batched = A.apply(x)

    for it in range(Nt):
        A_manual = _build_b0_operator(
            smaps, omega[..., it], b0map_hz, echo_times_s[..., it], L=L, nbins=10
        )
        y_manual = A_manual.apply(x[..., it])
        torch.testing.assert_close(y_batched[..., it], y_manual, atol=1e-5, rtol=1e-4)


def test_r2star_zero_map_matches_phase_only_operator():
    """r2star_map=torch.zeros(...) should reproduce the r2star_map=None
    operator's c_phasors exactly (up to float32 rounding). The two are
    built by genuinely different code paths -- None takes c_phasors
    straight from mri_exp_approx's own spatial output, r2star_map=zeros
    goes through this module's manual `torch.exp(tl * psi)` construction
    -- so this is a real cross-check of that construction against the
    library's own convention, not a tautology. Locks in operators.py's
    stated contract that r2star_map=None is a strict special case of
    r2star_map=0."""
    Nx, Ny, Nz, Nc, Nt, L = 5, 6, 4, 2, 3, 3
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=70)
    b0map_hz = _complex_randn(Nx, Ny, Nz, seed=71).real * 100

    omega = torch.stack([torch.rand(Nx, Ny, Nz, device=DEVICE) > 0.4 for _ in range(Nt)], dim=-1)
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

    A_none = build_encoding_operator_b0(smaps, omega, b0map_hz, echo_times_2d, L=L, nbins=10)
    r2star_zero = torch.zeros(Nx, Ny, Nz, device=DEVICE)
    A_zero = build_encoding_operator_b0(
        smaps, omega, b0map_hz, echo_times_2d, L=L, nbins=10,
        r2star_map=r2star_zero, t_ref_s=0.0,
    )

    x = _complex_randn(Nx, Ny, Nz, Nt, seed=73)
    torch.testing.assert_close(A_none.apply(x), A_zero.apply(x), atol=1e-5, rtol=1e-4)


def test_r2star_generalization_adjoint_is_self_consistent():
    """The check that discriminates this module's PHYSICAL sign
    (psi = i*2*pi*Δf(r) - R2*(r), decaying in the forward direction) from
    recon/lowres_calib_b0.py's flipped sign on the
    (unmerged) worktree-lowres-calib-recon branch (psi_recon =
    i*2*pi*Δf(r) + R2*(r)): the true adjoint identity <Ax,y> == <x,A^H y>
    holds for ANY complex c_phasors under GatheredSenseB0's `.conj()`
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
    b0map_hz = _complex_randn(Nx, Ny, Nz, seed=81).real * 150
    r2star_hz = _complex_randn(Nx, Ny, Nz, seed=82).real.abs() * 40  # 1/s, physically plausible
    t_frame = _complex_randn(Nx, Ny, Nz, seed=83).real.abs() * 0.05 + 0.01

    idx = torch.nonzero(samp.reshape(-1), as_tuple=False).squeeze(-1)
    t_ms = (t_frame.reshape(-1)[idx] * 1000).to(torch.float32)
    b0_neg = (-b0map_hz).to(torch.float32)
    b_by_echo, _c_phase_only, tl = mri_exp_approx(b0_neg, 20, L, t_ms)

    N = (Nx, Ny, Nz)
    psi = 1j * 2 * math.pi * b0map_hz.to(torch.complex64) - r2star_hz.to(torch.complex64)
    tl_c = tl.to(torch.complex64)
    c_phasors = torch.exp(tl_c.reshape((L,) + (1,) * len(N)) * psi[None, ...]).to(smaps.dtype)
    pos = torch.arange(b_by_echo.shape[0], device=DEVICE)
    A = GatheredSenseB0(smaps, samp, pos, b_by_echo.to(smaps.dtype), c_phasors)
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
    (recon/lowres_calib_b0.py's branch convention) would make
    |c_phasors| INCREASE with tl[l] instead of decrease (see the module
    docstring's measured tSNR-gets-worse regression).

    Checked directly on |c_phasors| via build_encoding_operator_b0's own
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
    b0map_hz = _complex_randn(Nx, Ny, Nz, seed=92).real * 150  # realistic-scale Δf, Hz
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

    A = build_encoding_operator_b0(
        smaps, omega, b0map_hz, echo_times_yz, L=L, nbins=20,
        r2star_map=r2star_hz, t_ref_s=t_ref_s,
    )
    c_phasors = A.A[0].c_phasors  # (L,Nx,Ny,Nz)

    # |c_phasors[l]| should be spatially uniform (r2star_hz is) and equal
    # exp(-r2_uniform * |tl[l] - t_ref_s|) -- checked via the spread/bound
    # below rather than re-deriving tl (build_encoding_operator_b0 doesn't
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
    GatheredSense with normalized smaps."""
    Nx, Ny, Nz, Nc = 8, 8, 6, 4
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=60)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)

    b_weights = torch.ones(Nx * Ny * Nz, 1, dtype=torch.complex64, device=DEVICE)
    pos = torch.arange(Nx * Ny * Nz, device=DEVICE)
    c_phasors = torch.ones(1, Nx, Ny, Nz, dtype=torch.complex64, device=DEVICE)
    A = GatheredSenseB0(smaps, full_mask, pos, b_weights, c_phasors)

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
    A = GatheredSenseB0(smaps, full_mask, pos, b_weights, c_phasors)

    x0 = _complex_randn(Nx, Ny, Nz, seed=63)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sigma1 = check_operator_unitary(A, x0, niter=30)
    assert abs(sigma1 - 1.0) < 1e-3


def test_check_operator_unitary_warns_for_real_b0_correction():
    """A genuine (L>1, real field map) B0-corrected operator is not
    guaranteed unitary -- regression guard for the 2026-09-18 finding that
    real GatheredSenseB0 operators measure sigma1 ~= 1.3, not ~1.0 (see
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
    b0map_hz = _complex_randn(Nx, Ny, Nz, seed=71).real * 200  # real-scale-ish field map, Hz
    t_frame = _complex_randn(Nx, Ny, Nz, seed=72).real.abs() * 0.05 + 0.01  # seconds

    A = _build_b0_operator(smaps, full_mask, b0map_hz, t_frame, L=L, nbins=40)
    x0 = _complex_randn(Nx, Ny, Nz, seed=73)

    with pytest.warns(UserWarning, match="not unitary"):
        sigma1 = check_operator_unitary(A, x0, niter=100)
    assert abs(sigma1 - 1.0) > 0.05
