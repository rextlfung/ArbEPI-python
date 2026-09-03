"""Validates recon/operators_b0.py's GatheredSenseB0/build_encoding_operator_b0
against the same brute-force, genuinely time-varying ground-truth forward
model tests/test_recon_b0_correction.py uses for the static (single-segment)
stage -- reused directly here (not reimplemented) so both stages are held to
the exact same ground truth."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from mirtorch.linear.mri import mri_exp_approx  # noqa: E402

from recon.operators_b0 import (  # noqa: E402
    GatheredSenseB0,
    build_encoding_operator_b0,
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
    match recon/b0_correction.py's static single-segment correction closely
    -- both are, in the end, one global per-voxel phase term applied before
    the FFT; this is the connective-tissue check between the two stages."""
    from recon.b0_correction import demodulate_smaps
    from recon.operators import GatheredSense

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


def test_more_segments_reduces_error_in_the_realistic_regime():
    """This repo's real acquisitions (B0 up to +-350 Hz, ETL=60 at ~1.2ms
    spacing) is exactly the regime tests/test_recon_b0_correction.py's
    test_realistic_regime_only_partly_corrects found static correction
    barely helps in (~5% error reduction). Time segmentation exists
    specifically to fix that -- confirm error drops substantially as L
    grows, well beyond what static (L=1) achieves."""
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
    _build_b0_operator does above, confirming the broadcast/gather plumbing
    (echo_times_s across Nx, per-frame idx) is wired correctly end to end."""
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

    A = build_encoding_operator_b0(smaps, omega, b0map_hz, echo_times_s, L=L, nbins=10)

    x = _complex_randn(Nx, Ny, Nz, Nt, seed=53)
    y_batched = A.apply(x)

    for it in range(Nt):
        A_manual = _build_b0_operator(
            smaps, omega[..., it], b0map_hz, echo_times_s[..., it], L=L, nbins=10
        )
        y_manual = A_manual.apply(x[..., it])
        torch.testing.assert_close(y_batched[..., it], y_manual, atol=1e-5, rtol=1e-4)


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
