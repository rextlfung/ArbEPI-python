"""Validates recon/b0_correction.py's static single-segment demodulation
against a brute-force, genuinely time-varying ground-truth forward model --
not just an internal-consistency check (which would pass even with the
wrong sign), since the whole reason to bother is external correctness. See
b0_correction.py's own docstring for the sign-convention derivation this
cross-checks empirically.

The ground truth models the same physical structure this repo's real EPI
readout has: off-resonance phase accrues with time-since-excitation, and
that time depends only on which echo (ky index, in this simplified toy)
is being played -- not which kx/kz sample within it -- matching
sequences/ArbEPI.py's echo_times (identical across an echo, varying only by
echo index). Built by brute force (one dense 3D FFT per ky, keeping only
that ky's slice of each), not by any shortcut that could hide a sign error.
"""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from recon.b0_correction import demodulate_smaps  # noqa: E402
from recon.operators import GatheredSense  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _complex_randn(*shape, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    real = torch.randn(*shape, generator=g, device=DEVICE)
    imag = torch.randn(*shape, generator=g, device=DEVICE)
    return (real + 1j * imag).to(torch.complex64)


def _brute_force_time_varying_ksp(
    img: torch.Tensor, smaps: torch.Tensor, b0map_hz: torch.Tensor, t_per_ky: torch.Tensor
) -> torch.Tensor:
    """img: (Nx,Ny,Nz). smaps: (Nc,Nx,Ny,Nz). b0map_hz: (Nx,Ny,Nz).
    t_per_ky: (Ny,) acquisition time (s) of the echo that encodes ky.
    Returns dense (Nc,Nx,Ny,Nz) k-space, the "true" signal this synthetic
    off-resonance model produces at every (kx,ky,kz) -- same convention
    (fftshift/ifftshift/ortho FFT) as GatheredSense._apply."""
    Nc, Nx, Ny, Nz = smaps.shape
    dims = (1, 2, 3)
    y_true = torch.zeros(Nc, Nx, Ny, Nz, dtype=torch.complex64, device=DEVICE)
    for iy in range(Ny):
        angle = (2 * math.pi * t_per_ky[iy].item()) * b0map_hz
        phasor = torch.exp(1j * angle).to(torch.complex64)
        demod = img * smaps * phasor  # (Nc,Nx,Ny,Nz)
        k_full = torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(demod, dim=dims), dim=dims, norm="ortho"), dim=dims
        )
        y_true[:, :, iy, :] = k_full[:, :, iy, :]
    return y_true


def _forward_rel_error(smaps: torch.Tensor, img: torch.Tensor, y_true_flat: torch.Tensor) -> float:
    full_mask = torch.ones(smaps.shape[1:], dtype=torch.bool, device=DEVICE)
    A = GatheredSense(smaps, full_mask)
    y_hat = A.apply(img)
    return (y_hat - y_true_flat).norm().item() / y_true_flat.norm().item()


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
    b0map_hz = (b0_max_hz * yy + 0.4 * b0_max_hz * zz**2).expand(Nx, Ny, Nz).contiguous()

    # Per-echo (per-ky) acquisition time: uniform spacing, matching
    # sequences/ArbEPI.py's echo_times model (one gro-duration step per
    # echo). Centered so the mean is a clean "nominal TE".
    te = 0.030  # s
    t_per_ky = te + (torch.arange(Ny, device=DEVICE, dtype=torch.float32) - (Ny - 1) / 2) * dt_echo

    y_true = _brute_force_time_varying_ksp(img, smaps, b0map_hz, t_per_ky)
    y_true_flat = y_true.reshape(Nc, -1).T  # (K,Nc), C-order -- matches GatheredSense's own flatten

    return img, smaps, b0map_hz, te, y_true_flat


def test_static_correction_reduces_forward_model_error_vs_uncorrected():
    img, smaps, b0map_hz, te, y_true_flat = _setup()

    err_uncorrected = _forward_rel_error(smaps, img, y_true_flat)
    smaps_corrected = demodulate_smaps(smaps, b0map_hz, te)
    err_corrected = _forward_rel_error(smaps_corrected, img, y_true_flat)

    assert err_corrected < 0.2 * err_uncorrected, (
        f"static B0 correction should nearly eliminate forward-model error "
        f"in this small-phase-excursion regime "
        f"(uncorrected={err_uncorrected:.4f}, corrected={err_corrected:.4f})"
    )


def test_wrong_sign_correction_does_not_help():
    """Regression guard for the sign convention documented in
    b0_correction.py -- flipping the field map's sign should make the
    forward-model fit worse than no correction at all, not better, since a
    wrong-sign demodulation adds phase error instead of removing it."""
    img, smaps, b0map_hz, te, y_true_flat = _setup(seed_offset=1)

    err_uncorrected = _forward_rel_error(smaps, img, y_true_flat)
    smaps_wrong_sign = demodulate_smaps(smaps, -b0map_hz, te)
    err_wrong_sign = _forward_rel_error(smaps_wrong_sign, img, y_true_flat)

    assert err_wrong_sign >= err_uncorrected - 1e-6, (
        f"wrong-sign correction unexpectedly helped "
        f"(uncorrected={err_uncorrected:.4f}, wrong_sign={err_wrong_sign:.4f}) "
        "-- check the sign convention in b0_correction.py"
    )


def test_realistic_regime_only_partly_corrects():
    """This repo's real acquisitions (see preprocessing/run_b0map.py's
    regenerated real datasets, and sequences/ArbEPI.py's echo_times) have
    B0 maps spanning roughly +-300-350 Hz over an ETL=60 echo train, ~1.2ms
    per echo -- a total phase excursion of tens of radians, far outside
    static correction's small-signal regime tested above. Confirm the
    honest, weaker claim Fable's plan makes for this regime: correction
    still helps (removes the TE-centered systematic term) but only
    partially, nowhere near test_static_correction_...'s near-elimination
    -- documenting exactly why full time-segmented correction (a separate,
    not-yet-implemented stage) is still needed for real data, not just a
    theoretical concern."""
    img, smaps, b0map_hz, te, y_true_flat = _setup(
        seed_offset=2, b0_max_hz=350.0, dt_echo=0.0012,
    )

    err_uncorrected = _forward_rel_error(smaps, img, y_true_flat)
    smaps_corrected = demodulate_smaps(smaps, b0map_hz, te)
    err_corrected = _forward_rel_error(smaps_corrected, img, y_true_flat)

    assert err_corrected < err_uncorrected, (
        f"static correction should still help even in the realistic large-"
        f"excursion regime (uncorrected={err_uncorrected:.4f}, "
        f"corrected={err_corrected:.4f})"
    )
    assert err_corrected > 0.2 * err_uncorrected, (
        "correction looks suspiciously close to fully correcting the "
        f"realistic regime (uncorrected={err_uncorrected:.4f}, "
        f"corrected={err_corrected:.4f}) -- re-check b0_max_hz/dt_echo "
        "still reflect real data, since this test exists to document that "
        "static correction is NOT sufficient at real scale"
    )


def test_demodulate_smaps_preserves_adjoint_self_consistency():
    """demodulate_smaps only rescales smaps by a unit-magnitude phasor --
    GatheredSense's own adjoint self-consistency (proven for arbitrary
    smaps in test_recon_operators.py) must still hold with corrected smaps."""
    Nx, Ny, Nz, Nc = 6, 7, 5, 3
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=20)
    b0map_hz = _complex_randn(Nx, Ny, Nz, seed=21).real * 100  # arbitrary Hz values
    smaps_corrected = demodulate_smaps(smaps, b0map_hz, te_s=0.02)

    mask = torch.rand(Nx, Ny, Nz, device=DEVICE) > 0.5
    A = GatheredSense(smaps_corrected, mask)
    K = A.idx.numel()

    x = _complex_randn(Nx, Ny, Nz, seed=22)
    y = _complex_randn(K, Nc, seed=23)

    lhs = torch.vdot(A.apply(x).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), A.adjoint(y).reshape(-1))
    assert abs(lhs - rhs).item() / abs(lhs).item() < 1e-5
