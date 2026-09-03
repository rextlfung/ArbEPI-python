"""One-off analysis: sweep the time-segmentation count L (recon/operators_b0.py)
against a synthetic ground truth scaled to this repo's REAL echo-train length
(ETL=60, ~1.2ms/echo -> ~72ms readout window) and REAL field-map range
(-300 to +70 Hz -- both numbers from operators_b0.py's own module docstring)
-- not the 12-distinct-echo-time toy grid
tests/test_recon_operators_b0.py's test_more_segments_reduces_error_in_the_
realistic_regime uses. That test's own "L=16 is ~exact" finding is an
artifact of its toy grid having only 12 distinct echo times (L>=12 trivially
resolves every one exactly); it says nothing about whether L=6 (the current
production default, params.py-adjacent choice in operators_b0.py/
run_b0_recon.py) is adequate at the real ETL=60 scale, where there are up to
60 distinct echo times spanning a much larger bandwidth-time product
(BT = Delta_f_range * T_readout ~= 370 Hz * 0.072 s ~= 27).

Ground-truth construction mirrors tests/test_recon_b0_correction.py's
_brute_force_time_varying_ksp (brute-force, genuinely time-varying, one
dense FFT per echo/ky -- not a shortcut that could hide a segmentation
error) and tests/test_recon_operators_b0.py's mri_exp_approx-based operator
construction, reimplemented here (not imported from tests/) so this stays a
standalone recon/ analysis script, not a test-suite dependency. Nx/Nz/Nc are
kept small for speed -- this is a synthetic forward-model sweep, not a real
reconstruction -- only Ny=ETL and the field-map range/echo spacing need to
match real values, since the per-echo off-resonance phase model here only
depends on t_per_ky (one time per ky row, matching sequences/ArbEPI.py's
echo_times) and b0map_hz's spatial values, not on Nx/Nz/Nc.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.sweep_time_segments
"""

import math

import torch
from mirtorch.linear.mri import mri_exp_approx

from recon.operators import GatheredSense
from recon.operators_b0 import GatheredSenseB0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Real values, both taken directly from operators_b0.py's module docstring.
ETL = 60
DT_ECHO_S = 0.0012
B0_MIN_HZ, B0_MAX_HZ = -300.0, 70.0
TE_S = 0.030  # nominal TE the echo train is centered on; only shifts all t_per_ky uniformly
NBINS = 128  # matches operators_b0.py/run_b0_recon.py's production default


def _complex_randn(*shape, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    real = torch.randn(*shape, generator=g, device=DEVICE)
    imag = torch.randn(*shape, generator=g, device=DEVICE)
    return (real + 1j * imag).to(torch.complex64)


def _brute_force_time_varying_ksp(
    img: torch.Tensor, smaps: torch.Tensor, b0map_hz: torch.Tensor, t_per_ky: torch.Tensor
) -> torch.Tensor:
    """Same construction as tests/test_recon_b0_correction.py's helper of the
    same name -- one dense 3D FFT per ky, keeping only that ky's slice, so
    the "ground truth" is genuinely time-varying rather than assembled from
    any segmented approximation."""
    Nc, Nx, Ny, Nz = smaps.shape
    dims = (1, 2, 3)
    y_true = torch.zeros(Nc, Nx, Ny, Nz, dtype=torch.complex64, device=DEVICE)
    for iy in range(Ny):
        angle = (2 * math.pi * t_per_ky[iy].item()) * b0map_hz
        phasor = torch.exp(1j * angle).to(torch.complex64)
        demod = img * smaps * phasor
        k_full = torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(demod, dim=dims), dim=dims, norm="ortho"), dim=dims
        )
        y_true[:, :, iy, :] = k_full[:, :, iy, :]
    return y_true


def _setup_real_scale(seed: int = 100):
    Nx, Ny, Nz, Nc = 16, ETL, 8, 4  # Ny=ETL matches the real echo train exactly
    img = _complex_randn(Nx, Ny, Nz, seed=seed)
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=seed + 1)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-6)

    # A smooth field map spanning exactly the documented real range, plus a
    # little curvature (bounded so the range stays close to documented) --
    # smooth and structured, like a real B0 map, not white noise.
    yy = torch.linspace(0, 1, Ny, device=DEVICE).reshape(1, Ny, 1)
    zz = torch.linspace(-1, 1, Nz, device=DEVICE).reshape(1, 1, Nz)
    b0map_hz = (B0_MIN_HZ + (B0_MAX_HZ - B0_MIN_HZ) * yy + 15.0 * zz**2).expand(Nx, Ny, Nz)
    b0map_hz = b0map_hz.contiguous().clamp(B0_MIN_HZ, B0_MAX_HZ + 15.0)

    t_per_ky = TE_S + (torch.arange(Ny, device=DEVICE, dtype=torch.float32) - (Ny - 1) / 2) * DT_ECHO_S

    y_true = _brute_force_time_varying_ksp(img, smaps, b0map_hz, t_per_ky)
    y_true_flat = y_true.reshape(Nc, -1).T  # (K,Nc), C-order -- matches GatheredSense's own flatten
    return img, smaps, b0map_hz, t_per_ky, y_true_flat


def _build_operator(smaps: torch.Tensor, b0map_hz: torch.Tensor, t_frame_s: torch.Tensor, L: int, nbins: int):
    """Same construction as tests/test_recon_operators_b0.py's
    _build_b0_operator, reimplemented here to avoid a recon/ -> tests/
    import."""
    Nx, Ny, Nz = smaps.shape[1:]
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)
    idx = torch.nonzero(full_mask.reshape(-1), as_tuple=False).squeeze(-1)
    t_ms = (t_frame_s.reshape(-1)[idx] * 1000).to(torch.float32)
    b0_neg = (-b0map_hz).to(torch.float32)  # sign convention, see b0_correction.py's module docstring
    b, c, _tl = mri_exp_approx(b0_neg, nbins, L, t_ms)
    N = (Nx, Ny, Nz)
    c = c.transpose(0, 1).reshape((L,) + N).to(smaps.dtype)
    pos = torch.arange(b.shape[0], device=b.device)  # identity gather (see GatheredSenseB0)
    return GatheredSenseB0(smaps, full_mask, pos, b.to(smaps.dtype), c)


def sweep(L_values: list[int], nbins: int = NBINS):
    img, smaps, b0map_hz, t_per_ky, y_true_flat = _setup_real_scale()
    Nx, Ny, Nz = smaps.shape[1:]
    t_frame = t_per_ky.reshape(1, Ny, 1).expand(Nx, Ny, Nz).contiguous()
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)

    err_uncorrected = (
        (GatheredSense(smaps, full_mask).apply(img) - y_true_flat).norm().item()
        / y_true_flat.norm().item()
    )

    results = []
    for L in L_values:
        A = _build_operator(smaps, b0map_hz, t_frame, L, nbins)
        y_hat = A.apply(img)
        err = (y_hat - y_true_flat).norm().item() / y_true_flat.norm().item()
        results.append((L, err))
    return err_uncorrected, results


if __name__ == "__main__":
    L_values = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 56, 60]
    err_uncorrected, results = sweep(L_values)

    print(f"ETL={ETL}, dt_echo={DT_ECHO_S * 1000:.2f} ms, "
          f"b0 range=[{B0_MIN_HZ:.0f}, {B0_MAX_HZ:.0f}] Hz, nbins={NBINS}")
    print(f"BT (bandwidth-time product) = {(B0_MAX_HZ - B0_MIN_HZ) * ETL * DT_ECHO_S:.1f}\n")
    print(f"uncorrected (no B0 correction): rel_error = {err_uncorrected:.4f}\n")

    err6 = dict(results).get(6)
    print(f"{'L':>4} {'rel_error':>10} {'vs uncorr':>10} {'vs L=6':>10}")
    for L, err in results:
        vs_uncorr = err / err_uncorrected
        vs_l6 = err / err6 if err6 else float("nan")
        marker = "  <- current default" if L == 6 else ""
        print(f"{L:>4} {err:>10.4f} {vs_uncorr:>9.2%} {vs_l6:>9.2%}{marker}")

    target = 0.01
    converged_L = next((L for L, err in results if err < target), None)
    print(f"\nSmallest swept L with rel_error < {target:.0%}: "
          f"{converged_L if converged_L is not None else 'none in this sweep'}")
