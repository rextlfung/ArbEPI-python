"""Checks for a scan-wide, spatially-uniform B0/center-frequency drift
(gradient/shim heating, per the fMRI-instability literature) that the
*static* B0 correction in lowres_calib_recon_b0.py cannot see.

That script's `GatheredSenseB0` demodulates each sample by the *static*
field map (measured once, from the deGRE scan) evaluated at that sample's
own echo time -- correcting the spatial off-resonance pattern's phase
accrual within one frame's echo train. It has no way to see a frame-to-
frame drift in the field *itself*: if the true center frequency shifts by
a few Hz between frame 0 and frame 29 (gradient-coil heating is the
textbook cause in EPI), the static map is simply wrong by that offset for
every later frame, and nothing in the B0-corrected recon corrects for it.

Test: reconstruct the B0-corrected calibration-region image per frame
(reusing lowres_calib_recon_b0.run_b0_corrected_calib_recon -- same
adjoint-only, no-iteration philosophy), then estimate each frame's ROI-
mean phase *offset* relative to a reference frame via a magnitude-weighted
circular mean (same estimator lowres_calib_phase_ramp_check.py uses for
its per-voxel spatial phase gradient, applied here across frames at a
fixed voxel instead of across voxels at a fixed frame):
    dphi[t] = angle( sum_{voxels in mask} conj(img[..., ref]) * img[..., t] )
If a global drift Δf(t) exists, dphi[t] should be proportional to
dTE[t] = mean_calib_echo_time[t] - mean_calib_echo_time[ref] (the frame's
own mean echo time already differs frame to frame, since a given (ky,kz)
calibration sample lands at a different echo index in different frames --
same fact lowres_calib_t2star_check.py's docstring documents): a phase
that accrues linearly with time since the reference is exactly what a
frequency offset predicts, dphi = 2*pi*Δf*dTE. Fitting dphi vs. dTE across
frames recovers Δf directly, in Hz -- and the residual after that fit
(phase not explained by dTE) plus the *spatial* uniformity of dphi (is it
the same phase shift at every voxel, or does it vary spatially?) together
say whether this looks like a real global-frequency-drift signature or
something else.

Sanity range: the fMRI-QA literature reports gradient/shim-heating center-
frequency drift on the order of 1-7 Hz over a scan at 3T -- a recovered
Δf far outside that range (e.g. << 0.1 Hz or >> 50 Hz) would suggest this
isn't the right mechanism, not that the fit is broken.

Prints results directly (this venv, .venv-recon, has no matplotlib -- see
lowres_calib_recon_b0.py's module docstring) and saves a .npz with the
per-frame arrays for optional plotting from .venv-preprocessing.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.lowres_calib_freq_drift_check <datdir> [--seqname ArbEPI] [--device cuda]
"""

import argparse
import os

import numpy as np

from recon.lowres_calib_recon_b0 import run_b0_corrected_calib_recon


def phase_offset_per_frame(img: np.ndarray, mask: np.ndarray, ref: int) -> np.ndarray:
    """img: [Nx,Ny,Nz,Nt] complex. mask: [Nx,Ny,Nz] bool. Returns [Nt] --
    magnitude-weighted mean phase of frame t relative to frame `ref`."""
    Nt = img.shape[-1]
    ref_conj = np.conj(img[..., ref])
    out = np.zeros(Nt)
    for t in range(Nt):
        out[t] = np.angle((ref_conj * img[..., t])[mask].sum())
    return out


def main(
    datdir: str, seqname: str = 'ArbEPI', L: int = 32, nbins: int = 128,
    device: str = 'cuda', skip_frames: int = 1,
) -> None:
    label = os.path.basename(os.path.normpath(datdir))
    result = run_b0_corrected_calib_recon(datdir, seqname, L, nbins, device)
    img = result['img_np'][..., skip_frames:]  # [Nx,Ny,Nz,Nt']
    mean_te_ms = result['mean_te_ms'][skip_frames:]  # [Nt']
    Nt = img.shape[-1]

    mask = np.abs(img).mean(axis=-1) > 0.2 * np.abs(img).mean(axis=-1).max()
    ref = 0  # first retained frame

    dphi = phase_offset_per_frame(img, mask, ref)  # [Nt'] radians
    dte = mean_te_ms - mean_te_ms[ref]  # [Nt'] ms

    # Linear fit dphi = slope * dte (through the origin at t=ref, where
    # both are 0 by construction) -- slope is rad/ms, i.e. 2*pi*Δf(Hz)/1000.
    slope, intercept = np.polyfit(dte, dphi, 1)
    dphi_fit = slope * dte + intercept
    resid = dphi - dphi_fit
    r = np.corrcoef(dte, dphi)[0, 1] if np.std(dte) > 0 else float('nan')
    delta_f_hz = slope / (2 * np.pi) * 1000  # rad/ms -> Hz

    # Spatial uniformity of the phase offset: per-voxel phase (not just the
    # ROI-mean) for the frame with the largest |dphi| against the same
    # reference, and its spatial std -- large std relative to the ROI-mean
    # value means the "phase offset" isn't spatially uniform, i.e. probably
    # not a pure global-frequency-offset effect.
    t_worst = int(np.argmax(np.abs(dphi)))
    per_voxel_phase = np.angle(np.conj(img[..., ref]) * img[..., t_worst])[mask]
    spatial_std_deg = np.rad2deg(per_voxel_phase.std(ddof=1))
    roi_mean_deg = np.rad2deg(dphi[t_worst])

    print(f'\n=== {label} ===')
    print(f'  reference frame (relative): index {ref} of {Nt} retained (skip_frames={skip_frames})')
    print(f'  dTE range across frames: [{dte.min():.3f}, {dte.max():.3f}] ms')
    print(f'  fit: dphi = {slope:.5f} rad/ms * dTE + {intercept:.4f}  (r = {r:.3f})')
    print(f'  implied global frequency drift: Δf = {delta_f_hz:.3f} Hz')
    print(f'  residual phase std after linear fit: {np.rad2deg(resid.std(ddof=1)):.3f} deg '
          f'(vs. raw dphi std {np.rad2deg(dphi.std(ddof=1)):.3f} deg)')
    print(f'  spatial uniformity check @ frame {t_worst} (largest |dphi|): '
          f'ROI-mean phase offset {roi_mean_deg:.2f} deg, spatial std across ROI {spatial_std_deg:.2f} deg')
    print('  (fMRI-QA literature: gradient/shim-heating center-frequency drift is typically '
          '~1-7 Hz over a scan at 3T -- compare Δf above against that range)')

    out_dir = os.path.join(datdir, 'recon', 'basic')
    os.makedirs(out_dir, exist_ok=True)
    fn_npz = os.path.join(out_dir, f'lowres_calib_freq_drift_check_skip{skip_frames}.npz')
    np.savez(
        fn_npz, dphi=dphi, dte_ms=dte, dphi_fit=dphi_fit, slope_rad_per_ms=slope,
        delta_f_hz=delta_f_hz, r=r, mean_te_ms=mean_te_ms,
    )
    print(f'  Wrote {fn_npz}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdir')
    parser.add_argument('--seqname', default='ArbEPI')
    parser.add_argument('--L', type=int, default=32)
    parser.add_argument('--nbins', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip-frames', type=int, default=1)
    args = parser.parse_args()
    main(args.datdir, args.seqname, args.L, args.nbins, args.device, args.skip_frames)
