"""Checks whether T2*/T1 relaxation decay -- real amplitude attenuation, not
phase -- explains the temporal instability preprocessing/lowres_temporal_
stability.py measures in lowres_calib_recon.py's output.

Same mechanism as recon/lowres_calib_recon_b0.py's motivation (a given
(ky,kz) calibration location is acquired at a different echo index, hence a
different time since RF excitation, in different frames -- confirmed
directly against this repo's echo_times array), but acting on magnitude
instead of phase: gradient-echo EPI does not refocus T2* decay, so a
calibration sample acquired late in one frame's echo train has genuinely
lost more signal than the same location acquired early in another frame's
train. The B0 time-segmented correction only demodulates phase -- it leaves
this amplitude effect completely uncorrected, which is why it didn't reduce
the measured fluctuation.

This script correlates each frame's mean calibration-region echo time
against that frame's ROI-mean signal (from the *uncorrected* lowres_calib_
recon.py output -- B0 correction is phase-only, so it shouldn't change this
correlation either way). A real T2*/T1 effect predicts a negative
correlation (later mean echo time -> lower signal); prints both the
correlation coefficient and the projected amplitude swing (regression slope
times the actual range of mean echo times observed) so a statistically real
but practically tiny effect isn't mistaken for the dominant cause.

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_calib_t2star_check <datdir> [datdir2 ...]
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.lowres_temporal_stability import load_lowres_calib_recon, object_mask


def calib_mean_echo_time_ms(fn_ksp: str) -> np.ndarray:
    """[Nframes] -- mean echo time (ms) of that frame's fully-sampled
    calibration-region samples (see lowres_calib_recon.py's compute_calib_mask)."""
    with h5py.File(fn_ksp, 'r') as f:
        omegas = f['omegas'][()]  # (Ny, Nz, Nt)
        echo_times = f['echo_times'][()]  # (Ny, Nz, Nt), seconds
    calib_mask = np.all(omegas, axis=-1)  # (Ny, Nz)
    return echo_times[calib_mask, :].mean(axis=0) * 1000  # (Nt,) ms


def main(datdirs: list[str], seqname: str = 'ArbEPI', skip_frames: int = 1) -> None:
    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))
        fn_ksp = os.path.join(datdir, 'recon', f'{seqname}_epi_zf.h5')
        mean_te_ms = calib_mean_echo_time_ms(fn_ksp)

        img, _meta = load_lowres_calib_recon(datdir, seqname)  # uncorrected recon
        mask = object_mask(img)
        roi_signal = img[mask].mean(axis=0)  # [Nframes]

        te = mean_te_ms[skip_frames:]
        sig = roi_signal[skip_frames:]

        r = np.corrcoef(te, sig)[0, 1]
        slope, intercept = np.polyfit(te, sig, deg=1)
        te_range = te.max() - te.min()
        predicted_swing_pct = 100 * abs(slope) * te_range / sig.mean()

        print(f'\n=== {label} ===')
        print(f'  mean calib echo time: {te.mean():.3f} ms, range [{te.min():.3f}, {te.max():.3f}] '
              f'({te_range:.3f} ms spread across frames)')
        print(f'  Pearson r(echo time, ROI signal) = {r:.3f}  (skip_frames={skip_frames})')
        print(f'  regression slope = {slope:.4g} signal/ms')
        print(f'  signal swing predicted by this slope over the observed TE range: '
              f'{predicted_swing_pct:.3f}%')

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(te, sig)
        te_fit = np.array([te.min(), te.max()])
        ax.plot(te_fit, slope * te_fit + intercept, 'r--', label=f'r={r:.2f}')
        ax.set_xlabel('frame mean calib-region echo time (ms)')
        ax.set_ylabel('ROI mean signal (a.u.)')
        ax.set_title(f'{label}: signal vs. echo time (skip_frames={skip_frames})')
        ax.legend()
        plt.tight_layout()
        out_dir = os.path.join(datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)
        fn_out = os.path.join(out_dir, f'lowres_calib_t2star_check_skip{skip_frames}.png')
        plt.savefig(fn_out, dpi=130)
        print(f'  Wrote {fn_out}')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdirs', nargs='+')
    parser.add_argument('--seqname', default='ArbEPI')
    parser.add_argument('--skip-frames', type=int, default=1)
    args = parser.parse_args()
    main(args.datdirs, args.seqname, args.skip_frames)
