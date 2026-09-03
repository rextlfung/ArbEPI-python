"""Checks for eddy-current / gradient-delay drift in lowres_calib_recon.py's
output, indirectly: `preprocessing/calibrate_delay.py`'s readout delay
correction (`cfg.delay`, applied uniformly to every shot/frame via
`preprocess.py`'s `apply_delay`) is estimated *once*, from a single short
calibration acquisition at the start of the scan session -- not per frame.
If the true gradient delay drifts over the ~1-minute EPI scan (e.g. from
gradient/eddy-current heating), the fixed correction becomes progressively
wrong, which should show up specifically as a small, time-varying spatial
shift of the reconstructed object along kx (the readout/x direction --
the one axis `apply_delay`'s correction actually parameterizes). y/z
(phase-encode directions, unaffected by readout delay) serve as a control:
if x drifts/fluctuates noticeably more than y/z, that's evidence consistent
with this mechanism; if all three axes behave similarly, it isn't.

Computes each frame's intensity-weighted centroid (in mm, relative to the
volume center) along each axis, within the object mask, from the plain
(uncorrected) lowres_calib_recon.py output.

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_calib_centroid_drift_check <datdir> [datdir2 ...]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from preprocessing.lowres_temporal_stability import load_lowres_calib_recon, object_mask


def centroid_per_frame(img: np.ndarray, mask: np.ndarray, voxel_mm: list[float]) -> np.ndarray:
    """img: [Nx,Ny,Nz,Nframes] magnitude. mask: [Nx,Ny,Nz] bool. Returns
    [Nframes, 3] -- intensity-weighted centroid (mm, relative to the
    geometric volume center) along (x,y,z), within mask, per frame."""
    Nx, Ny, Nz, Nframes = img.shape
    xs, ys, zs = np.meshgrid(
        (np.arange(Nx) - (Nx - 1) / 2) * voxel_mm[0],
        (np.arange(Ny) - (Ny - 1) / 2) * voxel_mm[1],
        (np.arange(Nz) - (Nz - 1) / 2) * voxel_mm[2],
        indexing='ij',
    )
    coords = np.stack([xs[mask], ys[mask], zs[mask]], axis=-1)  # [Nvox, 3]
    weights = img[mask]  # [Nvox, Nframes]
    total = weights.sum(axis=0)  # [Nframes]
    return (weights.T @ coords) / total[:, None]  # [Nframes, 3]


def main(datdirs: list[str], seqname: str = 'ArbEPI', skip_frames: int = 1) -> None:
    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))
        img, meta = load_lowres_calib_recon(datdir, seqname)
        mask = object_mask(img)
        voxel_mm = meta['native_voxel_mm']

        cen = centroid_per_frame(img, mask, voxel_mm)[skip_frames:]  # [Nframes', 3]
        t = np.arange(cen.shape[0])

        print(f'\n=== {label} ===')
        axis_names = ['x (readout)', 'y (phase-encode)', 'z (phase-encode)']
        for a, name in enumerate(axis_names):
            c = cen[:, a]
            slope, intercept = np.polyfit(t, c, 1)
            resid = c - (slope * t + intercept)
            print(f'  {name}: std={c.std(ddof=1):.4f} mm, range={c.max() - c.min():.4f} mm, '
                  f'linear drift over scan={abs(slope) * (len(t) - 1):.4f} mm, '
                  f'residual std after detrend={resid.std(ddof=1):.4f} mm')

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
        for a, (ax, name) in enumerate(zip(axes, axis_names)):
            ax.plot(t, cen[:, a], 'o-')
            ax.set_title(name)
            ax.set_xlabel('frame')
            ax.set_ylabel('centroid offset (mm)')
        fig.suptitle(f'{label}: per-frame object centroid, skip_frames={skip_frames}')
        plt.tight_layout()
        out_dir = os.path.join(datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)
        fn_out = os.path.join(out_dir, f'lowres_calib_centroid_drift_check_skip{skip_frames}.png')
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
