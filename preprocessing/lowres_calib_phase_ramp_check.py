"""Checks for eddy-current / gradient-delay drift in lowres_calib_recon.py's
reconstruction, correctly this time: a readout-delay (kx) error is a shift
in k-space, which by the Fourier shift theorem is a *linear phase ramp*
along x in image space, not a magnitude change or a centroid shift (a pure
phase term has |exp(i*phase)| == 1 everywhere -- see
lowres_calib_centroid_drift_check.py, which tested magnitude/centroid and
could not have detected this).

Estimates each frame's phase gradient along x (and y, z as controls -- a
pure kx-shift should show up only along x, since it doesn't touch ky/kz
encoding at all) via a finite-difference estimator robust to unwrapping:
    slope_axis[t] = angle( sum_{voxels in mask} conj(img[..., t]) * img_shifted_by_1_along_axis[..., t] )
i.e. the (magnitude-squared-weighted) average local phase increment between
adjacent voxels along that axis -- the same idea as an off-resonance/phase-
gradient estimator, avoiding the need to unwrap absolute phase across a
low-SNR, low-res object.

Needs the *complex* reconstruction, not the magnitude-only .nii.gz this
pipeline saves (NIfTI has no complex dtype -- see nifti_io.py's module
docstring) -- recomputes it directly via lowres_calib_recon.lowres_calib_recon
rather than loading from disk.

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_calib_phase_ramp_check <datdir> [datdir2 ...]
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.lowres_calib_recon import _load_chunked, compute_calib_mask, lowres_calib_recon
from preprocessing.config import load_config, load_seq_params, set_seq_paths


def phase_gradient_per_frame(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """img: [Nx,Ny,Nz,Nframes] complex. mask: [Nx,Ny,Nz] bool. Returns
    [Nframes, 3] -- the magnitude-weighted mean local phase increment
    (radians/voxel) between adjacent voxels along (x,y,z), within mask."""
    Nframes = img.shape[-1]
    out = np.zeros((Nframes, 3))
    for axis in range(3):
        shifted = np.roll(img, -1, axis=axis)
        # Only voxels where both the voxel and its +1 neighbor along `axis`
        # are inside the object mask contribute -- avoids wrap-around and
        # background-noise pairs contaminating the estimate.
        neighbor_mask = np.roll(mask, -1, axis=axis) & mask
        prod = np.conj(img) * shifted  # [Nx,Ny,Nz,Nframes]
        for t in range(Nframes):
            out[t, axis] = np.angle(prod[..., t][neighbor_mask].sum())
    return out


def main(datdirs: list[str], seqname: str = 'ArbEPI', skip_frames: int = 1) -> None:
    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))

        cfg = load_config(datdir=datdir, seqnames=[seqname])
        paths = set_seq_paths(cfg, seqname)
        seq_params = load_seq_params(paths)

        fn_epi_zf = paths.recon
        fn_smaps = os.path.join(datdir, 'recon', f'smaps_{seqname}_sigpy.h5')
        with h5py.File(fn_epi_zf, 'r') as f:
            ksp_epi_zf = _load_chunked(f, 'ksp_epi_zf')
            omegas = f['omegas'][()]
        calib_mask = compute_calib_mask(omegas)
        with h5py.File(fn_smaps, 'r') as f:
            smaps = f['smaps'][()]

        img, grid = lowres_calib_recon(ksp_epi_zf, calib_mask, smaps, seq_params.fov)  # complex
        img = img[..., skip_frames:]
        mask = np.abs(img).mean(axis=-1) > 0.2 * np.abs(img).mean(axis=-1).max()

        slopes = phase_gradient_per_frame(img, mask)  # [Nframes', 3] radians/voxel
        t = np.arange(slopes.shape[0])

        print(f'\n=== {label} ===')
        axis_names = ['x (readout)', 'y (phase-encode)', 'z (phase-encode)']
        for a, name in enumerate(axis_names):
            s = np.rad2deg(slopes[:, a])  # degrees/voxel, easier to read
            slope_fit, intercept = np.polyfit(t, s, 1)
            resid = s - (slope_fit * t + intercept)
            print(f'  {name}: mean={s.mean():.3f} deg/vox, std={s.std(ddof=1):.3f} deg/vox, '
                  f'range={s.max() - s.min():.3f} deg/vox, '
                  f'linear drift over scan={abs(slope_fit) * (len(t) - 1):.3f} deg/vox, '
                  f'residual std after detrend={resid.std(ddof=1):.3f} deg/vox')

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for a, (ax, name) in enumerate(zip(axes, axis_names)):
            ax.plot(t, np.rad2deg(slopes[:, a]), 'o-')
            ax.set_title(name)
            ax.set_xlabel('frame')
            ax.set_ylabel('phase gradient (deg/voxel)')
        fig.suptitle(f'{label}: per-frame local phase gradient, skip_frames={skip_frames}')
        plt.tight_layout()
        out_dir = os.path.join(datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)
        fn_out = os.path.join(out_dir, f'lowres_calib_phase_ramp_check_skip{skip_frames}.png')
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
