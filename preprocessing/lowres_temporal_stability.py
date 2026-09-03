"""Temporal stability analysis of lowres_calib_recon.py's output.

That reconstruction masks every frame down to the same fixed, fully-sampled
(ky, kz) calibration region before IFFT + smaps combine (see its module
docstring) -- so unlike a full-resolution reconstruction, there is no
frame-varying undersampling mask/trajectory in the signal path here at all.
Any temporal variation this script measures is therefore attributable to
the object/system itself (thermal noise, scanner drift, motion) and *not*
to which (ky, kz) locations a given frame happened to sample -- the
standard NEMA/fBIRN-style phantom stability decomposition (percent
fluctuation + linear drift from a per-frame ROI-mean signal curve, plus a
per-voxel tSNR map) is used to quantify that directly.

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_temporal_stability <datdir> [seqname]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_lowres_calib_recon(datdir: str, seqname: str = 'ArbEPI') -> tuple[np.ndarray, dict]:
    fn_base = os.path.join(datdir, 'recon', 'basic', f'{seqname}_recon_lowres_calib')
    img = np.asarray(nib.load(f'{fn_base}.nii.gz').dataobj)  # [Nx, Ny, Nz, Nframes], magnitude
    with open(f'{fn_base}.json') as f:
        meta = json.load(f)
    return img, meta


def object_mask(img: np.ndarray, thresh_frac: float = 0.2) -> np.ndarray:
    """[Nx, Ny, Nz] bool, thresholded on the time-mean magnitude -- the
    smaps eigenvalue mask already zeroes the true background (see
    lowres_calib_recon.py), so a simple relative threshold on what's left
    cleanly separates object from noise-only voxels."""
    mean_img = img.mean(axis=-1)
    return mean_img > thresh_frac * mean_img.max()


def temporal_stability(img: np.ndarray, mask: np.ndarray, tr_s: float) -> dict:
    """img: [Nx, Ny, Nz, Nframes] magnitude. mask: [Nx, Ny, Nz] bool.

    Returns a dict of the standard phantom-stability decomposition:
    - tsnr_map: [Nx, Ny, Nz], mean/std over time (NaN outside mask)
    - roi_signal: [Nframes], spatial mean over mask per frame
    - percent_fluctuation: 100 * std(residual after linear detrend) / mean(roi_signal)
    - percent_drift: 100 * (linear fit endpoint - start) / mean(roi_signal)
    - roi_tsnr: mean(roi_signal) / std(roi_signal) (no detrend -- the raw,
      undetrended ROI-average tSNR, for comparison against the
      detrend-and-decompose numbers above)
    """
    Nframes = img.shape[-1]
    t = np.arange(Nframes)

    voxels = img[mask]  # [Nvox, Nframes]
    mean_t = voxels.mean(axis=-1)
    std_t = voxels.std(axis=-1, ddof=1)
    tsnr_map = np.full(mask.shape, np.nan, dtype=np.float64)
    tsnr_map[mask] = np.divide(mean_t, std_t, out=np.zeros_like(mean_t), where=std_t > 0)

    roi_signal = img[mask].mean(axis=0)  # [Nframes]

    coeffs = np.polyfit(t, roi_signal, deg=1)
    fit = np.polyval(coeffs, t)
    residual = roi_signal - fit
    grand_mean = roi_signal.mean()

    percent_fluctuation = 100 * residual.std(ddof=1) / grand_mean
    percent_drift = 100 * (fit[-1] - fit[0]) / grand_mean
    roi_tsnr = grand_mean / roi_signal.std(ddof=1)

    return dict(
        tsnr_map=tsnr_map,
        roi_signal=roi_signal,
        time_s=t * tr_s,
        linear_fit=fit,
        percent_fluctuation=percent_fluctuation,
        percent_drift=percent_drift,
        roi_tsnr=roi_tsnr,
        median_voxel_tsnr=np.nanmedian(tsnr_map),
    )


def main(
    datdirs: list[str], seqname: str = 'ArbEPI', tr_s: float = 2.0, skip_frames: int = 0
) -> None:
    results = {}
    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))
        img, meta = load_lowres_calib_recon(datdir, seqname)
        if skip_frames:
            print(f'(dropping first {skip_frames} frame(s) as a non-steady-state transient)')
            img = img[..., skip_frames:]
        mask = object_mask(img)
        stats = temporal_stability(img, mask, tr_s)
        results[label] = (img, mask, stats)

        print(f'\n=== {label} ===')
        print(f'  object mask: {mask.sum()} voxels ({100 * mask.sum() / mask.size:.1f}% of volume)')
        print(f'  ROI-mean signal: {stats["roi_signal"].mean():.4g} +/- {stats["roi_signal"].std(ddof=1):.4g}')
        print(f'  percent fluctuation (detrended): {stats["percent_fluctuation"]:.3f}%')
        print(f'  percent drift (linear, over {(img.shape[-1] - 1) * tr_s:.0f}s): {stats["percent_drift"]:.3f}%')
        print(f'  ROI tSNR (undetrended): {stats["roi_tsnr"]:.1f}')
        print(f'  median per-voxel tSNR: {stats["median_voxel_tsnr"]:.1f}')

    # Comparison figures (one panel per dataset) go in the datdirs' common
    # parent when there's more than one, or that single dataset's own
    # recon/basic/ otherwise.
    out_dir = (
        os.path.commonpath([os.path.normpath(d) for d in datdirs])
        if len(datdirs) > 1
        else os.path.join(datdirs[0], 'recon', 'basic')
    )
    labels = list(results.keys())
    suffix = f'_skip{skip_frames}' if skip_frames else ''

    # --- Figure 1: per-frame ROI-mean signal + linear fit, one panel per dataset ---
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 4), squeeze=False)
    for ax, label in zip(axes[0], labels):
        _, _, stats = results[label]
        ax.plot(stats['time_s'], stats['roi_signal'], 'o-', label='ROI mean signal')
        ax.plot(stats['time_s'], stats['linear_fit'], '--', label='linear fit')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('ROI mean signal (a.u.)')
        ax.set_title(
            f'{label}\nfluct={stats["percent_fluctuation"]:.2f}%, '
            f'drift={stats["percent_drift"]:.2f}%, tSNR={stats["roi_tsnr"]:.0f}'
        )
        ax.legend()
    plt.tight_layout()
    fn1 = os.path.join(out_dir, f'lowres_temporal_stability_roi_signal{suffix}.png')
    plt.savefig(fn1, dpi=130)
    print(f'\nWrote {fn1}')

    # --- Figure 2: tSNR map, central slice, one panel per dataset ---
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4.5), squeeze=False)
    for ax, label in zip(axes[0], labels):
        img, mask, stats = results[label]
        iz = img.shape[2] // 2
        im = ax.imshow(stats['tsnr_map'][:, :, iz].T, origin='lower', cmap='viridis', vmin=0)
        ax.set_title(f'{label}: tSNR, z={iz}')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fn2 = fn1.replace('roi_signal', 'tsnr_map')
    plt.savefig(fn2, dpi=130)
    print(f'Wrote {fn2}')

    # --- Figure 3: frame-to-frame difference vs frame 0, central slice ---
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4.5), squeeze=False)
    for ax, label in zip(axes[0], labels):
        img, mask, stats = results[label]
        iz = img.shape[2] // 2
        diff = img[:, :, iz, -1] - img[:, :, iz, 0]
        vmax = np.percentile(np.abs(diff[mask[:, :, iz]]), 99) if mask[:, :, iz].any() else 1
        im = ax.imshow(diff.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{label}: frame -1 minus frame 0, z={iz}')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fn3 = fn1.replace('roi_signal', 'last_minus_first')
    plt.savefig(fn3, dpi=130)
    print(f'Wrote {fn3}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdirs', nargs='+')
    parser.add_argument('--seqname', default='ArbEPI')
    parser.add_argument('--tr', type=float, default=2.0, help='volume TR in seconds')
    parser.add_argument(
        '--skip-frames', type=int, default=0,
        help='drop this many leading frames before computing stats (non-steady-state transient)',
    )
    args = parser.parse_args()
    main(args.datdirs, args.seqname, args.tr, args.skip_frames)
