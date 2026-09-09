"""Visualizes the distribution of echo times (TE) across frames, per
(ky, kz) sample, within the fully-sampled calibration region (see
lowres_calib_recon.py's module docstring for how that region is
identified: the intersection of every frame's sampling mask).

This is the transpose of lowres_calib_t2star_check.py's
calib_mean_echo_time_ms, which averages *across the calibration region*
for each frame (one number per frame, [Nt]). Here, for each (ky, kz)
*location*, this reduces *across frames* instead (one number per
location, [Ny, Nz]) -- directly visualizing why a fixed calibration
location's echo time varies frame to frame in the first place (each
frame's shot order assigns that location to a different point in the
echo train), which is the root mechanism behind the T2*/off-resonance
temporal-instability findings elsewhere in this investigation.

Two maps per dataset: the per-location mean TE (diverging colormap
centered on TE_nominal -- the reference the complex-field B0+T2*
correction targets, see recon/lowres_calib_recon_b0complex.py) and the
per-location std TE across frames. Both datasets share one color scale
per map (computed jointly), since the point of comparing laminar vs.
radial is that their echo-time spans differ, not to let per-panel
autoscaling hide that.

The k-space-center cell is marked explicitly on both maps -- but its
(y, z) index is *detected*, not assumed to be (Ny//2, Nz//2): measured
directly on this repo's real data, the true center sits at (116, 22),
four cells off Ny//2=120 in y (Nz//2=22 happens to match). mask2epi_radial
pins the true-center sample to echo index (ETL-1)//2 on every single
shot (see this repo's CLAUDE.md mask2epi_radial paragraph), so that one
cell's std(TE) across frames is exactly 0 in the radial dataset --
measured 7e-15 (float noise) at (116, 22), while (120, 22) measures
1.89 ms, i.e. not the pinned cell at all. This script locates the center
as the calibration-mask cell with the smallest std(TE) across whichever
datasets are passed in (radial's exact pinning wins that search whenever
a radial dataset is included) and reuses that same absolute grid index
to mark every dataset's panel, since the underlying (ky, kz) grid is
shared between mask2epi variants -- only the shot-order-to-location
assignment differs. If no dataset shows a near-zero-std cell (e.g.
laminar-only), it prints a warning and falls back to (Ny//2, Nz//2)
rather than silently mismarking the plot.

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_calib_te_distribution_check <datdir> [datdir2 ...]
"""

import argparse
import copy
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.lowres_calib_recon import compute_calib_mask
from preprocessing.matio import read_mat


def nominal_te_s(scan_info_path: str, etl: int) -> float:
    """The prescribed-TE echo's acquisition time (seconds since RF
    excitation), frame/shot-invariant by construction -- same as
    recon/lowres_calib_recon_b0complex.py's nominal_te_s, reimplemented
    here rather than imported so this script stays off that module's
    torch/mirtorch imports (it runs in .venv-preprocessing, not
    .venv-recon)."""
    raw = read_mat(scan_info_path, ['schedules'])['schedules']  # (Nframes,Nshots,ETL,3)
    return float(raw[0, 0, (etl - 1) // 2, 2])


def te_mean_std_maps(fn_ksp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mean_ms, std_ms, calib_mask), each [Ny, Nz]. mean_ms/std_ms
    are NaN outside the calibration mask (not 0 -- a 0 would read as
    "acquired at t=0" and skew the color scale)."""
    with h5py.File(fn_ksp, 'r') as f:
        omegas = f['omegas'][()]  # (Ny, Nz, Nt) bool
        echo_times = f['echo_times'][()]  # (Ny, Nz, Nt) s
    calib_mask = compute_calib_mask(omegas)
    mean_ms = echo_times.mean(axis=-1) * 1000
    std_ms = echo_times.std(axis=-1, ddof=1) * 1000
    mean_ms = np.where(calib_mask, mean_ms, np.nan)
    std_ms = np.where(calib_mask, std_ms, np.nan)
    return mean_ms, std_ms, calib_mask


def crop_to_bbox(arr2d: np.ndarray, calib_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ys, zs = np.nonzero(calib_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    z0, z1 = int(zs.min()), int(zs.max()) + 1
    return arr2d[y0:y1, z0:z1], (y0, y1, z0, z1)


def main(datdirs: list[str], seqname: str = 'ArbEPI') -> None:
    results = {}
    te_nominal_ms = None
    center_candidates = []  # (std_value, (y, z), Ny, Nz) across all datasets' full grids

    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))
        cfg = load_config(datdir=datdir, seqnames=[seqname])
        paths = set_seq_paths(cfg, seqname)
        seq_params = load_seq_params(paths)

        mean_ms, std_ms, calib_mask = te_mean_std_maps(paths.recon)
        Ny, Nz = calib_mask.shape

        flat_idx = np.nanargmin(std_ms)
        loc = tuple(int(v) for v in np.unravel_index(flat_idx, std_ms.shape))
        center_candidates.append((float(std_ms[loc]), loc, Ny, Nz))

        te_nom = nominal_te_s(paths.scan_info, seq_params.ETL) * 1000
        if te_nominal_ms is None:
            te_nominal_ms = te_nom
        else:
            assert abs(te_nominal_ms - te_nom) < 1e-6, (
                f'TE_nominal differs between datasets ({te_nominal_ms} vs {te_nom} ms) -- '
                'the shared-color-scale assumption below breaks'
            )

        mean_crop, bbox = crop_to_bbox(mean_ms, calib_mask)
        std_crop, _ = crop_to_bbox(std_ms, calib_mask)

        n_calib = int(calib_mask.sum())
        bbox_size = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
        print(f'\n=== {label} ===')
        print(f'  calibration region: {n_calib} / {bbox_size} cells in bounding box '
              f'({bbox[1] - bbox[0]} x {bbox[3] - bbox[2]}), {100 * n_calib / bbox_size:.1f}% fill')
        print(f'  TE_nominal = {te_nom:.3f} ms (echo index {(seq_params.ETL - 1) // 2} of {seq_params.ETL})')
        print(f'  mean TE across mask: [{np.nanmin(mean_crop):.3f}, {np.nanmax(mean_crop):.3f}] ms')
        print(f'  std TE across mask:  [{np.nanmin(std_crop):.4f}, {np.nanmax(std_crop):.4f}] ms')
        print(f'  this dataset\'s own min-std cell: (y={loc[0]}, z={loc[1]}), std={std_ms[loc]:.4e} ms, '
              f'mean={mean_ms[loc]:.4f} ms')

        results[label] = dict(mean=mean_crop, std=std_crop, bbox=bbox, Ny=Ny, Nz=Nz)

    # Detect the true k-space-center cell: whichever dataset shows the
    # smallest std(TE) wins (radial's exact per-shot pinning drives this to
    # ~0 when present; see module docstring). Reused as one shared absolute
    # grid index across every dataset's panel, since the (ky,kz) grid itself
    # is shared -- only the shot-order assignment differs by variant.
    best_std, best_loc, best_Ny, best_Nz = min(center_candidates, key=lambda c: c[0])
    zero_thresh_ms = 1e-3
    if best_std < zero_thresh_ms:
        center = best_loc
        print(f'\nDetected true k-space center at (y={center[0]}, z={center[1]}) '
              f'(std={best_std:.2e} ms, below the {zero_thresh_ms} ms pinning threshold).')
    else:
        center = (best_Ny // 2, best_Nz // 2)
        print(f'\nWARNING: no dataset showed a near-zero-std cell (smallest was {best_std:.4f} ms) -- '
              f'falling back to the geometric-center guess (y={center[0]}, z={center[1]}), which may be wrong.')

    for label, r in results.items():
        y0, y1, z0, z1 = r['bbox']
        center_crop = (center[0] - y0, center[1] - z0)
        if 0 <= center_crop[0] < r['mean'].shape[0] and 0 <= center_crop[1] < r['mean'].shape[1]:
            print(f'  [{label}] k-space center (y={center[0]}, z={center[1]}): '
                  f'mean={r["mean"][center_crop]:.4f} ms, std={r["std"][center_crop]:.4f} ms')
        else:
            print(f'  [{label}] WARNING: k-space center falls outside this dataset\'s calibration bounding box')
        r['center'] = center_crop

    # Shared color scales across datasets -- see module docstring for why.
    max_dev = max(np.nanmax(np.abs(r['mean'] - te_nominal_ms)) for r in results.values())
    mean_vmin, mean_vmax = te_nominal_ms - max_dev, te_nominal_ms + max_dev
    std_vmax = max(np.nanmax(r['std']) for r in results.values())

    cmap_mean = copy.copy(plt.get_cmap('RdBu_r'))
    cmap_mean.set_bad('lightgray')
    cmap_std = copy.copy(plt.get_cmap('viridis'))
    cmap_std.set_bad('lightgray')

    labels = list(results.keys())
    out_dirs = [os.path.join(d, 'recon', 'basic') for d in datdirs]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)

    def _save(fig, fn_name: str) -> None:
        for out_dir in out_dirs:
            fig.savefig(os.path.join(out_dir, fn_name), dpi=130)
        print(f'Wrote {fn_name} to: {", ".join(out_dirs)}')

    fig, axes = plt.subplots(1, len(labels), figsize=(6.5 * len(labels), 5.2), squeeze=False)
    for ax, label in zip(axes[0], labels):
        r = results[label]
        im = ax.imshow(r['mean'].T, origin='lower', cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax)
        ax.plot(*r['center'], marker='+', color='black', markersize=14, markeredgewidth=2,
                label='k-space center')
        ax.set_title(f'{label}\nmean TE (ms)')
        ax.set_xlabel('ky index (cropped)')
        ax.set_ylabel('kz index (cropped)')
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, label='mean TE (ms)')
    fig.suptitle(f'Mean echo time per (ky,kz) sample, across {len(labels)} dataset(s)\'s frames -- '
                 f'TE_nominal = {te_nominal_ms:.3f} ms')
    plt.tight_layout()
    _save(fig, 'lowres_calib_te_mean_map.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, len(labels), figsize=(6.5 * len(labels), 5.2), squeeze=False)
    for ax, label in zip(axes[0], labels):
        r = results[label]
        im = ax.imshow(r['std'].T, origin='lower', cmap=cmap_std, vmin=0, vmax=std_vmax)
        ax.plot(*r['center'], marker='+', color='red', markersize=14, markeredgewidth=2,
                label='k-space center')
        ax.set_title(f'{label}\nstd(TE) across frames (ms)')
        ax.set_xlabel('ky index (cropped)')
        ax.set_ylabel('kz index (cropped)')
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, label='std TE (ms)')
    fig.suptitle('Standard deviation of echo time per (ky,kz) sample, across frames')
    plt.tight_layout()
    _save(fig, 'lowres_calib_te_std_map.png')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdirs', nargs='+')
    parser.add_argument('--seqname', default='ArbEPI')
    args = parser.parse_args()
    main(args.datdirs, args.seqname)
