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

Axes are physical k-space units (m^-1), not array indices -- same
deltak = 1/fov, k = (index - N/2) * deltak convention
plotting/plotting.py already uses for the sampling-mask/trajectory
plots, so this stays consistent with the rest of the repo rather than
introducing a second axis convention.

No k-space-center marker: an earlier version of this script tried to
mark it by finding the calibration cell with std(TE) == 0, reasoning
that mask2epi_radial pins "the" k-space-center sample to echo index
(ETL-1)//2 on every shot. That reasoning doesn't hold in general --
which exact discrete (ky,kz) sample lands there varies shot to shot and
frame to frame (per user correction), so treating one fixed cell as
"the center" and annotating it as such would be misleading rather than
informative.

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


def bbox_extent_m(bbox: tuple[int, int, int, int], Ny: int, Nz: int,
                   deltak_y: float, deltak_z: float) -> tuple[float, float, float, float]:
    """imshow extent (left, right, bottom, top) in m^-1 for a [y0:y1, z0:z1]
    crop of a full (Ny, Nz) grid, using the same k = (index - N/2) * deltak
    convention as plotting/plotting.py."""
    y0, y1, z0, z1 = bbox
    ky_left = (y0 - Ny / 2) * deltak_y
    ky_right = (y1 - Ny / 2) * deltak_y
    kz_bottom = (z0 - Nz / 2) * deltak_z
    kz_top = (z1 - Nz / 2) * deltak_z
    return ky_left, ky_right, kz_bottom, kz_top


def main(datdirs: list[str], seqname: str = 'ArbEPI') -> None:
    results = {}
    te_nominal_ms = None

    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))
        cfg = load_config(datdir=datdir, seqnames=[seqname])
        paths = set_seq_paths(cfg, seqname)
        seq_params = load_seq_params(paths)

        mean_ms, std_ms, calib_mask = te_mean_std_maps(paths.recon)
        Ny, Nz = calib_mask.shape
        deltak_y = 1 / seq_params.fov[1]
        deltak_z = 1 / seq_params.fov[2]

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
        extent = bbox_extent_m(bbox, Ny, Nz, deltak_y, deltak_z)

        n_calib = int(calib_mask.sum())
        bbox_size = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
        print(f'\n=== {label} ===')
        print(f'  calibration region: {n_calib} / {bbox_size} cells in bounding box '
              f'({bbox[1] - bbox[0]} x {bbox[3] - bbox[2]}), {100 * n_calib / bbox_size:.1f}% fill')
        print(f'  deltak_y = {deltak_y:.3f} m^-1, deltak_z = {deltak_z:.3f} m^-1')
        print(f'  TE_nominal = {te_nom:.3f} ms (echo index {(seq_params.ETL - 1) // 2} of {seq_params.ETL})')
        print(f'  mean TE across mask: [{np.nanmin(mean_crop):.3f}, {np.nanmax(mean_crop):.3f}] ms')
        print(f'  std TE across mask:  [{np.nanmin(std_crop):.4f}, {np.nanmax(std_crop):.4f}] ms')

        results[label] = dict(mean=mean_crop, std=std_crop, extent=extent)

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
        im = ax.imshow(r['mean'].T, origin='lower', cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax,
                        extent=r['extent'], aspect='auto')
        ax.set_title(f'{label}\nmean TE (ms)')
        ax.set_xlabel('ky (m$^{-1}$)')
        ax.set_ylabel('kz (m$^{-1}$)')
        plt.colorbar(im, ax=ax, fraction=0.046, label='mean TE (ms)')
    fig.suptitle(f'Mean echo time per (ky,kz) sample, across {len(labels)} dataset(s)\'s frames -- '
                 f'TE_nominal = {te_nominal_ms:.3f} ms')
    plt.tight_layout()
    _save(fig, 'lowres_calib_te_mean_map.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, len(labels), figsize=(6.5 * len(labels), 5.2), squeeze=False)
    for ax, label in zip(axes[0], labels):
        r = results[label]
        im = ax.imshow(r['std'].T, origin='lower', cmap=cmap_std, vmin=0, vmax=std_vmax,
                        extent=r['extent'], aspect='auto')
        ax.set_title(f'{label}\nstd(TE) across frames (ms)')
        ax.set_xlabel('ky (m$^{-1}$)')
        ax.set_ylabel('kz (m$^{-1}$)')
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
