"""Quick low-res sanity-check reconstruction from the fully-sampled k-space
calibration region, for a fast look at a dataset without running the full
iterative Stage-2 pipeline (recon_frames.py / recon/).

Every frame's (ky, kz) sampling mask (see sampling/pd_sample.py's
`calib_frac`) always includes a small, fully-sampled centered ellipse --
the calibration region ESPIRiT itself is calibrated from (via the deGRE
scan, not this one). That same guarantee holds for the *EPI* acquisition's
own per-frame mask: `omegas[..., t]` is `calib_mask | (extra incoherent
samples)` for every frame `t`, so `np.all(omegas, axis=-1)` (the
intersection across all frames) recovers exactly that calibration region --
no need to know `params.pd_calib_frac`/`R` ahead of time, or assume they
match the current defaults for a dataset acquired under different settings.
Verified empirically on both `20260822ball_*` datasets: 362/10800 (ky, kz)
locations, a centered ellipse, identical between the radial/laminar
variants (they share the same underlying (ky, kz) mask, only the per-frame
EPI shot ordering differs).

Since that region is exactly, not approximately, fully sampled, no
iterative reconstruction is needed: masking `ksp_epi_zf` down to it,
inverse-FFTing, and combining coils with the existing ESPIRiT sensitivity
maps (`recon/smaps_<seqname>_sigpy.h5`, already normalized so
`sum_c |s_c|^2 <= 1`, see smaps.py's process_smaps) is the correct linear
estimate directly -- `img = sum_c conj(s_c) * ifft(ksp_c)`, no
regularization or iteration.

**Reconstructs at native resolution, not zero-padded to the full (Nx, Ny,
Nz) acquisition grid.** Standard Cartesian MRI relation: resolution =
FOV/N (Delta_k = 1/FOV, and N samples span a k-space extent of N*Delta_k =
N/FOV). The calibration region's (ky, kz) bounding box -- 49 x 10 samples
on both `20260822ball_*` datasets, out of the full 240 x 45 -- caps the
achievable in-plane resolution at FOV_y/49 = 4.41 mm and FOV_z/10 = 4.05 mm,
far coarser than the full acquisition's 0.9 mm. Zero-padding that region up
to the full (Nx, Ny, Nz) grid before IFFT (an earlier version of this
script did exactly that) is pure sinc interpolation -- it doesn't add any
real information, and it makes neighboring voxels highly correlated by
construction (heavily oversampled relative to the true resolution), which
inflates variance-based diagnostics run on the result (e.g. an SVD/PCA
decomposition's "fraction of variance in the top component" -- see
preprocessing/lowres_calib_gain_drift_check.py). Reconstructing directly at
the native grid size gives the same true image content without the
redundant interpolation. `kx` (the readout direction, fully sampled on
every echo, not calibration-limited) is *also* cropped to match, to the
same effective sample count as `ky` -- `Nx_eff = round(Ny_eff * FOV_x /
FOV_y)`, which on these two datasets (FOV_x == FOV_y) works out to exactly
49, matching `Ny_eff` -- an explicit choice to keep the two in-plane axes
at matched resolution rather than leaving `kx` at full resolution while
`ky`/`kz` are calibration-limited.

Same centered-IFFT convention as run_rss.py's/gre_diagnostics.py's _ift3,
and the same FOV-preserving resize (`grid_resize.resize_to_epi_grid`) the
rest of this pipeline already uses to move smaps between grids of the same
FOV at different resolutions.

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_calib_recon <datdir> [seqname]
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.grid_resize import resize_to_epi_grid
from preprocessing.nifti_io import save_recon_nifti


def _ift3(d: np.ndarray) -> np.ndarray:
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(d, axes=axes), axes=axes), axes=axes)


def _load_chunked(f: h5py.File, key: str) -> np.ndarray:
    """Read a dataset chunk-by-chunk along its last axis when chunked there.
    A whole-dataset `d[()]` call on this repo's own ksp_epi_zf (chunked one
    frame per chunk) was measured at ~7 MB/s -- an h5py/HDF5 chunk-cache
    pathology, not a disk-speed limit -- versus ~500 MB/s reading one chunk
    at a time; see recon/reconstruct.py's `_load_array`, which this mirrors."""
    d = f[key]
    if d.chunks is not None and d.chunks[-1] < d.shape[-1]:
        out = np.empty(d.shape, dtype=d.dtype)
        step = d.chunks[-1]
        for start in range(0, d.shape[-1], step):
            out[..., start : start + step] = d[..., start : start + step]
        return out
    return np.asarray(d[()])


def compute_calib_mask(omegas: np.ndarray) -> np.ndarray:
    """omegas: [Ny, Nz, Nframes] bool. Returns [Ny, Nz] bool: the (ky, kz)
    locations sampled in *every* frame -- the fully-sampled calibration
    region every frame's mask is built around (see module docstring)."""
    return np.all(omegas, axis=-1)


def native_calib_grid(calib_mask: np.ndarray, fov: tuple[float, float, float], Nx_full: int) -> dict:
    """calib_mask: [Ny, Nz] bool. fov: (fx, fy, fz) m, the full acquisition
    FOV (unchanged by any of this -- only resolution/N changes). Nx_full:
    the full readout matrix size (240 on these datasets).

    Returns the native (resolution-matched, not zero-padded) grid size and
    the centered crop slices into the full (Nx, Ny, Nz) k-space array --
    see module docstring for the FOV/N resolution relation and why kx is
    cropped too, to Ny's effective sample count."""
    ys, zs = np.nonzero(calib_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    z0, z1 = int(zs.min()), int(zs.max()) + 1
    Ny_eff, Nz_eff = y1 - y0, z1 - z0

    fov_x, fov_y, _fov_z = fov
    Nx_eff = round(Ny_eff * fov_x / fov_y)
    x0 = Nx_full // 2 - Nx_eff // 2
    x1 = x0 + Nx_eff

    return dict(
        Nx_eff=Nx_eff, Ny_eff=Ny_eff, Nz_eff=Nz_eff,
        x_slice=slice(x0, x1), y_slice=slice(y0, y1), z_slice=slice(z0, z1),
    )


def lowres_calib_recon(
    ksp_epi_zf: np.ndarray, calib_mask: np.ndarray, smaps: np.ndarray,
    fov: tuple[float, float, float],
) -> tuple[np.ndarray, dict]:
    """ksp_epi_zf: [Nx, Ny, Nz, Nvcoils, Nframes]. calib_mask: [Ny, Nz] bool.
    smaps: [Nx, Ny, Nz, Nvcoils] complex, sum_c|s_c|^2 <= 1 normalized, on
    the same (Nx, Ny, Nz) grid as ksp_epi_zf. fov: (fx, fy, fz) m.

    Returns (img, grid): img is [Nx_eff, Ny_eff, Nz_eff, Nframes] complex --
    per-frame, sensitivity-map-weighted coil combination of the
    calibration-region-only image, reconstructed at native resolution (see
    module docstring) rather than zero-padded to the full grid. grid is
    native_calib_grid's return value.
    """
    Nx_full = ksp_epi_zf.shape[0]
    grid = native_calib_grid(calib_mask, fov, Nx_full)
    xs, ys, zs = grid['x_slice'], grid['y_slice'], grid['z_slice']

    ksp_crop = ksp_epi_zf[xs, ys, zs, :, :]  # [Nx_eff, Ny_eff, Nz_eff, Nvcoils, Nframes]
    calib_mask_crop = calib_mask[ys, zs]  # [Ny_eff, Nz_eff] -- the ellipse within its bounding box
    ksp_crop = ksp_crop * calib_mask_crop[None, :, :, None, None]  # zero any stray non-calib sample

    img_coils = _ift3(ksp_crop)  # [Nx_eff, Ny_eff, Nz_eff, Nvcoils, Nframes]

    n_target = (grid['Nx_eff'], grid['Ny_eff'], grid['Nz_eff'])
    smaps_native = resize_to_epi_grid(smaps, fov, fov, n_target, order=3)
    img = np.sum(np.conj(smaps_native)[..., None] * img_coils, axis=3)
    return img, grid


def main(datdir: str, seqname: str = 'ArbEPI') -> None:
    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    seq_params = load_seq_params(paths)

    fn_epi_zf = paths.recon
    fn_smaps = os.path.join(datdir, 'recon', f'smaps_{seqname}_sigpy.h5')

    print(f'Loading {fn_epi_zf}...')
    with h5py.File(fn_epi_zf, 'r') as f:
        ksp_epi_zf = _load_chunked(f, 'ksp_epi_zf')  # [Nx, Ny, Nz, Nvcoils, Nframes]
        omegas = f['omegas'][()]  # [Ny, Nz, Nframes]

    calib_mask = compute_calib_mask(omegas)
    n_calib = int(calib_mask.sum())
    print(f'Calibration region: {n_calib} / {calib_mask.size} (ky, kz) locations')
    if n_calib == 0:
        raise RuntimeError(
            'lowres_calib_recon: empty calibration region (no (ky, kz) location is '
            'sampled in every frame) -- is this dataset actually fully sampled at '
            'k-space center?'
        )

    print(f'Loading {fn_smaps}...')
    with h5py.File(fn_smaps, 'r') as f:
        smaps = f['smaps'][()]  # [Nx, Ny, Nz, Nvcoils]

    print('Reconstructing at native (resolution-matched) grid size...')
    img, grid = lowres_calib_recon(
        ksp_epi_zf, calib_mask, smaps, seq_params.fov
    )  # [Nx_eff, Ny_eff, Nz_eff, Nframes]
    voxel_mm = [1000 * seq_params.fov[a] / img.shape[a] for a in range(3)]
    print(f'  Native grid: {img.shape[:3]}  (voxel size {voxel_mm[0]:.3f} x '
          f'{voxel_mm[1]:.3f} x {voxel_mm[2]:.3f} mm)')

    out_dir = os.path.join(datdir, 'recon', 'basic')
    os.makedirs(out_dir, exist_ok=True)
    fn_out = os.path.join(out_dir, f'{seqname}_recon_lowres_calib')
    save_recon_nifti(
        fn_out, img, fov=seq_params.fov, seqname=seqname,
        n_calib_samples=n_calib, n_ky_kz=int(calib_mask.size),
        native_grid=list(img.shape[:3]), native_voxel_mm=voxel_mm,
        note='IFFT + smaps-weighted coil combine of the fully-sampled k-space center only, '
             'reconstructed at native (resolution-matched) grid size, not zero-padded',
    )
    print(f'Wrote {fn_out}.nii.gz + .json')

    Nx, Ny, Nz, Nframes = img.shape
    iz = Nz // 2
    mag = np.abs(img[:, :, iz, :])
    vmax = np.percentile(mag, 99.5)
    frame_idxs = np.linspace(0, Nframes - 1, min(6, Nframes)).astype(int)
    fig, axes = plt.subplots(1, len(frame_idxs), figsize=(3 * len(frame_idxs), 3.5))
    for ax, t in zip(np.atleast_1d(axes), frame_idxs):
        ax.imshow(mag[:, :, t].T, origin='lower', cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(f'frame {t}')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'{seqname}: low-res calib-region recon, z={iz}')
    plt.tight_layout()
    fn_png = os.path.join(out_dir, f'{seqname}_recon_lowres_calib_z{iz}.png')
    plt.savefig(fn_png, dpi=130)
    print(f'Wrote {fn_png}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdir')
    parser.add_argument('seqname', nargs='?', default='ArbEPI')
    args = parser.parse_args()
    main(args.datdir, args.seqname)
