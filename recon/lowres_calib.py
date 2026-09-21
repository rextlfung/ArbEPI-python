"""Low-resolution reconstruction of the fully sampled calibration region and its
temporal-stability check (.venv-preprocessing; no torch). Subcommands:

    .venv-preprocessing/bin/python -m recon.lowres_calib {calib,stability} ...

The sections below are the original module docstrings, kept verbatim.

Formerly recon/lowres_calib/lowres_calib_recon.py
-------------------------------------------------
Quick low-res sanity-check reconstruction from the fully-sampled k-space
calibration region, for a fast look at a dataset without running the full
iterative Stage-2 pipeline (sigpy_recon.py / recon/).

Every frame's (ky, kz) sampling mask (see sampling/pd_sample.py's
`calib_frac`) always includes a small, fully-sampled centered region --
the calibration region ESPIRiT itself is calibrated from (via the deGRE
scan, not this one). That same guarantee holds for the *EPI* acquisition's
own per-frame mask: `omegas[..., t]` is `calib_mask | (extra incoherent
samples)` for every frame `t`, so `np.all(omegas, axis=-1)` (the
intersection across all frames) recovers exactly that calibration region --
no need to know `params.pd_calib_frac`/`R` ahead of time, or assume they
match the current defaults for a dataset acquired under different settings,
and no dependence on the region's shape either (a plain set intersection,
which works the same whether `calib_mask` is an ellipse or a rectangle).
Verified empirically on both `20260822ball_*` datasets: 362/10800 (ky, kz)
locations, a centered ellipse, identical between the radial/laminar
variants (they share the same underlying (ky, kz) mask, only the per-frame
EPI shot ordering differs) -- both acquired under `pd_sample.py`'s original
area-matched-ellipse `calib_frac` semantics (fraction of the sample
budget); a since-reverted intermediate version briefly made calib_frac a
fixed fraction of k-space instead (independent of R), but that let the
calibration region consume nearly the entire sample budget at high
acceleration, so `_calib_side_frac` restored the fraction-of-sample-budget
sizing -- now realized as a rectangle rather than the original ellipse
(see that module's docstring and docs/review-findings.md item 195). This
paragraph's specific numbers are a historical record of those two
(ellipse-shaped) datasets, not a current claim about the shape a fresh
acquisition's calibration region will have.

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

Same centered-IFFT convention as sigpy_recon.py's/gre_diagnostics.py's _ift3,
and the same FOV-preserving resize (`grid_resize.resize_to_epi_grid`) the
rest of this pipeline already uses to move smaps between grids of the same
FOV at different resolutions.

Usage (from repo root, .venv-preprocessing -- this module needs sigpy/h5py/
matplotlib, not torch/mirtorch, despite living under recon/ alongside the
torch-based MSLR pipeline; see CLAUDE.md's recon/ section for the venv
split rationale):
    .venv-preprocessing/bin/python -m recon.lowres_calib calib <datdir> [seqname]


Formerly recon/lowres_calib/lowres_temporal_stability.py
--------------------------------------------------------
Temporal stability analysis of lowres_calib.py's output.

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

variant='' reads recon/lowres_calib.py's plain output;
variant='b0' reads recon/lowres_calib_b0.py's B0-corrected output --
see that module's docstring for why per-frame B0-induced phase, not just
system drift/noise, is expected to show up here as apparent instability
for a static object.

Usage (from repo root, .venv-preprocessing -- matplotlib/nibabel, not
torch/mirtorch, despite comparing recon/lowres_calib_b0.py's
.venv-recon-only output; see CLAUDE.md's recon/ section "not a
single-venv package" note):
    .venv-preprocessing/bin/python -m recon.lowres_calib stability <datdir> [--seqname ArbEPI] [--variant b0]
"""

import argparse
import json
import os
import sys

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from preprocessing.config import PreprocessingConfig, load_config, load_seq_params, set_seq_paths
from preprocessing.grid_resize import resize_to_epi_grid
from preprocessing.nifti_io import save_recon_nifti
from recon.hdf5_chunked_io import read_frames_cropped


def _ift3(d: np.ndarray) -> np.ndarray:
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(d, axes=axes), axes=axes), axes=axes)


def compute_calib_mask(omegas: np.ndarray) -> np.ndarray:
    """omegas: [Ny, Nz, Nframes] bool. Returns [Ny, Nz] bool: the (ky, kz)
    locations sampled in *every* frame -- the fully-sampled calibration
    region every frame's mask is built around (see module docstring)."""
    return np.all(omegas, axis=-1)


def native_calib_grid(
    calib_mask: np.ndarray, fov: tuple[float, float, float], Nx_full: int
) -> dict:
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
    ksp_crop: np.ndarray, calib_mask: np.ndarray, grid: dict, smaps: np.ndarray,
    fov: tuple[float, float, float],
) -> np.ndarray:
    """ksp_crop: [Nx_eff, Ny_eff, Nz_eff, Nvcoils, Nframes] -- already
    cropped to `grid`'s slices. Callers compute `grid` (via
    native_calib_grid) from `calib_mask` *before* reading k-space, so the
    HDF5 read itself can crop to this bounding box (recon/hdf5_chunked_io.py's
    read_frames_cropped) instead of loading the full dense array and
    cropping after -- see docs/review-findings.md item 204 for why that
    matters (the full array is ~201GB at this pipeline's real 0.8mm/R~93.5
    scale, vs. a few hundred calibration-region samples actually needed).
    calib_mask: [Ny, Nz] bool, full-grid shape (cropped internally via
    grid's y/z slices). smaps: [Nx, Ny, Nz, Nvcoils] complex, sum_c|s_c|^2
    <= 1 normalized, full grid. fov: (fx, fy, fz) m.

    Returns img: [Nx_eff, Ny_eff, Nz_eff, Nframes] complex -- per-frame,
    sensitivity-map-weighted coil combination of the calibration-region-only
    image, reconstructed at native resolution (see module docstring) rather
    than zero-padded to the full grid.
    """
    ys, zs = grid['y_slice'], grid['z_slice']
    # [Ny_eff, Nz_eff] -- calib_mask's own shape within its bounding box. A
    # no-op for a rectangular calib_mask (its bounding box is itself, fully
    # true), but real masking for e.g. an older ellipse-shaped one, whose
    # own bounding box has corners outside the region -- kept generic since
    # calib_mask's shape isn't assumed here.
    calib_mask_crop = calib_mask[ys, zs]
    ksp_crop = ksp_crop * calib_mask_crop[None, :, :, None, None]  # zero any stray non-calib sample

    img_coils = _ift3(ksp_crop)  # [Nx_eff, Ny_eff, Nz_eff, Nvcoils, Nframes]

    n_target = (grid['Nx_eff'], grid['Ny_eff'], grid['Nz_eff'])
    smaps_native = resize_to_epi_grid(smaps, fov, fov, n_target, order=3)
    img = np.sum(np.conj(smaps_native)[..., None] * img_coils, axis=3)
    return img


def _recon_one(cfg: PreprocessingConfig, seqname: str) -> None:
    datdir = cfg.datdir
    paths = set_seq_paths(cfg, seqname)
    seq_params = load_seq_params(paths)

    fn_epi_zf = paths.recon
    fn_smaps = os.path.join(datdir, 'recon', f'smaps_{seqname}_sigpy.h5')

    print(f'Loading sampling mask ({fn_epi_zf})...')
    with h5py.File(fn_epi_zf, 'r') as f:
        omegas = f['omegas'][()]  # [Ny, Nz, Nframes]
        Nx_full = f['ksp_epi_zf'].shape[0]

    calib_mask = compute_calib_mask(omegas)
    n_calib = int(calib_mask.sum())
    print(f'Calibration region: {n_calib} / {calib_mask.size} (ky, kz) locations')
    if n_calib == 0:
        raise RuntimeError(
            'lowres_calib_recon: empty calibration region (no (ky, kz) location is '
            'sampled in every frame) -- is this dataset actually fully sampled at '
            'k-space center?'
        )

    # Computed before the k-space load (not after, as an earlier version of
    # this function did internally) so the HDF5 read itself can crop to
    # this bounding box -- see docs/review-findings.md item 204.
    grid = native_calib_grid(calib_mask, seq_params.fov, Nx_full)
    xs, ys, zs = grid['x_slice'], grid['y_slice'], grid['z_slice']

    print(f'Loading {fn_epi_zf} (cropped to the calibration-region bounding box)...')
    ksp_crop = read_frames_cropped(
        fn_epi_zf, 'ksp_epi_zf', spatial_slices=(xs, ys, zs)
    )  # [Nx_eff, Ny_eff, Nz_eff, Nvcoils, Nframes]

    print(f'Loading {fn_smaps}...')
    with h5py.File(fn_smaps, 'r') as f:
        smaps = f['smaps'][()]  # [Nx, Ny, Nz, Nvcoils]

    print('Reconstructing at native (resolution-matched) grid size...')
    img = lowres_calib_recon(
        ksp_crop, calib_mask, grid, smaps, seq_params.fov
    )  # [Nx_eff, Ny_eff, Nz_eff, Nframes]
    voxel_mm = [1000 * seq_params.fov[a] / img.shape[a] for a in range(3)]
    print(f'  Native grid: {img.shape[:3]}  (voxel size {voxel_mm[0]:.3f} x '
          f'{voxel_mm[1]:.3f} x {voxel_mm[2]:.3f} mm)')

    out_dir = os.path.join(datdir, 'recon', 'lowres_calib')
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
    plt.close(fig)
    print(f'Wrote {fn_png}')


def run_lowres_calib_recon(cfg: PreprocessingConfig) -> None:
    """Batch driver, mirroring sigpy_recon.py/run_preprocessing.py's per-sequence
    try/except pattern -- runs _recon_one for every cfg.seqnames, continuing
    past a single sequence's failure rather than aborting the batch."""
    print(f'Batch: {len(cfg.seqnames)} sequence(s) in {cfg.datdir}')
    for i, seqname in enumerate(cfg.seqnames, start=1):
        print(f'\n[{i}/{len(cfg.seqnames)}] {seqname}')
        try:
            _recon_one(cfg, seqname)
        except Exception as e:  # noqa: BLE001 -- mirrors sigpy_recon.py's per-sequence try/catch
            print(f"ERROR [{seqname}]: {e}\nSkipping...")
    print('\nBatch complete.')


def main_calib(datdir: str, seqname: str = 'ArbEPI') -> None:
    """Single-sequence convenience wrapper (unchanged CLI) around
    run_lowres_calib_recon -- lets a single-dataset failure raise directly
    instead of being caught-and-printed, useful for interactive/ad hoc use."""
    cfg = load_config(datdir=datdir, seqnames=[seqname])
    _recon_one(cfg, seqname)

def _cli_calib() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('datdir')
    parser.add_argument('seqname', nargs='*', default=['ArbEPI'],
                         help='one or more seqnames to batch (default: ArbEPI)')
    args = parser.parse_args()
    if len(args.seqname) == 1:
        main_calib(args.datdir, args.seqname[0])
    else:
        run_lowres_calib_recon(load_config(datdir=args.datdir, seqnames=args.seqname))

def load_lowres_calib_recon(
    datdir: str, seqname: str = 'ArbEPI', variant: str = ''
) -> tuple[np.ndarray, dict]:
    """variant: '' for the plain (uncorrected) recon, 'b0' for
    recon/lowres_calib_b0.py's B0-corrected output."""
    suffix = f'_{variant}' if variant else ''
    fn_base = os.path.join(datdir, 'recon', 'lowres_calib', f'{seqname}_recon_lowres_calib{suffix}')
    img = np.asarray(nib.load(f'{fn_base}.nii.gz').dataobj)  # [Nx, Ny, Nz, Nframes], magnitude
    with open(f'{fn_base}.json') as f:
        meta = json.load(f)
    return img, meta


def object_mask(img: np.ndarray, thresh_frac: float = 0.2) -> np.ndarray:
    """[Nx, Ny, Nz] bool, thresholded on the time-mean magnitude -- the
    smaps eigenvalue mask already zeroes the true background (see
    lowres_calib.py), so a simple relative threshold on what's left
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


def main_stability(
    datdirs: list[str], seqname: str = 'ArbEPI', tr_s: float = 2.0, skip_frames: int = 0,
    variant: str = '',
) -> None:
    results = {}
    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))
        if variant:
            label = f'{label} ({variant})'
        img, meta = load_lowres_calib_recon(datdir, seqname, variant)
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

    # Comparison figures (one panel per dataset) are saved next to each
    # dataset's own .nii.gz -- i.e. a copy in every datdir's recon/lowres_calib/,
    # not a shared parent directory, so each dataset's folder stays
    # self-contained even when the figure itself compares multiple datasets.
    out_dirs = [os.path.join(d, 'recon', 'lowres_calib') for d in datdirs]
    labels = list(results.keys())
    suffix = (f'_{variant}' if variant else '') + (f'_skip{skip_frames}' if skip_frames else '')

    def _save_fig(name: str) -> None:
        fn = f'lowres_temporal_stability_{name}{suffix}.png'
        for out_dir in out_dirs:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, fn), dpi=130)
        print(f'Wrote {fn} to: {", ".join(out_dirs)}')

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
    _save_fig('roi_signal')

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
    _save_fig('tsnr_map')

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
    _save_fig('last_minus_first')

def _cli_stability() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('datdirs', nargs='+')
    parser.add_argument('--seqname', default='ArbEPI')
    parser.add_argument('--tr', type=float, default=2.0, help='volume TR in seconds')
    parser.add_argument(
        '--skip-frames', type=int, default=0,
        help='drop this many leading frames before computing stats (non-steady-state transient)',
    )
    parser.add_argument(
        '--variant', default='',
        help="'' for the plain recon, 'b0' for recon/lowres_calib_b0.py's output",
    )
    args = parser.parse_args()
    main_stability(args.datdirs, args.seqname, args.tr, args.skip_frames, args.variant)


_COMMANDS = {
    'calib': _cli_calib,
    'stability': _cli_stability,
}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        sys.exit('usage: python -m recon.lowres_calib {' + ','.join(_COMMANDS) + '} ...')
    _cmd = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {_cmd}'
    _COMMANDS[_cmd]()
