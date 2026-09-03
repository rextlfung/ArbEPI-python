"""Coil sensitivity map estimation and post-processing.

Ports makeSmaps.m's 'bart' branch -- now sigpy.mri.app.EspiritCalib, BART's
ecalib was dropped, see CLAUDE.md -- and process_smaps.m's 4-step
mask/crop/resize/normalize pipeline. makeSmaps.m's 'pisco' branch is not
ported: SENSEmethod is effectively always sigpy-ESPIRiT in this port, and
PISCO (a separate, large MATLAB toolbox) has no available Python port.

ESPIRiT: Uecker M, Lai P, Murphy MJ, et al. "ESPIRiT -- an eigenvalue
approach to autocalibrating parallel MRI: where SENSE meets GRAPPA." Magn
Reson Med. 2014;71(3):990-1001.

Convention throughout this repo is coils-last ([..., Ncoils]); sigpy's
EspiritCalib expects coils-first ([Ncoils, ...]), so estimate_smaps
transposes at its boundary rather than propagating that convention further.
"""

import os

import h5py
import numpy as np
import sigpy as sp
import sigpy.mri.app as mri_app

from preprocessing.config import PreprocessingConfig, SeqParams, SeqPaths
from preprocessing.grid_resize import resize_to_epi_grid
from preprocessing.nifti_io import save_recon_nifti


def estimate_smaps(
    ksp_gre: np.ndarray,
    calib_width: int = 24,
    thresh: float = 0.02,
    crop: float = 0.8,
    cal_size: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """ESPIRiT sensitivity map estimation from fully-sampled GRE k-space.

    ksp_gre: [Nx, Ny, Nz, Ncoils] complex.
    Returns (smaps, emap): smaps [cal_size, cal_size, cal_size, Ncoils]
    complex; emap [cal_size, cal_size, cal_size] float, the dominant-
    eigenvalue map (sigpy's EspiritCalib "currently only supports
    outputting one set of maps", so unlike BART's ecalib there's no
    emaps(...,end)-style selection among several map sets to do).

    cal_size: center-crop ksp_gre's spatial dims to this matrix size (per
        axis) before running ESPIRiT, rather than passing the full
        acquisition grid. sigpy's EspiritCalib allocates a coil-covariance
        array sized to its *entire* input k-space's spatial shape (not
        just calib_width -- `AHA = zeros(image_shape + (Ncoils, Ncoils))`
        in its own source), so passing a full 3D acquisition grid directly
        is impractical: confirmed against real project data, a
        108^3/12-coil GRE volume allocated 2.9GB for that one array alone
        and, combined with a non-vectorized per-calibration-kernel Python
        loop, thrashed 14GB+ of memory and never completed in over an hour
        (killed). Cropping k-space reduces resolution while preserving
        FOV -- process_smaps() already resizes the result up to the EPI
        target grid afterward regardless of what resolution ESPIRiT itself
        ran at, so there's no need to calibrate at full acquisition
        resolution. Default matches calib_width (24) so EspiritCalib's own
        internal calibration-region crop becomes a no-op and the
        covariance/eigenmap stage runs at the same small size throughout.
    """
    ksp_coils_first = np.moveaxis(ksp_gre, -1, 0)
    if cal_size is not None:
        ncoils = ksp_coils_first.shape[0]
        ksp_coils_first = sp.resize(ksp_coils_first, (ncoils, cal_size, cal_size, cal_size))
    calib = mri_app.EspiritCalib(
        ksp_coils_first,
        calib_width=calib_width,
        thresh=thresh,
        crop=crop,
        show_pbar=False,
        output_eigenvalue=True,
    )
    mps, emap = calib.run()
    smaps = np.moveaxis(mps, 0, -1)
    return smaps, emap[0]


def process_smaps(
    smaps_raw: np.ndarray,
    emap: np.ndarray,
    fov_gre: tuple[float, float, float],
    fov: tuple[float, float, float],
    n_target: tuple[int, int, int],
    threshold_mask: float,
) -> np.ndarray:
    """Mask, z-crop, resize, and RSS-normalize raw sensitivity maps to the
    EPI acquisition grid. Ports process_smaps.m's 'bart' eigenvalue
    convention (high eigenvalue = inside object) -- the only convention
    relevant here since PISCO isn't ported.

    Assumption carried over from process_smaps.m: the GRE and EPI
    acquisitions share the same isocenter, so a symmetric z-crop is valid.

    smaps_raw: [Nx_gre, Ny_gre, Nz_gre, Ncoils]
    emap: [Nx_gre, Ny_gre, Nz_gre]
    fov_gre, fov: (fx, fy, fz) in meters
    n_target: (Nx, Ny, Nz), the EPI acquisition grid
    """
    # 1. Eigenvalue support mask.
    eig_mask = emap > threshold_mask
    smaps = smaps_raw * eig_mask[..., None]

    # 2+3. Crop z to match EPI FOV, then interpolate (cubic spline) to the
    # EPI grid -- see grid_resize.py's module docstring for why this
    # deGRE-grid-to-EPI-grid crop+resize is shared with run_b0map.py.
    smaps = resize_to_epi_grid(smaps, fov_gre, fov, n_target, order=3)

    # 4. Normalize: divide by the cross-coil RSS so sum(|s_c|^2) <= 1
    # everywhere, matching the ESPIRiT convention regularized SENSE recon
    # (sigpy or otherwise) expects when no explicit step size is given.
    rss = np.sqrt(np.sum(np.abs(smaps) ** 2, axis=-1))
    rss[rss < np.finfo(rss.dtype).eps] = 1
    return smaps / rss[..., None]


def load_smaps(
    cfg: PreprocessingConfig, paths: SeqPaths, seq_params: SeqParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(smaps, smaps_degre, emap_degre, nvcoils): sensitivity maps on both
    the EPI grid (the SENSE encoding operator's own grid) and the *deGRE*
    grid (for preprocessing/julia/b0map.jl's `smap` argument -- see its
    module docstring for why passing real smaps there, instead of leaving
    B0 field-map estimation to MRIFieldmaps' phase-contrast coil-combine
    fallback, is expected to reduce field-map noise in this pipeline's real
    low-per-coil-SNR object-center regions); `emap_degre` is ESPIRiT's own
    dominant-eigenvalue map, also resized to the deGRE grid, for an
    optional ESPIRiT-informed image-support mask in b0map.jl (thresholded
    the same way process_smaps already does for `eig_mask`) -- a
    complement to its own magnitude-based mask, not a guaranteed fix for
    the same reason a raw magnitude threshold already has known limits
    near low-SNR/partial-volume voxels (see CLAUDE.md's recon/ section).

    All three deGRE-grid arrays are resized from the same `cal_size`-
    cropped ESPIRiT calibration (`smaps_raw`/`emap`, see `estimate_smaps`)
    via `process_smaps`/`resize_to_epi_grid` -- deGRE-grid uses `fov_gre`
    as both source *and* target FOV (only resolution changes, no z-crop),
    EPI-grid is the existing crop+resize. Loads from/writes
    `<datdir>/recon/smaps_<seqname>_sigpy.h5` (was `recon_frames.py`'s
    private `_load_smaps` -- moved here, and extended with the
    `smaps_degre`/`emap_degre` datasets, so `run_b0map.py` can reuse the
    same cache instead of re-running ESPIRiT). An older cache written
    before these existed is backfilled in place rather than re-estimating
    from scratch (`smaps_raw`/`emap` are already cached).
    """
    fn_smaps = os.path.join(cfg.datdir, 'recon', f'smaps_{paths.seqname}_sigpy.h5')
    fn_smaps_nifti = fn_smaps[: -len('.h5')] + '.nii.gz'
    fn_gre = os.path.join(cfg.datdir, 'recon', f'{paths.seqname}_gre.h5')
    fov_degre = tuple(seq_params.fov_degre)
    n_target_degre = (seq_params.Nx_degre, seq_params.Ny_degre, seq_params.Nz_degre)

    # Guard against a stale cache from a run with a different Nvcoils (e.g.
    # coil-compression settings changed since the cache was written) --
    # matching preprocess.py's own smaps_valid check. Falls through to
    # re-estimation below on mismatch, same as if fn_smaps didn't exist. If
    # fn_gre isn't available to check against (e.g. already cleaned up),
    # trust the existing cache rather than failing outright.
    smaps_cache_valid = os.path.exists(fn_smaps)
    if smaps_cache_valid and os.path.exists(fn_gre):
        with h5py.File(fn_smaps, 'r') as f:
            cached_nvcoils = int(f.attrs['Nvcoils'])
        with h5py.File(fn_gre, 'r') as f:
            current_nvcoils = f['ksp_gre'].shape[-1]
        smaps_cache_valid = cached_nvcoils == current_nvcoils

    if smaps_cache_valid:
        print(f'Loading precomputed sensitivity maps from {fn_smaps}')
        with h5py.File(fn_smaps, 'r') as f:
            smaps, nvcoils = f['smaps'][()], int(f.attrs['Nvcoils'])
            has_degre = 'smaps_degre' in f and 'emap_degre' in f
            if has_degre:
                smaps_degre, emap_degre = f['smaps_degre'][()], f['emap_degre'][()]
            else:
                smaps_raw, emap = f['smaps_raw'][()], f['emap'][()]
        if not has_degre:
            print(f'  Backfilling deGRE-grid smaps/emap into {fn_smaps}...')
            smaps_degre = process_smaps(
                smaps_raw, emap, fov_degre, fov_degre, n_target_degre, cfg.threshold_mask,
            )
            emap_degre = resize_to_epi_grid(emap, fov_degre, fov_degre, n_target_degre, order=3)
            with h5py.File(fn_smaps, 'a') as f:
                f.create_dataset('smaps_degre', data=smaps_degre)
                f.create_dataset('emap_degre', data=emap_degre)
        if not os.path.exists(fn_smaps_nifti):
            # Backfill: cache was written before the NIfTI export existed.
            save_recon_nifti(
                fn_smaps[: -len('.h5')], smaps, fov=seq_params.fov,
                seqname=paths.seqname, Nvcoils=nvcoils,
            )
        return smaps, smaps_degre, emap_degre, nvcoils

    if os.path.exists(fn_smaps):
        print(f'Cached sensitivity maps at {fn_smaps} have stale Nvcoils -- re-estimating.')
    print('Sensitivity maps not found. Estimating via sigpy ESPIRiT...')
    with h5py.File(fn_gre, 'r') as f:
        ksp_gre = f['ksp_gre'][()]
    nvcoils = ksp_gre.shape[-1]
    smaps_raw, emap = estimate_smaps(ksp_gre)
    smaps = process_smaps(
        smaps_raw, emap, fov_degre, tuple(seq_params.fov),
        (seq_params.Nx, seq_params.Ny, seq_params.Nz), cfg.threshold_mask,
    )
    smaps_degre = process_smaps(
        smaps_raw, emap, fov_degre, fov_degre, n_target_degre, cfg.threshold_mask,
    )
    emap_degre = resize_to_epi_grid(emap, fov_degre, fov_degre, n_target_degre, order=3)
    with h5py.File(fn_smaps, 'w') as f:
        f.create_dataset('smaps_raw', data=smaps_raw)
        f.create_dataset('emap', data=emap)
        f.create_dataset('smaps', data=smaps)
        f.create_dataset('smaps_degre', data=smaps_degre)
        f.create_dataset('emap_degre', data=emap_degre)
        f.attrs['Nvcoils'] = nvcoils
    # Coil axis stands in for save_recon_nifti's "frames" axis -- FSLeyes'
    # volume slider then scrolls through per-coil maps, magnitude-only
    # (NIfTI has no complex dtype; see nifti_io module docstring).
    save_recon_nifti(
        fn_smaps[: -len('.h5')], smaps, fov=seq_params.fov, seqname=paths.seqname, Nvcoils=nvcoils,
    )
    return smaps, smaps_degre, emap_degre, nvcoils
