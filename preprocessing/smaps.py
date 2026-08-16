"""Coil sensitivity map estimation and post-processing.

Ports makeSmaps.m's 'bart' branch -- now sigpy.mri.app.EspiritCalib, BART's
ecalib was dropped, see CLAUDE.md -- and process_smaps.m's 4-step
mask/crop/resize/normalize pipeline. makeSmaps.m's 'pisco' branch is not
ported: SENSEmethod is effectively always sigpy-ESPIRiT in this port, and
PISCO (a separate, large MATLAB toolbox) has no available Python port.

Convention throughout this repo is coils-last ([..., Ncoils]); sigpy's
EspiritCalib expects coils-first ([Ncoils, ...]), so estimate_smaps
transposes at its boundary rather than propagating that convention further.
"""

import numpy as np
import sigpy as sp
import sigpy.mri.app as mri_app
from scipy import ndimage


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


def _matlab_round(x: float) -> int:
    """See preprocessing/oephase.py's _matlab_round -- MATLAB rounds half
    away from zero; only ever called here on a non-negative value."""
    return int(np.floor(x + 0.5))


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
    Nx_gre, Ny_gre, Nz_gre, _ = smaps_raw.shape
    Nx, Ny, Nz = n_target
    if fov_gre[2] < fov[2]:
        raise ValueError(
            f'process_smaps: EPI z-FOV ({fov[2]:.3f} m) exceeds '
            f'GRE z-FOV ({fov_gre[2]:.3f} m).'
        )

    # 1. Eigenvalue support mask.
    eig_mask = emap > threshold_mask
    smaps = smaps_raw * eig_mask[..., None]

    # 2. Crop z to match EPI FOV. z_frac*Nz_gre is always in [0, Nz_gre/2)
    # given the FOV check above, so plain floor(x+0.5) rounding suffices.
    z_frac = (fov_gre[2] - fov[2]) / fov_gre[2] / 2
    z_start = _matlab_round(z_frac * Nz_gre)
    z_end = _matlab_round(Nz_gre - z_frac * Nz_gre)
    if z_start < 0 or z_end > Nz_gre or z_start >= z_end:
        raise ValueError(
            f'process_smaps: computed z crop [{z_start}, {z_end}) '
            f'is out of range [0, {Nz_gre}).'
        )
    smaps = smaps[:, :, z_start:z_end, :]

    # 3. Interpolate to EPI grid. Cubic spline, matching MATLAB imresize3's
    # default; not bit-identical to it (different implementation).
    zoom = (Nx / smaps.shape[0], Ny / smaps.shape[1], Nz / smaps.shape[2], 1)
    smaps = ndimage.zoom(smaps.real, zoom, order=3) + 1j * ndimage.zoom(smaps.imag, zoom, order=3)

    # 4. Normalize: divide by the cross-coil RSS so sum(|s_c|^2) <= 1
    # everywhere, matching the ESPIRiT convention regularized SENSE recon
    # (sigpy or otherwise) expects when no explicit step size is given.
    rss = np.sqrt(np.sum(np.abs(smaps) ** 2, axis=-1))
    rss[rss < np.finfo(rss.dtype).eps] = 1
    return smaps / rss[..., None]
