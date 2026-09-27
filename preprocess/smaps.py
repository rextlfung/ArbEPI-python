"""Coil sensitivity maps: ESPIRiT (Uecker et al., MRM 2014) via sigpy, then
masking, resizing to a target grid, smoothing and RSS normalization.

Ports makeSmaps.m's ESPIRiT branch (with sigpy's EspiritCalib in place of
BART's ecalib) and process_smaps.m. Arrays are coils-last here; sigpy wants
coils first, so estimate_smaps transposes at its boundary.
"""

import numpy as np
import sigpy as sp
import sigpy.mri.app as mri_app
from scipy import ndimage

from preprocess.grid_resize import resize_to_epi_grid


def _default_device() -> sp.Device:
    """GPU if sigpy/cupy sees one, else CPU."""
    if sp.config.cupy_enabled:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return sp.Device(0)
    return sp.cpu_device


def estimate_smaps(
    ksp_gre: np.ndarray,
    calib_width: int = 24,
    thresh: float = 0.02,
    crop: float = 0.95,
    cal_size: int = 24,
    device: sp.Device | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """ESPIRiT maps from fully sampled k-space [Nx, Ny, Nz, Ncoils].

    Returns (smaps [cal_size]*3 + [Ncoils], emap [cal_size]*3): the maps and
    ESPIRiT's dominant-eigenvalue map, at the calibration resolution.

    cal_size: k-space is center-cropped (or zero-padded) to cal_size^3 before
        calibration. sigpy allocates a covariance array the size of its whole
        input, so a full 108^3 x 32-coil volume does not fit; the maps are
        smooth and get resized to the target grid by process_smaps anyway.
    crop: eigenvalue threshold; maps are zeroed where emap <= crop. sigpy's
        default 0.95, stricter than BART's 0.8. On 01_fullsamp_4p55mm it gives
        a support covering 43% of the volume against the object's 39%; higher
        values start cutting into the object, which is worse for SENSE than a
        slightly large support. process_smaps must get the same value.
    device: None picks a GPU when available. Returns numpy arrays either way.
    """
    device = _default_device() if device is None else sp.Device(device)
    ksp_coils_first = np.moveaxis(ksp_gre, -1, 0)
    if cal_size is not None:
        ncoils = ksp_coils_first.shape[0]
        ksp_coils_first = sp.resize(ksp_coils_first, (ncoils, cal_size, cal_size, cal_size))
    calib = mri_app.EspiritCalib(
        ksp_coils_first,
        calib_width=calib_width,
        thresh=thresh,
        crop=crop,
        device=device,
        show_pbar=False,
        output_eigenvalue=True,
    )
    mps, emap = calib.run()
    mps = sp.to_device(mps, sp.cpu_device)
    emap = sp.to_device(emap, sp.cpu_device)
    return np.moveaxis(mps, 0, -1), emap[0]


def _masked_gaussian_smooth(
    smaps: np.ndarray, mask: np.ndarray, sigma_vox: np.ndarray
) -> np.ndarray:
    """Per-coil Gaussian smoothing normalized by the smoothed mask, so the
    zero background doesn't darken the edge. Real and imaginary parts are
    smoothed separately. The result extends slightly past the mask; callers
    re-mask."""
    weight = mask.astype(np.float64)
    denom = ndimage.gaussian_filter(weight, sigma_vox)
    denom[denom < 1e-6] = 1
    out = np.empty_like(smaps)
    for c in range(smaps.shape[-1]):
        num_real = ndimage.gaussian_filter(smaps[..., c].real * weight, sigma_vox)
        num_imag = ndimage.gaussian_filter(smaps[..., c].imag * weight, sigma_vox)
        out[..., c] = (num_real + 1j * num_imag) / denom
    return out


def process_smaps(
    smaps_raw: np.ndarray,
    emap: np.ndarray,
    fov_gre: tuple[float, float, float],
    fov: tuple[float, float, float],
    n_target: tuple[int, int, int],
    crop: float,
    smooth_sigma_mm: float = 6.0,
    zero_pad_z: bool = False,
) -> np.ndarray:
    """estimate_smaps' output -> maps on the target grid n_target (fov, m).

    1. Resize maps and emap from the deGRE grid (resize_to_epi_grid: z crop,
       then cubic spline).
    2. Mask where the *resized* emap > crop. Thresholding the continuous
       eigenvalue map on the fine grid gives a round boundary; binarizing at
       the 24^3 calibration resolution first gave a blocky one.
    3. Optionally smooth (smooth_sigma_mm, on the target grid) to remove the
       rippling texture ESPIRiT leaves at its calibration resolution; real
       coil sensitivities vary over centimeters. Re-mask afterwards.
    4. Divide by the per-voxel root-sum-of-squares over coils.

    The exact re-masks matter: the cubic spline leaks ~1e-6 values outside the
    object, which step 4 (and recon's own RSS normalization) would otherwise
    scale back up to unit magnitude.
    """
    smaps = resize_to_epi_grid(smaps_raw, fov_gre, fov, n_target, order=3, zero_pad_z=zero_pad_z)
    emap_resized = resize_to_epi_grid(emap, fov_gre, fov, n_target, order=3, zero_pad_z=zero_pad_z)
    target_mask = emap_resized > crop
    smaps = smaps * target_mask[..., None]

    if smooth_sigma_mm > 0:
        vox_mm = np.array(fov) / np.array(n_target) * 1000
        smaps = _masked_gaussian_smooth(smaps, target_mask, smooth_sigma_mm / vox_mm)
        smaps = smaps * target_mask[..., None]

    rss = np.sqrt(np.sum(np.abs(smaps) ** 2, axis=-1))
    rss[rss < np.finfo(rss.dtype).eps] = 1
    return smaps / rss[..., None]
