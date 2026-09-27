"""Move a volume from the deGRE grid onto the EPI grid.

Sensitivity, B0 and R2* maps are all estimated on the deGRE grid and needed
on the EPI grid. The two acquisitions share an isocenter and their x/y FOV
(params.py's fov_degre tracks fov in x/y), so only z needs cropping: the
central part of the deGRE slab matching the EPI z-FOV is kept, then the
volume is resampled to the EPI matrix size.

Voxel convention: FOV/edge-aligned. N voxels tile the FOV edge to edge, which
is what the proportional z crop assumes, so the resample uses
scipy.ndimage.zoom(grid_mode=True). scipy's default (grid_mode=False) anchors
the first/last voxel centers instead and misplaced a linear ramp by 0.27 mm
on average (0.63 mm max) at the real 108 -> 240 resize; grid_mode=True gets
0.006 mm (review item 12; tests/test_preprocess_grid_resize.py).
mode='nearest' holds the edge value where the target grid extends past the
outermost source voxel centers.
"""

import numpy as np
from scipy import ndimage

from preprocess.utils import matlab_round


def resize_to_epi_grid(
    vol: np.ndarray,
    fov_src: tuple[float, float, float],
    fov: tuple[float, float, float],
    n_target: tuple[int, int, int],
    order: int = 3,
    zero_pad_z: bool = False,
) -> np.ndarray:
    """vol [Nx_src, Ny_src, Nz_src, ...] -> [Nx, Ny, Nz, ...] on the target
    grid n_target with FOV `fov` (m). Trailing axes pass through. The x/y FOVs
    must match.

    order: spline order; 3 (cubic, like MATLAB imresize3) for maps, 0 for
        masks so they stay binary.
    zero_pad_z: if the target z-FOV is larger than the source's (normally an
        error), resample onto the inner target slices the source covers and
        zero the rest (review items 196, 203: an EPI resolution whose rounded
        z-FOV slightly exceeds the deGRE slab).
    """
    Nx_src, Ny_src, Nz_src = vol.shape[:3]
    Nx, Ny, Nz = n_target
    if not np.allclose(fov_src[:2], fov[:2], rtol=1e-6, atol=1e-6):
        raise ValueError(
            f'resize_to_epi_grid: source x/y FOV {fov_src[:2]} does not match '
            f'target x/y FOV {fov[:2]} -- only z is cropped/resized here, so '
            f'x/y must already agree.'
        )
    if fov_src[2] < fov[2]:
        if not zero_pad_z:
            raise ValueError(
                f'resize_to_epi_grid: target z-FOV ({fov[2]:.4f} m) exceeds '
                f'source z-FOV ({fov_src[2]:.4f} m).'
            )
        return _resize_with_zero_pad_z(vol, fov_src, fov, n_target, order)

    z_frac = (fov_src[2] - fov[2]) / fov_src[2] / 2
    z_start = matlab_round(z_frac * Nz_src)
    z_end = matlab_round(Nz_src - z_frac * Nz_src)
    if z_start < 0 or z_end > Nz_src or z_start >= z_end:
        raise ValueError(
            f'resize_to_epi_grid: computed z crop [{z_start}, {z_end}) '
            f'is out of range [0, {Nz_src}).'
        )
    vol = vol[:, :, z_start:z_end, ...]

    return _zoom_grid_aligned(vol, (Nx, Ny, Nz), order)


def _zoom_grid_aligned(vol: np.ndarray, n_target: tuple[int, int, int], order: int) -> np.ndarray:
    """scipy.ndimage.zoom with grid_mode=True, mode='nearest' (see the module
    docstring); complex volumes are resampled as real and imaginary parts."""
    Nx, Ny, Nz = n_target
    zoom = (Nx / vol.shape[0], Ny / vol.shape[1], Nz / vol.shape[2]) + (1.0,) * (vol.ndim - 3)
    zoom_kwargs = dict(order=order, grid_mode=True, mode='nearest')
    if np.iscomplexobj(vol):
        return (ndimage.zoom(vol.real, zoom, **zoom_kwargs)
                + 1j * ndimage.zoom(vol.imag, zoom, **zoom_kwargs))
    return ndimage.zoom(vol.astype(np.float64), zoom, **zoom_kwargs)


def _resize_with_zero_pad_z(
    vol: np.ndarray,
    fov_src: tuple[float, float, float],
    fov: tuple[float, float, float],
    n_target: tuple[int, int, int],
    order: int,
) -> np.ndarray:
    """zero_pad_z path: resample onto the inner target slices inside the
    source's z coverage (rounded inward, so a slice that is only partly
    covered gets zero) and zero-fill the rest."""
    Nx, Ny, Nz = n_target
    z_frac = (fov[2] - fov_src[2]) / fov[2] / 2  # uncovered fraction of the target, per side
    z_start = int(np.ceil(z_frac * Nz - 1e-9))
    z_end = int(np.floor(Nz - z_frac * Nz + 1e-9))
    if z_start < 0 or z_end > Nz or z_start >= z_end:
        raise ValueError(
            f'resize_to_epi_grid: zero-pad inner target range [{z_start}, {z_end}) '
            f'is out of range [0, {Nz}).'
        )

    inner = _zoom_grid_aligned(vol, (Nx, Ny, z_end - z_start), order)
    out = np.zeros((Nx, Ny, Nz) + vol.shape[3:], dtype=inner.dtype)
    out[:, :, z_start:z_end, ...] = inner
    return out
