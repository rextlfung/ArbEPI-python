"""Crop/resize a volume from the deGRE acquisition grid onto the EPI
acquisition grid.

Both this repo's ESPIRiT sensitivity maps (smaps.py's process_smaps) and its
B0 field maps (run_b0map.py) are estimated from the same deGRE acquisition
(chosen for GRE's speed/robustness, not spatial fidelity -- see params.py's
res_degre vs res) and need moving onto the finer, larger EPI grid before
either is usable inside an EPI reconstruction. Both share the same geometry
assumption: deGRE and EPI share an isocenter and an x/y FOV (params.py's
fov_degre always tracks fov's x/y exactly, only z differs -- see its "N_degre
... tracks the EPI FOV" comment), so only z ever needs cropping. This module
factors that one crop-then-resize routine out so process_smaps and
run_b0map's field-map resize don't each carry their own copy.
"""

import numpy as np
from scipy import ndimage


def _matlab_round(x: float) -> int:
    """MATLAB's round() rounds half away from zero; only ever called here on
    a non-negative value (see preprocessing/oephase.py's own copy, which
    handles negatives too, for the general case)."""
    return int(np.floor(x + 0.5))


def resize_to_epi_grid(
    vol: np.ndarray,
    fov_src: tuple[float, float, float],
    fov: tuple[float, float, float],
    n_target: tuple[int, int, int],
    order: int = 3,
) -> np.ndarray:
    """vol: [Nx_src, Ny_src, Nz_src, ...] -- any trailing axes (e.g. coils)
    pass through unresized. fov_src, fov: (fx, fy, fz) in meters, x/y assumed
    equal between the two (only z is cropped -- see module docstring).
    n_target: (Nx, Ny, Nz), the EPI acquisition grid.

    order: scipy.ndimage.zoom spline order (3 = cubic, matching MATLAB
    imresize3's default, not bit-identical to it -- same choice
    process_smaps has always made for coil maps). Pass order=0 (nearest) for
    a boolean/label volume, e.g. a validity mask, so no fractional values are
    invented at the resample.
    """
    Nx_src, Ny_src, Nz_src = vol.shape[:3]
    Nx, Ny, Nz = n_target
    if fov_src[2] < fov[2]:
        raise ValueError(
            f'resize_to_epi_grid: target z-FOV ({fov[2]:.4f} m) exceeds '
            f'source z-FOV ({fov_src[2]:.4f} m).'
        )

    # z_frac*Nz_src is always in [0, Nz_src/2) given the FOV check above, so
    # plain floor(x+0.5) rounding suffices.
    z_frac = (fov_src[2] - fov[2]) / fov_src[2] / 2
    z_start = _matlab_round(z_frac * Nz_src)
    z_end = _matlab_round(Nz_src - z_frac * Nz_src)
    if z_start < 0 or z_end > Nz_src or z_start >= z_end:
        raise ValueError(
            f'resize_to_epi_grid: computed z crop [{z_start}, {z_end}) '
            f'is out of range [0, {Nz_src}).'
        )
    vol = vol[:, :, z_start:z_end, ...]

    zoom = (Nx / vol.shape[0], Ny / vol.shape[1], Nz / vol.shape[2]) + (1.0,) * (vol.ndim - 3)
    if np.iscomplexobj(vol):
        return (ndimage.zoom(vol.real, zoom, order=order)
                + 1j * ndimage.zoom(vol.imag, zoom, order=order))
    return ndimage.zoom(vol.astype(np.float64), zoom, order=order)
