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

**Voxel-grid convention: FOV/edge-aligned, not pixel-center-aligned.** The
z-crop above already assumes N voxels tile the FOV edge-to-edge -- voxel i
spans physical extent `[i/N, (i+1)/N) * FOV` -- since `z_start`/`z_end` are
computed as plain proportional indices (`z_frac * Nz_src`) with no
pixel-center offset. `scipy.ndimage.zoom` must use the *same* convention for
the resize step, or the two steps disagree on where a given voxel physically
sits: its default (`grid_mode=False`) instead anchors the first/last pixel
*centers* to the array's endpoints (a length-N axis is treated as spanning
N-1 units, matching MATLAB's `imresize3` with `Antialiasing`/default corner
convention off), which is wrong here since deGRE and EPI cover the *same*
physical FOV per axis (params.py's `fov_degre` tracks `fov`) -- two grids of
the same FOV are edge-aligned by construction, not center-aligned.
`grid_mode=True` matches the crop step's own convention (confirmed
numerically: at this repo's real deGRE 108 -> EPI 240 x-resize, 216mm FOV, a
linear ramp resized with `grid_mode=False` disagrees with the analytic
voxel-center positions by a systematic ~0.27mm mean / 0.63mm max error;
`grid_mode=True` cuts that to ~0.006mm mean, with the residual ~0.39mm error
confined to the two outermost voxels on each axis, an unavoidable
extrapolation artifact of upsampling past the source grid's own edge voxel
centers -- see `tests/test_preprocessing_grid_resize.py`'s
`test_resize_to_epi_grid_matches_analytic_ramp_at_voxel_centers`).
`mode='nearest'` clamps that edge extrapolation to the boundary voxel's own
value rather than blending toward 0 (`grid_mode=True`'s other `mode` options
mix in wrap-around or reflected samples that make even less physical sense
for a truncated anatomical/field-map volume).

Note this changes the resize step's output values relative to this
function's pre-existing behavior (previously `grid_mode=False`, i.e.
pixel-center-aligned) -- CLAUDE.md's item 12 finding. No test elsewhere in
this repo pinned a specific interpolated value against the old convention:
the one real end-to-end validation on real data (`run_rss.py` against a
MATLAB/BART RSS reference, see CLAUDE.md) reconstructs via
root-sum-of-squares, which never reads `smaps` at all (`_rss_recon` ignores
its `_smaps` argument), so that validation is silent on this function's
alignment convention either way.
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
    # grid_mode=True + mode='nearest': FOV/edge-aligned, matching the z-crop
    # above's own convention -- see module docstring for why grid_mode=False
    # (scipy's default) is wrong for two grids covering the same FOV.
    zoom_kwargs = dict(order=order, grid_mode=True, mode='nearest')
    if np.iscomplexobj(vol):
        return (ndimage.zoom(vol.real, zoom, **zoom_kwargs)
                + 1j * ndimage.zoom(vol.imag, zoom, **zoom_kwargs))
    return ndimage.zoom(vol.astype(np.float64), zoom, **zoom_kwargs)
