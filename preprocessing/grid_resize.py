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
pixel-center-aligned) -- docs/review-findings.md's item 12 finding. No test elsewhere in
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
    zero_pad_z: bool = False,
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

    zero_pad_z: when the *target* z-FOV exceeds the source's (the case this
    function otherwise raises on -- see below), setting this True resizes
    onto only the inner target-grid slices that fall within the source's
    real coverage and zero-fills the rest, instead of raising. Use only
    when a real, already-acquired dataset has this mismatch (this function
    cannot retroactively acquire the missing coverage) and the caller has
    made a deliberate call that a zero (not fabricated) sensitivity/field
    value at those edge slices is acceptable -- see
    docs/review-findings.md item 196 for the real case this was added for
    (deGRE's fixed z-FOV slightly smaller than one EPI resolution variant's
    own, rounding-driven z-FOV) and why zero, not edge-replication, is the
    right fill value (no real coil-sensitivity data exists there to
    replicate from).
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

    return _zoom_grid_aligned(vol, (Nx, Ny, Nz), order)


def _zoom_grid_aligned(vol: np.ndarray, n_target: tuple[int, int, int], order: int) -> np.ndarray:
    """The shared grid_mode=True/mode='nearest' zoom step -- see module
    docstring for why this convention, not scipy's default, is correct for
    two grids covering the same physical FOV."""
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
    """zero_pad_z=True path of resize_to_epi_grid (fov_src[2] < fov[2]):
    resize the source (no z-crop needed -- it's already <= the target FOV)
    onto only the inner target-grid z-slices that fall within the source's
    own real coverage, then embed that into a full-size zero array. The
    inner range is computed conservatively (ceil the inner start, floor the
    inner end) so a boundary slice straddling real/fake coverage is zeroed
    entirely rather than credited with partial real data.
    """
    Nx_src, Ny_src, Nz_src = vol.shape[:3]
    Nx, Ny, Nz = n_target

    # Fraction of the TARGET z-extent, per side, that lies outside the
    # source's real coverage.
    z_frac = (fov[2] - fov_src[2]) / fov[2] / 2
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
