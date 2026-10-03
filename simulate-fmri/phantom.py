"""Phantoms and coil sensitivities for the ArbEPI simulations.

- brainweb_phantom: SNAKE's BrainWeb phantom (downloaded on first use, then
  cached) reduced to white matter, gray matter and CSF with 3 T relaxation
  times.
- ellipsoid_phantom: the same three tissues as nested ellipsoids, built
  analytically. Needs no download; used by the tests and for quick runs.
- ellipsoid_phantom_roi: where to put the activation in that phantom.
- centered_fov: places an acquisition field of view on a phantom's tissue.
- to_acquisition_grid: resamples a phantom onto the simulation grid and
  attaches coil sensitivities defined on that grid.
- birdcage_smaps: rings of coils around the z axis.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from snake.core.phantom import Phantom
from snake.core.simulation import FOVConfig, HardwareConfig, SimConfig

from .handlers import ellipsoid_mask

# name: (T1 ms, T2 ms, T2* ms, proton density, susceptibility ppm), the column
# order of SNAKE's tissue tables. SNAKE ships 1.5 T and 7 T tables only; these
# are approximate 3 T values. T1 and T2: Wansapura et al., J Magn Reson Imaging
# 1999;9:531 (gray 1331/110 ms, white 832/80 ms). T2*: Peters et al., Magn
# Reson Imaging 2007;25:748 (gray 66 ms, white 53 ms). CSF, density and
# susceptibility are SNAKE's own. Only T1, T2* and density enter the
# gradient-echo signal.
TISSUE_PROPS_3T = {
    'wm': (832.0, 80.0, 53.0, 0.77, -9.05),
    'gm': (1331.0, 110.0, 66.0, 0.86, -9.05),
    'csf': (4000.0, 2000.0, 2000.0, 1.0, -9.05),
}
TISSUES = ('wm', 'gm', 'csf')


def _props(labels) -> NDArray[np.float32]:
    return np.array([TISSUE_PROPS_3T[str(name)] for name in labels], dtype=np.float32)


def brainweb_phantom(sub_id: int = 4, output_res: float = 1.0) -> Phantom:
    """BrainWeb subject `sub_id`'s white matter, gray matter and CSF maps at
    `output_res` mm, with TISSUE_PROPS_3T, on BrainWeb's own grid and without
    coil sensitivities (see to_acquisition_grid)."""
    # tissue_7T is SNAKE's wm/gm/csf table; its 7 T values are replaced below.
    # A one-coil config keeps from_brainweb from making its own sensitivities.
    single_coil = SimConfig(hardware=HardwareConfig(n_coils=1))
    phantom = Phantom.from_brainweb(
        sub_id=sub_id,
        sim_conf=single_coil,
        tissue_file='tissue_7T',
        tissue_select=list(TISSUES),
        output_res=output_res,
    )
    phantom.props = _props(phantom.labels)
    return phantom


def ellipsoid_phantom(
    shape: tuple[int, int, int] = (64, 64, 48), res_mm: float = 3.0
) -> Phantom:
    """A head-sized analytic phantom centered on the world origin: a white
    matter core inside a gray matter shell, with two CSF ventricles. Tissue
    fractions are 0 or 1 and sum to at most 1."""
    affine = np.diag([res_mm, res_mm, res_mm, 1.0]).astype(np.float32)
    affine[:3, 3] = -(np.array(shape) - 1) / 2 * res_mm
    half = np.array(shape) * res_mm / 2

    def ellipsoid(center, semi_axes):
        return ellipsoid_mask(shape, affine, center, semi_axes)

    brain = ellipsoid((0, 0, 0), 0.80 * half)
    core = ellipsoid((0, 0, 0), 0.62 * half)
    ventricles = ellipsoid((-0.18 * half[0], 0, 0), 0.10 * half * (1, 2.5, 1.5)) | ellipsoid(
        (0.18 * half[0], 0, 0), 0.10 * half * (1, 2.5, 1.5)
    )
    masks = {
        'wm': core & ~ventricles,
        'gm': brain & ~core,
        'csf': ventricles,
    }
    return Phantom(
        name='ellipsoid',
        masks=np.stack([masks[name] for name in TISSUES]).astype(np.float32),
        labels=np.array(TISSUES),
        props=_props(TISSUES),
        affine=affine,
    )


def ellipsoid_phantom_roi(
    shape: tuple[int, int, int] = (64, 64, 48), res_mm: float = 3.0
) -> dict[str, tuple[float, float, float]]:
    """An activation ellipsoid for ellipsoid_phantom(shape, res_mm): a patch of
    its posterior gray matter shell, as EllipsoidActivationHandler arguments."""
    half = np.array(shape) * res_mm / 2
    return {
        'center_mm': (0.0, float(-0.71 * half[1]), 0.0),
        'semi_axes_mm': tuple(float(v) for v in (0.40, 0.12, 0.30) * half),
        'euler_angles': (0.0, 0.0, 0.0),
    }


def centered_fov(
    phantom: Phantom, shape: tuple[int, int, int], res_mm: tuple[float, float, float]
) -> FOVConfig:
    """The (shape, res_mm) field of view centered on the bounding box of the
    phantom's tissue. Axis-aligned with the phantom."""
    occupied = phantom.masks.sum(axis=0) > 0.1
    center_vox = [
        (np.flatnonzero(occupied.any(axis=tuple(a for a in range(3) if a != ax)))[[0, -1]]).mean()
        for ax in range(3)
    ]
    center_mm = phantom.affine[:3, :3] @ np.array(center_vox) + phantom.affine[:3, 3]
    res = np.asarray(res_mm, dtype=float)
    offset = center_mm - (np.array(shape) - 1) / 2 * res
    return FOVConfig(
        size=tuple(float(s) for s in np.array(shape) * res),
        res_mm=tuple(float(r) for r in res),
        offset=tuple(float(o) for o in offset),
    )


def birdcage_smaps(
    shape: tuple[int, int, int], n_coils: int, coils_per_ring: int = 8, r: float = 1.5
) -> NDArray[np.complex64]:
    """(n_coils, X, Y, Z) sensitivities of rings of coils around the z axis,
    unit root-sum-of-squares at every voxel.

    sigpy.mri.sim.birdcage_maps' model (1 / distance magnitude, azimuthal
    phase), with the rings around the last array axis and stacked along it.
    SNAKE's own get_smaps puts a single ring around the first axis, which for
    these arrays is x: no coil would then distinguish positions along z, one
    of the two axes ArbEPI undersamples.

    r: ring radius in units of the half field of view. Rings are spaced half a
    field of view apart along z, centered on the volume.
    """
    nx, ny, nz = shape
    n_rings = math.ceil(n_coils / coils_per_ring)
    c = np.arange(n_coils).reshape(-1, 1, 1, 1)
    ring = c // coils_per_ring
    angle = c * (2 * np.pi / coils_per_ring)
    x, y, z = np.meshgrid(
        (np.arange(nx) - nx / 2) / (nx / 2),
        (np.arange(ny) - ny / 2) / (ny / 2),
        (np.arange(nz) - nz / 2) / (nz / 2),
        indexing='ij',
    )
    x_co = x - r * np.cos(angle)
    y_co = y - r * np.sin(angle)
    z_co = z - (ring - 0.5 * (n_rings - 1))
    phase = np.arctan2(x_co, -y_co) - (c + ring) * (2 * np.pi / coils_per_ring)
    out = np.exp(1j * phase) / np.sqrt(x_co**2 + y_co**2 + z_co**2)
    out /= np.sqrt(np.sum(np.abs(out) ** 2, axis=0))
    return out.astype(np.complex64)


def to_acquisition_grid(
    phantom: Phantom, sim_conf: SimConfig, coils_per_ring: int = 8
) -> Phantom:
    """`phantom` resampled onto sim_conf's field of view, with
    sim_conf.hardware.n_coils birdcage sensitivities defined on that grid (none
    for a single coil)."""
    # mode='constant': air, not the phantom's edge voxels, beyond its own grid.
    # n_jobs: one process per tissue (SNAKE's default starts one per CPU).
    on_grid = phantom.resample(
        new_affine=sim_conf.fov.affine, new_shape=sim_conf.shape, mode='constant',
        n_jobs=len(phantom.masks),
    )
    on_grid.masks = np.clip(on_grid.masks, 0, 1)  # cubic resampling overshoots
    n_coils = sim_conf.hardware.n_coils
    on_grid.smaps = (
        birdcage_smaps(sim_conf.shape, n_coils, coils_per_ring) if n_coils > 1 else None
    )
    return on_grid
