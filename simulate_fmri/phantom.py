"""Phantoms and coil sensitivities for the ArbEPI simulations.

- brainweb_phantom: SNAKE's BrainWeb phantom (downloaded on first use, then
  cached) reduced to white matter, gray matter and CSF with 3 T relaxation
  times.
- ellipsoid_phantom: the same three tissues as nested ellipsoids, built
  analytically. Needs no download; used by the tests and for quick runs.
- ellipsoid_phantom_roi, ellipsoid_phantom_rois: where to put activations in
  that phantom.
- brainweb_anatomy, ellipsoid_anatomy: either phantom with the outline of its
  head and air cavities, for the field map of the raw-data simulation.
- place_fov: places an acquisition field of view on a phantom's tissue.
- to_acquisition_grid: resamples a phantom onto the simulation grid and
  attaches coil sensitivities defined on that grid.
- birdcage_smaps: rings of coils around the z axis.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from snake.core.phantom import Phantom
from snake.core.phantom.static import SNAKE_CACHE_DIR
from snake.core.simulation import FOVConfig, HardwareConfig, SimConfig

from .handlers import ellipsoid_mask

# name: (T1 ms, T2 ms, T2* ms, proton density, susceptibility ppm), the column
# order of SNAKE's tissue tables. SNAKE ships 1.5 T and 7 T tables only; these
# are 3 T values from the literature.
# - T1 and T2: Wansapura et al., J Magn Reson Imaging 1999;9:531 (gray 1331 and
#   80 ms, white 832 and 110 ms).
# - T2*: Peters et al., Proc ISMRM 14 (2006) 926, the conference version of
#   Magn Reson Imaging 2007;25:748: cortical gray 59.7 ms and white 54.6 ms,
#   with the signal loss from through-slice dephasing fitted and removed
#   (47.1 and 44.0 ms without that correction). The corrected values are the
#   ones to use where the field inhomogeneity is simulated separately.
#   Other 3 T reports: Wansapura 1999, with no such correction, 41.6-51.8 ms
#   in gray and 44.7-48.4 ms in white matter; van der Zwaag et al., NeuroImage
#   2009;47:1425 imply R2* = 18 1/s (55 ms) in active motor cortex.
# - CSF (its 1.5 T table), density and susceptibility are SNAKE's own.
# Only T1, T2* and density enter the gradient-echo signal.
TISSUE_PROPS_3T = {
    'wm': (832.0, 110.0, 54.6, 0.77, -9.05),
    'gm': (1331.0, 80.0, 59.7, 0.86, -9.05),
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


@dataclass
class Anatomy:
    """A phantom and what the raw-data simulation needs around it.

    phantom: the tissue maps (wm, gm, csf) on their own grid.
    head, head_affine: fraction of each voxel that is tissue of any kind
        (brain, skull, scalp; 0 = air), on a coarser grid, and that grid's
        voxel-to-world matrix. The susceptibility model (b0.py).
    cavities: air cavities to carve out of the head.
    rois: the regions that can be activated, by name: an ellipsoid
        (EllipsoidActivationHandler's center_mm, semi_axes_mm, euler_angles)
        or a list of them (a bilateral region).
    """

    phantom: Phantom
    head: NDArray
    head_affine: NDArray
    cavities: tuple = ()
    rois: dict = field(default_factory=dict)

    @property
    def head_brain(self) -> NDArray:
        """Brain mask on the head grid."""
        from .b0 import resample

        brain = self.phantom.masks.sum(axis=0)
        return resample(brain, self.phantom.affine, self.head_affine, self.head.shape) > 0.5


def brainweb_anatomy(sub_id: int = 4, output_res: float = 1.0) -> Anatomy:
    """BrainWeb subject `sub_id`: brainweb_phantom plus the outline of the
    whole head at 2 mm (every tissue class of its 12-class model) and
    b0.BRAINWEB_CAVITIES."""
    from brainweb_dl import get_mri

    from . import handlers as h
    from .b0 import BRAINWEB_CAVITIES

    phantom = brainweb_phantom(sub_id, output_res)
    cache = os.path.join(os.environ.get('SNAKE_CACHE_DIR', SNAKE_CACHE_DIR),
                         f'arbepi_head_{sub_id:02d}.npz')
    if os.path.exists(cache):
        with np.load(cache) as f:
            head, affine = f['head'], f['affine']
    else:
        classes, affine05 = get_mri(sub_id, contrast='fuzzy', with_affine=True)
        tissue = np.zeros(classes.T.shape[1:], dtype=np.float32)
        for c in classes.T[1:]:  # class 0 is background; as SNAKE does, (class, x, y, z)
            tissue += c
        f = 4  # 0.5 mm -> 2 mm
        nx, ny, nz = (n // f * f for n in tissue.shape)
        head = np.clip(tissue[:nx, :ny, :nz], 0, 1).reshape(
            nx // f, f, ny // f, f, nz // f, f).mean(axis=(1, 3, 5))
        affine = np.diag([0.5 * f, 0.5 * f, 0.5 * f, 1.0])
        affine[:3, 3] = np.asarray(affine05)[:3, 3] + 0.5 * (f - 1) / 2
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, head=head, affine=affine)
    rois = {
        'occipital': {'center_mm': h.OCCIPITAL_CENTER_MM,
                      'semi_axes_mm': h.OCCIPITAL_SEMI_AXES_MM,
                      'euler_angles': h.OCCIPITAL_EULER_ANGLES},
        'motor': [{'center_mm': center, 'semi_axes_mm': h.MOTOR_SEMI_AXES_MM,
                   'euler_angles': h.MOTOR_EULER_ANGLES} for center in h.MOTOR_CENTERS_MM],
    }
    return Anatomy(phantom, head, affine, BRAINWEB_CAVITIES, rois)


def ellipsoid_anatomy(
    shape: tuple[int, int, int] = (64, 64, 48), res_mm: float = 3.0
) -> Anatomy:
    """ellipsoid_phantom with a head 12% larger than its brain on every axis
    and one air cavity below the front of the brain."""
    from .b0 import Cavity

    phantom = ellipsoid_phantom(shape, res_mm)
    half = np.array(shape) * res_mm / 2
    head_shape = tuple(int(np.ceil(1.15 * s)) for s in shape)
    affine = np.diag([res_mm, res_mm, res_mm, 1.0])
    affine[:3, 3] = -(np.array(head_shape) - 1) / 2 * res_mm
    head = ellipsoid_mask(head_shape, affine, (0, 0, 0), 0.92 * half).astype(np.float32)
    cavity = Cavity((0.0, float(0.45 * half[1]), float(-0.86 * half[2])),
                    tuple(float(v) for v in (0.25, 0.3, 0.12) * half))
    return Anatomy(phantom, head, affine, (cavity,), ellipsoid_phantom_rois(shape, res_mm))


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


def ellipsoid_phantom_rois(
    shape: tuple[int, int, int] = (64, 64, 48), res_mm: float = 3.0
) -> dict[str, dict]:
    """The regions of ellipsoid_phantom(shape, res_mm) by name, as
    brainweb_anatomy has them: 'occipital' (ellipsoid_phantom_roi) and 'motor',
    a patch of the gray matter shell up and to each side."""
    half = np.array(shape) * res_mm / 2
    motor = [{
        'center_mm': (float(side * 0.50 * half[0]), 0.0, float(0.50 * half[2])),
        'semi_axes_mm': tuple(float(v) for v in (0.25, 0.30, 0.25) * half),
        'euler_angles': (0.0, 0.0, 0.0),
    } for side in (-1, 1)]
    return {'occipital': ellipsoid_phantom_roi(shape, res_mm), 'motor': motor}


def place_fov(
    phantom: Phantom, shape: tuple[int, int, int], res_mm: tuple[float, float, float]
) -> FOVConfig:
    """The (shape, res_mm) field of view on a phantom, axis-aligned with it and
    centered on the bounding box of its tissue -- except along z when the
    tissue is taller than the field of view (BrainWeb's brain, brainstem and
    cord against a 144 mm slab): the slab then keeps the top of the head, one
    voxel clear of its upper edge, and loses the inferior end, the way a scan
    would be prescribed. Assumes +z is superior (BrainWeb, ellipsoid_phantom)."""
    occupied = phantom.masks.sum(axis=0) > 0.1
    res = np.asarray(res_mm, dtype=float)
    n = np.array(shape)
    offset = np.empty(3)
    for ax in range(3):
        idx = np.flatnonzero(occupied.any(axis=tuple(a for a in range(3) if a != ax)))
        step, origin = phantom.affine[ax, ax], phantom.affine[ax, 3]
        lo, hi = origin + idx[0] * step, origin + idx[-1] * step  # voxel centers, mm
        offset[ax] = (lo + hi) / 2 - (n[ax] - 1) / 2 * res[ax]
        if ax == 2 and hi - lo + step > n[ax] * res[ax]:
            top_edge = hi + step / 2 + res[ax]
            offset[ax] = top_edge - (n[ax] - 0.5) * res[ax]
    return FOVConfig(
        size=tuple(float(v) for v in n * res),
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
