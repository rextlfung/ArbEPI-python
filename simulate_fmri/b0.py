"""A static B0 field map for the phantom, from its magnetic susceptibility.

The field inside a head is set mostly by where the air is: outside the head,
and in the sinuses and ear canals next to the brain. Given a susceptibility
distribution chi (ppm, relative to vacuum), the field shift along B0 is the
dipole convolution (Salomir et al., Concepts Magn Reson B 2003;19:26; Marques &
Bowtell, Concepts Magn Reson B 2005;25:65)

    dB/B0 = F^-1[ (1/3 - kz^2 / |k|^2) F[chi] ],        B0 along z,

the 1/3 being the sphere-of-Lorentz correction. head_field_hz builds chi from a
map of where tissue is (-9.05 ppm, water; air +0.36 ppm) with air cavities
carved out of the non-brain tissue, evaluates that, and removes what the
scanner's shim would (spherical harmonics up to shim_order, fitted over the
brain).

BrainWeb's head model has no air inside it (its 12 tissue classes fill the
head), so the cavities are ellipsoids placed by hand in the bone and soft
tissue below the frontal lobes and beside the temporal lobes (BRAINWEB_CAVITIES).
They are not anatomy; they are sized so the field in the brain resembles a
measured one: on a 3 T head scan (20260922xiaokai, deGRE field map) the field
over the brain mask had a standard deviation of 45 Hz and 0.1-99.9 percentiles
of -267 and +190 Hz, strongest above the sinuses and in the inferior slices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

AIR_PPM = 0.36
TISSUE_PPM = -9.05
GAMMA_BAR_HZ_PER_T = 42.577478e6


@dataclass(frozen=True)
class Cavity:
    """An air-filled ellipsoid, in world mm."""

    center_mm: tuple[float, float, float]
    semi_axes_mm: tuple[float, float, float]


# BrainWeb world coordinates: +x right, +y anterior, +z superior; the brain
# spans x +-74, y -106..75, z -72..87 mm.
BRAINWEB_CAVITIES = (
    Cavity((0.0, 79.0, -8.0), (14.0, 6.0, 12.0)),  # frontal sinus, in the frontal bone
    Cavity((0.0, 32.0, -47.0), (17.0, 24.0, 12.0)),  # ethmoid and sphenoid sinuses
    Cavity((0.0, 48.0, -66.0), (13.0, 30.0, 10.0)),  # nasal cavity
    Cavity((52.0, -24.0, -42.0), (11.0, 10.0, 10.0)),  # mastoid air cells, right
    Cavity((-52.0, -24.0, -42.0), (11.0, 10.0, 10.0)),  # left
    Cavity((76.0, -18.0, -40.0), (16.0, 5.0, 5.0)),  # ear canal, right
    Cavity((-76.0, -18.0, -40.0), (16.0, 5.0, 5.0)),  # left
)


def world_coords(shape: tuple[int, int, int], affine: NDArray) -> NDArray[np.float64]:
    """(3, X, Y, Z) world coordinates (mm) of the voxel centers of an
    axis-aligned grid."""
    a = np.asarray(affine, dtype=np.float64)
    axes = [a[i, 3] + a[i, i] * np.arange(n) for i, n in enumerate(shape)]
    return np.stack(np.meshgrid(*axes, indexing='ij'))


def dipole_field_ppm(chi_ppm: NDArray, voxel_mm: tuple[float, float, float]) -> NDArray:
    """Field shift dB/B0 (ppm) of a susceptibility distribution (ppm), B0 along
    the last axis. Periodic: pad chi so that images of the object are far
    away. The mean (k = 0) is set to zero."""
    k = [np.fft.fftfreq(n, d) for n, d in zip(chi_ppm.shape, voxel_mm)]
    kx, ky, kz = np.meshgrid(*k, indexing='ij', sparse=True)
    k2 = kx**2 + ky**2 + kz**2
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = 1.0 / 3.0 - kz**2 / k2
    kernel[0, 0, 0] = 0.0
    return np.fft.ifftn(np.fft.fftn(chi_ppm) * kernel).real


def shim_basis(coords_mm: NDArray, order: int) -> NDArray[np.float64]:
    """(n_terms, ...) real spherical harmonics up to `order` (0, 1 or 2) at
    the given coordinates (3, ...), mm."""
    x, y, z = coords_mm
    terms = [np.ones_like(x)]
    if order >= 1:
        terms += [x, y, z]
    if order >= 2:
        terms += [z**2 - (x**2 + y**2) / 2, x * z, y * z, x**2 - y**2, x * y]
    if order > 2:
        raise ValueError('shim_order must be 0, 1 or 2')
    return np.stack(terms)


def shim(field: NDArray, mask: NDArray, coords_mm: NDArray, order: int = 1) -> NDArray:
    """`field` minus its least-squares fit by shim_basis over `mask`."""
    basis = shim_basis(coords_mm, order)
    a = basis[:, mask].T
    coef, *_ = np.linalg.lstsq(a, field[mask], rcond=None)
    return field - np.tensordot(coef, basis, axes=1)


def head_field_hz(
    tissue: NDArray,
    brain: NDArray,
    affine: NDArray,
    cavities: tuple[Cavity, ...] = (),
    shim_order: int = 1,
    field_strength_t: float = 3.0,
    pad_factor: float = 2.0,
) -> NDArray[np.float32]:
    """The field offset (Hz) on an axis-aligned grid whose last axis is the
    direction of B0, inferior to superior.

    tissue: (X, Y, Z) fraction of each voxel that is tissue (0 = air).
    brain: (X, Y, Z) bool, where the field is shimmed (and air is never
        carved).
    affine: voxel-to-world matrix (mm).
    cavities: air ellipsoids to carve out of the tissue outside `brain`.
    shim_order: spherical-harmonic order removed over `brain` (1: the linear
        shims a routine prescan sets; 2 adds second order).
    pad_factor: the periodic box is this many times the grid along each axis.
        The lowest slice is repeated down to the bottom of the box: a head
        does not end in air at the neck.
    """
    shape = tissue.shape
    voxel = tuple(abs(float(affine[i, i])) for i in range(3))
    coords = world_coords(shape, affine)
    tissue = np.clip(np.asarray(tissue, dtype=np.float64), 0, 1)
    for cav in cavities:
        d = sum(
            ((coords[i] - cav.center_mm[i]) / cav.semi_axes_mm[i]) ** 2 for i in range(3)
        )
        tissue = np.where((d <= 1) & ~brain, 0.0, tissue)
    chi = AIR_PPM + (TISSUE_PPM - AIR_PPM) * tissue

    box = tuple(int(np.ceil(n * pad_factor / 2)) * 2 for n in shape)
    lo = tuple((b - n) // 2 for b, n in zip(box, shape))
    padded = np.full(box, AIR_PPM)
    sl = tuple(slice(o, o + n) for o, n in zip(lo, shape))
    padded[sl] = chi
    padded[sl[0], sl[1], : lo[2]] = chi[:, :, :1]  # the neck continues downward
    field_ppm = dipole_field_ppm(padded, voxel)[sl]
    field = field_ppm * 1e-6 * GAMMA_BAR_HZ_PER_T * field_strength_t
    return shim(field, brain, coords, shim_order).astype(np.float32)


def resample(vol: NDArray, affine: NDArray, new_affine: NDArray, new_shape: tuple[int, ...],
             order: int = 1) -> NDArray:
    """An axis-aligned volume on another axis-aligned grid (world mm affines).
    Beyond the source grid the edge value is held."""
    a, b = np.asarray(affine, float), np.asarray(new_affine, float)
    scale = np.array([b[i, i] / a[i, i] for i in range(3)])
    offset = np.array([(b[i, 3] - a[i, 3]) / a[i, i] for i in range(3)])
    return ndimage.affine_transform(
        vol, scale, offset=offset, output_shape=tuple(new_shape), order=order, mode='nearest'
    )
