"""A receive array defined in space, and its noise covariance.

The four scans of a session are simulated on different grids (the EPI's and
the deGRE's, each refined), so the sensitivities are functions of position,
not arrays tied to one grid: loops on rings around z, each with the on-axis
field of a circular loop of radius a at distance d,

    |S| = a^2 / (a^2 + d^2)^(3/2),

and the azimuthal phase of sigpy's birdcage model. Unlike phantom.py's
birdcage maps (1 / d, loops far from the head), this falls off within the
head, as the small elements of a 32-channel head array do; with 32 loops in
four staggered rings it gives parallel imaging something to work with along
both y and z. The maps are scaled to unit root-sum-of-squares at every point,
so the object an ideal reconstruction returns is the object itself.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def coil_centers(
    n_coils: int, coils_per_ring: int = 8, radius_mm: float = 125.0, ring_spacing_mm: float = 45.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(positions (n_coils, 3) mm from isocenter, phase offsets (n_coils,)).
    Rings are centered on z = 0; alternate rings are rotated by half a coil."""
    c = np.arange(n_coils)
    ring = c // coils_per_ring
    n_rings = math.ceil(n_coils / coils_per_ring)
    angle = (c % coils_per_ring + 0.5 * (ring % 2)) * (2 * np.pi / coils_per_ring)
    pos = np.stack(
        [radius_mm * np.cos(angle), radius_mm * np.sin(angle),
         (ring - 0.5 * (n_rings - 1)) * ring_spacing_mm], axis=1,
    )
    return pos, -angle - ring * (2 * np.pi / coils_per_ring)


def coil_maps(
    coords_mm: tuple[NDArray, NDArray, NDArray],
    n_coils: int,
    coils_per_ring: int = 8,
    radius_mm: float = 125.0,
    ring_spacing_mm: float = 45.0,
    loop_radius_mm: float = 45.0,
) -> NDArray[np.complex64]:
    """(n_coils, X, Y, Z) sensitivities on the grid whose voxel centers are
    coords_mm = (x, y, z), 1D arrays in mm from isocenter. Unit
    root-sum-of-squares at every voxel. A single coil is uniform."""
    x, y, z = (np.asarray(v, dtype=np.float32) for v in coords_mm)
    shape = (len(x), len(y), len(z))
    if n_coils == 1:
        return np.ones((1, *shape), dtype=np.complex64)
    pos, phase0 = coil_centers(n_coils, coils_per_ring, radius_mm, ring_spacing_mm)
    out = np.empty((n_coils, *shape), dtype=np.complex64)
    for c in range(n_coils):
        dx = x[:, None, None] - pos[c, 0]
        dy = y[None, :, None] - pos[c, 1]
        dz = z[None, None, :] - pos[c, 2]
        mag = loop_radius_mm**2 / (loop_radius_mm**2 + dx**2 + dy**2 + dz**2) ** 1.5
        out[c] = mag * np.exp(1j * (np.arctan2(dx, -dy) + phase0[c]))
    out /= np.sqrt(np.sum(np.abs(out) ** 2, axis=0))
    return out


def noise_covariance(
    n_coils: int, rng: np.random.Generator, gain_spread: float = 0.06, correlation: float = 0.02
) -> NDArray[np.complex128]:
    """A coil noise covariance matrix with unit mean variance: per-coil noise
    standard deviations spread by `gain_spread` (relative), and complex
    correlations of magnitude up to about `correlation` between coils.

    The sizes are those of a 32-channel head array measured through
    preprocess/ (W of 20260922xiaokai and 20260930ballfat): standard
    deviations within a ratio of 1.10-1.17 of each other, |correlation| 0.007
    on average and 0.02 at most.
    """
    std = 1 + gain_spread * rng.uniform(-1, 1, n_coils)
    a = (rng.normal(size=(n_coils, n_coils)) + 1j * rng.normal(size=(n_coils, n_coils))) / 2
    corr = np.eye(n_coils) + correlation / 2 * (a + a.conj().T) * (1 - np.eye(n_coils))
    psi = corr * np.outer(std, std)
    w = np.linalg.eigvalsh(psi)
    if w.min() <= 0:  # only for an absurd `correlation`
        psi = psi + (1e-6 - w.min()) * np.eye(n_coils)
    return psi / np.mean(np.real(np.diag(psi)))
