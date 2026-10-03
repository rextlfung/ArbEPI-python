"""Ported from ../ArbEPI/lib/rand_sample.m.

Weighted random sampling of a 2D grid via the Efraimidis-Spirakis
key-sorting method for weighted sampling without replacement.
"""

from typing import Optional, Sequence

import numpy as np


def rand_sample(
    dims: Sequence[int],
    R: float,
    pdf_in: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Parameters
    ----------
    dims : (Nx, Ny)
    R : acceleration factor (>= 1)
    pdf_in : (Nx, Ny) array of sampling weights, or None for uniform weights.
    rng : numpy random Generator used to draw the sampling keys.

    Returns
    -------
    mask : (Nx, Ny) boolean array.
    """
    assert len(dims) == 2 and all(d >= 1 for d in dims), (
        'dims must be a 2-element vector of positive integers.'
    )
    assert R >= 1, 'R must be a scalar acceleration factor >= 1.'
    Nx, Ny = dims[0], dims[1]

    total_points = Nx * Ny
    num_samples = round(total_points / R)

    if pdf_in is None:
        weights = np.ones((Nx, Ny))
    else:
        weights = pdf_in
        if weights.shape != (Nx, Ny):
            raise ValueError('PDF dimensions must match the input [Nx, Ny] vector.')

    u = rng.random((Nx, Ny))

    safe_weights = weights + np.finfo(float).eps
    keys = u ** (1 / safe_weights)

    sort_idx = np.argsort(keys, axis=None)[::-1]

    mask = np.zeros((Nx, Ny), dtype=bool)
    chosen_indices = sort_idx[:num_samples]
    mask.flat[chosen_indices] = True

    return mask
