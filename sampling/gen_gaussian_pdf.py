"""Ported from ../ArbEPI/lib/gen_gaussian_pdf.m."""

import math
from typing import Sequence, Union

import numpy as np


def gen_gaussian_pdf(dims: Sequence[int], sigma: Union[float, Sequence[float]]) -> np.ndarray:
    """Generate a 2D Gaussian density map.

    Parameters
    ----------
    dims : (Nx, Ny)
    sigma : scalar (isotropic) or (sig_x, sig_y) (anisotropic), in pixels.

    Returns
    -------
    pdf : (Nx, Ny) array with values in [0, 1].
    """
    Nx, Ny = dims[0], dims[1]

    if np.isscalar(sigma):
        sig_x = sig_y = float(sigma)
    else:
        sig_x, sig_y = sigma[0], sigma[1]

    # Centered grid: matches MATLAB's ceil(N/2)-centered [1:N]-cx convention.
    cx = math.ceil(Nx / 2)
    cy = math.ceil(Ny / 2)
    x = np.arange(1, Nx + 1) - cx
    y = np.arange(1, Ny + 1) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')

    pdf = np.exp(-((X**2) / (2 * sig_x**2) + (Y**2) / (2 * sig_y**2)))
    pdf = pdf / pdf.max()

    return pdf
