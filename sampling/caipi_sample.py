"""Ported from ../ArbEPI/lib/caipi_sample.m.

Generates a regular CAIPI-shifted 2D sampling mask.
"""

import math
from typing import Sequence

import numpy as np


def balanced_factors(N: Sequence[int], R: int) -> tuple[int, int]:
    """Integer factorization Ry * Rz = R, chosen to best match the
    FOV-weighted ratio Ry/Rz = Ny/Nz, not the most-square split.

    For an axis with FOV and local reduction factor R_axis, the achieved
    k-space sample spacing is R_axis/FOV, so the resulting image-domain
    aliasing period is FOV/R_axis. Splitting R evenly (Ry ~= Rz, the
    previous behavior here) gives *unequal* aliasing periods whenever
    FOV_y != FOV_z -- worse (tighter, more severe) on whichever axis has
    the smaller FOV. Equalizing the periods instead
    (FOV_y/Ry = FOV_z/Rz) requires Ry/Rz = FOV_y/FOV_z = Ny/Nz (equal
    resolution assumed), which is also exactly what `sampling/pd_sample.py`'s
    aspect-matched exclusion ellipse already does continuously -- this
    picks the closest achievable *integer* factor pair to that same ratio,
    since CAIPI's regular decimation can't do fractional Ry/Rz. At this
    repo's real (Ny, Nz, R) = (240, 45, 9), that's (Ry, Rz) = (9, 1), not
    the (3, 3) a most-square split would give -- worst-case aliasing
    period improves from 40.5/3 = 13.5mm to 40.5/1 = 40.5mm on the
    small-FOV axis, at the cost of 216/9 = 24mm (still better than
    the equal-period ideal ~31mm, but far better than the 13.5mm floor
    the naive split leaves on the axis that can least afford it)."""
    Ny, Nz = N[0], N[1]
    target_log_ratio = math.log(Ny / Nz)

    best = (1, R)
    best_score = math.inf
    for Ry in range(1, R + 1):
        if R % Ry != 0:
            continue
        Rz = R // Ry
        score = abs(math.log(Ry / Rz) - target_log_ratio)
        if score < best_score:
            best_score = score
            best = (Ry, Rz)
    return best


def caipi_sample(N: Sequence[int], R: int, shift_offset: int = 0) -> np.ndarray:
    Ny, Nz = N[0], N[1]

    assert Ny >= 1 and Nz >= 1, 'Dimensions must be >= 1'
    assert R >= 1 and round(R) == R, 'R must be a positive integer'
    assert round(shift_offset) == shift_offset and shift_offset >= 0, (
        'shift_offset must be a non-negative integer'
    )

    Ry, Rz = balanced_factors(N, R)
    caipi_z = Ry

    omega = np.zeros((Ny, Nz))
    omega[0:Ny:Ry, 0:Nz:Rz] = 1

    for z0 in range(caipi_z):
        shift_amount = (z0 + shift_offset) % caipi_z
        cols = np.arange(Rz * z0, Nz, caipi_z * Rz)
        if cols.size == 0:
            continue
        omega[:, cols] = np.roll(omega[:, cols], shift_amount, axis=0)

    return omega
