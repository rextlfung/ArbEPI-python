"""Ported from ../ArbEPI/lib/caipi_sample.m.

Generates a regular CAIPI-shifted 2D sampling mask.
"""

from typing import Sequence

import numpy as np


def caipi_sample(N: Sequence[int], R: int, shift_offset: int = 0) -> np.ndarray:
    Ny, Nz = N[0], N[1]

    assert Ny >= 1 and Nz >= 1, 'Dimensions must be >= 1'
    assert R >= 1 and round(R) == R, 'R must be a positive integer'
    assert round(shift_offset) == shift_offset and shift_offset >= 0, (
        'shift_offset must be a non-negative integer'
    )

    # Most balanced factorization Ry * Rz = R; larger factor goes to larger dim
    Rsmall = int(np.floor(np.sqrt(R)))
    while R % Rsmall != 0:
        Rsmall -= 1
    Rlarge = R // Rsmall
    if Ny >= Nz:
        Ry, Rz = Rlarge, Rsmall
    else:
        Ry, Rz = Rsmall, Rlarge
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
