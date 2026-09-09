"""Ported from ../ArbEPI/lib/caipi_sample.m.

Generates a regular CAIPI-shifted 2D sampling mask.
"""

import math
from typing import Sequence

import numpy as np


def balanced_factors(N: Sequence[int], R: int) -> tuple[int, int]:
    """Integer factorization Ry * Rz = R, chosen among the pairs that evenly
    divide (Ny, Nz) to best match the FOV-weighted ratio Ry/Rz = Ny/Nz, not
    the most-square split.

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
    since CAIPI's regular decimation can't do fractional Ry/Rz.

    The search is restricted to (Ry, Rz) pairs that evenly divide (Ny, Nz):
    `caipi_sample`'s actual sample count is ceil(Ny/Ry)*ceil(Nz/Rz), which
    only equals the assumed Ny*Nz/R -- the invariant callers like
    `lib/mask2epi.py` and `ticaipi_sample`'s own coverage guarantee both
    depend on -- when Ry | Ny and Rz | Nz exactly. Picking a
    better-FOV-matched pair that *doesn't* divide evenly (an earlier version
    of this function did) silently breaks that invariant instead of merely
    giving a suboptimal split. At (Ny, Nz, R) = (240, 60, 4) (this repo's
    default `res`), the dividing candidates are (1,4)/(2,2)/(4,1); the
    FOV-weighted pick among them is (4, 1) (exact ratio match, score 0),
    not the most-square (2, 2) -- worst-case aliasing period improves from
    54/2 = 27mm (unequal: 216/2 = 108mm on the other axis) to 54/1 = 54mm
    on both axes exactly, hitting the equal-period ideal precisely. Raises
    if no factor pair of R evenly divides (Ny, Nz) at all -- this can
    happen (e.g. balanced_factors([64, 64], 9): none of (1,9)/(3,3)/(9,1)
    divide 64) -- callers needing regular CAIPI decimation have no valid
    split to fall back to in that case."""
    Ny, Nz = N[0], N[1]
    target_log_ratio = math.log(Ny / Nz)

    dividing = [
        (Ry, R // Ry)
        for Ry in range(1, R + 1)
        if R % Ry == 0 and Ny % Ry == 0 and Nz % (R // Ry) == 0
    ]
    if not dividing:
        raise ValueError(
            f'balanced_factors: no factor pair (Ry, Rz) of R={R} evenly '
            f'divides (Ny, Nz)=({Ny}, {Nz}) -- regular CAIPI decimation '
            'requires an exact divisor pair. Choose a different R, '
            'resolution, or sampling_method.'
        )
    return min(dividing, key=lambda rr: abs(math.log(rr[0] / rr[1]) - target_log_ratio))


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
