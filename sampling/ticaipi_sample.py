"""Ported from ../ArbEPI/lib/ticaipi_sample.m.

Generates a temporally-interleaved CAIPI sampling mask.

Cycles through all R = Ry * Rz unique (y_shift, kz_offset) combinations so
that every frame gets a distinct mask and the union over R consecutive
frames covers k-space exactly once.

The CAIPI pattern is fixed (shift_offset = 0). Full coverage is achieved by
two independent global circular shifts applied to that base pattern:
    kz_offset (0..Rz-1) shifts the sampled kz columns (axis 1)
    y_shift   (0..Ry-1) shifts the sampled ky rows    (axis 0)

Note: the MATLAB original takes a 1-based `frame` and computes
`frame_idx = mod(frame-1, R)` internally. This port instead takes a 0-based
`frame_idx` directly (i.e. callers loop `for frame_idx in range(Nframes)`),
which removes that 1-based artifact rather than re-introducing it in Python.
"""

from typing import Sequence

import numpy as np

from sampling.caipi_sample import balanced_factors, caipi_sample


def ticaipi_sample(N: Sequence[int], R: int, frame_idx: int) -> np.ndarray:
    Ny, Nz = N[0], N[1]
    Ry, Rz = balanced_factors(N, R)
    # The "union over R frames covers k-space exactly once" guarantee only
    # holds when the balanced (Ry, Rz) factorization evenly divides
    # (Ny, Nz): the global np.roll below only permutes CAIPI residue classes
    # correctly in that case. When it doesn't, some locations are sampled
    # more than once across the R-frame cycle and others never -- silently,
    # with no error (see docs/review-findings.md item 103, which measured
    # this failing on ~44% of a swept (N, R) grid). This repo's own shipped
    # config never hits it (sampling_method='pd', not 'ticaipi'), but
    # 'ticaipi' is a fully supported, documented option.
    if Ny % Ry != 0 or Nz % Rz != 0:
        raise ValueError(
            f'ticaipi_sample: (Ny, Nz)=({Ny}, {Nz}) is not evenly divided by '
            f'the balanced factorization (Ry, Rz)=({Ry}, {Rz}) of R={R} -- '
            'the "covers k-space exactly once over R frames" guarantee only '
            'holds when Ny % Ry == 0 and Nz % Rz == 0. Choose an R whose '
            'balanced factors divide (Ny, Nz) evenly.'
        )

    frame_cycle = frame_idx % R
    kz_offset = frame_cycle % Rz
    y_shift = frame_cycle // Rz

    omega = caipi_sample(N, R, 0)
    omega = np.roll(omega, y_shift, axis=0)
    omega = np.roll(omega, kz_offset, axis=1)
    return omega
