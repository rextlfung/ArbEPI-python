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
    _, Rz = balanced_factors(N, R)

    frame_cycle = frame_idx % R
    kz_offset = frame_cycle % Rz
    y_shift = frame_cycle // Rz

    omega = caipi_sample(N, R, 0)
    omega = np.roll(omega, y_shift, axis=0)
    omega = np.roll(omega, kz_offset, axis=1)
    return omega
