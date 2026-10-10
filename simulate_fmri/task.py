"""The task and the BOLD response to it.

A block paradigm p(t) is +1 while the subject performs the task and -1 while
they rest (including before the first block and after the last). The BOLD
response of a region is the canonical haemodynamic response function h
convolved with it and shifted by that region's delay d:

    w(t) = (h * p)(t - d),

which sits at -1 during long rest, rises to about +1 during a long task block,
and is what a region's fractional signal change is proportional to
(session.Activation.amplitude times w).

h is SPM's canonical HRF: a gamma density peaking at 5 s minus one sixth of a
gamma density peaking at 15 s (shapes 6 and 16, unit scale), with unit area.
Because p is piecewise constant the convolution is exact, through the HRF's
running integral, with no time grid to choose.

Regional delays. The canonical HRF's own time to peak (5.0 s) is what Lin et
al. (NeuroImage 2013;78:372; 3 T, 100 ms sampling, 21 subjects, a visuomotor
reaction task) measured in visual cortex (5.0 +- 0.4 s). In the same task the
motor cortex's response started later and reached half its peak 0.6 s after the
visual cortex's (3.4 vs 2.8 s; onset 2.2 vs 1.0 s; peak 5.2 to 5.5 s), most of
it neuronal (the reaction time) rather than vascular. MOTOR_DELAY_S is that
0.6 s. Differences between people are larger than between regions (a range of
about 4 s in time to peak and to onset: Handwerker et al., NeuroImage
2004;21:1639; Aguirre et al., NeuroImage 1998;8:360), so these are typical
values, not constants.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from analyze.design import canonical_hrf, hrf_integral  # noqa: F401 (canonical_hrf re-exported)

VISUAL_DELAY_S = 0.0
MOTOR_DELAY_S = 0.6

# The HRF and its integral live in analyze/design.py, shared with the analysis.
_hrf_integral = hrf_integral


def block_starts(task_s: float, rest_s: float, duration: float, onset: float = 0.0,
                 n_cycles: int | None = None) -> NDArray[np.float64]:
    """Start time of every task block: onset, onset + task_s + rest_s, ...,
    for n_cycles blocks, or (None) as many as begin before `duration`."""
    if task_s <= 0 or rest_s < 0:
        raise ValueError(f'task_s={task_s}, rest_s={rest_s}: need task_s > 0 and rest_s >= 0')
    period = task_s + rest_s
    n = int(np.ceil((duration - onset) / period)) if n_cycles is None else n_cycles
    return onset + period * np.arange(max(n, 0))


def block_paradigm(t: NDArray, task_s: float, rest_s: float, duration: float, onset: float = 0.0,
                   n_cycles: int | None = None) -> NDArray[np.float64]:
    """p(t): +1 during the task blocks, -1 otherwise."""
    t = np.asarray(t, dtype=np.float64)
    starts = block_starts(task_s, rest_s, duration, onset, n_cycles)
    on = ((t[..., None] >= starts) & (t[..., None] < starts + task_s)).any(axis=-1)
    return np.where(on, 1.0, -1.0)


def bold_response(t: NDArray, task_s: float, rest_s: float, duration: float, onset: float = 0.0,
                  delay: float = 0.0, n_cycles: int | None = None) -> NDArray[np.float64]:
    """w(t) = (h * p)(t - delay) for the block paradigm p (see the module
    docstring): -1 at rest, about +1 late in a long block."""
    t = np.asarray(t, dtype=np.float64) - delay
    starts = block_starts(task_s, rest_s, duration, onset, n_cycles)
    box = (_hrf_integral(t[..., None] - starts)
           - _hrf_integral(t[..., None] - starts - task_s)).sum(axis=-1)
    return 2 * box - 1
