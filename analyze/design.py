"""The GLM design matrix: HRF-convolved task regressors plus nuisance terms.

A task condition is a list of (onset, duration) blocks. Its regressor is the
canonical haemodynamic response function h convolved with the boxcar of those
blocks, sampled at the centre of each frame. Because the boxcar is piecewise
constant the convolution is exact, through h's running integral, with no time
grid to choose (the same construction `simulate_fmri/task.py` uses to generate
the signal, so an analysis of simulated data models exactly what was injected).

Nuisance terms follow SPM: a constant and a discrete cosine transform (DCT)
basis that models slow drift. With cutoff period T_c the basis holds the cosines
of period longer than T_c, `floor(2 * n_frames * tr / T_c)` of them (SPM's
default T_c is 128 s). Without a cutoff the nuisance terms are a constant and a
linear trend. Drift is far below a block task's frequency (a 20 s on / 20 s off
task is 0.025 Hz; T_c = 128 s removes below 0.0078 Hz), so the basis models
what would otherwise inflate the residual, not the task.

Times are in seconds from the start of the first frame kept (after
`n_discard`). `Params.n_frames_discard` of the sequence itself is already gone
from a reconstruction; `n_discard` here is for further frames to drop, such as
instruction frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gamma

_PEAK_SHAPE, _UNDERSHOOT_SHAPE, _UNDERSHOOT_RATIO = 6.0, 16.0, 1 / 6
_AREA = 1 - _UNDERSHOOT_RATIO


def canonical_hrf(t: NDArray) -> NDArray[np.float64]:
    """SPM's canonical HRF at times t (s after a brief event), unit area: a gamma
    density peaking at 5 s minus one sixth of one peaking at 15 s (shapes 6 and
    16, unit scale)."""
    t = np.asarray(t, dtype=np.float64)
    h = gamma.pdf(t, _PEAK_SHAPE) - _UNDERSHOOT_RATIO * gamma.pdf(t, _UNDERSHOOT_SHAPE)
    return h / _AREA


def hrf_integral(t: NDArray) -> NDArray[np.float64]:
    """Integral of canonical_hrf from 0 to t (0 for t <= 0, 1 as t grows)."""
    t = np.maximum(np.asarray(t, dtype=np.float64), 0.0)
    return (gamma.cdf(t, _PEAK_SHAPE) - _UNDERSHOOT_RATIO * gamma.cdf(t, _UNDERSHOOT_SHAPE)) / _AREA


def block_response(t: NDArray, onsets: NDArray, durations: NDArray) -> NDArray[np.float64]:
    """(h * boxcar)(t) for blocks [onset, onset + duration): 0 before the first
    block, about 1 late in a long block, back to 0 after the last."""
    t = np.asarray(t, dtype=np.float64)[..., None]
    onsets = np.asarray(onsets, dtype=np.float64)
    durations = np.broadcast_to(np.asarray(durations, dtype=np.float64), onsets.shape)
    return (hrf_integral(t - onsets) - hrf_integral(t - onsets - durations)).sum(axis=-1)


def dct_matrix(nt: int) -> NDArray[np.float64]:
    """(nt, nt) orthonormal DCT-II; row k is the cosine of k half-cycles."""
    t = np.arange(nt)
    C = np.cos(np.pi * t[:, None] * (t[None, :] + 0.5) / nt) * np.sqrt(2.0 / nt)
    C[0] /= np.sqrt(2.0)
    return C


def drift_basis(nt: int, tr: float, cutoff_s: float | None) -> NDArray[np.float64]:
    """(nt, K) drift regressors: DCT cosines of period longer than cutoff_s
    (excluding the constant), or, with cutoff_s=None, a mean-free linear trend."""
    if cutoff_s is None:
        return np.linspace(-1, 1, nt)[:, None]
    k = int(np.floor(2 * nt * tr / cutoff_s))
    return dct_matrix(nt)[1 : k + 1].T


@dataclass
class Condition:
    name: str
    onsets: tuple[float, ...]  # s from the first kept frame
    durations: tuple[float, ...]  # s; one value is broadcast to every onset

    def __post_init__(self):
        self.onsets = tuple(float(o) for o in self.onsets)
        d = tuple(float(x) for x in np.atleast_1d(self.durations))
        self.durations = d * len(self.onsets) if len(d) == 1 else d
        if len(self.durations) != len(self.onsets):
            raise ValueError(f"{self.name}: {len(self.onsets)} onsets, "
                             f"{len(self.durations)} durations")
        if any(x <= 0 for x in self.durations):
            raise ValueError(f"{self.name}: durations must be positive")


@dataclass
class ExperimentParams:
    """What a session's analysis needs to know about its paradigm.

    tr: volume repetition time (s). conditions: the task blocks. n_discard:
    leading frames to drop before fitting. drift_cutoff_s: SPM's high-pass
    period (None -> a linear trend instead). contrast: weights on the
    conditions (default: the first one against baseline)."""

    tr: float
    conditions: list[Condition] = field(default_factory=list)
    n_discard: int = 0
    drift_cutoff_s: float | None = 128.0
    contrast: tuple[float, ...] | None = None

    @classmethod
    def block(cls, tr: float, onsets, duration: float, name: str = "task",
              **kw) -> "ExperimentParams":
        """One condition from block onsets and a common duration."""
        return cls(tr=tr, conditions=[Condition(name, tuple(onsets), (duration,))], **kw)


def frame_times(nt: int, tr: float) -> NDArray[np.float64]:
    """Centre of every frame, s from the start of the first."""
    return (np.arange(nt) + 0.5) * tr


def build_design_matrix(params: ExperimentParams, nt: int) -> tuple[NDArray, list[str]]:
    """(nt, P) design and its column names: the conditions (HRF-convolved), then
    the drift terms, then a constant last. nt counts frames after n_discard."""
    if not params.conditions:
        raise ValueError("ExperimentParams has no conditions")
    t = frame_times(nt, params.tr)
    cols = [block_response(t, c.onsets, c.durations) for c in params.conditions]
    names = [c.name for c in params.conditions]
    drift = drift_basis(nt, params.tr, params.drift_cutoff_s)
    cols += list(drift.T)
    names += [f"drift{k}" for k in range(drift.shape[1])]
    cols.append(np.ones(nt))
    names.append("constant")
    return np.stack(cols, axis=1), names


def contrast_vector(params: ExperimentParams, names: list[str]) -> NDArray[np.float64]:
    """The contrast over the design's columns: params.contrast weights the
    conditions in order; every other column gets 0."""
    n = len(params.conditions)
    w = np.zeros(len(names))
    w[:n] = (1.0,) + (0.0,) * (n - 1) if params.contrast is None else params.contrast
    return w
