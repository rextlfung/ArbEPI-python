"""Analysis of reconstructed simulated runs against their truth.

- activation maps: t_map (a GLM on the task regressor, in the low band so that
  its degrees of freedom are those the data has) and overlay, which lays the
  thresholded map over the T1-weighted anatomical image of the truth file;
- ROC curves against the truth: roc;
- test-retest reliability without the truth, from M repetitions of the same
  experiment: the mixed-binomial model of Genovese, Noll and Eddy (Magn Reson
  Med 1997;38:497), levels and fit_mixed_binomial;
- spectra: the true signal of any voxel at every excitation (true_signal), and
  where a frequency lands once sampled at the volume rate (alias).

Truth files are session.py's (<name>_truth.h5); reconstructions are
recon.sense's (<name>_recon.h5, dataset X_recon).
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from recon.testbed import _dct, _glm

# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


@dataclass
class Truth:
    """What analysis needs from a truth file (see session.py for each)."""

    x0: NDArray  # (Nx, Ny, Nz) time-averaged noise-free magnitude at TE
    image_rest: NDArray  # the same without the activations' mean
    brain: NDArray  # bool, voxels at least half tissue
    tissues: dict[str, NDArray]
    roi_masks: NDArray  # (R, Nx, Ny, Nz) bool
    roi_names: list[str]
    waveforms: NDArray  # (R, nt)
    amps: NDArray  # (R,)
    amp_map: NDArray
    canonical: NDArray  # (R, nt) the task response without regional delays, per frame
    response: NDArray  # (R, n_exc) with them, per excitation
    paradigm: NDArray  # (R, n_exc)
    mode_names: list[str]
    mode_gain: NDArray  # (M, Nx, Ny, Nz)
    mode_course: NDArray  # (M, n_exc)
    t1w: NDArray  # anatomical image, `factor` voxels per acquisition voxel per axis
    factor: int
    volume_tr: float
    tr_shot: float
    te: float

    @property
    def n_frames(self) -> int:
        return self.waveforms.shape[1]

    @property
    def n_shots(self) -> int:
        return self.mode_course.shape[1] // self.n_frames


def load_truth(fn: str) -> Truth:
    with h5py.File(fn, 'r') as f:
        g = f['truth']
        labels = [str(s) for s in g.attrs['tissue_labels']]
        return Truth(
            x0=g['x0'][()], image_rest=g['image_rest'][()], brain=g['brain_mask'][()],
            tissues=dict(zip(labels, g['tissues'][()])),
            roi_masks=g['roi_masks'][()], roi_names=[str(s) for s in g.attrs['roi_names']],
            waveforms=g['waveforms'][()], amps=np.atleast_1d(g.attrs['amps']),
            amp_map=g['amp_map'][()], canonical=g['task/canonical'][()],
            response=g['task/response'][()], paradigm=g['task/paradigm'][()],
            mode_names=[str(s) for s in g['modes'].attrs['names']],
            mode_gain=g['modes/gain'][()], mode_course=g['modes/course'][()],
            t1w=g['anat/t1w'][()].astype(np.float32), factor=int(g['anat'].attrs['factor']),
            volume_tr=float(f.attrs['volume_tr']), tr_shot=float(g.attrs['tr_shot']),
            te=float(g.attrs['TE']),
        )


def load_series(fn_recon: str, mask: NDArray) -> NDArray[np.float32]:
    """(voxels in mask, frames) magnitude time series of a reconstruction."""
    with h5py.File(fn_recon, 'r') as f:
        d = f['X_recon']
        out = np.empty((int(mask.sum()), d.shape[-1]), dtype=np.float32)
        step = 64
        for start in range(0, d.shape[-1], step):
            out[:, start:start + step] = np.abs(d[..., start:start + step])[mask]
    return out


def true_signal(truth: Truth, mask: NDArray, modes: list[str] | None = None,
                per_frame: bool = False) -> NDArray[np.float64]:
    """The noise-free signal of the voxels in mask, relative to image_rest
    (1 = rest), at every excitation: 1 + sum over modes of gain x course.
    Returns (voxels, n_exc), or (voxels, frames) with per_frame (the mean over
    each frame's excitations, which is what a multi-shot frame measures at the
    center of k-space). modes: only these, by name (default: all)."""
    keep = [i for i, name in enumerate(truth.mode_names) if modes is None or name in modes]
    gain = truth.mode_gain[keep][:, mask].astype(np.float64)  # (M, V)
    signal = 1 + gain.T @ truth.mode_course[keep]
    if per_frame:
        signal = signal.reshape(len(signal), truth.n_frames, truth.n_shots).mean(axis=2)
    return signal


# ---------------------------------------------------------------------------
# activation maps
# ---------------------------------------------------------------------------


def low_band(n_frames: int, tr: float, cutoff_hz: float = 0.15) -> NDArray[np.float64]:
    """(k, n_frames) orthonormal DCT components up to cutoff_hz."""
    keep = int((np.arange(n_frames) / (2 * n_frames * tr) <= cutoff_hz).sum())
    return _dct(n_frames)[:keep]


def t_threshold(n_frames: int, tr: float, p: float = 0.001, cutoff_hz: float | None = 0.15,
                two_sided: bool = True) -> float:
    """The t value of significance p in t_map's GLM (three regressors)."""
    dof = (len(low_band(n_frames, tr, cutoff_hz)) if cutoff_hz else n_frames) - 3
    return float(stats.t.ppf(1 - (p / 2 if two_sided else p), dof))


def t_map(series: NDArray, regressor: NDArray, tr: float,
          cutoff_hz: float | None = 0.15) -> tuple[NDArray, NDArray]:
    """GLM of each voxel's series (V, nt) on a constant, a linear trend and
    the task regressor (nt,). Returns (beta, t): beta as a fraction of the
    voxel's mean signal per unit of the regressor.

    cutoff_hz: fit on the temporal frequencies up to it only (recon.testbed's
    low-band GLM). A reconstruction with a temporal penalty, and physiological
    noise, leave residuals that are far from white, so the plain GLM's t
    (cutoff_hz=None) counts degrees of freedom the data does not have."""
    series = np.asarray(series, dtype=np.float64)
    mean = np.maximum(series.mean(axis=1, keepdims=True), 1e-30)
    basis = low_band(series.shape[1], tr, cutoff_hz) if cutoff_hz else None
    return _glm(series / mean, np.asarray(regressor, dtype=np.float64), basis)


def to_volume(values: NDArray, mask: NDArray, fill: float = 0.0) -> NDArray:
    out = np.full(mask.shape, fill, dtype=np.float32)
    out[mask] = values
    return out


# ---------------------------------------------------------------------------
# ROC against the truth
# ---------------------------------------------------------------------------


def truth_classes(truth: Truth, mask: NDArray, inactive_below: float = 0.05
                  ) -> tuple[NDArray, NDArray]:
    """(active, inactive) booleans over the voxels of mask: active are the
    truth's activated voxels (any region's roi_mask); inactive, those whose
    true change is below inactive_below of the smallest region amplitude.
    Voxels in between (partly activated) are in neither."""
    active = truth.roi_masks.any(axis=0)[mask]
    inactive = (np.abs(truth.amp_map)[mask] < inactive_below * truth.amps.min()) & ~active
    return active, inactive


def roc(score: NDArray, active: NDArray, inactive: NDArray
        ) -> tuple[NDArray, NDArray, NDArray, float]:
    """ROC curve of `score` (higher = more active) for telling `active`
    voxels from `inactive` ones. Returns (false positive rate, true positive
    rate, thresholds, area under the curve); the curve runs from (0, 0) at
    the highest threshold to (1, 1)."""
    s = np.concatenate([score[active], score[inactive]])
    y = np.concatenate([np.ones(int(active.sum())), np.zeros(int(inactive.sum()))])
    order = np.argsort(-s, kind='stable')
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # one point per distinct score
    tpr = np.r_[0.0, np.cumsum(y)[last] / max(y.sum(), 1)]
    fpr = np.r_[0.0, np.cumsum(1 - y)[last] / max((1 - y).sum(), 1)]
    thresholds = np.r_[np.inf, s[last]]
    return fpr, tpr, thresholds, float(np.trapezoid(tpr, fpr))


def rates(score: NDArray, active: NDArray, inactive: NDArray, thresholds: NDArray
          ) -> tuple[NDArray, NDArray]:
    """(false positive rate, true positive rate) at each threshold."""
    thresholds = np.asarray(thresholds, dtype=np.float64)
    a, i = np.sort(score[active]), np.sort(score[inactive])
    tpr = 1 - np.searchsorted(a, thresholds, side='left') / max(len(a), 1)
    fpr = 1 - np.searchsorted(i, thresholds, side='left') / max(len(i), 1)
    return fpr, tpr


# ---------------------------------------------------------------------------
# test-retest reliability: the mixed-binomial model (Genovese et al. 1997)
# ---------------------------------------------------------------------------


def levels(scores: NDArray, thresholds: NDArray) -> NDArray[np.int64]:
    """How many of the (increasing) thresholds each voxel's score reaches in
    each repetition: scores (M, V) -> (M, V) integers in 0..K. A voxel is
    classified active at threshold k when its level is at least k + 1."""
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if np.any(np.diff(thresholds) <= 0):
        raise ValueError('thresholds must increase')
    return np.searchsorted(thresholds, scores, side='right')


def reliability_map(scores: NDArray, threshold: float) -> NDArray[np.int64]:
    """R_v: in how many of the M repetitions each voxel is classified active."""
    return (np.asarray(scores) >= threshold).sum(axis=0)


@dataclass
class MixedBinomial:
    """A fitted mixed-binomial model.

    lam: the proportion of truly active voxels.
    p_active, p_inactive: (K,) the probability that a truly active (inactive)
        voxel is classified active at each threshold: the true and false
        positive rates, estimated without knowing which voxels are active.
    q_active, q_inactive: (K + 1,) the probability of each level (of reaching
        exactly that many thresholds) in one repetition; the model's own
        parameters, p = the tail sums of q.
    posterior: (V,) the probability that each voxel is truly active, given
        how it was classified in the M repetitions.
    loglik: the log-likelihood (without the multinomial coefficients).
    """

    lam: float
    p_active: NDArray
    p_inactive: NDArray
    q_active: NDArray
    q_inactive: NDArray
    posterior: NDArray
    loglik: float


def fit_mixed_binomial(level: NDArray, n_thresholds: int | None = None, n_init: int = 8,
                       max_iter: int = 2000, tol: float = 1e-10, seed: int = 0) -> MixedBinomial:
    """Maximum-likelihood fit of Genovese, Noll and Eddy's model to M
    repetitions of an experiment classified at K thresholds.

    level: (M, V) integers in 0..K from levels(): the number of thresholds
    each voxel reached in each repetition (K = 1: 0/1, the single-threshold
    model, where a voxel's count of 1s is the reliability map R_v).

    The model: a voxel is truly active with probability lam; its level in each
    repetition is drawn independently from q_active or q_inactive accordingly,
    the same for every voxel. With one threshold this is a mixture of two
    binomials for R_v, lam Binomial(M, p_A) + (1 - lam) Binomial(M, p_I); with
    several it is the "dependent likelihood" that shares lam between the
    thresholds, 2 K + 1 parameters. M >= 3 identifies the single-threshold
    model; the paper recommends more.

    Fitted by EM from n_init starting points (the likelihood has two labelings
    and can have local maxima); the class with the higher levels is called
    active.
    """
    level = np.asarray(level)
    m, n_vox = level.shape
    k = int(level.max()) if n_thresholds is None else n_thresholds
    counts = np.stack([(level == j).sum(axis=0) for j in range(k + 1)], axis=1)  # (V, K + 1)
    patterns, inverse, weight = np.unique(counts, axis=0, return_inverse=True, return_counts=True)
    inverse = np.ravel(inverse)
    patterns = patterns.astype(np.float64)
    rng = np.random.default_rng(seed)
    overall = np.maximum(counts.sum(axis=0) / counts.sum(), 1e-6)
    ramp = np.linspace(-1, 1, k + 1)
    best = None
    for init in range(n_init):
        tilt = 1.5 + rng.uniform(0, 3) if init else 2.0
        qa = overall * np.exp(tilt * ramp)
        qi = overall * np.exp(-0.5 * tilt * ramp)
        qa, qi = qa / qa.sum(), qi / qi.sum()
        lam = 0.05 if init == 0 else rng.uniform(0.01, 0.4)
        previous = -np.inf
        for _ in range(max_iter):
            la = np.log(lam) + patterns @ np.log(qa)
            li = np.log1p(-lam) + patterns @ np.log(qi)
            top = np.maximum(la, li)
            norm = top + np.log(np.exp(la - top) + np.exp(li - top))
            loglik = float(weight @ norm)
            resp = np.exp(la - norm)  # P(active | pattern)
            wa, wi = weight * resp, weight * (1 - resp)
            lam = float(np.clip(wa.sum() / n_vox, 1e-9, 1 - 1e-9))
            qa = np.maximum(wa @ patterns, 1e-12)
            qi = np.maximum(wi @ patterns, 1e-12)
            qa, qi = qa / qa.sum(), qi / qi.sum()
            if loglik - previous < tol * abs(loglik):
                break
            previous = loglik
        if best is None or loglik > best[0]:
            best = (loglik, lam, qa, qi, resp)
    loglik, lam, qa, qi, resp = best
    if qa @ np.arange(k + 1) < qi @ np.arange(k + 1):  # the labels came out swapped
        lam, qa, qi, resp = 1 - lam, qi, qa, 1 - resp
    tail = lambda q: np.cumsum(q[::-1])[::-1][1:]  # noqa: E731 -- P(level >= j), j = 1..K
    return MixedBinomial(lam, tail(qa), tail(qi), qa, qi, resp[inverse], loglik)


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------


def alias(f_hz: float | NDArray, fs_hz: float) -> NDArray[np.float64]:
    """Where a frequency appears once sampled at fs_hz (0..fs/2)."""
    f = np.asarray(f_hz, dtype=np.float64)
    return np.abs(f - fs_hz * np.round(f / fs_hz))


def frame_average_gain(f_hz: float | NDArray, frame_s: float) -> NDArray[np.float64]:
    """How much of an oscillation at f_hz survives averaging over a frame of
    frame_s seconds (a multi-shot frame collects k-space over its whole
    length): |sinc(f x frame)|."""
    return np.abs(np.sinc(np.asarray(f_hz, dtype=np.float64) * frame_s))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def overlay(truth: Truth, stat: NDArray, threshold: float, centers: dict[str, NDArray],
            vmax: float | None = None, outline: NDArray | None = None, title: str = '',
            two_sided: bool = True, label: str = 't'):
    """The statistic map, thresholded, over the T1-weighted anatomical image:
    one row of sagittal, coronal and axial slices per entry of centers (name ->
    voxel index on the acquisition grid). Voxels with stat >= threshold are
    drawn in warm colors (and, two_sided, stat <= -threshold in cool ones);
    outline (acquisition grid, bool) is drawn as a contour; label names the
    colorbar. Returns the figure.
    """
    import matplotlib.pyplot as plt

    g = truth.factor
    up = lambda a: np.repeat(np.repeat(np.repeat(a, g, 0), g, 1), g, 2)  # noqa: E731
    stat_up = up(np.asarray(stat, dtype=np.float32))
    line = up(outline.astype(np.float32)) if outline is not None else None
    vmax = vmax or max(2 * threshold, float(np.percentile(np.abs(stat), 99.9)))
    anat_max = float(np.percentile(truth.t1w, 99.5))
    fig, axes = plt.subplots(len(centers), 3, figsize=(10, 3.2 * len(centers)), squeeze=False)
    hot = plt.get_cmap('autumn')
    cool = plt.get_cmap('winter_r')
    for row, (name, center) in zip(axes, centers.items()):
        c = np.asarray(center) * g + g // 2
        for j, (ax, view) in enumerate(zip(row, ('sagittal', 'coronal', 'axial'))):
            sl = [slice(None)] * 3
            sl[j] = int(c[j])
            sl = tuple(sl)
            ax.imshow(np.rot90(truth.t1w[sl]), cmap='gray', vmin=0, vmax=anat_max)
            s = np.rot90(stat_up[sl])
            im = ax.imshow(np.ma.masked_less(s, threshold), cmap=hot, vmin=threshold, vmax=vmax,
                           interpolation='nearest')
            if two_sided:
                ax.imshow(np.ma.masked_greater(s, -threshold), cmap=cool, vmin=-vmax,
                          vmax=-threshold, interpolation='nearest')
            if line is not None and line[sl].any():
                ax.contour(np.rot90(line[sl]), levels=[0.5], colors='lime', linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(view if row is axes[0] else '', fontsize=9)
        row[0].set_ylabel(name)
        fig.colorbar(im, ax=row[2], fraction=0.046, label=label)
    fig.suptitle(title)
    return fig
