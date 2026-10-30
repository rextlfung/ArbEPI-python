"""Voxel-wise GLM: contrast t-scores, z-scores and multiple-comparison correction.

`fit_glm` fits Y (V, nt) = X (nt, P) beta with an optional AR(1) noise model
estimated the way SPM does: one autocorrelation coefficient shared by all
voxels, not one per voxel, and the data and design are whitened by it before
the fit. Two simplifications relative to SPM12's `spm_est_non_sphericity`, which
fits an AR(1)-plus-white covariance by ReML to the pooled residual covariance:
rho is the pooled lag-1 autocorrelation of each voxel's normalised residuals,
and it is iterated (Cochrane-Orcutt) rather than solved by ReML. For rho in the
range fMRI shows (0.2-0.5) the difference is small.

The whitening is the exact AR(1) (Prais-Winsten) filter,
    y*_0 = sqrt(1 - rho^2) y_0,   y*_t = y_t - rho y_{t-1},
applied to Y and X alike. Degrees of freedom are nt - rank(X); the whitened
model is treated as ordinary least squares from there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

_CHUNK = 50_000  # voxels per block of the fit; bounds float64 temporaries


@dataclass
class GLMResult:
    beta: NDArray  # (V, P) float32
    t: NDArray  # (V,) contrast t-scores, float32
    z: NDArray  # (V,) z-scores equivalent to t at `dof`
    dof: float
    rho: float  # AR(1) coefficient used (0 without prewhitening)
    sigma2: NDArray  # (V,) residual variance of the whitened model
    contrast_beta: NDArray  # (V,) c' beta


def whiten(a: NDArray, rho: float, axis: int = -1, starts=(0,)) -> NDArray:
    """The AR(1) prewhitening filter along `axis`, restarted (the Prais-Winsten
    first-sample scaling) at each index of `starts`: more than the first frame
    only for a series made of separate stretches of frames."""
    a = np.moveaxis(np.asarray(a), axis, -1)
    out = np.empty_like(a, dtype=np.result_type(a.dtype, np.float32))
    out[..., 1:] = a[..., 1:] - rho * a[..., :-1]
    out[..., list(starts)] = np.sqrt(1 - rho**2) * a[..., list(starts)]
    return np.moveaxis(out, -1, axis)


def t_to_z(t: NDArray, dof: float) -> NDArray:
    """z with the same tail probability as t on `dof` degrees of freedom,
    sign(t) * Phi^-1(F_t(|t|)). Computed in log space, so it does not saturate
    at 8.3 as the naive form does once F_t rounds to 1."""
    t = np.asarray(t, dtype=np.float64)
    log_sf = stats.t.logsf(np.abs(t), dof)
    return np.sign(t) * -special.ndtri_exp(log_sf)


def t_to_p(t: NDArray, dof: float, two_tailed: bool = True) -> NDArray:
    p = stats.t.sf(np.abs(np.asarray(t, dtype=np.float64)), dof)
    return 2 * p if two_tailed else p


def t_threshold(dof: float, p: float = 0.001, two_sided: bool = True) -> float:
    """The |t| of significance p on `dof` degrees of freedom."""
    return float(stats.t.ppf(1 - (p / 2 if two_sided else p), dof))


def fdr_threshold(p: NDArray, q: float = 0.05) -> float:
    """Benjamini-Hochberg: the largest p-value that survives, or 0 if none."""
    p = np.sort(np.asarray(p, dtype=np.float64).ravel())
    ok = np.nonzero(p <= q * np.arange(1, p.size + 1) / p.size)[0]
    return float(p[ok[-1]]) if ok.size else 0.0


def _blocks(n: int):
    for s in range(0, n, _CHUNK):
        yield slice(s, min(s + _CHUNK, n))


def _estimate_rho(Y: NDArray, X: NDArray, rho: float, starts=(0,)) -> float:
    """Pooled lag-1 autocorrelation of the residuals of the GLS fit at `rho`,
    in the original (unwhitened) frame, each voxel normalised to unit variance
    so that bright voxels do not set it. The pair (i-1, i) is no lag-1 pair at
    a restart `i` in `starts`."""
    lag_ok = np.ones(X.shape[0] - 1, bool)
    lag_ok[np.asarray(starts[1:], dtype=int) - 1] = False
    Xw = whiten(X, rho, axis=0, starts=starts)
    pinv = np.linalg.pinv(Xw)
    num = den = 0.0
    for sl in _blocks(Y.shape[0]):
        y = Y[sl].astype(np.float64)
        beta = whiten(y, rho, starts=starts) @ pinv.T
        r = y - beta @ X.T
        r -= r.mean(axis=1, keepdims=True)
        sd = np.sqrt((r**2).mean(axis=1, keepdims=True))
        r = r / np.maximum(sd, 1e-30)
        lagged = r[:, 1:] * r[:, :-1]
        num += float((lagged if lag_ok.all() else lagged[:, lag_ok]).sum())
        den += float((r**2).sum())
    return num / den


def fit_glm(Y: NDArray, X: NDArray, contrast: NDArray, ar1: bool = True,
            max_iter: int = 5, starts=(0,)) -> GLMResult:
    """GLM of every row of Y (V, nt) on X (nt, P); t-score of `contrast` (P,).

    ar1: prewhiten with a pooled AR(1) coefficient (iterated until it moves by
    less than 1e-3, at most max_iter times). starts: the frames that begin a
    new stretch when the nt frames are not all adjacent in time (a subset of
    cycles); the AR(1) filter restarts at each."""
    Y = np.asarray(Y)
    X = np.asarray(X, dtype=np.float64)
    c = np.asarray(contrast, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != X.shape[0]:
        raise ValueError(f"Y {Y.shape} and X {X.shape} disagree on the number of frames")
    if c.shape != (X.shape[1],):
        raise ValueError(f"contrast has {c.size} weights, the design {X.shape[1]} columns")
    nt = Y.shape[1]
    rank = np.linalg.matrix_rank(X)
    if nt <= rank:
        raise ValueError(f"{nt} frames cannot fit {rank} independent regressors")

    rho = 0.0
    if ar1:
        for _ in range(max_iter):
            new = _estimate_rho(Y, X, rho, starts)
            done = abs(new - rho) < 1e-3
            rho = new
            if done:
                break
    Xw = whiten(X, rho, axis=0, starts=starts) if rho else X
    XtXi = np.linalg.pinv(Xw.T @ Xw)
    var_c = float(c @ XtXi @ c)
    dof = nt - rank

    V = Y.shape[0]
    beta = np.empty((V, X.shape[1]), np.float32)
    sigma2 = np.empty(V, np.float32)
    for sl in _blocks(V):
        y = Y[sl].astype(np.float64)
        if rho:
            y = whiten(y, rho, starts=starts)
        b = y @ Xw @ XtXi
        r = y - b @ Xw.T
        beta[sl] = b
        sigma2[sl] = (r**2).sum(axis=1) / dof
    cb = beta @ c.astype(np.float32)
    t = cb / np.maximum(np.sqrt(sigma2 * var_c), 1e-30)
    return GLMResult(beta=beta, t=t.astype(np.float32), z=t_to_z(t, dof).astype(np.float32),
                     dof=float(dof), rho=float(rho), sigma2=sigma2, contrast_beta=cb)


def to_volume(values: NDArray, mask: NDArray, fill: float = 0.0) -> NDArray:
    """Scatter per-voxel `values` back into a volume where `mask` is True."""
    out = np.full(mask.shape, fill, dtype=np.float32)
    out[mask] = values
    return out
