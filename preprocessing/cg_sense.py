"""CG-SENSE: unregularized conjugate-gradient SENSE reconstruction.

Ports cg_sense.m near-verbatim -- CG on the normal equations E^H E x = E^H y,
with E = mask . centered-unitary-FFT . (coil sensitivities) over every axis
except the last (coil) axis.

Note on broadcasting: MATLAB implicitly right-pads a lower-rank array with
singleton dimensions when broadcasting (`Mask .* Y` with Mask missing a
trailing coil axis just works). numpy broadcasts by *left*-padding missing
axes instead, which would silently misalign Mask against the wrong axes
here -- so `mask` is kept as an explicit trailing size-1 axis throughout
(`keepdims=True`) rather than dropped, unlike MATLAB's implicit version.
"""

import numpy as np


def _fftc(x: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    res = x
    for d in axes:
        n = res.shape[d]
        fx = np.fft.fft(np.fft.ifftshift(res, axes=d), axis=d)
        res = np.fft.fftshift(fx, axes=d) / np.sqrt(n)
    return res


def _ifftc(x: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    res = x
    for d in axes:
        n = res.shape[d]
        fx = np.fft.ifft(np.fft.ifftshift(res, axes=d), axis=d)
        res = np.fft.fftshift(fx, axes=d) * np.sqrt(n)
    return res


def cg_sense(
    kdata_zf: np.ndarray, sens_maps: np.ndarray, num_iter: int, tol: float = 1e-9
) -> np.ndarray:
    """Img_recon = cg_sense(kdata_zf, sens_maps, num_iter, tol)

    kdata_zf, sens_maps: [*spatial, Ncoils] -- coils are the last axis,
    every other axis is an independent FFT dimension (mirrors cg_sense.m's
    own num_dims/coil_dim/spatial_dims, which is derived from whatever rank
    KData_zf happens to have -- works for 2D+coil or 3D+coil alike).

    Returns an array of shape [*spatial, 1] (the trailing singleton coil
    axis is kept, matching cg_sense.m's literal output shape -- callers
    that want a plain [*spatial] image should index `[..., 0]`).
    """
    num_dims = kdata_zf.ndim
    coil_dim = num_dims - 1
    spatial_dims = tuple(range(num_dims - 1))

    # RSS across all coils, not a single coil's exact-zero check: a single
    # coil can legitimately be all-zero at a sampled location (e.g. after
    # PCA coil compression concentrates signal in the dominant virtual
    # coils), which would otherwise silently mark real samples as
    # unsampled and drive CG's first step to a 0/0 NaN. Matches
    # recon_sigpy.py's `sp.rss(ksp_cf, axes=(0,)) > 0` mask, coil axis
    # adjusted for this module's [*spatial, Ncoils] layout.
    mask = np.sqrt(np.sum(np.abs(kdata_zf) ** 2, axis=coil_dim, keepdims=True)) > 0

    def E(x):
        return mask * _fftc(sens_maps * x, spatial_dims)

    def Eh(y):
        return np.sum(
            np.conj(sens_maps) * _ifftc(mask * y, spatial_dims), axis=coil_dim, keepdims=True
        )

    B = Eh(kdata_zf)
    X = np.zeros_like(B)
    R = B.copy()
    P = R.copy()
    rsold = np.sum(np.abs(R) ** 2)
    rs0 = rsold

    for _ in range(num_iter):
        EP = E(P)
        EHEP = Eh(EP)
        alpha = rsold / np.sum(np.real(np.conj(P) * EHEP))
        X = X + alpha * P
        R = R - alpha * EHEP
        rsnew = np.sum(np.abs(R) ** 2)
        rel_res = np.sqrt(rsnew / rs0)
        if rel_res < tol:
            break
        beta = rsnew / rsold
        P = R + beta * P
        rsold = rsnew

    return X
