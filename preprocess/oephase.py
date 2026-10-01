"""Odd/even EPI ghost-correction phase estimation and correction.

Ports hmriutils' getoephase.m, epiphasecorrect.m, and smooth_custom.m --
compact (~30-50 line) self-contained algorithms, translated directly rather
than redesigned.
"""

import numpy as np

from preprocess.utils import matlab_round


def smooth_custom(x: np.ndarray, span: int = 5) -> np.ndarray:
    """Centered moving-average smoother with a shrinking window at the
    edges (not zero-padded) -- ports smooth_custom.m exactly."""
    if span % 2 == 0:
        raise ValueError('span must be an odd number')
    n = len(x)
    half = span // 2
    y = np.empty_like(x, dtype=x.dtype)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        y[i] = np.mean(x[lo:hi])
    return y


def getoephase(
    x: np.ndarray, echo_order: int = 1, min_pixels: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the odd/even EPI echo phase difference (constant + linear-in-x
    terms, each a polynomial in echo-pair index) from one EPI echo train
    acquired without phase-encoding blips.

    x: [nx, etl, nCoils] complex, etl even, inverse-FFT'd along the readout
       (1st) axis already (i.e. x holds 1D spatial profiles per echo/coil).
    echo_order: degree of the polynomial in echo-pair index (0 = one
       correction for the whole train, 1 = linear drift along it).
    min_pixels: raise if fewer pixels carry signal (see below).

    Returns (a, th):
      a   [etl//2, 2]: for echo pair j (echoes 2j, 2j+1), a[j, 0] is the
          constant phase offset (rad) and a[j, 1] the linear term (rad/fov;
          the corresponding k-space shift in samples is a[j, 1] / (2*pi)).
          epiphasecorrect applies a[j] to even echo 2j+1.
      th  [nx, etl//2] measured odd/even phase mismatch per echo pair,
          before the fit (for diagnostics/plotting).

    The phase difference drifts along the echo train (on 20260930ballfat, a0
    moved by 0.05-0.09 rad and a1 by up to ~0.1 rad/fov between the first and
    last pair, the same way in every run; docs/review-findings.md item 259),
    so one value per train is biased toward whichever echoes it is fit on.
    Every pair is used, weighted by its coil-combined |signal|^2 (the inverse
    of the phase variance), over the central half of x. The ported
    getoephase.m instead fit only pairs etl/4..etl/2-1 with a hard mask taken
    from echoes 3*etl/4..etl-1 at 10% of the train's peak; on a long, decayed
    train that mask was empty and np.linalg.lstsq returned a = [0, 0]
    silently (no correction).
    """
    nx, etl, ncoils = x.shape
    if etl % 2:
        raise ValueError('etl must be even')

    # Off-resonance phase from the odd echoes (MATLAB 1-based; Python 0::2) at
    # the center readout row, magnitude-squared coil-weighted (as in
    # phase-contrast MRI). getoephase.m used the even echoes, which carry the
    # odd/even offset itself: harmless while that offset is the same for every
    # pair, but once it drifts, the accrual fit absorbs half of the drift.
    row0 = nx // 2 - 1  # MATLAB's x(end/2, ...), 1-based end/2 -> 0-based end/2-1
    th_echo = np.zeros(etl // 2, dtype=complex)
    for ic in range(ncoils):
        xe = x[row0, 0::2, ic]
        tmp = np.unwrap(np.angle(xe))
        tmp = tmp - tmp[(len(tmp) - 1) // 2]
        th_echo = th_echo + np.abs(xe) ** 2 * np.exp(1j * tmp)

    ma_span = 1 if etl // 2 < 10 else 5
    th_echo = smooth_custom(np.unwrap(np.angle(th_echo)), ma_span)

    # Linear fit of off-resonance phase accrual vs. echo index, then
    # subtract that evolution from every echo (same term at every x).
    echo_idx = np.arange(1, etl + 1, 2, dtype=float)  # MATLAB's 1:2:etl
    B = np.stack([np.ones_like(echo_idx), echo_idx], axis=1)
    a_echo, *_ = np.linalg.lstsq(B, th_echo, rcond=None)
    xy = np.tile(np.arange(1, etl + 1, dtype=float), (nx, 1))  # [nx, etl]
    dph = a_echo[0] + a_echo[1] * xy
    xc = x * np.exp(-1j * dph[:, :, None])

    # Odd/even phase mismatch for all neighboring echo pairs, per x, and its
    # coil-combined weight (~ sum_c |xe|^2).
    thc = np.zeros((nx, etl // 2), dtype=complex)
    for ic in range(ncoils):
        xo = xc[:, 0::2, ic]
        xe = xc[:, 1::2, ic]
        with np.errstate(divide='ignore', invalid='ignore'):  # 0/0 where there is no signal
            thc = thc + np.abs(xe) ** 2 * np.exp(1j * np.angle(xe / xo))  # assumes no phase wrap
    th = np.angle(thc)
    w = np.abs(thc)

    # Central half of x only (exclude edge background).
    w[:matlab_round(nx / 4), :] = 0
    w[matlab_round(3 * nx / 4) - 1:, :] = 0
    if np.count_nonzero(w > 1e-3 * w.max(initial=0)) < min_pixels:
        raise ValueError(
            f'getoephase: fewer than {min_pixels} pixels with signal in the central half '
            'of x; the calibration data cannot determine the odd/even phase'
        )

    # Weighted least squares over every (x, pair):
    # th(x, j) = sum_k (c0k + c1k * x) * t_j^k, with t_j the pair index
    # centered on mid-train, so the k = 0 terms are the mid-train (TE) values.
    x_coord = (np.arange(nx) - nx / 2 + 0.5) / nx
    t = np.arange(etl // 2) - (etl // 2 - 1) / 2
    xx, tt = np.meshgrid(x_coord, t, indexing='ij')
    powers = tt[..., None] ** np.arange(echo_order + 1)  # [nx, npairs, order+1]
    H = np.concatenate([powers, xx[..., None] * powers], axis=-1)  # [nx, npairs, 2*(order+1)]
    sw = np.sqrt(w).ravel()
    c, *_ = np.linalg.lstsq(H.reshape(-1, H.shape[-1]) * sw[:, None], th.ravel() * sw, rcond=None)

    tp = t[:, None] ** np.arange(echo_order + 1)  # [npairs, order+1]
    a = np.stack([tp @ c[:echo_order + 1], tp @ c[echo_order + 1:]], axis=1)  # [npairs, 2]
    return a, th


def epiphasecorrect(d: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Odd/even phase correction: in image space along x, multiply even echo
    2j+1 by exp(-i(a[j, 0] + a[j, 1] x)) (getoephase's model). Ports
    epiphasecorrect.m.

    d: [nx, etl, ...] Cartesian EPI k-space.
    a: [etl//2, 2], one (constant, linear) pair per even echo, as returned by
       getoephase; or [2], the same correction for every even echo (the
       pre-item-259 model, e.g. `oephase_a` in older preprocessed files).

    Uses the standard centered-FFT pairing along x, which round-trips exactly
    for odd nx too (the MATLAB source's fftshift-on-both-sides spelling only
    does for even nx; review item 44). preprocess.compute_oephase estimates
    `a` in this same frame (item 251).
    """
    nx, etl = d.shape[0], d.shape[1]
    d_shape = d.shape
    d = d.reshape(nx, etl, -1)

    x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(d, axes=0), axis=0), axes=0)

    a = np.asarray(a, dtype=float)
    if a.shape == (2,):
        a = np.broadcast_to(a, (etl // 2, 2))
    if a.shape != (etl // 2, 2):
        raise ValueError(
            f'epiphasecorrect: a has shape {a.shape}, expected (2,) or ({etl // 2}, 2)'
        )
    x_coord = (np.arange(nx) - nx / 2 + 0.5) / nx
    th = a[None, :, 0] + a[None, :, 1] * x_coord[:, None]  # [nx, etl//2], one column per even echo

    x[:, 1::2, :] = x[:, 1::2, :] * np.exp(-1j * th)[:, :, None]

    dc = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=0), axis=0), axes=0)
    return dc.reshape(d_shape)
