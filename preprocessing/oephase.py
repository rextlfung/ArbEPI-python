"""Odd/even EPI ghost-correction phase estimation and correction.

Ports hmriutils' getoephase.m, epiphasecorrect.m, and smooth_custom.m --
compact (~30-50 line) self-contained algorithms, translated directly rather
than redesigned.
"""

import numpy as np


def _matlab_round(x: float) -> int:
    """MATLAB's round() rounds half away from zero; numpy/Python round to
    even. Only matters at exact .5 ties, but replicated for fidelity since
    it gates mask boundaries in getoephase below."""
    return int(np.floor(x + 0.5)) if x >= 0 else int(np.ceil(x - 0.5))


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


def getoephase(x: np.ndarray, threshold: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Estimate odd/even EPI echo phase difference (constant + linear-in-x
    terms) from one EPI echo train acquired without phase-encoding blips.

    x: [nx, etl, nCoils] complex, etl even, inverse-FFT'd along the readout
       (1st) axis already (i.e. x holds 1D spatial profiles per echo/coil).

    Returns (a, th):
      a[0]  constant phase offset (rad)
      a[1]  linear term (rad/fov); the corresponding k-space shift in
            samples is a[1] / (2*pi)
      th    [nx, etl//2] measured odd/even phase mismatch per echo pair,
            before the final spatial linear fit (for diagnostics/plotting).
    """
    nx, etl, ncoils = x.shape
    if etl % 2:
        raise ValueError('etl must be even')

    # Off-resonance phase from even echoes at the center readout row,
    # magnitude-squared coil-weighted (as in phase-contrast MRI).
    row0 = nx // 2 - 1  # MATLAB's x(end/2, ...), 1-based end/2 -> 0-based end/2-1
    th_echo = np.zeros(etl // 2, dtype=complex)
    for ic in range(ncoils):
        xe = x[row0, 1::2, ic]
        tmp = np.unwrap(np.angle(xe))
        tmp = tmp - tmp[(len(tmp) - 1) // 2]
        th_echo = th_echo + np.abs(xe) ** 2 * np.exp(1j * tmp)

    ma_span = 1 if etl // 2 < 10 else 5
    th_echo = smooth_custom(np.unwrap(np.angle(th_echo)), ma_span)

    # Linear fit of off-resonance phase accrual vs. echo index, then
    # subtract that evolution from every echo (same term at every x).
    echo_idx = np.arange(2, etl + 1, 2, dtype=float)  # MATLAB's 2:2:etl
    B = np.stack([np.ones_like(echo_idx), echo_idx], axis=1)
    a_echo, *_ = np.linalg.lstsq(B, th_echo, rcond=None)
    xy = np.tile(np.arange(1, etl + 1, dtype=float), (nx, 1))  # [nx, etl]
    dph = a_echo[0] + a_echo[1] * xy
    xc = x * np.exp(-1j * dph[:, :, None])

    # Spatial mask: keep only the center half of x (exclude edge background).
    rssim = np.sqrt(np.sum(np.abs(xc) ** 2, axis=2))  # [nx, etl]
    mask = rssim > threshold * rssim.max()
    lo = _matlab_round(nx / 4)
    hi = _matlab_round(3 * nx / 4)
    mask[:lo, :] = False
    mask[hi - 1:, :] = False

    # Odd/even phase mismatch for all neighboring echo pairs, per x.
    th = np.zeros((nx, etl // 2), dtype=complex)
    for ic in range(ncoils):
        xo = xc[:, 0::2, ic]
        xe = xc[:, 1::2, ic]
        th = th + np.abs(xe) ** 2 * np.exp(1j * np.angle(xe / xo))  # assumes no phase wrap
    th = np.angle(th)

    # Linear fit (offset + spatial-linear term) to the later, more stable
    # echo pairs -- mirrors getoephase.m's column-slice reuse of `mask`.
    mask = mask[:, etl // 2:]
    mask[:, :mask.shape[1] // 2] = False
    x_coord = (np.arange(nx) - nx / 2 + 0.5) / nx
    x_grid = np.tile(x_coord[:, None], (1, th.shape[1]))
    H = np.stack([np.ones(mask.sum()), x_grid[mask]], axis=1)
    a, *_ = np.linalg.lstsq(H, th[mask], rcond=None)

    return a, th


def epiphasecorrect(d: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Odd/even phase correction for EPI: subtract the a[0]+a[1]*x linear
    phase model (from getoephase) from every even-indexed echo, in image
    space. Ports epiphasecorrect.m.

    d: [nx, etl, ...] raw (Cartesian) EPI data.
    """
    nx, etl = d.shape[0], d.shape[1]
    d_shape = d.shape
    d = d.reshape(nx, etl, -1)

    x = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(d, axes=0), axis=0), axes=0)

    x_coord = (np.arange(nx) - nx / 2 + 0.5) / nx
    th = (a[0] + a[1] * x_coord)[:, None]  # [nx, 1], same for every even echo

    x[:, 1::2, :] = x[:, 1::2, :] * np.exp(-1j * th)[:, :, None]

    dc = np.fft.fftshift(np.fft.fft(np.fft.fftshift(x, axes=0), axis=0), axes=0)
    return dc.reshape(d_shape)
