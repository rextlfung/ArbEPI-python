"""Minimal Shinnar-Le Roux (SLR) RF pulse design: the 90-degree excitation
('ex') subset of John Pauly's `dzrf`, for the fat-sat pulse.

Pauly J, Le Roux P, Nishimura D, Macovski A. "Parameter relations for the
Shinnar-Le Roux selective excitation pulse design algorithm." IEEE Trans Med
Imaging. 1991;10(1):53-65.

Ported from SigPy's `sigpy.mri.rf.slr` (BSD-3-Clause, Copyright (c) 2016,
Frank Ong), itself a port of Pauly's MATLAB `rf_tools` that
`toppe.utils.rf.makeslr` (the MATLAB original's fat-sat design, see
`lib/make_fatsat_rf.py`) calls. Vendored rather than depending on sigpy for the
same reason `sampling/pd_sample.py` is: the sequence side only needs ~100
lines of it, and sigpy pulls in its own FFT wrappers that silently cast real
input to complex64 (this port stays in float64/complex128 throughout).

Only what `dzrf(n, tb, 'ex', ftype, d1, d2)` needs is here: `ftype` 'min'
(minimum phase, via a factored Parks-McClellan filter) or 'ls' (linear phase,
least squares), then the inverse SLR transform. Validated against sigpy 0.1.27's
`dzrf` and MATLAB's `makeslr` (see tests/test_slr.py).
"""

import numpy as np
from scipy import signal


def dzrf_ex(
    n: int, tb: float, ftype: str = 'min', d1: float = 0.01, d2: float = 0.01
) -> np.ndarray:
    """Design a 90-degree SLR excitation pulse.

    Parameters
    ----------
    n : number of time samples.
    tb : time-bandwidth product.
    ftype : 'min' (minimum phase) or 'ls' (least-squares linear phase).
    d1, d2 : passband / stopband ripple of the magnetization profile.

    Returns
    -------
    rf : length-n real array, each sample the flip (rad) of one hard pulse of
        duration T/n; sum(rf) is the total on-resonance flip (~pi/2).
    """
    # calc_ripples('ex'): ripple of the Mxy profile -> ripple of the beta filter
    bsf = np.sqrt(0.5)
    d1 = np.sqrt(d1 / 2)
    d2 = d2 / np.sqrt(2)
    if ftype == 'min':
        b = _dzmp(n, tb, d1, d2)[::-1]
    elif ftype == 'ls':
        b = _dzls(n, tb, d1, d2)
    else:
        raise ValueError(f"ftype must be 'min' or 'ls', got {ftype!r}")
    b = bsf * b
    return np.real(_ab2rf(_b2a(b), b))


def _dinf(d1: float, d2: float) -> float:
    """D-infinity (Pauly 1991 eq. 25): transition width x time-bandwidth for a
    linear-phase filter with ripples d1, d2."""
    l1, l2 = np.log10(d1), np.log10(d2)
    return (
        (5.309e-3 * l1**2 + 7.114e-2 * l1 - 4.761e-1) * l2
        + (-2.66e-3 * l1**2 - 5.941e-1 * l1 - 4.278e-1)
    )


def _dzls(n, tb, d1, d2):
    w = _dinf(d1, d2) / tb
    f = np.array([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), n / 2]) / (n / 2)
    h = signal.firls(n + 1, f, [1, 1, 0, 0], weight=[1, d1 / d2])
    # shift the filter half a sample to make it symmetric, like MATLAB's firls
    k = np.concatenate([np.arange(0, n / 2 + 1), np.arange(-n / 2, 0)])
    c = np.exp(1j * 2 * np.pi / (2 * (n + 1)) * k)
    return np.real(np.fft.ifft(np.fft.fft(h) * c))[:n]


def _dzmp(n, tb, d1, d2):
    # design a linear-phase filter with squared ripple specs, then factor out
    # its minimum-phase half
    n2 = 2 * n - 1
    di = 0.5 * _dinf(2 * d1, 0.5 * d2 * d2)
    w = di / tb
    f = np.array([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), n / 2]) / n
    hl = signal.remez(n2, f, [1, 0], weight=[1, 2 * d1 / (0.5 * d2 * d2)])
    return _fmp(hl)


def _fmp(h):
    """Minimum-phase factor of a linear-phase filter."""
    ll = h.size
    lp = int(128 * 2 ** np.ceil(np.log2(ll)))
    hp = np.pad(h, (int(np.ceil((lp - ll) / 2)), int(np.floor((lp - ll) / 2))))
    hpf = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(hp)))  # centered FFT
    hpfs = hpf - np.min(np.real(hpf)) * 1.000001
    hpfmp = _mag2mp(np.sqrt(np.abs(hpfs)))
    hpmp = np.fft.ifft(np.fft.ifftshift(np.conj(hpfmp)))
    return hpmp[: (ll + 1) // 2]


def _mag2mp(x):
    """Minimum-phase spectrum with magnitude |x| (folded cepstrum)."""
    n = x.size
    xlf = np.fft.fft(np.log(np.abs(x)))
    xlf[1 : n // 2] *= 2  # double positive quefrencies
    xlf[n // 2 + 1 :] = 0  # zero negative quefrencies
    return np.exp(np.fft.ifft(xlf))


def _b2a(b):
    """Minimum-phase alpha polynomial with |alpha|^2 + |beta|^2 = 1."""
    n = b.size
    npad = 16 * n
    bf = np.fft.fft(np.concatenate([b, np.zeros(npad - n)]).astype(complex))
    bfmax = np.max(np.abs(bf))
    if bfmax >= 1:
        bf = bf / (1e-7 + bfmax)
    afa = _mag2mp(np.sqrt(1 - np.abs(bf) ** 2))
    a = np.fft.fft(afa) / npad
    return a[:n][::-1]


def _ab2rf(a, b):
    """Inverse SLR transform: peel off one hard-pulse rotation per sample."""
    n = a.size
    rf = np.zeros(n, dtype=complex)
    a = a.astype(complex)
    b = b.astype(complex)
    for ii in range(n - 1, -1, -1):
        cj = np.sqrt(1 / (1 + np.abs(b[ii] / a[ii]) ** 2))
        sj = np.conj(cj * b[ii] / a[ii])
        rf[ii] = 2 * np.arctan2(np.abs(sj), cj) * np.exp(1j * np.angle(sj))
        if ii > 0:
            at = cj * a + sj * b
            bt = -np.conj(sj) * a + cj * b
            a = at[1 : ii + 1]
            b = bt[:ii]
    return rf
