"""Port of ../PulCeq/matlab/+pge2/pns.m -- peripheral nerve stimulation
prediction using the impulse-response model in IEC 60601-2-33:2022.

This is GE's PNS model, not the Siemens SAFE model pypulseq already ships
in pypulseq.utils.safe_pns_prediction (different parameterization --
GE's is a single chronaxie/rheobase pair per axis via `pge2.opts.m`'s coil
table; Siemens' SAFE model uses a 3-time-constant a1/a2/a3/tau1/tau2/tau3
decomposition read from a per-scanner .asc file. Not interchangeable, so
this is a from-scratch port of GE's simpler model, not a reuse of
pypulseq's.
"""

import numpy as np
from scipy.signal import fftconvolve


def pns(
    s_min: float,
    chronaxie: float,
    g: np.ndarray,
    dt: float,
    wt: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    s_min : stimulation threshold (rheobase) for constant slew of infinite
        duration. In GE terms, s_min = rheobase / alpha (see scanners.py /
        ../PulCeq/matlab/+pge2/opts.m's coil table).
    chronaxie : nerve impulse response time constant (s).
    g : (3, n) x/y/z gradient waveform (T/m), uniformly sampled at `dt`.
    dt : gradient raster (sample) time (s).
    wt : x/y/z channel weights (IEC 60601-2-33:2022 section 12 default for
        gradient coils: (0.8, 1.0, 0.7); PulCeq's default is (1, 1, 1)).

    Returns (pt, p): pt is the channel-combined PNS waveform (percent of
    stimulation threshold), shape (n,); p is per-channel, shape (3, n).
    """
    assert g.shape[0] == 3, 'g must be shape (3, n)'
    n = g.shape[1]

    # Contribution of a slew impulse at time 0 to PNS at time tau
    # (IEC 60601-2-33:2022 Eq. AA.21). No need to extend beyond ~20x the
    # chronaxie -- the response has decayed to a small fraction by then.
    n_tau = round(20 * chronaxie / dt)
    tau = (np.arange(n_tau) + 1) * dt
    f = (dt / s_min) * chronaxie / (chronaxie + tau) ** 2

    s = np.diff(g, axis=1) / dt  # T/m/s
    p = np.zeros((3, n))
    for ch in range(3):
        # fftconvolve, not np.convolve -- direct convolution is O(n*m) and
        # much too slow for a full ~60s sequence (n ~1e7) against a ~2000
        # sample kernel; FFT convolution is O(n log n). Same math, some
        # floating-point roundoff (still passes ge/validate_pns.py's
        # cross-check against MATLAB's conv()).
        p[ch] = (wt[ch] * 100 * fftconvolve(s[ch], f))[:n]

    pt = np.sqrt((p ** 2).sum(axis=0))
    return pt, p
