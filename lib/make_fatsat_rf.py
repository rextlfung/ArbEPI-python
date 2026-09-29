"""Adapted from ../ArbEPI/lib/make_fatsat_rf.m.

Spectrally selective fat-saturation pulse: a 90-degree Shinnar-Le Roux pulse
(`lib/slr.py`) played at the fat frequency, the same design route as the
MATLAB original's `toppe.utils.rf.makeslr(..., 'type', 'ex')` -- real part of
the SLR waveform, designed at 200 samples, resampled to the RF raster, area
scaled to the prescribed flip.

The pulse must leave water alone, not just saturate fat: it plays every shot,
and whatever it tips water by becomes coherence that the per-shot random
spoilers refocus somewhere different each frame (docs/review-findings.md item
255). The port's earlier pypulseq `make_gauss_pulse` (90 deg, 4 ms, TBW 3) tipped
on-resonance water by 26 deg and was the main source of the ~3% temporal
fluctuation in static phantom scans (median voxel CV 3.20% -> 0.58% with the
pulse off, `20260924ball`). The MATLAB original's own parameters (min-phase
SLR, 4 ms, TBW 3, a 750 Hz band centered 447 Hz from water) would still tip
water 9 deg on resonance and 28 deg at -100 Hz. `params.py`'s defaults (see its
`fatsat` comment) trade a longer pulse and a narrower band for a stop band
that covers measured in-object B0; use `flip_profile` below to check any
change before scanning.
"""

import math
from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from lib.slr import dzrf_ex
from params import FatsatParams

_N_DESIGN = 200  # SLR design length, as in makeslr; resampled to the RF raster


def make_fatsat_rf(fatsat: FatsatParams, sys: pp.Opts, fat_offres_freq: float) -> SimpleNamespace:
    """Create a fat-saturation RF pulse object.

    Parameters
    ----------
    fatsat : flip (deg), tbw, dur (s), ftype ('min' or 'ls')
    sys : pypulseq system (Opts)
    fat_offres_freq : fat off-resonance frequency (Hz), positive; the pulse
        plays at -fat_offres_freq.
    """
    design = dzrf_ex(_N_DESIGN, fatsat.tbw, fatsat.ftype)
    n = round(fatsat.dur / sys.rf_raster_time)
    # Each design sample is one hard pulse of dur/_N_DESIGN; interpolate between
    # their centers onto the raster's sample centers.
    t_design = (np.arange(_N_DESIGN) + 0.5) / _N_DESIGN
    t_raster = (np.arange(n) + 0.5) / n
    signal = np.interp(t_raster, t_design, design)
    rfsat = pp.make_arbitrary_rf(
        signal,
        fatsat.flip / 180 * math.pi,
        system=sys,
        use='saturation',
    )
    rfsat.freq_offset = -fat_offres_freq  # Hz
    return rfsat


def flip_profile(rf: SimpleNamespace, df_hz) -> np.ndarray:
    """Bloch-simulate the flip angle (deg, from Mz) that `rf` produces on spins
    at off-resonance `df_hz` (Hz, relative to the scanner frequency, so fat is
    at about -fat_offres_freq). Relaxation is ignored.

    Hard-pulse (Cayley-Klein) rotation per RF sample, vectorized over df_hz.
    `rf` must be sampled on a uniform raster at sample centers, as
    `make_arbitrary_rf`/`make_gauss_pulse` produce (not `make_block_pulse`'s
    two-point shape).
    """
    df = np.atleast_1d(np.asarray(df_hz, dtype=float))
    dt = np.diff(rf.t).mean()
    if len(rf.t) < 3 or not np.allclose(np.diff(rf.t), dt) or not np.isclose(rf.t[0], dt / 2):
        raise ValueError('flip_profile needs an RF waveform sampled at uniform raster centers')
    w0 = 2 * np.pi * (df - rf.freq_offset) * dt
    a = np.ones_like(w0, dtype=complex)
    b = np.zeros_like(w0, dtype=complex)
    b1 = rf.signal * np.exp(1j * rf.phase_offset)
    for s in b1:
        wxy = 2 * np.pi * s * dt
        phi = np.sqrt(np.abs(wxy) ** 2 + w0**2)
        sinc_half = np.where(phi > 0, np.sin(phi / 2) / np.where(phi > 0, phi, 1), 0.5)
        ca = np.cos(phi / 2) - 1j * w0 * sinc_half
        cb = -1j * wxy * sinc_half
        a, b = ca * a - np.conj(cb) * b, cb * a + np.conj(ca) * b
    mz = np.abs(a) ** 2 - np.abs(b) ** 2
    return np.degrees(np.arccos(np.clip(mz, -1, 1)))
