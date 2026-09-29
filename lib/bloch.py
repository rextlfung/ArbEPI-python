"""Bloch simulation of a slab-selective excitation, for checking RF designs.

`simulate_excitation` plays an excitation block (rf + slab gradient) and its
rephaser block exactly as the sequence does, for spins at slab position z and
off-resonance f, and returns the resulting transverse magnetization and flip.
Relaxation is ignored. Hard-pulse (Cayley-Klein) rotation per RF raster step.
"""

from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pypulseq as pp


def _grad_at(g: SimpleNamespace, t: np.ndarray) -> np.ndarray:
    """Gradient amplitude (Hz/m) of a pypulseq trap or extended-trapezoid/
    arbitrary gradient at block times t (s)."""
    if g.type == 'trap':
        tt = g.delay + np.cumsum([0, g.rise_time, g.flat_time, g.fall_time])
        wf = np.array([0, g.amplitude, g.amplitude, 0])
    else:
        tt, wf = g.delay + g.tt, g.waveform
    return np.interp(t, tt, wf, left=0.0, right=0.0)


def simulate_excitation(
    rf: SimpleNamespace,
    gz: SimpleNamespace,
    gz_rephase: SimpleNamespace,
    df_hz,
    z_m,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mxy (complex, M0 = 1) and flip (deg, from Mz) after the [rf, gz] block
    and the [gz_rephase] block, on a (len(z_m), len(df_hz)) grid. df_hz is
    relative to the scanner frequency; rf.freq_offset is honored."""
    z = np.atleast_1d(np.asarray(z_m, float))[:, None]
    df = np.atleast_1d(np.asarray(df_hz, float))[None, :]
    dt = np.diff(rf.t).mean()
    if len(rf.t) < 3 or not np.allclose(np.diff(rf.t), dt):
        raise ValueError('simulate_excitation needs an RF waveform on a uniform raster')

    n1 = int(round(pp.calc_duration(rf, gz) / dt))
    t1 = (np.arange(n1) + 0.5) * dt
    b1 = np.interp(t1, rf.delay + rf.t, rf.signal, left=0, right=0) * np.exp(1j * rf.phase_offset)
    b1[(t1 < rf.delay) | (t1 > rf.delay + rf.shape_dur)] = 0
    g1 = _grad_at(gz, t1)
    n2 = int(round(pp.calc_duration(gz_rephase) / dt))
    g2 = _grad_at(gz_rephase, (np.arange(n2) + 0.5) * dt)

    a = np.ones(np.broadcast(z, df).shape, complex)
    b = np.zeros_like(a)
    for bb, gg in zip(b1, g1):
        w0 = 2 * np.pi * (df - rf.freq_offset + gg * z) * dt
        wxy = 2 * np.pi * bb * dt
        phi = np.sqrt(np.abs(wxy) ** 2 + w0**2)
        s = np.where(phi > 0, np.sin(phi / 2) / np.where(phi > 0, phi, 1), 0.5)
        ca = np.cos(phi / 2) - 1j * w0 * s
        cb = -1j * wxy * s
        a, b = ca * a - np.conj(cb) * b, cb * a + np.conj(ca) * b
    mxy = 2 * np.conj(a) * b
    mz = np.abs(a) ** 2 - np.abs(b) ** 2
    # Rephaser: free precession in the gradient (no RF), which in this
    # rotation convention multiplies Mxy by exp(+i 2 pi (f + G z) t) -- the
    # same sense as the RF loop above. Off-resonance phase accrued here is
    # common to every z and left in.
    mxy = mxy * np.exp(2j * np.pi * (g2.sum() * dt * z + df * n2 * dt))
    return mxy, np.degrees(np.arccos(np.clip(mz, -1, 1)))
