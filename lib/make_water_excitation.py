"""Slab-selective binomial water excitation: an alternative to fat-sat +
slab-selective excitation (`params.excitation = 'water'`).

Binomial subpulses (default 1-3-3-1) spaced by half a fat precession period,
tau = 1 / (2 * fat_offres_freq) (1.12 ms at 3 T): water sees the subpulses
add up to the full flip, while fat, 180 deg further along at each subpulse,
sees them cancel (mriquestions.com/water-excitation.html; the "fast water
excitation" of Stirnberg et al., NeuroImage 2017;163:81-92, is a non-selective
single-rect variant, Stirnberg et al., MRM 2016;76:1517-1523). Each subpulse is
a sinc on the flat top of its own slab-select lobe, for the same 0.9 * fov_z
slab as `make_excitation_pulse`'s single sinc. Hanning-apodized by default: an
unapodized TBW 8 subpulse overshoots the slab profile by 16% near the edges,
Hanning by 0.5%.

Monopolar lobes, each followed by a triangular rewinder of its full area, not
bipolar lobes: every subpulse then excites fat in the same chemical-shift-
displaced slab, so fat cancels across the whole slab. With bipolar lobes the
displacement alternates sign between subpulses, and fat at the slab edges
(within ~f L / BW of them) saw only one polarity's subpulses: Bloch-simulated,
1-3-3-1 at TBW 8 left up to ~4 deg of fat excitation (~24% of a 17 deg water
flip) at z = +-65 mm, vs <= 0.8 deg anywhere with monopolar lobes. Raising the
subpulse TBW only narrows that band. Monopolar lobes are also immune to an
odd/even gradient delay shifting alternate subpulses' slabs.

The whole train is one RF event (zero during ramps and rewinders) on one
extended-trapezoid gradient, so the GE RF dead time and ringdown are paid once
rather than per subpulse; it drops into the same (rf, gz_ss, gz_ssr) slot as
`make_excitation_pulse`. `rf.center` is the train's weighted center, so
`calc_te_tr_delays` anchors TE there.

Trade-off: water off resonance is excited less, as cos^3(pi f tau) for 1-3-3-1
(Bloch at 17 deg: 0.89 at +-80 Hz, 0.64 at -150 Hz, 0.44 at -200 Hz).
"""

import math
from types import SimpleNamespace
from typing import Sequence, Tuple

import numpy as np
import pypulseq as pp

from lib.trap4ge import trap4ge


def _ceil_to(t: float, r: float) -> float:
    return math.ceil(t / r - 1e-9) * r


def make_water_excitation(
    alpha: float,
    tbw: float,
    fov: Sequence[float],
    sys: pp.Opts,
    crt: float,
    fat_offres_freq: float,
    binomial: Sequence[int] = (1, 3, 3, 1),
    apodization: float = 0.5,
) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    """Create a slab-selective binomial water-excitation pulse.

    Parameters
    ----------
    alpha : total flip angle on resonance water (degrees)
    tbw : time-bandwidth product of each sinc subpulse
    fov : field of view [x, y, z] (m); slab thickness = 0.9*fov[2]
    sys : pypulseq system (Opts); its max_slew sets every ramp
    crt : common raster time for GE compatibility (s)
    fat_offres_freq : fat off-resonance frequency (Hz, positive)
    binomial : subpulse flip weights
    apodization : window on each sinc subpulse, as pypulseq's make_sinc_pulse
        (0 = none, 0.5 = Hanning)

    Returns
    -------
    rf, gz_ss, gz_ssr : the RF train, its slab-select gradient (extended
        trapezoid, delay-synced to rf) and the slab rephaser -- the same slot
        as make_excitation_pulse's return.
    """
    thk = 0.9 * fov[2]
    tau = round(1 / (2 * fat_offres_freq) / crt) * crt  # subpulse spacing
    dt = sys.rf_raster_time
    slew = sys.max_slew

    # Longest subpulse (flat top d) whose lobe + rewinder fit in tau, with the
    # idle time around the rewinder splitting evenly on the raster.
    d = _ceil_to(tau, crt)
    while True:
        d -= crt
        if d <= 0:
            raise ValueError(f'no subpulse fits the {tau * 1e6:.0f} us binomial spacing')
        G = tbw / d / thk  # Hz/m
        r = _ceil_to(G / slew, crt)  # lobe ramp
        area = G * (d + r)  # full lobe area
        tr = _ceil_to(math.sqrt(area / slew), crt)  # rewinder triangle half-width
        idle = tau - (2 * r + d) - 2 * tr  # gap between lobes not used by the rewinder
        if idle >= -1e-12 and round(idle / crt) % 2 == 0:
            break
    Gr = area / tr  # rewinder peak (<= slew * tr by construction)
    pad = idle / 2

    # Gradient corners, t = 0 at the first lobe's ramp start. Each rewinder is
    # centered midway between two subpulse centers, so the moment from any such
    # midpoint (and from the train's center, rf.center) to the next subpulse
    # center is zero: pypulseq's calculate_kspace, which resets k at rf.center,
    # then agrees with where the spins actually are.
    n = len(binomial)
    times, amps = [0.0], [0.0]
    for k in range(n):
        t0 = k * tau
        t_end = t0 + 2 * r + d
        times += [t0 + r, t0 + r + d, t_end]
        amps += [G, G, 0.0]
        if k < n - 1:
            if pad > 1e-12:
                times += [t_end + pad]
                amps += [0.0]
            times += [t_end + pad + tr, t_end + pad + 2 * tr]
            amps += [-Gr, 0.0]
            if pad > 1e-12:
                times += [t0 + tau]
                amps += [0.0]
    times = np.round(np.array(times) / crt) * crt

    # RF: from the first flat top's start to the last one's end; each subpulse a
    # sinc of duration d centered on its flat top, zero in between.
    n_rf = round(((n - 1) * tau + d) / dt)
    t_rf = (np.arange(n_rf) + 0.5) * dt
    signal = np.zeros(n_rf)
    bw = tbw / d
    for k, w in enumerate(binomial):
        tt = t_rf - (k * tau + d / 2)
        on = np.abs(tt) < d / 2
        window = 1 - apodization + apodization * np.cos(2 * np.pi * tt[on] / d)
        signal[on] += w * window * np.sinc(bw * tt[on])
    center = float(np.dot(binomial, np.arange(n) * tau) / np.sum(binomial)) + d / 2

    rf = pp.make_arbitrary_rf(
        signal, alpha / 180 * math.pi, system=sys, use='excitation', center=center
    )
    assert rf.freq_offset == 0, 'a shifted slab would need per-subpulse modulation'
    delay = max(rf.delay, _ceil_to(r, crt))
    rf.delay = delay
    gz_ss = pp.make_extended_trapezoid(
        channel='z', times=times, amplitudes=np.array(amps), system=sys
    )
    gz_ss.delay = delay - r  # first flat top starts with the RF

    # Rephase the last subpulse's second half (flat d/2 + fall ramp); earlier
    # subpulses' moments to it are zero (each lobe is fully rewound).
    gz_ssr = pp.make_trapezoid(channel='z', area=-G * (d / 2 + r / 2), system=sys)
    gz_ssr = trap4ge(gz_ssr, crt, sys)
    return rf, gz_ss, gz_ssr
