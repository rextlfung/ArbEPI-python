"""Ported from ../ArbEPI/lib/make_prephasers.m.

Duration is set large enough to support the maximum-area direction without
exceeding the PNS limit (scale factor < 1 avoids max slew rate). All three
axes share that one (longest-needed) duration rather than each keeping its
own independently-computed one, since a pypulseq block's duration is set by
its longest gradient event anyway -- a shorter axis-specific duration would
just mean that axis's own trapezoid finishes early and idles for the rest
of the block, with no benefit (a real, if not previously live, consistency
bug in this port -- see docs/review-findings.md item 28).
"""

from types import SimpleNamespace
from typing import Sequence, Tuple

import pypulseq as pp

from lib.trap4ge import trap4ge


def make_prephasers(
    Nx: int, Ny: int, Nz: int, fov: Sequence[float], sys: pp.Opts, crt: float
) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    deltak = [1 / f for f in fov]
    tmp = 0.5  # scale factor < 1 to avoid PNS
    channels = ('x', 'y', 'z')
    ns = (Nx, Ny, Nz)

    # Virtual (pre-scale_grad) area per axis -- built at area/tmp so that
    # scaling the resulting trapezoid's amplitude by tmp afterward yields
    # the true target area with slower (PNS-friendlier) ramps than a
    # direct area=-N/2*deltak build would use.
    virtual_areas = [-n / 2 * dk / tmp for n, dk in zip(ns, deltak)]
    natural = [pp.make_trapezoid(ch, system=sys, area=a) for ch, a in zip(channels, virtual_areas)]
    duration = max(pp.calc_duration(g) for g in natural)

    gx_pre, gy_pre, gz_pre = (
        trap4ge(
            pp.scale_grad(pp.make_trapezoid(ch, system=sys, area=a, duration=duration), tmp, sys),
            crt,
            sys,
        )
        for ch, a in zip(channels, virtual_areas)
    )
    return gx_pre, gy_pre, gz_pre
