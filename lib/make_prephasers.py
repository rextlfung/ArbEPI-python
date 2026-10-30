"""Ported from ../ArbEPI/lib/make_prephasers.m.

Duration is set large enough to support the maximum-area direction without
exceeding the PNS limit (scale factor < 1 avoids max slew rate). All three
axes share that one (longest-needed) duration rather than each keeping its
own independently-computed one, since a pypulseq block's duration is set by
its longest gradient event anyway -- a shorter axis-specific duration would
just mean that axis's own trapezoid finishes early and idles for the rest
of the block, with no benefit (a real, if not previously live, consistency
bug in this port; tests/test_make_prephasers.py guards it).
"""

import math
from types import SimpleNamespace
from typing import Sequence, Tuple

import pypulseq as pp

from lib.trap4ge import ceil_to_raster, trap4ge


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


def stretched_prephasers(
    pre_areas: dict, te_base: float, te_target: float, sys: pp.Opts, crt: float
) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, float]:
    """x/y/z prephasers (areas by axis in `pre_areas`) that share one duration, the
    *longest* that still lets the echo reach `te_target` -- they absorb the TE
    padding instead of playing at full slew and then waiting. TE is fixed, so a
    longer prephaser costs no time, and it lowers their amplitude and slew-time,
    lowering PNS (sequences/deGRE.py: 85.5% peak PNS shortest-duration vs 79.2%
    stretched, 2026-09-30). `te_base` is the time from the RF center to the start
    of the prephasers plus what follows them to the echo. Floored at the shortest
    duration that fits the largest area; the caller raises if even that misses the
    target. Returns (gx_pre, gy_pre, gz_pre, block duration)."""

    def make(duration):
        return tuple(
            trap4ge(pp.make_trapezoid(ax, system=sys, area=a, duration=duration), crt, sys)
            for ax, a in pre_areas.items()
        )

    def block_duration(pre):
        return max(pp.calc_duration(g) for g in pre)

    t_min = ceil_to_raster(
        max(
            pp.calc_duration(pp.make_trapezoid(ax, system=sys, area=a))
            for ax, a in pre_areas.items()
        ),
        crt,
    )
    t_pre = max(t_min, math.floor((te_target - te_base) / crt + 1e-9) * crt)
    pre = make(t_pre)
    # trap4ge rounds each ramp/flat segment up to crt, which can overshoot t_pre
    # by a raster step or two; back off until the echo fits.
    while t_pre > t_min and te_base + block_duration(pre) > te_target + 1e-9:
        t_pre -= crt
        pre = make(t_pre)
    return (*pre, block_duration(pre))


def prephase_rewinders(
    pre_areas: dict, duration: float, sys: pp.Opts, crt: float
) -> Tuple[SimpleNamespace, SimpleNamespace]:
    """y and z rewinders of `pre_areas`' prephasers, `duration` long (they play in
    the spoiler's block, so the prephasers' stretch never lengthens TR)."""
    gy, gz = (
        trap4ge(pp.make_trapezoid(ax, system=sys, area=-pre_areas[ax], duration=duration), crt, sys)
        for ax in ('y', 'z')
    )
    return gy, gz
