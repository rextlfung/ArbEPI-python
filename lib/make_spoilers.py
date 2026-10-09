"""Ported from ../ArbEPI/lib/make_spoilers.m, later reworked to express
spoiler gradient area directly in cycles of phase twist per voxel (the
physical quantity spoiler design is usually specified by in the
literature, e.g. Nielsen et al., MRM 2016) instead of via an
Nx/fov/deltak indirection: a gradient of area A dephases spins by A*res
cycles across one voxel of size res, so area = n_cycles / res exactly.

Callers that want shot-to-shot spoiler variation (to avoid an exact
RF-spoiling-cycle lock -- see params.py's Params.spoil_cycles_min comment) should
build at the maximum cycles/voxel value they'll use and scale down per
shot via pp.scale_grad, matching the pattern sequences/ArbEPI.py already
uses for the mandatory ky/kz rewind.
"""

from types import SimpleNamespace
from typing import Sequence, Tuple

import pypulseq as pp

from lib.trap4ge import trap4ge


def make_spoilers(
    res: Sequence[float],
    n_cycles_spoil: Sequence[float],
    sys: pp.Opts,
    crt: float,
) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    """
    res : [res_x, res_y, res_z], voxel dimensions (m).
    n_cycles_spoil : [n_x, n_y, n_z], cycles of gradient-induced phase
        twist across one voxel, along each axis.

    All three trapezoids share one duration, the minimum that fits the
    largest-area channel under the hardware slew/amplitude limits.
    """
    tmp = 0.5  # scale factor < 1 to avoid PNS
    channels = ('x', 'y', 'z')

    # Virtual (pre-scale_grad) area per axis, as in make_prephasers.py.
    virtual_areas = [n_cyc / res_ax / tmp for res_ax, n_cyc in zip(res, n_cycles_spoil)]

    # One shared duration for all three channels: the shortest that
    # accommodates the channel with the greatest area requirement. A
    # pypulseq block lasts as long as its longest gradient anyway, so a
    # shorter axis would only finish early and idle (while ramping faster,
    # for more PNS, than it needs to); docs/review-findings.md item 142.
    duration = max(
        pp.calc_duration(pp.make_trapezoid(ch, system=sys, area=a))
        for ch, a in zip(channels, virtual_areas)
    )

    return tuple(
        trap4ge(
            pp.scale_grad(pp.make_trapezoid(ch, system=sys, area=a, duration=duration), tmp, sys),
            crt,
            sys,
        )
        for ch, a in zip(channels, virtual_areas)
    )
