"""Ported from ../ArbEPI/lib/make_spoilers.m, later reworked to express
spoiler gradient area directly in cycles of phase twist per voxel (the
physical quantity spoiler design is usually specified by in the
literature, e.g. Nielsen et al., MRM 2016) instead of via an
Nx/fov/deltak indirection: a gradient of area A dephases spins by A*res
cycles across one voxel of size res, so area = n_cycles / res exactly.

Callers that want shot-to-shot spoiler variation (to avoid an exact
RF-spoiling-cycle lock -- see CLAUDE.md's RF-spoiling section) should
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
    """
    tmp = 0.5  # scale factor < 1 to avoid PNS

    spoilers = []
    for axis, res_ax, n_cyc in zip(('x', 'y', 'z'), res, n_cycles_spoil):
        area = n_cyc / res_ax
        spoilers.append(
            trap4ge(
                pp.scale_grad(pp.make_trapezoid(axis, system=sys, area=area / tmp), tmp, sys),
                crt,
                sys,
            )
        )
    return tuple(spoilers)
