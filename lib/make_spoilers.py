"""Ported from ../ArbEPI/lib/make_spoilers.m."""

from types import SimpleNamespace
from typing import Sequence, Tuple

import pypulseq as pp

from lib.trap4ge import trap4ge


def make_spoilers(
    Nx: int,
    Ny: int,
    Nz: int,
    fov: Sequence[float],
    n_cycles_spoil: int,
    sys: pp.Opts,
    crt: float,
) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    deltak = [1 / f for f in fov]
    tmp = 0.5  # scale factor < 1 to avoid PNS

    gx_spoil = trap4ge(
        pp.scale_grad(
            pp.make_trapezoid('x', system=sys, area=Nx * deltak[0] * n_cycles_spoil / tmp), tmp, sys
        ),
        crt,
        sys,
    )
    gy_spoil = trap4ge(
        pp.scale_grad(
            pp.make_trapezoid('y', system=sys, area=Ny * deltak[1] * n_cycles_spoil / tmp), tmp, sys
        ),
        crt,
        sys,
    )
    gz_spoil = trap4ge(
        pp.scale_grad(
            pp.make_trapezoid('z', system=sys, area=Nz * deltak[2] * n_cycles_spoil / tmp), tmp, sys
        ),
        crt,
        sys,
    )
    return gx_spoil, gy_spoil, gz_spoil
