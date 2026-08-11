"""Ported from ../ArbEPI/lib/make_prephasers.m.

Duration is set large enough to support the maximum-area direction without
exceeding the PNS limit (scale factor < 1 avoids max slew rate).
"""

from types import SimpleNamespace
from typing import Sequence, Tuple

import pypulseq as pp

from trap4ge import trap4ge


def make_prephasers(
    Nx: int, Ny: int, Nz: int, fov: Sequence[float], sys: pp.Opts, crt: float
) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    deltak = [1 / f for f in fov]
    tmp = 0.5  # scale factor < 1 to avoid PNS

    gx_pre = trap4ge(
        pp.scale_grad(pp.make_trapezoid('x', system=sys, area=-Nx / 2 * deltak[0] / tmp), tmp, sys),
        crt,
        sys,
    )
    gy_pre = trap4ge(
        pp.scale_grad(pp.make_trapezoid('y', system=sys, area=-Ny / 2 * deltak[1] / tmp), tmp, sys),
        crt,
        sys,
    )
    gz_pre = trap4ge(
        pp.scale_grad(pp.make_trapezoid('z', system=sys, area=-Nz / 2 * deltak[2] / tmp), tmp, sys),
        crt,
        sys,
    )
    return gx_pre, gy_pre, gz_pre
