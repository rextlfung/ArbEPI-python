"""Ported from ../ArbEPI/lib/calc_te_tr_delays.m.

Computes TE and TR padding delays for an EPI sequence. Issues a warning
(does not raise) if the prescribed TE/TR is unachievable.
"""

import math
import warnings
from types import SimpleNamespace
from typing import Tuple

import pypulseq as pp


def calc_te_tr_delays(
    rf: SimpleNamespace,
    rfsat: SimpleNamespace,
    gz_ss: SimpleNamespace,
    gz_ssr: SimpleNamespace,
    gx_pre: SimpleNamespace,
    gy_pre: SimpleNamespace,
    gz_pre: SimpleNamespace,
    gro: SimpleNamespace,
    gx_spoil: SimpleNamespace,
    gy_spoil: SimpleNamespace,
    gz_spoil: SimpleNamespace,
    ETL: int,
    TE: float,
    TR: float,
    sys: pp.Opts,
) -> Tuple[float, float, float, float]:
    min_te = (
        0.5 * pp.calc_duration(rf)
        + pp.calc_duration(gz_ssr)
        + max(pp.calc_duration(gx_pre), pp.calc_duration(gy_pre), pp.calc_duration(gz_pre))
        + (ETL / 2 - 0.5) * pp.calc_duration(gro)
    )

    if TE >= min_te:
        te_delay = math.floor((TE - min_te) / sys.block_duration_raster) * sys.block_duration_raster
    else:
        warnings.warn(
            f'Minimum achievable TE ({min_te * 1e3:.3f} ms) exceeds prescribed TE ({TE * 1e3:.3f} ms).'
        )
        te_delay = 0.0

    min_tr = (
        pp.calc_duration(rfsat)
        + max(pp.calc_duration(gx_spoil), pp.calc_duration(gz_spoil))
        + max(pp.calc_duration(rf), pp.calc_duration(gz_ss))
        + pp.calc_duration(gz_ssr)
        + te_delay
        + max(pp.calc_duration(gx_pre), pp.calc_duration(gy_pre), pp.calc_duration(gz_pre))
        + ETL * pp.calc_duration(gro)
        + max(pp.calc_duration(gx_spoil), pp.calc_duration(gy_spoil), pp.calc_duration(gz_spoil))
    )

    if TR >= min_tr:
        tr_delay = math.floor((TR - min_tr) / sys.block_duration_raster) * sys.block_duration_raster
    else:
        warnings.warn(
            f'Minimum achievable TR ({min_tr * 1e3:.3f} ms) exceeds prescribed TR ({TR * 1e3:.3f} ms).'
        )
        tr_delay = 0.0

    return te_delay, tr_delay, min_te, min_tr
