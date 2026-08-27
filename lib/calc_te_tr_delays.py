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
    echo_offset: float = 0.0,
) -> Tuple[float, float, float, float]:
    # echo_offset (ReadoutGrads.echo_offset) is the time from the start of
    # the gro1 lead-in block to the true in-block kx = 0 crossing of each
    # echo, so the nominal echo (continuous echo-train index ETL/2 - 0.5)
    # is anchored at the actual echo, not at the composite block's center.
    # The first term is the time from the true RF center to the end of the
    # RF block (the old `0.5 * pp.calc_duration(rf)` measured from the
    # block's *middle*, which sits (ringdown - dead_time)/2 away from the
    # physical RF center). Before these fixes, the realized TE ran
    # ~0.6-0.7 ms later than prescribed at default params (gro1 block +
    # in-block crossing time + RF-center offset, all silently omitted).
    min_te = (
        max(pp.calc_duration(rf), pp.calc_duration(gz_ss)) - (rf.delay + pp.calc_rf_center(rf)[0])
        + pp.calc_duration(gz_ssr)
        + max(pp.calc_duration(gx_pre), pp.calc_duration(gy_pre), pp.calc_duration(gz_pre))
        + echo_offset
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
