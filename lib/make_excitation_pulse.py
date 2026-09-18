"""Ported from ../ArbEPI/lib/make_excitation_pulse.m."""

import math
from types import SimpleNamespace
from typing import Sequence, Tuple

import pypulseq as pp

from lib.trap4ge import trap4ge


def make_excitation_pulse(
    alpha: float,
    rf_dur: float,
    rf_tb: float,
    fov: Sequence[float],
    sys: pp.Opts,
    crt: float,
) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    """Create a slab-selective sinc excitation pulse.

    Parameters
    ----------
    alpha : flip angle (degrees)
    rf_dur : RF pulse duration (s)
    rf_tb : RF time-bandwidth product
    fov : field of view [x, y, z] (m); slab thickness = 0.9*fov[2]
    sys : pypulseq system (Opts)
    crt : common raster time for GE compatibility (s)
    """
    # Target a slightly thinner slab to alleviate aliasing.
    rf, gz_ss, gz_ssr = pp.make_sinc_pulse(
        alpha / 180 * math.pi,
        duration=rf_dur,
        slice_thickness=0.9 * fov[2],
        time_bw_product=rf_tb,
        system=sys,
        use='excitation',
        return_gz=True,
    )
    gz_ss = trap4ge(gz_ss, crt, sys)
    # trap4ge always resets gz_ss.delay to 0 (it rebuilds the trapezoid from
    # scratch via pp.make_trapezoid regardless of whether raster-rounding
    # changed anything), so this resync is needed every time, not just when
    # crt actually perturbs the timing -- see this module's own git history/
    # docs/review-findings.md item 63. rf.delay is fixed at pulse-construction
    # time (effectively max(pre-trap4ge gz.rise_time, sys.rf_dead_time)); if a
    # future crt/RF-timing change ever makes trap4ge's rounded-up
    # gz_ss.rise_time exceed that margin, the subtraction goes negative and
    # pypulseq's own seq.check_timing() would only report it downstream as an
    # opaque NEGATIVE_DELAY error with no indication of the actual cause --
    # fail loudly and specifically here instead (docs/review-findings.md item
    # 168).
    assert gz_ss.rise_time <= rf.delay, (
        f'make_excitation_pulse: trap4ge-rounded gz_ss.rise_time ({gz_ss.rise_time * 1e6:.2f} us) '
        f'exceeds rf.delay ({rf.delay * 1e6:.2f} us) -- the RF/slice-select resync '
        f'(gz_ss.delay = rf.delay - gz_ss.rise_time) would go negative. This means crt '
        f'({crt * 1e6:.2f} us) is rounding the slice-select ramp up further than rf.delay\'s '
        f'margin can absorb -- reduce crt or otherwise shorten gz_ss.rise_time.'
    )
    gz_ss.delay = rf.delay - gz_ss.rise_time  # sync RF onset with slice-select gradient
    gz_ssr = trap4ge(gz_ssr, crt, sys)
    return rf, gz_ss, gz_ssr
