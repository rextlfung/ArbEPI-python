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
    gz_ss.delay = rf.delay - gz_ss.rise_time  # sync RF onset with slice-select gradient
    gz_ssr = trap4ge(gz_ssr, crt, sys)
    return rf, gz_ss, gz_ssr
