"""Ported from ../PulCeq/matlab/trap4ge.m."""

import math
from types import SimpleNamespace

import pypulseq as pp


def trap4ge(gin: SimpleNamespace, common_raster_time: float, sys: pp.Opts) -> SimpleNamespace:
    """Extend trapezoid rise/flat/fall times to a common raster time boundary.

    Ensures sample points land on both the Siemens (10us) and GE (4us) raster
    boundaries, so interpolation between them is accurate. This matters most
    for CAIPI EPI blips, where blip area must be preserved after interpolation.
    """
    # Subtract a small epsilon before ceil so a time already on-raster (up to
    # float64 rounding noise in the division) doesn't get bumped up to the
    # next raster step -- e.g. 0.002 / 4e-6 can evaluate to 500.00000000000006
    # instead of 500.0, which would otherwise silently pad by one extra step.
    def _round_up_to_raster(t: float) -> float:
        return math.ceil(t / common_raster_time - 1e-9) * common_raster_time

    gout = pp.make_trapezoid(
        channel=gin.channel,
        system=sys,
        amplitude=gin.amplitude,  # dummy value, rescaled below
        rise_time=_round_up_to_raster(gin.rise_time),
        flat_time=_round_up_to_raster(gin.flat_time),
        fall_time=_round_up_to_raster(gin.fall_time),
    )

    if abs(gin.area) > 1e-6:
        gout.amplitude = gout.amplitude * gin.area / gout.area

    gout.area = gin.area

    return gout
