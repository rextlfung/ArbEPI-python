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
    gout = pp.make_trapezoid(
        channel=gin.channel,
        system=sys,
        amplitude=gin.amplitude,  # dummy value, rescaled below
        rise_time=math.ceil(gin.rise_time / common_raster_time) * common_raster_time,
        flat_time=math.ceil(gin.flat_time / common_raster_time) * common_raster_time,
        fall_time=math.ceil(gin.fall_time / common_raster_time) * common_raster_time,
    )

    if abs(gin.area) > 1e-6:
        gout.amplitude = gout.amplitude * gin.area / gout.area

    gout.area = gin.area

    return gout
