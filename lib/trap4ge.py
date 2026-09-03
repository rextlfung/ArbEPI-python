"""Ported from ../PulCeq/matlab/trap4ge.m."""

import math
from types import SimpleNamespace

import pypulseq as pp


def trap4ge(gin: SimpleNamespace, common_raster_time: float, sys: pp.Opts) -> SimpleNamespace:
    """Extend trapezoid rise/flat/fall times to a common raster time boundary.

    Ensures sample points land on the raster boundary given by
    `common_raster_time` (`params.crt`), so interpolation is accurate --
    matters most for CAIPI EPI blips, where blip area must be preserved
    after interpolation. `crt` is GE-only now (`4e-6`), not the lcm of
    Siemens' 10us and GE's 4us rasters this docstring used to describe --
    see CLAUDE.md's `trap4ge` paragraph for why, and note that at
    `crt == grad_raster_time` (today's setting) this whole function is a
    measured no-op (0 of 11 real call sites change any rise/flat/fall time
    or amplitude): pypulseq already puts every trapezoid on-raster.
    Kept anyway -- it's the right net if `crt` ever reverts to `20e-6` for
    Siemens dual-raster support (docs/review-findings.md item 17).
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
        gout.flat_area = gout.amplitude * gout.flat_time

    gout.area = gin.area

    return gout
