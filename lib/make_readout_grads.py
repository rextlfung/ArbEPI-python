"""Ported from ../ArbEPI/lib/make_readout_grads.m.

Creates EPI readout gradients, blips, and the ADC event. Blips are stored
at unit amplitude and scaled at assembly time via `pp.scale_grad`. The
readout trapezoid is circularly shifted so blips fit within each Pulseq
block boundary.
"""

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import pypulseq as pp

from trap4ge import trap4ge


@dataclass
class ReadoutGrads:
    gro: SimpleNamespace
    gro1: SimpleNamespace
    gro2: SimpleNamespace
    gy_blip: SimpleNamespace
    gz_blip: SimpleNamespace
    adc: SimpleNamespace
    Tread: float
    Nfid: int
    blip_duration: float
    deltak: Sequence[float]
    max_blip_area: float


def make_readout_grads(
    max_ky_step: float,
    max_kz_step: float,
    Nx: int,
    fov: Sequence[float],
    dwell: float,
    sys: pp.Opts,
    crt: float,
) -> ReadoutGrads:
    """
    Parameters
    ----------
    max_ky_step : largest ky blip step (in k-space index units)
    max_kz_step : largest kz blip step (in k-space index units)
    Nx : readout matrix size
    fov : field of view [x, y, z] (m)
    dwell : ADC sample dwell time (s)
    sys : pypulseq system (Opts)
    crt : common raster time for GE compatibility (s)
    """
    deltak = [1 / f for f in fov]

    # Size blips to support the largest steps across all frames/shots.
    gy_blip = pp.make_trapezoid('y', system=sys, area=max_ky_step * deltak[1])
    gy_blip = pp.scale_grad(gy_blip, 1 / max_ky_step, sys)
    gy_blip = trap4ge(gy_blip, crt, sys)
    gz_blip = pp.make_trapezoid('z', system=sys, area=max_kz_step * deltak[2])
    gz_blip = pp.scale_grad(gz_blip, 1 / max_kz_step, sys)
    gz_blip = trap4ge(gz_blip, crt, sys)

    # Match blip durations so they always fit within one readout block.
    if pp.calc_duration(gy_blip) > pp.calc_duration(gz_blip):  # y blip is longer
        max_blip_area = max_ky_step * deltak[1]
        blip_duration = pp.calc_duration(gy_blip)
        gz_blip = pp.make_trapezoid('z', system=sys, area=max_kz_step * deltak[2], duration=blip_duration)
        gz_blip = pp.scale_grad(gz_blip, 1 / max_kz_step, sys)
    else:  # z blip is longer
        max_blip_area = max_kz_step * deltak[2]
        blip_duration = pp.calc_duration(gz_blip)
        gy_blip = pp.make_trapezoid('y', system=sys, area=max_ky_step * deltak[1], duration=blip_duration)
        gy_blip = pp.scale_grad(gy_blip, 1 / max_ky_step, sys)

    # Readout trapezoid sized for ramp-sampling and to contain the blip area.
    systmp = copy.deepcopy(sys)
    systmp.max_grad = deltak[0] / dwell  # enforce Nyquist sampling on flat top
    gro = trap4ge(
        pp.make_trapezoid('x', system=systmp, area=Nx * deltak[0] + max_blip_area), crt, sys
    )

    # Circularly shift gro so blips are contained within each Pulseq block:
    # the waveform is split and reassembled as [second half, -first half].
    gro1, gro2 = pp.split_gradient_at(gro, blip_duration / 2, system=sys)
    gro2.delay = 0
    gro1.delay = gro2.shape_dur
    gro = pp.add_gradients([gro2, pp.scale_grad(gro1, -1)], system=sys)
    gro1.delay = 0  # leading piece played before the first echo line

    # ADC event covering the flat-top portion.
    Tread = pp.calc_duration(gro) - blip_duration
    Nfid = round(Tread / dwell / 4) * 4  # round to multiple of 4 for GE
    adc = pp.make_adc(Nfid, dwell=dwell)

    # Delay blips to play after the ADC window closes.
    gy_blip.delay = Tread
    gz_blip.delay = Tread

    return ReadoutGrads(
        gro=gro,
        gro1=gro1,
        gro2=gro2,
        gy_blip=gy_blip,
        gz_blip=gz_blip,
        adc=adc,
        Tread=Tread,
        Nfid=Nfid,
        blip_duration=blip_duration,
        deltak=deltak,
        max_blip_area=max_blip_area,
    )
