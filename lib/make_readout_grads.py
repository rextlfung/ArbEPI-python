"""Ported from ../ArbEPI/lib/make_readout_grads.m, extended for POPE.

Creates EPI readout gradients, blips, and the ADC event. Blips are stored
at unit amplitude and scaled at assembly time via `pp.scale_grad`. The
readout trapezoid is circularly shifted so blips fit within each Pulseq
block boundary.

The readout trapezoid supports *asymmetric* rise/fall slew rates (POPE:
"PNS Optimized Pulses for EPI", Huber et al., bioRxiv 2026,
doi 10.64898/2026.07.22.739360): under nerve-integration PNS models (GE's
IEC 60601-2-33 chronaxie-kernel model in ge/pns.py, same idea as Siemens'
SAFE), predicted PNS peaks at the *end* of each sustained slew event -- in
an EPI train that event is [fall of lobe n -> blip window -> rise of lobe
n+1], one continuous dG/dt excursion from +A to -A -- so only the ramp-up
slew needs throttling while the ramp-down can stay at/near hardware slew.
In the circularly-shifted composite waveform each lobe's own `rise_time`
is exactly the "ramp-up" half of a transition event and its `fall_time`
the "ramp-down" half, so POPE maps directly onto per-trapezoid
rise/fall slews with no waveform reordering.

Geometry notation used throughout (see also the derivation in the repo's
CLAUDE.md): t_s = blip_duration/2; r/flat/d = rise/flat/fall times; A =
amplitude; a1 = A*t_s^2/(2r) = area of the first t_s of the rise ramp
(= gro1's area); a_d = A*t_s^2/(2d) = area of the last t_s of the fall
ramp; S = full trapezoid area. The composite (circularly shifted) lobe
crosses gx = 0 at time D - t_s (the blip-window center), and the two
readout parities sample the kx windows [k_lo + a1, k_hi - a_d] (odd) and
[k_lo + a_d, k_hi - a1] (even), where k_hi - k_lo = S. Prephasing to
exactly -S/2 (see `gx_pre_scale`) makes those windows symmetric about 0,
makes both parities cross kx = 0 at the same in-block time t0 (solving
integral_0^t0 g = S/2 - a1), and reduces full +-kmax coverage to
S >= Nx*deltak + 2*max(a1, a_d). The symmetric-slew special case of all
this is the original design: with readout ramps at the blip slew,
a1 = a_d = max_blip_area/2 and S = Nx*deltak + max_blip_area.
"""

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import pypulseq as pp

from lib.trap4ge import trap4ge


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
    # Dead under the current POPE (asymmetric-ramp) geometry, which sizes
    # the flat top from `2 * max(a1, a_d)` instead (see below) -- but this
    # was load-bearing pre-POPE (`S = Nx*deltak + max_blip_area`, the
    # symmetric-ramp case where a1 == a_d == max_blip_area/2) and kept
    # deliberately rather than deleted, in case a future change reverts to
    # a symmetric-slew readout where it's needed again.
    max_blip_area: float
    # Scale factor for the x prephaser: gx_pre is built with area exactly
    # -Nx/2*deltak[0] (lib/make_prephasers.py), but centered coverage needs
    # a pre-wind to -S/2 where S is the readout trapezoid's area (module
    # docstring) -- apply via pp.scale_grad(gx_pre, rg.gx_pre_scale) at
    # assembly. Before this existed, sampling was off-center by
    # +a1 (~21 lines at default params): an unintended one-sided partial
    # Fourier inherited from the MATLAB original.
    gx_pre_scale: float
    # Time from the start of the gro1 lead-in block to the kx = 0 crossing
    # within each subsequent full-lobe block's own frame, i.e. per-echo
    # acquisition time = (time at gro1 block start) + echo_offset +
    # echo_index * calc_duration(gro). Identical for both readout parities
    # (module docstring). calc_te_tr_delays consumes this so prescribed TE
    # lands on the true echo, not the block center.
    echo_offset: float


def _ceil_to_raster(t: float, raster: float) -> float:
    # Same epsilon rationale as lib/trap4ge.py's _round_up_to_raster:
    # float64 division noise must not bump an already-on-raster time up by
    # one extra raster step.
    return math.ceil(t / raster - 1e-9) * raster


def _composite_segments(
    A: float, r: float, flat: float, d: float, t_s: float
) -> List[Tuple[float, float, float]]:
    """Piecewise-linear segments (duration, g_start, g_end) of the
    circularly-shifted composite lobe: mid-rise -> peak -> flat -> fall to
    zero -> start of the next (negated) lobe's rise."""
    g_split = A * t_s / r  # amplitude at the split point inside the rise
    return [
        (r - t_s, g_split, A),  # remainder of the rise ramp
        (flat, A, A),  # flat top
        (d, A, 0.0),  # full fall ramp (ends at the kx turnaround)
        (t_s, 0.0, -g_split),  # first t_s of the next lobe's (negated) rise
    ]


def _integrate_to(segments: List[Tuple[float, float, float]], t: float) -> float:
    """Integral of the piecewise-linear waveform from 0 to t."""
    total = 0.0
    for dur, g0, g1 in segments:
        if dur <= 0:
            continue
        if t <= dur:
            return total + g0 * t + (g1 - g0) * t**2 / (2 * dur)
        total += (g0 + g1) / 2 * dur
        t -= dur
    return total


def _invert_integral(segments: List[Tuple[float, float, float]], target: float) -> float:
    """First time t at which the integral from 0 reaches `target` (must lie
    within the segments' monotonically-increasing portion)."""
    elapsed = 0.0
    for dur, g0, g1 in segments:
        if dur <= 0:
            continue
        seg_area = (g0 + g1) / 2 * dur
        if target > seg_area + 1e-12:
            target -= seg_area
            elapsed += dur
            continue
        # Solve a*tau^2 + b*tau - target = 0 within this segment, in the
        # numerically stable form tau = 2*target / (b + sqrt(b^2 + 4a*target))
        # (valid for either sign of a while b >= 0, which holds everywhere
        # the integral is still increasing).
        a = (g1 - g0) / (2 * dur)
        b = g0
        if abs(a) < 1e-12 * abs(b):
            return elapsed + target / b
        disc = b**2 + 4 * a * target
        return elapsed + 2 * target / (b + math.sqrt(max(disc, 0.0)))
    raise ValueError('target area exceeds total waveform area')


def make_readout_grads(
    max_ky_step: float,
    max_kz_step: float,
    Nx: int,
    fov: Sequence[float],
    dwell: float,
    sys: pp.Opts,
    crt: float,
    slew_rise: float | None = None,
    slew_fall: float | None = None,
) -> ReadoutGrads:
    """
    Parameters
    ----------
    max_ky_step : largest ky blip step (in k-space index units)
    max_kz_step : largest kz blip step (in k-space index units)
    Nx : readout matrix size
    fov : field of view [x, y, z] (m)
    dwell : ADC sample dwell time (s)
    sys : pypulseq system (Opts). Its max_slew governs the *blips* (they
        play centered on the kx turnaround, a separate PNS lever from the
        readout ramps); its max_grad caps the readout flat top.
    crt : common raster time for GE compatibility (s)
    slew_rise : readout ramp-up slew (Hz/m/s; None -> sys.max_slew). The
        POPE-throttled slew -- PNS peaks at the end of each ramp-up.
    slew_fall : readout ramp-down slew (Hz/m/s; None -> sys.max_slew).
        Hardware-limited, not PNS-limited, so it can exceed slew_rise.
    """
    deltak = [1 / f for f in fov]
    if slew_rise is None:
        slew_rise = sys.max_slew
    if slew_fall is None:
        slew_fall = sys.max_slew
    assert slew_rise <= slew_fall, (
        'POPE throttles the ramp-up (where accumulated PNS peaks); a rise slew '
        'above the fall slew inverts that and is never what you want.'
    )

    # Size blips to support the largest steps across all frames/shots. A
    # zero step means no shot ever needs a blip on that axis (mask2epi
    # returns (0.0, 0.0) for ETL == 1, and any axis with Ny or Nz == 1
    # hits this too) -- build a zero-area placeholder instead of dividing
    # by zero to reach "unit amplitude": since every real step on that
    # axis is also 0, `pp.scale_grad(rg.*_blip, step_size)` at assembly
    # time always scales by 0 regardless of this blip's own amplitude, so
    # only its duration (for matching against the other axis, below)
    # needs to come out right.
    gy_blip = pp.make_trapezoid('y', system=sys, area=max_ky_step * deltak[1])
    if max_ky_step != 0:
        gy_blip = pp.scale_grad(gy_blip, 1 / max_ky_step, sys)
    gy_blip = trap4ge(gy_blip, crt, sys)
    gz_blip = pp.make_trapezoid('z', system=sys, area=max_kz_step * deltak[2])
    if max_kz_step != 0:
        gz_blip = pp.scale_grad(gz_blip, 1 / max_kz_step, sys)
    gz_blip = trap4ge(gz_blip, crt, sys)

    # Match blip durations so they always fit within one readout block.
    if pp.calc_duration(gy_blip) > pp.calc_duration(gz_blip):  # y blip is longer
        max_blip_area = max_ky_step * deltak[1]
        blip_duration = pp.calc_duration(gy_blip)
        gz_blip = pp.make_trapezoid('z', system=sys, area=max_kz_step * deltak[2], duration=blip_duration)
        if max_kz_step != 0:
            gz_blip = pp.scale_grad(gz_blip, 1 / max_kz_step, sys)
        gz_blip = trap4ge(gz_blip, crt, sys)
    else:  # z blip is longer
        max_blip_area = max_kz_step * deltak[2]
        blip_duration = pp.calc_duration(gz_blip)
        gy_blip = pp.make_trapezoid('y', system=sys, area=max_ky_step * deltak[1], duration=blip_duration)
        if max_ky_step != 0:
            gy_blip = pp.scale_grad(gy_blip, 1 / max_ky_step, sys)
        gy_blip = trap4ge(gy_blip, crt, sys)

    # The circular shift below splits gro at t_s = blip_duration/2;
    # split_gradient_at rounds its split point to the gradient raster, so an
    # off-raster t_s would silently move the split and desync every
    # a1/a_d/t0 quantity computed from it.
    t_s = blip_duration / 2
    assert abs(t_s / crt - round(t_s / crt)) < 1e-9, (
        f'blip_duration ({blip_duration * 1e6:.1f} us) must be an even multiple '
        f'of crt ({crt * 1e6:.1f} us) so the split point lands on-raster'
    )

    # Readout trapezoid, asymmetric rise/fall (module docstring). Flat top
    # is capped at sys.max_grad, not always the exact-Nyquist rate
    # deltak[0]/dwell: if max_grad is below that rate, samples land denser
    # than deltak[0] (safe oversampling), and Tread grows to still reach the
    # same kmax -- the ramp-sample gridding recon already handles that.
    # Times are computed as crt multiples directly, NOT via trap4ge: its
    # area-preserving amplitude rescale would perturb the a1/a_d geometry
    # the pre-wind and echo timing are derived from.
    A = min(deltak[0] / dwell, sys.max_grad)
    r = max(_ceil_to_raster(A / slew_rise, crt), t_s)
    d = max(_ceil_to_raster(A / slew_fall, crt), t_s)
    # (r, d >= t_s also guarantees the split lands inside the rise ramp and
    # the fall spans the first half of the blip window -- the turnaround
    # geometry the composite construction relies on.)
    a1 = A * t_s**2 / (2 * r)
    a_d = A * t_s**2 / (2 * d)
    M = max(a1, a_d)

    # Smallest flat top whose area covers Nx*deltak plus the M lost off each
    # end of the ADC window (both parities must reach +-kmax).
    flat = _ceil_to_raster((Nx * deltak[0] + 2 * M - A * (r + d) / 2) / A, crt)
    assert flat >= 0, (
        'Readout lobe would be triangular (ramps alone exceed the required '
        'area) -- unsupported; would need A = sqrt(2*S/(1/slew_rise + 1/slew_fall)).'
    )

    # The ADC does not span exactly [0, Tread]: Nfid rounds Tread/dwell to a
    # multiple of 4 (either direction), and samples sit at (n + 0.5)*dwell.
    # Verify the realized first/last SAMPLES still cover +-(Nx/2 - 0.5)*deltak
    # in both parities, bumping the flat top a raster step at a time if not
    # (at most a couple of iterations -- the ceil above already leaves
    # sub-raster slack).
    k_need = (Nx / 2 - 0.5) * deltak[0]
    for _bump in range(10):
        S = A * ((r + d) / 2 + flat)
        D = r + flat + d
        Tread = D - blip_duration
        Nfid = round(Tread / dwell / 4) * 4  # multiple of 4 for GE
        segments = _composite_segments(A, r, flat, d, t_s)
        k_start = -S / 2 + a1  # odd-parity kx at composite-block start
        k_first = k_start + _integrate_to(segments, 0.5 * dwell)
        k_last = k_start + _integrate_to(segments, (Nfid - 0.5) * dwell)
        # Even parity samples the exact negation of these, so one check
        # covers both parities.
        if k_first <= -k_need and k_last >= k_need:
            break
        flat += crt
    else:
        raise AssertionError('flat-top coverage bump did not converge')

    gro = pp.make_trapezoid(
        'x',
        amplitude=A,
        rise_time=r,
        flat_time=flat,
        fall_time=d,
        system=sys,
        # The fall may exceed sys.max_slew (that's the point of POPE); the
        # tiny headroom factor covers the 1e-9-raster epsilon in
        # _ceil_to_raster, which can leave realized slew above slew_fall by
        # ~1e-11 relative.
        max_slew=slew_fall * (1 + 1e-9),
    )

    # In-block kx = 0 crossing time, identical for both parities thanks to
    # the -S/2 pre-wind (module docstring): integral_0^t0 g = S/2 - a1.
    t0 = _invert_integral(segments, S / 2 - a1)

    # Circularly shift gro so blips are contained within each Pulseq block:
    # the waveform is split and reassembled as [second half, -first half].
    sys_fast = copy.deepcopy(sys)
    sys_fast.max_slew = max(sys.max_slew, slew_fall * (1 + 1e-9))
    gro1, gro2 = pp.split_gradient_at(gro, blip_duration / 2, system=sys_fast)
    gro2.delay = 0
    gro1.delay = gro2.shape_dur
    gro = pp.add_gradients([gro2, pp.scale_grad(gro1, -1)], system=sys_fast)
    gro1.delay = 0  # leading piece played before the first echo line

    assert abs(pp.calc_duration(gro) - D) < 1e-9

    # ADC event covering the flat-top portion.
    adc = pp.make_adc(Nfid, dwell=dwell)

    # Delay blips to start at Tread, intended to be after the ADC window
    # closes. In practice `Nfid = round(Tread / dwell / 4) * 4` rounds to
    # the *nearest* multiple of 4, so when it rounds up the ADC window
    # (Nfid * dwell) extends up to 2 samples past Tread and the blip is
    # already ramping for those samples -- worst-case ky error is 0.17% of
    # one k-space step at this repo's default params (see
    # docs/review-findings.md item 63). Not fixed here (would need to floor
    # instead of round, which costs a slightly larger flat top for the same
    # k-space coverage); this comment now states what actually happens
    # rather than the intent.
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
        gx_pre_scale=S / (Nx * deltak[0]),
        echo_offset=pp.calc_duration(gro1) + t0,
    )
