"""Python port of `../PulCeq/matlab/+pge2/writeceq.m` -- serializes a `Ceq`
(see ge/ceq.py) to a GE `.pge` binary file.

Pure binary struct-packing, no algorithm: every MATLAB `fwrite(fid, x,
'type')` call below is a direct `struct.pack` translation, matching the
mirror-image C reader documented in ../PulCeq/src/pulCeq.h /
write_ceq.m's own inline comments.

Byte order: writeceq.m's "big endian (network byte order)" comment is
stale/wrong -- `fopen(fn, 'wb')` alone does not force endianness, and a
real .pge regenerated via the MATLAB path (output/ArbEPI.pge, 2026-08-11)
starts with bytes `2f 00`, i.e. the little-endian encoding of the int16
sentinel value 47 that the format writes first specifically so a reader
can detect endianness. This port always writes little-endian ('<').

Two deliberate deviations from literal MATLAB, both because a single
Python call does not have MATLAB's separate check()-then-writeceq() call
structure:
  - Skips DataHash-based staleness checking (see pge2.check.m/writeceq.m)
    -- nothing to go stale across separate calls here.
  - Skips the assertion that a fresh max-gradient scan over ceq.loop
    matches a `params.gmax` computed by a prior pge2.check() call -- that
    was the same kind of staleness guard, not a hardware-limit check
    (which is enforced independently, at .seq construction time, by
    `sys` being built from the same ScannerSpec -- see params.py).
    maxB1/maxGrad are still computed from ceq.loop and written to the
    header, since the .pge format requires them regardless.

The header's `maxSlew` field is the sequence's *realized* peak slew rate
(confirmed empirically against a real MATLAB-written .pge: a sequence with
no gradients writes maxSlew=0, not the hardware ceiling) -- writeceq.m
gets this from `params.smax`, itself computed by `pge2.check()` walking
every segment instance's actual (per-instance-scaled) waveforms. This
port computes the same thing directly from `ceq` instead of requiring a
`pge2.check()`-equivalent: each parent block's slew is amp/rise_time or
amp/fall_time for a trapezoid, or diff(waveform)/diff(tt) for an arbitrary
shape, in both cases scaled per loop-table row by that row's actual
instance amplitude (since blips share a parent block at unit shape but
vary in amplitude per instance -- see ge/blocks.py).
"""

import struct
from typing import BinaryIO

import numpy as np

from ge.ceq import Ceq, ParentBlock

NMAXBLOCKSFORGRADHEATCHECK = 40000


def _w(fid: BinaryIO, fmt: str, *values) -> None:
    fid.write(struct.pack('<' + fmt, *values))


def write_ceq(
    ceq: Ceq,
    fn: str,
    pislquant: int = 1,
) -> None:
    with open(fn, 'wb') as fid:
        _w(fid, 'h', 47)  # endianness-detection sentinel

        _w(fid, 'h', ceq.nParentBlocks)
        for pb in ceq.parentBlocks:
            _write_block(fid, pb)

        _w(fid, 'h', ceq.nSegments)
        parent_by_id = {pb.ID: pb for pb in ceq.parentBlocks}
        for seg in ceq.segments:
            _write_segment(fid, seg, parent_by_id)

        _w(fid, 'i', ceq.nMax)
        _w(fid, 'h', ceq.loop.shape[1])  # nColumnsInLoopArray
        for row in ceq.loop:
            _w(fid, f'{len(row)}f', *row)

        max_b1 = float(abs(ceq.loop[:, 2]).max()) if ceq.nMax else 0.0
        max_grad = float(abs(ceq.loop[:, [5, 7, 9]]).max()) if ceq.nMax else 0.0
        max_slew = _max_realized_slew(ceq)

        _w(fid, 'f', 1.0)  # maxRfPower, G^2*sec -- not used
        _w(fid, 'f', max_b1)
        _w(fid, 'f', max_grad)  # Hz/m
        _w(fid, 'f', max_slew)  # Hz/m/s
        _w(fid, 'f', ceq.duration)
        _w(fid, 'i', ceq.nReadouts)
        _w(fid, 'i', pislquant)

        # Number of blocks (rows) to use for the sliding-window gradient/RF
        # heating check -- last complete segment instance within the cap.
        segment_by_id = {s.ID: s for s in ceq.segments}
        n = 1
        while n < min(ceq.nMax, NMAXBLOCKSFORGRADHEATCHECK):
            seg = segment_by_id[int(ceq.loop[n - 1, 0])]
            n += seg.nBlocksInSegment
            if n > NMAXBLOCKSFORGRADHEATCHECK:
                n -= seg.nBlocksInSegment
                break
        _w(fid, 'i', n - 1)


def _max_realized_slew(ceq: Ceq) -> float:
    """Peak realized |slew| (Hz/m/s) across every segment instance, per
    axis, scaled by that instance's actual amplitude -- see module
    docstring for why this can't just echo the hardware slew limit."""
    amp_cols = {'gx': 5, 'gy': 7, 'gz': 9}

    # unit_slew[parent_id][axis] = peak slew per unit amplitude (1/s),
    # i.e. multiply by a loop row's actual instance amplitude to get that
    # instance's realized slew on that axis.
    unit_slew: dict[int, dict[str, float]] = {}
    for pb in ceq.parentBlocks:
        unit_slew[pb.ID] = {}
        for ax in ('gx', 'gy', 'gz'):
            g = getattr(pb.block, ax)
            if g is None:
                continue
            if g.type == 'trap':
                unit_slew[pb.ID][ax] = max(1 / g.rise_time, 1 / g.fall_time)
            else:
                peak = np.abs(g.waveform).max()
                slew_shape = np.diff(g.waveform) / np.diff(g.tt)
                unit_slew[pb.ID][ax] = float(np.abs(slew_shape).max() / peak)

    max_slew = 0.0
    for row in ceq.loop:
        p = int(row[1])
        if p < 1:
            continue
        per_axis = unit_slew[p]
        for ax, col in amp_cols.items():
            if ax not in per_axis:
                continue
            max_slew = max(max_slew, per_axis[ax] * abs(row[col]))
    return max_slew


def _write_block(fid: BinaryIO, pb: ParentBlock) -> None:
    b = pb.block
    _w(fid, 'i', pb.ID)
    _w(fid, 'f', b.block_duration)
    _write_rf(fid, b.rf)
    _write_grad(fid, b.gx)
    _write_grad(fid, b.gy)
    _write_grad(fid, b.gz)
    _write_adc(fid, b.adc)
    _write_trig(fid)


def _write_rf(fid: BinaryIO, rf) -> None:
    if rf is None:
        _w(fid, 'h', 0)
        return

    n_samples = len(rf.signal)
    magnitude = np.abs(rf.signal) / np.abs(rf.signal).max()
    phase = np.angle(rf.signal)
    raster = rf.t[1] - rf.t[0]  # sub_rf2shape.m computes this unconditionally

    if n_samples > 3 and _is_regular_raster(rf.t):
        # sampled on center of raster times
        _w(fid, 'h', 1)
        _w(fid, 'h', 1)  # complex flag
        _write_arbitrary(fid, n_samples, raster, rf.t, magnitude, phase,
                          complexflag=True, regular_raster=True)
    else:
        # corner points (extended trapezoid rf)
        _w(fid, 'h', 2)
        _w(fid, 'h', 1)
        _write_arbitrary(fid, n_samples, raster, rf.t, magnitude, phase,
                          complexflag=True, regular_raster=False)

    energy = float((magnitude[:-1] ** 2 * np.diff(rf.t)).sum())
    _w(fid, 'f', rf.shape_dur)
    _w(fid, 'f', rf.delay)
    _w(fid, 'f', energy)


def _write_grad(fid: BinaryIO, g) -> None:
    if g is None:
        _w(fid, 'h', 0)
        return

    if g.type == 'trap':
        _w(fid, 'h', 1)
        _w(fid, 'f', g.delay)
        _w(fid, 'f', g.rise_time)
        _w(fid, 'f', g.flat_time)
        _w(fid, 'f', g.fall_time)
        return

    n_samples = len(g.waveform)
    # signed, unlike RF's magnitude/phase decomposition -- sub_writegrad.m's
    # "magnitude" is actually the signed waveform normalized by its peak
    # abs value, not an abs(waveform); gradients have no separate phase
    # field to carry polarity, so the sign must stay here.
    magnitude = g.waveform / np.abs(g.waveform).max()

    if g.tt[0] > 0:
        # sampled on center of raster times
        raster = g.tt[1] - g.tt[0]
        _w(fid, 'h', 2)
        _w(fid, 'f', g.delay)
        _w(fid, 'i', n_samples)
        _w(fid, 'f', raster)
        _w(fid, f'{n_samples}f', *magnitude)
    else:
        # corner points
        _w(fid, 'h', 3)
        _w(fid, 'f', g.delay)
        _w(fid, 'i', n_samples)
        _w(fid, 'f', 0.0)  # dummy raster time, ignored by interpreter
        _w(fid, f'{n_samples}f', *g.tt)
        _w(fid, f'{n_samples}f', *magnitude)


def _write_adc(fid: BinaryIO, adc) -> None:
    if adc is None:
        _w(fid, 'h', 0)
        return
    _w(fid, 'h', 1)
    _w(fid, 'i', adc.num_samples)
    _w(fid, 'f', adc.dwell)
    _w(fid, 'f', adc.delay)


def _write_trig(fid: BinaryIO) -> None:
    # write_to_ge_from_seq.m's caller (writeceq.m's sub_writeblock) always
    # overwrites trig.type to 0 immediately before writing, regardless of
    # what the branch above it computed -- this repo's sequences never use
    # trigger blocks either, so type 0 (none) is written unconditionally,
    # matching the reference implementation's actual (if likely
    # unintentional) behavior byte-for-byte.
    _w(fid, 'h', 0)


def _write_segment(fid: BinaryIO, seg, parent_by_id: dict) -> None:
    _w(fid, 'h', seg.ID)
    _w(fid, 'h', seg.nBlocksInSegment)
    _w(fid, f'{seg.nBlocksInSegment}h', *[int(x) for x in seg.blockIDs])
    _w(fid, 'i', seg.Emax_n)

    has_grad_events = any(
        parent_by_id[p].block.gx is not None
        or parent_by_id[p].block.gy is not None
        or parent_by_id[p].block.gz is not None
        for p in seg.blockIDs if p > 0
    )
    _w(fid, 'h', 1 if has_grad_events else 0)


def _write_arbitrary(fid, n_samples, raster, t, magnitude, phase, complexflag, regular_raster) -> None:
    _w(fid, 'i', n_samples)
    _w(fid, 'f', raster)
    if not regular_raster:
        _w(fid, f'{n_samples}f', *t)
    _w(fid, f'{n_samples}f', *magnitude)
    if complexflag:
        _w(fid, f'{n_samples}f', *phase)


def _is_regular_raster(t) -> bool:
    d2 = np.diff(np.diff(t))
    return bool((np.abs(d2) < 10 * np.finfo(float).eps).all())
