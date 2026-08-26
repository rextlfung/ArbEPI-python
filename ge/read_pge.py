"""Read a .pge file back into plain Python structures -- the mirror image
of ge/writeceq.py, used only to validate write_ceq's output field-by-field
against a MATLAB-written reference .pge (see ge/validate_against_matlab.py).
Not used by the write path itself.
"""

import struct
from typing import BinaryIO


def _r(fid: BinaryIO, fmt: str):
    size = struct.calcsize('<' + fmt)
    data = fid.read(size)
    values = struct.unpack('<' + fmt, data)
    return values[0] if len(values) == 1 else values


def read_pge(fn: str) -> dict:
    with open(fn, 'rb') as fid:
        sentinel = _r(fid, 'h')
        assert sentinel == 47, f'bad endianness sentinel: {sentinel}'

        n_parent_blocks = _r(fid, 'h')
        parent_blocks = [_read_block(fid) for _ in range(n_parent_blocks)]

        n_segments = _r(fid, 'h')
        segments = [_read_segment(fid) for _ in range(n_segments)]

        n_max = _r(fid, 'i')
        n_cols = _r(fid, 'h')
        loop = [_r(fid, f'{n_cols}f') for _ in range(n_max)]

        max_rf_power = _r(fid, 'f')
        max_b1 = _r(fid, 'f')
        max_grad = _r(fid, 'f')
        max_slew = _r(fid, 'f')
        duration = _r(fid, 'f')
        n_readouts = _r(fid, 'i')
        pislquant = _r(fid, 'i')
        n_heat_check = _r(fid, 'i')

        trailing = fid.read()
        assert trailing == b'', f'{len(trailing)} unread trailing bytes'

    return dict(
        n_parent_blocks=n_parent_blocks, parent_blocks=parent_blocks,
        n_segments=n_segments, segments=segments,
        n_max=n_max, n_cols=n_cols, loop=loop,
        max_rf_power=max_rf_power, max_b1=max_b1, max_grad=max_grad,
        max_slew=max_slew, duration=duration, n_readouts=n_readouts,
        pislquant=pislquant, n_heat_check=n_heat_check,
    )


def _read_block(fid: BinaryIO) -> dict:
    return dict(
        ID=_r(fid, 'i'),
        block_duration=_r(fid, 'f'),
        rf=_read_rf(fid),
        gx=_read_grad(fid),
        gy=_read_grad(fid),
        gz=_read_grad(fid),
        adc=_read_adc(fid),
        trig=_read_trig(fid),
    )


def _read_rf(fid: BinaryIO):
    flag = _r(fid, 'h')
    if flag == 0:
        return None
    complexflag = _r(fid, 'h')
    n_samples, raster, time, magnitude, phase = _read_arbitrary(
        fid, complexflag=bool(complexflag), regular_raster=(flag == 1)
    )
    shape_dur = _r(fid, 'f')
    delay = _r(fid, 'f')
    energy = _r(fid, 'f')
    return dict(type=flag, n_samples=n_samples, raster=raster, time=time,
                magnitude=magnitude, phase=phase, shape_dur=shape_dur,
                delay=delay, energy=energy)


def _read_grad(fid: BinaryIO):
    flag = _r(fid, 'h')
    if flag == 0:
        return None
    if flag == 1:
        delay = _r(fid, 'f')
        rise_time = _r(fid, 'f')
        flat_time = _r(fid, 'f')
        fall_time = _r(fid, 'f')
        return dict(type='trap', delay=delay, rise_time=rise_time,
                    flat_time=flat_time, fall_time=fall_time)

    delay = _r(fid, 'f')
    n_samples = _r(fid, 'i')
    raster = _r(fid, 'f')
    if flag == 2:
        magnitude = _r(fid, f'{n_samples}f')
        return dict(type='raster', delay=delay, n_samples=n_samples,
                    raster=raster, magnitude=magnitude)
    else:  # flag == 3, corner points
        tt = _r(fid, f'{n_samples}f')
        magnitude = _r(fid, f'{n_samples}f')
        return dict(type='corner', delay=delay, n_samples=n_samples,
                    tt=tt, magnitude=magnitude)


def _read_adc(fid: BinaryIO):
    flag = _r(fid, 'h')
    if flag == 0:
        return None
    num_samples = _r(fid, 'i')
    dwell = _r(fid, 'f')
    delay = _r(fid, 'f')
    return dict(num_samples=num_samples, dwell=dwell, delay=delay)


def _read_trig(fid: BinaryIO):
    trig_type = _r(fid, 'h')
    if trig_type == 0:
        return dict(type=0)
    channel = _r(fid, 'i')
    delay = _r(fid, 'f')
    duration = _r(fid, 'f')
    return dict(type=trig_type, channel=channel, delay=delay, duration=duration)


def _read_segment(fid: BinaryIO) -> dict:
    seg_id = _r(fid, 'h')
    n_blocks = _r(fid, 'h')
    block_ids = _r(fid, f'{n_blocks}h')
    if n_blocks == 1:
        block_ids = (block_ids,)
    emax_n = _r(fid, 'i')
    has_grad = _r(fid, 'h')
    return dict(ID=seg_id, nBlocksInSegment=n_blocks, blockIDs=block_ids,
                Emax_n=emax_n, has_grad_events=bool(has_grad))


def _read_arbitrary(fid: BinaryIO, complexflag: bool, regular_raster: bool):
    n_samples = _r(fid, 'i')
    raster = _r(fid, 'f')
    time = None
    if not regular_raster:
        time = _r(fid, f'{n_samples}f')
    magnitude = _r(fid, f'{n_samples}f')
    phase = None
    if complexflag:
        phase = _r(fid, f'{n_samples}f')
    return n_samples, raster, time, magnitude, phase
