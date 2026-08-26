"""Ported from ../PulCeq/matlab/getblocktype.m, compareblocks.m, getdynamics.m.

Per-block helpers for seq2ceq: classify a block, test two blocks for
"sameness" (parent-block dedup), and extract one row of the dynamic loop
table. Block/row numbering is 1-indexed throughout (see ge/ceq.py).

Gradient "sameness" is deliberately about shape/timing only, never
amplitude -- that's what lets differently-scaled blips (pp.scale_grad on a
unit-amplitude blip) dedup to a single parent block. The per-instance
amplitude is read separately in get_dynamics.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pypulseq as pp


@dataclass
class BlockType:
    no_events: bool  # no rf/gradient/adc events at all
    has_trid: bool
    has_trigger: bool  # physio1 trigger
    pure_delay: bool  # getblocktype.m's T(4) = T(1) (its docstring is stale)


def get_block_type(block: SimpleNamespace) -> BlockType:
    no_events = (
        block.rf is None and block.adc is None
        and block.gx is None and block.gy is None and block.gz is None
    )

    has_trid = False
    if block.label is not None:
        has_trid = any(lbl.label == 'TRID' for lbl in block.label.values())

    trig = getattr(block, 'trig', None)
    has_trigger = trig is not None and trig.channel == 'physio1'

    return BlockType(no_events, has_trid, has_trigger, pure_delay=no_events)


def compare_blocks(seq: pp.Sequence, n1: int, n2: int) -> bool:
    """True if blocks n1 and n2 are instances of the same parent block."""
    b1 = seq.get_block(n1)
    b2 = seq.get_block(n2)

    if abs(b1.block_duration - b2.block_duration) > 1e-6:
        return False

    for ax in ('gx', 'gy', 'gz'):
        if not _compare_gradients(getattr(b1, ax), getattr(b2, ax)):
            return False

    if not _compare_adc(b1.adc, b2.adc):
        return False

    return _compare_rf(seq, b1, b2, n1, n2)


def _compare_rf(seq: pp.Sequence, b1, b2, n1: int, n2: int) -> bool:
    if b1.rf is None and b2.rf is None:
        return True
    if (b1.rf is None) != (b2.rf is None):
        return False

    # Equal iff mag and phase shape IDs match. seq.block_events rows are
    # [dur, rf, gx, gy, gz, adc, ext]; rf_library data tuples are
    # (amplitude, mag_shape_id, phase_shape_id, time_shape_id, ...) --
    # both confirmed against pypulseq's register_rf_event/read code.
    ev1 = seq.rf_library.data[seq.block_events[n1][1]]
    ev2 = seq.rf_library.data[seq.block_events[n2][1]]
    return ev1[1] == ev2[1] and ev1[2] == ev2[2]


def _compare_gradients(g1, g2) -> bool:
    if g1 is None and g2 is None:
        return True
    if (g1 is None) != (g2 is None):
        return False
    if g1.type != g2.type:
        return False

    if g1.type == 'trap':
        # timing only, never amplitude (scaled blips share a parent)
        return (
            g1.rise_time == g2.rise_time
            and g1.flat_time == g2.flat_time
            and g1.fall_time == g2.fall_time
            and g1.delay == g2.delay
        )
    return g1.shape_id == g2.shape_id


def _compare_adc(adc1, adc2) -> bool:
    if adc1 is None and adc2 is None:
        return True
    if (adc1 is None) != (adc2 is None):
        return False
    return (
        adc1.num_samples == adc2.num_samples
        and adc1.dwell == adc2.dwell
        and adc1.delay == adc2.delay
    )


def get_dynamics(
    block: SimpleNamespace,
    segment_id: int,
    parent_block_id: int,
    physio_trigger: bool,
    parent_block: SimpleNamespace | None,
) -> np.ndarray:
    """One row of ceq.loop (see ge/ceq.py's LOOP_COLUMNS), in Pulseq units.

    Gradient energies are (Hz/m)^2*sec. The rotation matrix is always
    identity: Pulseq rotation events are not supported by this port (this
    repo's sequences never use them, and pypulseq's get_block() doesn't
    expose a .rotation field).
    """
    rfamp = 0.0
    rfphs = 0.0
    rffreq = 0.0
    amp = {'gx': 0.0, 'gy': 0.0, 'gz': 0.0}
    energy = {'gx': 0.0, 'gy': 0.0, 'gz': 0.0}
    recphs = 0.0

    if block.rf is not None:
        assert parent_block.rf is not None, (
            f'(virtual segment {segment_id}) Expected RF event not found '
            f'in base block {parent_block_id}'
        )
        rfamp = np.max(np.abs(block.rf.signal))
        rfphs = block.rf.phase_offset
        rffreq = block.rf.freq_offset

    for ax in ('gx', 'gy', 'gz'):
        g = getattr(block, ax)
        if g is None:
            continue
        assert getattr(parent_block, ax) is not None, (
            f'(virtual segment {segment_id}) Expected {ax} event not found '
            f'in base block {parent_block_id}'
        )
        if g.type == 'trap':
            amp[ax] = g.amplitude
            energy[ax] = (
                g.amplitude ** 2 / 3 * g.rise_time
                + g.amplitude ** 2 * g.flat_time
                + g.amplitude ** 2 / 3 * g.fall_time
            )
        else:  # arbitrary gradient or extended trapezoid
            energy[ax] = np.sum(g.waveform[:-1] ** 2 * np.diff(g.tt))
            amp[ax] = np.max(np.abs(g.waveform))
            # Polarity vs the normalized parent shape (the shape loaded into
            # the interpreter's waveform memory). getdynamics.m tests
            # `if w2 .* g.waveform < 1` -- an elementwise all() that only
            # reliably detects sign flips for large-amplitude waveforms and
            # misfires for eps-scaled gradients; a dot-product sign test is
            # the robust equivalent of the intent (deliberate deviation).
            w1 = getattr(parent_block, ax).waveform
            w2 = w1 / np.max(np.abs(w1))
            if np.dot(w2, g.waveform) < 0:
                amp[ax] = -amp[ax]

    if block.adc is not None:
        assert parent_block.adc is not None, (
            f'(virtual segment {segment_id}) Expected ADC event not found '
            f'in base block {parent_block_id}'
        )
        recphs = block.adc.phase_offset
        rffreq = block.adc.freq_offset  # save ADC frequency for ADC events

    return np.concatenate((
        [segment_id, parent_block_id,
         rfamp, rfphs, rffreq,
         amp['gx'], energy['gx'],
         amp['gy'], energy['gy'],
         amp['gz'], energy['gz'],
         recphs,
         block.block_duration,
         float(physio_trigger)],
        np.eye(3).ravel(),
    ))
