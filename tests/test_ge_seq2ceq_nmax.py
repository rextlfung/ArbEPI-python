"""ge/seq2ceq.py's final-segment-instance handling (review items 107, 122),
ge/writeceq.py's heating-check window (126) and ge/blocks.py's trigger
detection (137), all on small synthetic sequences."""

import numpy as np
import pypulseq as pp

from ge.blocks import get_block_type
from ge.ceq import Ceq, Segment
from ge.seq2ceq import seq2ceq
from ge.writeceq import NMAXBLOCKSFORGRADHEATCHECK, _heat_check_nblocks


def _seq_with_increasing_tr(blocks_per_tr: int, n_tr: int = 4) -> pp.Sequence:
    sys = pp.Opts()
    seq = pp.Sequence(system=sys)
    for i in range(n_tr):
        amp = (i + 1) * 1e3
        for j in range(blocks_per_tr):
            gx = pp.make_trapezoid('x', system=sys, amplitude=amp, flat_time=100e-6)
            if j == 0:
                seq.add_block(gx, pp.make_label(label='TRID', type='SET', value=1))
            else:
                seq.add_block(gx)
    return seq


def test_emax_n_reaches_last_multiblock_instance():
    # Energy grows each TR, so the worst instance is the last one (rows 7-8).
    ceq = seq2ceq(_seq_with_increasing_tr(blocks_per_tr=2))
    assert ceq.nMax == 8
    assert ceq.segments[0].Emax_n == 7


def test_emax_n_reaches_last_single_block_instance():
    # Item 122: a one-block segment whose last instance starts at row nMax.
    ceq = seq2ceq(_seq_with_increasing_tr(blocks_per_tr=1))
    assert ceq.nMax == 4
    assert ceq.segments[0].Emax_n == 4


def _ceq_with_segment(nb: int, n_instances: int) -> Ceq:
    ceq = Ceq()
    ceq.segments = [
        Segment(ID=1, TRID=1, nBlocksInSegment=nb, blockIDs=np.ones(nb), rows=np.arange(1, nb + 1))
    ]
    ceq.nMax = nb * n_instances
    ceq.loop = np.zeros((ceq.nMax, 23))
    ceq.loop[:, 0] = 1
    return ceq


def test_heat_check_window_includes_instance_ending_exactly_at_cap():
    # Item 126: nb divides the cap evenly -> the full cap, not one instance less.
    ceq = _ceq_with_segment(nb=40, n_instances=1005)
    assert ceq.nMax > NMAXBLOCKSFORGRADHEATCHECK
    assert _heat_check_nblocks(ceq) == NMAXBLOCKSFORGRADHEATCHECK


def test_heat_check_window_drops_instance_crossing_cap():
    ceq = _ceq_with_segment(nb=69, n_instances=700)
    n = _heat_check_nblocks(ceq)
    assert n == 69 * (NMAXBLOCKSFORGRADHEATCHECK // 69)
    assert n <= NMAXBLOCKSFORGRADHEATCHECK


def test_physio_trigger_block_is_detected():
    seq = pp.Sequence()
    seq.add_block(pp.make_trigger(channel='physio1', duration=100e-6))
    seq.add_block(pp.make_delay(100e-6))
    assert get_block_type(seq.get_block(1)).has_trigger
    assert not get_block_type(seq.get_block(2)).has_trigger
