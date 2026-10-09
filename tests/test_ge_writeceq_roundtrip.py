"""write_ceq -> read_pge round trip (docs/review-findings.md item 134).

Real noise.seq / ArbEPI.seq (built_seq_dir) check header scalars, parent
blocks, segments and the loop table survive serialization; a synthetic Ceq
checks the heat-check block count where a segment's block count divides
NMAXBLOCKSFORGRADHEATCHECK evenly (item 126).
"""

from types import SimpleNamespace

import numpy as np
import pypulseq as pp
import pytest

from ge.ceq import N_LOOP_COLUMNS, Ceq, ParentBlock, Segment
from ge.read_pge import read_pge
from ge.seq2ceq import seq2ceq
from ge.writeceq import NMAXBLOCKSFORGRADHEATCHECK, write_ceq


@pytest.mark.parametrize('seq_name', ['noise.seq', 'ArbEPI.seq'])
def test_write_read_roundtrip_real_sequences(built_seq_dir, tmp_path, seq_name):
    seq = pp.Sequence()
    seq.read(str(built_seq_dir / seq_name))
    ceq = seq2ceq(seq)
    fn = str(tmp_path / 'x.pge')
    write_ceq(ceq, fn, pislquant=7)
    d = read_pge(fn)

    assert d['n_parent_blocks'] == ceq.nParentBlocks
    assert [b['ID'] for b in d['parent_blocks']] == [p.ID for p in ceq.parentBlocks]
    for b, pb in zip(d['parent_blocks'], ceq.parentBlocks):
        assert b['block_duration'] == pytest.approx(pb.block.block_duration, rel=1e-6)
        assert (b['adc'] is None) == (pb.block.adc is None)
        for ax in ('gx', 'gy', 'gz'):
            assert (b[ax] is None) == (getattr(pb.block, ax) is None)
        assert (b['rf'] is None) == (pb.block.rf is None)
        if pb.block.adc is not None:
            assert b['adc']['num_samples'] == int(pb.block.adc.num_samples)

    assert d['n_segments'] == ceq.nSegments
    for s, cs in zip(d['segments'], ceq.segments):
        assert s['ID'] == cs.ID
        assert s['nBlocksInSegment'] == cs.nBlocksInSegment
        assert list(s['blockIDs']) == [int(x) for x in cs.blockIDs]
        assert s['Emax_n'] == cs.Emax_n

    assert d['n_max'] == ceq.nMax
    assert d['n_cols'] == N_LOOP_COLUMNS
    np.testing.assert_array_equal(np.array(d['loop']), ceq.loop.astype(np.float32))
    assert d['n_readouts'] == ceq.nReadouts
    assert d['pislquant'] == 7
    assert d['duration'] == pytest.approx(ceq.duration, rel=1e-6)
    assert d['max_b1'] == pytest.approx(np.abs(ceq.loop[:, 2]).max(), rel=1e-6)
    assert d['max_grad'] == pytest.approx(np.abs(ceq.loop[:, [5, 7, 9]]).max(), rel=1e-6)
    assert 0 < d['n_heat_check'] <= ceq.nMax


def _delay_ceq(n_blocks_per_seg, n_instances):
    pb = ParentBlock(ID=1, row=1, block=SimpleNamespace(
        block_duration=1e-3, rf=None, gx=None, gy=None, gz=None, adc=None))
    seg = Segment(ID=1, TRID=1, nBlocksInSegment=n_blocks_per_seg,
                  blockIDs=np.ones(n_blocks_per_seg, dtype=int),
                  rows=np.arange(1, n_blocks_per_seg + 1))
    n = n_blocks_per_seg * n_instances
    loop = np.zeros((n, N_LOOP_COLUMNS))
    loop[:, 0] = 1
    loop[:, 1] = 1
    return Ceq(nParentBlocks=1, parentBlocks=[pb], nSegments=1, segments=[seg],
               loop=loop, nMax=n, nReadouts=0, duration=n * 1e-3)


@pytest.mark.parametrize('blocks_per_seg', [40, 64, 7])
def test_heat_check_count_is_whole_segments_within_cap(tmp_path, blocks_per_seg):
    # 40 divides NMAXBLOCKSFORGRADHEATCHECK evenly (item 126's repro).
    n_inst = 2 * (NMAXBLOCKSFORGRADHEATCHECK // blocks_per_seg) + 3
    ceq = _delay_ceq(blocks_per_seg, n_inst)
    fn = str(tmp_path / 'syn.pge')
    write_ceq(ceq, fn)
    d = read_pge(fn)
    n_hc = d['n_heat_check']
    assert n_hc <= NMAXBLOCKSFORGRADHEATCHECK
    assert n_hc % blocks_per_seg == 0
    assert n_hc == (NMAXBLOCKSFORGRADHEATCHECK // blocks_per_seg) * blocks_per_seg
    assert d['max_slew'] == 0.0
