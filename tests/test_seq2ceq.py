"""Self-consistency smoke test for ge/seq2ceq.py against real generated
sequences in output/ (run `uv run python main.py` first if missing).

noise.seq exercises pure-delay blocks and multi-instance segments;
ArbEPI.seq additionally exercises the two mechanisms the port hinges on:
scaled trap blips deduping to one parent block, and arbitrary-gradient
polarity detection (the alternating EPI readout). seq2ceq's own
segment-consistency check warns on a mis-ported polarity or shape
comparison, so the no-warnings assertion is the real test here.
"""

from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest

from ge.seq2ceq import seq2ceq

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'


@pytest.mark.parametrize('seq_name', ['noise.seq', 'ArbEPI.seq'])
def test_seq2ceq_self_consistency(seq_name, recwarn):
    seq_path = OUTPUT_DIR / seq_name
    if not seq_path.exists():
        pytest.skip(f'{seq_path} not found; run `uv run python main.py` first')

    seq = pp.Sequence()
    seq.read(str(seq_path))
    ceq = seq2ceq(seq)

    assert ceq.nMax == len(seq.block_events)
    assert 0 < ceq.nParentBlocks < ceq.nMax  # dedup actually happened
    assert ceq.nSegments == len(ceq.segments)
    assert len(ceq.parentBlocks) == ceq.nParentBlocks

    for seg in ceq.segments:
        assert len(seg.blockIDs) == seg.nBlocksInSegment
        for bid in seg.blockIDs:
            assert bid == -1 or 0 <= bid <= ceq.nParentBlocks

    assert ceq.loop.shape == (ceq.nMax, 23)
    seg_ids = ceq.loop[:, 0]
    assert np.all((seg_ids >= 0) & (seg_ids <= ceq.nSegments))

    n_adc = sum(
        seq.get_block(n).adc is not None for n in range(1, ceq.nMax + 1)
    )
    assert ceq.nReadouts == n_adc

    assert ceq.duration > 0

    inconsistent = [
        w for w in recwarn if 'inconsistent segment definitions' in str(w.message)
    ]
    assert not inconsistent
