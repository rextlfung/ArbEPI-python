"""Ported from ../PulCeq/matlab/seq2ceq.m.

Converts a pypulseq Sequence into a Ceq struct (see ge/ceq.py): unique
parent blocks + a per-row loop table of dynamic scale/phase values, grouped
into segments by TRID labels. Rotation events are not supported (this
repo's sequences never use them; pypulseq's get_block() doesn't expose a
.rotation field), so the loop table's rotation matrix is always identity.

Row numbering n is 1-indexed (pypulseq's get_block convention); the loop
table row for block n is ceq.loop[n - 1].
"""

import warnings

import numpy as np
import pypulseq as pp

from ge.blocks import compare_blocks, get_block_type, get_dynamics
from ge.ceq import N_LOOP_COLUMNS, Ceq, ParentBlock, Segment

_INCONSISTENT_MSG = (
    'Sequence contains inconsistent segment definitions. This may occur due '
    'to programming error (possibly fatal), or if an arbitrary gradient '
    'resembles that from another block except with opposite sign or scaled '
    'by zero (which is probably ok). Often, a solution to this is to scale '
    'gradients to "eps" instead of identically zero, when calling '
    'pp.scale_grad().'
)


def seq2ceq(seq: pp.Sequence, verbose: bool = False) -> Ceq:
    ceq = Ceq()
    ceq.nMax = len(seq.block_events)

    # Pass 1: count ADC events, collect TRID labels and their rows
    trids = np.zeros(ceq.nMax + 1, dtype=int)  # index by row n; 0 = no label
    trid_rows = []  # rows carrying a TRID label, in stream order
    for n in range(1, ceq.nMax + 1):
        b = seq.get_block(n)
        if b.adc is not None:
            ceq.nReadouts += 1
        if b.label is not None:
            for lbl in b.label.values():
                if lbl.label == 'TRID':
                    trids[n] = int(lbl.value)
                    trid_rows.append(n)
                    break
    trid_rows = np.array(trid_rows)

    # Virtual segments: one per distinct TRID value, defined by its first
    # instance; repeated TRIDs are repeated instances of the same segment
    unique_trids, first_occurrence = np.unique(trids[trid_rows], return_index=True)
    n_blocks_per_instance = np.diff(np.append(trid_rows, ceq.nMax + 1))
    ceq.nSegments = len(unique_trids)
    seg_index = {}  # TRID value -> 0-based index into ceq.segments
    for i in range(ceq.nSegments):
        nb = int(n_blocks_per_instance[first_occurrence[i]])
        row0 = int(trid_rows[first_occurrence[i]])
        ceq.segments.append(Segment(
            ID=i + 1,
            TRID=int(unique_trids[i]),
            nBlocksInSegment=nb,
            blockIDs=np.zeros(nb, dtype=int),
            rows=row0 + np.arange(nb),
        ))
        seg_index[int(unique_trids[i])] = i

    # Detect variable delay blocks: a pure-delay block whose duration
    # differs across instances of its segment
    max_nb = max(seg.nBlocksInSegment for seg in ceq.segments)
    is_variable_delay = np.zeros((ceq.nSegments, max_nb), dtype=bool)
    block_duration = -np.ones((ceq.nSegments, max_nb))
    n = int(trid_rows[0])
    while n <= ceq.nMax:
        i = seg_index[trids[n]]
        for j in range(ceq.segments[i].nBlocksInSegment):
            b = seq.get_block(n)
            if block_duration[i, j] == -1:
                block_duration[i, j] = b.block_duration
            elif b.block_duration != block_duration[i, j]:
                if get_block_type(b).pure_delay:
                    is_variable_delay[i, j] = True
                else:
                    raise ValueError(
                        f'(row {n}: segment {i + 1}, block {j + 1}) Non-delay '
                        f'blocks must have the same duration in all segment '
                        f'instances'
                    )
            n += 1

    # Parent blocks, from the first instance of each segment. Pure delay
    # blocks get blockID 0 (static) or -1 (variable), not a parent block
    for seg in ceq.segments:
        for j in range(seg.nBlocksInSegment):
            n = int(seg.rows[j])
            b = seq.get_block(n)

            if get_block_type(b).pure_delay:
                seg.blockIDs[j] = -1 if is_variable_delay[seg.ID - 1, j] else 0
                continue

            for pb in ceq.parentBlocks:
                if compare_blocks(seq, n, pb.row):
                    seg.blockIDs[j] = pb.ID
                    break
            else:
                if verbose:
                    print(f'Found new parent block on line {n}')
                ceq.nParentBlocks += 1
                b.ID = ceq.nParentBlocks
                ceq.parentBlocks.append(ParentBlock(ID=ceq.nParentBlocks, row=n, block=b))
                seg.blockIDs[j] = ceq.nParentBlocks

    # Dynamic scan information (loop table)
    ceq.loop = np.zeros((ceq.nMax, N_LOOP_COLUMNS))
    n = int(trid_rows[0])
    while n <= ceq.nMax:
        if trids[n] == 0:  # not the start of a segment instance
            n += 1
            continue
        seg = ceq.segments[seg_index[trids[n]]]
        for j in range(seg.nBlocksInSegment):
            b = seq.get_block(n)
            p = int(seg.blockIDs[j])
            parent = ceq.parentBlocks[p - 1].block if p >= 1 else None
            ceq.loop[n - 1] = get_dynamics(
                b, seg.ID, p, get_block_type(b).has_trigger, parent
            )
            n += 1
        # seq2ceq.m writes this instance's rotation matrix into the last
        # block's loop row here; rotation events are unsupported in this
        # port, so the identity get_dynamics already wrote stands

    ceq.duration = seq.duration()[0]

    # Check that block execution throughout the sequence is consistent with
    # the segment definitions
    n = int(trid_rows[0])
    while n < ceq.nMax:
        seg_id = int(ceq.loop[n - 1, 0])
        assert seg_id >= 1, f'row {n}: block outside any segment instance'
        seg = ceq.segments[seg_id - 1]
        if n + seg.nBlocksInSegment > ceq.nMax:
            break
        for j in range(seg.nBlocksInSegment):
            p = int(ceq.loop[n - 1, 1])
            p_ij = int(seg.blockIDs[j])
            if p != p_ij:
                warnings.warn(
                    f'{_INCONSISTENT_MSG}\nExpected parent block ID {p_ij}, '
                    f'found {p} (block {n})'
                )
            n += 1

    # Gradient heating: find each segment's instance with max combined
    # energy. seq2ceq.m sums 1-based loop columns 11:13 (energy_gz, recphs,
    # blockDuration) -- stale indices from an older loop layout; the intent
    # is the three energy columns (deliberate deviation)
    n = int(trid_rows[0])
    while n < ceq.nMax:
        seg_id = int(ceq.loop[n - 1, 0])
        assert seg_id >= 1, f'row {n}: block outside any segment instance'
        seg = ceq.segments[seg_id - 1]
        n_first = n
        e_total = 0.0
        for j in range(seg.nBlocksInSegment):
            e_total += ceq.loop[n - 1, 6] + ceq.loop[n - 1, 8] + ceq.loop[n - 1, 10]
            n += 1
        if e_total > seg.Emax_val:
            seg.Emax_val = e_total
            seg.Emax_n = n_first

    return ceq
