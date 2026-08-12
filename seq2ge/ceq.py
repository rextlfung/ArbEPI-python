"""Python port of PulCeq's `Ceq` struct.

See ../PulCeq/matlab/seq2ceq.m (builds it), ../PulCeq/matlab/+pge2/writeceq.m
(serializes it to a .pge binary file), and ../PulCeq/src/pulCeq.h (the
struct's C-side description -- note pulCeq.h's `loop` comment is stale
relative to what getdynamics.m/seq2ceq.m actually build; LOOP_COLUMNS below
is the ground truth, derived directly from getdynamics.m's return value).

This is the shared data contract between ge/seq2ceq.py (builds a Ceq from a
pypulseq Sequence) and ge/writeceq.py (serializes a Ceq to a .pge file), so
it's defined once, up front, for both to depend on.

Block/row numbering throughout is 1-indexed, matching both MATLAB Pulseq
and pypulseq's own `Sequence.get_block(n)` convention (n starts at 1) --
no index-base translation needed here, unlike mask2epi.py's deliberate
0-based departure from MATLAB (see CLAUDE.md).
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

# ceq.loop column layout (0-indexed here; getdynamics.m's return order).
# amp.g* is signed peak amplitude (Hz/m); energy.g* is (Hz/m)^2*sec.
LOOP_COLUMNS = [
    'segmentID', 'parentBlockID',
    'rfamp', 'rfphs', 'rffreq',
    'amp_gx', 'energy_gx',
    'amp_gy', 'energy_gy',
    'amp_gz', 'energy_gz',
    'recphs',
    'blockDuration',
    'physioTrigger',
    'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22',
]
N_LOOP_COLUMNS = len(LOOP_COLUMNS)  # 23
assert N_LOOP_COLUMNS == 23


@dataclass
class ParentBlock:
    ID: int
    row: int  # 1-indexed row where this parent block was first encountered
    block: SimpleNamespace  # pypulseq get_block() result (block.ID set to ID)


@dataclass
class Segment:
    ID: int
    TRID: int
    nBlocksInSegment: int
    # length nBlocksInSegment; 0 = static delay, -1 = variable delay, >0 = parent block ID
    blockIDs: np.ndarray
    rows: np.ndarray  # 1-indexed rows of this (first) segment instance
    Emax_val: float = 0.0
    Emax_n: int = 1  # row index of the segment instance with max gradient energy


@dataclass
class Ceq:
    nParentBlocks: int = 0
    parentBlocks: list[ParentBlock] = field(default_factory=list)
    nSegments: int = 0
    segments: list[Segment] = field(default_factory=list)
    loop: np.ndarray = None  # shape (nMax, N_LOOP_COLUMNS), float64
    nMax: int = 0
    nReadouts: int = 0
    duration: float = 0.0
