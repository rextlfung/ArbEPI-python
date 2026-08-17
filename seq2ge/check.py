"""Pure-Python GE feasibility check: hardware limits (gradient/slew/RF
amplitude), PNS (ge/pns.py), and acoustic resonance (ge/acoustics.py) --
run directly on a pypulseq Sequence's real waveforms via `get_gradients()`,
with no MATLAB round trip.

PNS and hardware limits are checked over the *entire* sequence -- cheap
even for a full ~60s ArbEPI.seq (a few seconds, via FFT convolution).
Acoustics is checked over a bounded window, unlike PNS, since its FFT cost
scales with window length (this repo's default zero-fill factor makes it
O(n log n) on ~7x the sample count -- a full ~60s/15M-sample sequence takes
minutes for no benefit). The window is not an arbitrary fixed duration --
`_blockrange_window_s` reproduces the exact block-selection semantics of
MATLAB's own check (`pge2.plot(ceq, sys, 'blockRange', [1 10], ...)` inside
write_to_ge_from_seq.m: walk `ceq.loop` rows in segment order, starting at
row 1, including whole segments until the next segment's start row would be
>= `block_range[1]`), computed from this port's own `seq2ceq(seq)` (see
ge/seq2ceq.py) rather than a hardcoded time. The one difference from
MATLAB's window -- per-segment `segment_dead_time`/`segment_ringdown_time`
padding added by `pge2.getsegmentinstance.m` (~117us/segment, a
GE-Ceq-interpreter artifact with no Pulseq-timeline equivalent) -- is not
reproduced, and empirically accounts for the entire residual gap: computed
against real MATLAB output (`matlab_reference/dump_acoustics_blockrange.m`) on this
repo's four sequences, the window duration matches MATLAB's to within one
4us raster sample (GRE.seq: 173766 vs MATLAB's 173767 samples; ArbEPI.seq:
25000 vs 25001), and the resulting acoustics number matches to <0.04%
relative error (GRE.seq: 0.4024 here vs MATLAB's 0.402213; ArbEPI.seq:
0.028146 here vs MATLAB's 0.02814424) -- see CLAUDE.md for the full
reproduction record.
"""

from dataclasses import dataclass

import numpy as np
import pypulseq as pp

from scanners import ScannerSpec
from seq2ge.acoustics import THRESHOLD as ACOUSTICS_THRESHOLD
from seq2ge.acoustics import check_grad_acoustics
from seq2ge.ceq import Ceq
from seq2ge.pns import pns
from seq2ge.seq2ceq import seq2ceq

# Both thresholds are IEC 60601-2-33:2022's own operating-mode boundaries
# (see pge2.pns.m's header: wt = [0.8 1.0 0.7] "From IEC 60601-2-33:2022
# section (12)"), not values GE or this port derived: <=80% is Normal
# Operating Mode (ordinary clinical scanning), 80-100% is First Level
# Controlled Operating Mode (permitted, but calls for extra
# safeguards/operator awareness under the standard), >100% is predicted to
# actually stimulate.
#
# ../PulCeq/matlab/+pge2/checksegment.m throws at *either* line (`if
# max(pt) > 100 ... throw` then `if max(pt) > 80 ... throw`) -- GE's real
# scanner refuses to operate anywhere above Normal Operating Mode at all,
# a stricter, GE-specific policy layered on top of the standard. This port
# deliberately does not replicate the 80% throw: unlike the real scanner,
# there is no interlock consequence to a Python check running over a .seq
# file, so blocking `--ge` export at 80% only prevented inspecting/using
# sequences that are still within a standard-defined, permitted operating
# mode. `.ok` therefore only gates on the 100% line (genuinely predicted
# stimulation); 80-100% is surfaced as WARN via .summary(), same treatment
# as acoustics below. This does not change what MATLAB's real
# write_to_ge_from_seq.m path would do with the same sequence -- see
# CLAUDE.md's "Open finding" for why PNS in the 80-115% range across this
# repo's sequences is still a real, unresolved sequence-design problem to
# revisit before any human scan, warning-only or not.
PNS_NORMAL_MODE_THRESHOLD = 80.0
PNS_FIRST_CONTROLLED_MODE_THRESHOLD = 100.0


@dataclass
class FeasibilityReport:
    max_grad_mT_m: float
    max_slew_T_m_s: float
    max_b1_gauss: float
    peak_pns_percent: float
    acoustics_max_in_band: float
    spec_max_grad: float
    spec_max_slew: float
    spec_b1_max: float

    @property
    def ok(self) -> bool:
        # Acoustics is deliberately excluded: ../ArbEPI/lib/check_grad_acoustics.m
        # only ever calls MATLAB's `warning(...)` when magb>threshold (see its
        # line ~159) -- it has never blocked a real MATLAB `--ge` export, so
        # this port doesn't block on it either. Surfaced via .summary() as a
        # WARN, not folded into .ok.
        #
        # PNS only gates on the 100% ("first controlled mode") line, not 80%
        # ("normal mode") -- see the PNS_NORMAL_MODE_THRESHOLD comment above
        # for why this diverges from MATLAB's real 80% throw. 80-100% is
        # WARN via .summary(), same treatment as acoustics.
        return (
            self.max_grad_mT_m <= self.spec_max_grad
            and self.max_slew_T_m_s <= self.spec_max_slew
            and self.max_b1_gauss <= self.spec_b1_max
            and self.peak_pns_percent <= PNS_FIRST_CONTROLLED_MODE_THRESHOLD
        )

    def summary(self) -> str:
        pns_flag = (
            'FAIL' if self.peak_pns_percent > PNS_FIRST_CONTROLLED_MODE_THRESHOLD
            else 'WARN' if self.peak_pns_percent > PNS_NORMAL_MODE_THRESHOLD
            else 'OK  '
        )
        pns_tier = (
            f'exceeds first controlled mode ({PNS_FIRST_CONTROLLED_MODE_THRESHOLD:.0f}%)'
            if self.peak_pns_percent > PNS_FIRST_CONTROLLED_MODE_THRESHOLD
            else f'exceeds normal mode ({PNS_NORMAL_MODE_THRESHOLD:.0f}%)'
            if self.peak_pns_percent > PNS_NORMAL_MODE_THRESHOLD
            else 'within normal mode'
        )
        acoustics_flag = 'OK  ' if self.acoustics_max_in_band <= ACOUSTICS_THRESHOLD else 'WARN'
        lines = [
            f'{"OK  " if self.max_grad_mT_m <= self.spec_max_grad else "FAIL"} '
            f'max grad: {self.max_grad_mT_m:.2f} mT/m (limit {self.spec_max_grad:.2f})',
            f'{"OK  " if self.max_slew_T_m_s <= self.spec_max_slew else "FAIL"} '
            f'max slew: {self.max_slew_T_m_s:.1f} T/m/s (limit {self.spec_max_slew:.1f})',
            f'{"OK  " if self.max_b1_gauss <= self.spec_b1_max else "FAIL"} '
            f'max B1: {self.max_b1_gauss:.4f} G (limit {self.spec_b1_max:.4f})',
            f'{pns_flag} peak PNS: {self.peak_pns_percent:.1f}% ({pns_tier})',
            f'{acoustics_flag} acoustics: {self.acoustics_max_in_band:.4f} (limit {ACOUSTICS_THRESHOLD}'
            f'{", non-blocking -- MATLAB only warns on this" if acoustics_flag == "WARN" else ""})',
        ]
        return '\n'.join(lines)


def sample_gradients_tesla_per_m(
    seq: pp.Sequence, time_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Returns (gw, dt): gw shape (3, n) in T/m, uniformly sampled at the
    sequence's gradient raster time. Same sampling pattern as pypulseq's
    own Sequence/calc_pns.py."""
    dt = seq.grad_raster_time
    gw_pp = seq.get_gradients(time_range=list(time_range) if time_range else None)
    ends = [g.x[-1] for g in gw_pp if g is not None]
    t_start = time_range[0] if time_range else 0.0
    if time_range is not None:
        # Trust the caller's requested end directly -- max(ends) would
        # under-run it whenever the last block(s) in the window happen to
        # carry no gradient on any of the 3 channels (e.g. a trailing
        # TRID-label/delay block), silently truncating the sampled window.
        max_t = time_range[1] - 1e-10
    elif ends:
        max_t = max(ends) - 1e-10
    else:
        # No gradients anywhere in the sequence (e.g. noise.seq) -- keeps
        # the waveform array non-degenerate so diff()/convolution below
        # don't need their own empty-input special cases; the waveform
        # itself is correctly all-zero either way.
        max_t = seq.duration()[0]
    nt = max(int(np.ceil((max_t - t_start) / dt)), 2)
    t = t_start + (np.arange(nt) + 0.5) * dt

    gw_hz_per_m = np.zeros((3, nt))
    for i in range(3):
        if gw_pp[i] is not None:
            gw_hz_per_m[i] = gw_pp[i](t)

    return gw_hz_per_m / seq.system.gamma, dt


def _max_b1_gauss(seq: pp.Sequence, n_blocks: int) -> float:
    max_amp_hz = 0.0
    for n in range(1, n_blocks + 1):
        rf = seq.get_block(n).rf
        if rf is not None:
            max_amp_hz = max(max_amp_hz, float(np.abs(rf.signal).max()))
    return max_amp_hz / seq.system.gamma * 1e4  # Hz -> T -> Gauss


def _blockrange_window_s(ceq: Ceq, block_range: tuple[int, int] = (1, 10)) -> float:
    """Reproduce PulCeq's `pge2.plot.m` blockRange row-selection: process
    whole segments in `ceq.loop`-row order starting at row 1, continuing
    while the next segment's start row is still less than `block_range[1]`
    (see ../PulCeq/matlab/+pge2/plot.m's `while n < arg.blockRange(2)`
    loop). Returns the elapsed sequence time -- the sum of raw Pulseq block
    durations for exactly the rows MATLAB's blockRange would plot -- which
    corresponds directly to a `seq.get_gradients(time_range=(0, this))`
    window, since Pulseq blocks are contiguous with no gaps on the real
    timeline (unlike MATLAB's per-segment dead/ringdown padding, see this
    module's docstring)."""
    segments_by_id = {seg.ID: seg for seg in ceq.segments}
    n = 1
    total_s = 0.0
    while n < block_range[1]:
        seg = segments_by_id[int(ceq.loop[n - 1, 0])]
        n2 = n - 1 + seg.nBlocksInSegment
        if n < block_range[0]:
            n = n2 + 1
            continue
        total_s += float(ceq.loop[n - 1 : n2, 12].sum())
        n = n2 + 1
    return total_s


def check_ge_feasibility(
    seq: pp.Sequence,
    spec: ScannerSpec,
    pns_wt: tuple[float, float, float] = (1.0, 1.0, 1.0),
    acoustics_block_range: tuple[int, int] = (1, 10),
) -> FeasibilityReport:
    # Grad/slew/B1/PNS: cheap even over the whole sequence (a few seconds
    # for a full 60s/15M-sample ArbEPI.seq), and more thorough than
    # windowing since PNS in particular has no periodicity assumption to
    # lean on -- check everything.
    gw_tm, dt = sample_gradients_tesla_per_m(seq)

    max_grad_mT_m = float(np.abs(gw_tm).max()) * 1e3
    max_slew_T_m_s = float(np.abs(np.diff(gw_tm, axis=1) / dt).max())
    max_b1_gauss = _max_b1_gauss(seq, len(seq.block_events))

    s_min = spec.rheobase / spec.alpha
    pt, _ = pns(s_min, spec.chronaxie, gw_tm, dt, wt=pns_wt)
    peak_pns_percent = float(pt.max())

    # Acoustics: windowed to MATLAB's own blockRange, see
    # _blockrange_window_s and this module's docstring.
    ceq = seq2ceq(seq)
    window = min(_blockrange_window_s(ceq, acoustics_block_range), seq.duration()[0])
    gw_acoustics, dt_acoustics = sample_gradients_tesla_per_m(seq, time_range=(0.0, window))
    grad_for_acoustics = gw_acoustics.T.reshape(gw_acoustics.shape[1], 1, 3)
    acoustics = check_grad_acoustics(grad_for_acoustics, spec.ge_coil, dt_acoustics)

    return FeasibilityReport(
        max_grad_mT_m=max_grad_mT_m,
        max_slew_T_m_s=max_slew_T_m_s,
        max_b1_gauss=max_b1_gauss,
        peak_pns_percent=peak_pns_percent,
        acoustics_max_in_band=acoustics.max_in_band,
        spec_max_grad=spec.max_grad,
        spec_max_slew=spec.max_slew,
        spec_b1_max=spec.b1_max,
    )
