"""Self-consistency smoke test for seq2ge/check.py (seq2ge/pns.py + seq2ge/acoustics.py
wired together) against real generated sequences in output/ (run
`uv run python main.py` first if missing).

Numerical correctness of the underlying algorithms is validated against
real MATLAB output separately (seq2ge/validate_pns.py, seq2ge/acoustics.py's
module docstring, and CLAUDE.md's PNS finding) -- this only checks the
pipeline runs and returns sane values on real sequences, since that
cross-check needs a local MATLAB install this suite can't assume.
"""

from pathlib import Path

import pypulseq as pp
import pytest

from scanners import SCANNERS
from seq2ge.check import PNS_NORMAL_MODE_THRESHOLD, FeasibilityReport, check_ge_feasibility

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'


def _clean_report(**overrides) -> FeasibilityReport:
    fields = dict(
        max_grad_mT_m=0.0, max_slew_T_m_s=0.0, max_b1_gauss=0.0,
        peak_pns_percent=0.0, acoustics_max_in_band=0.0,
        spec_max_grad=100.0, spec_max_slew=200.0, spec_b1_max=0.25,
    )
    fields.update(overrides)
    return FeasibilityReport(**fields)


def test_acoustics_over_threshold_does_not_block_ok():
    # ../ArbEPI/lib/check_grad_acoustics.m only ever calls MATLAB's
    # warning(...) when over threshold, never error(...) -- it has never
    # blocked a real MATLAB --ge export, so .ok must not gate on it either
    # (see CLAUDE.md's "Real behavioral bug caught while wiring this in").
    report = _clean_report(acoustics_max_in_band=0.9)
    assert report.ok
    assert 'WARN' in report.summary()


def test_pns_over_threshold_blocks_ok():
    # ../PulCeq/matlab/+pge2/checksegment.m really does throw above 80%.
    report = _clean_report(peak_pns_percent=PNS_NORMAL_MODE_THRESHOLD + 1)
    assert not report.ok
    assert 'FAIL' in report.summary()


@pytest.mark.parametrize('seq_name', ['noise.seq', 'ArbEPI.seq'])
def test_check_ge_feasibility_runs(seq_name):
    seq_path = OUTPUT_DIR / seq_name
    if not seq_path.exists():
        pytest.skip(f'{seq_path} not found; run `uv run python main.py` first')

    seq = pp.Sequence()
    seq.read(str(seq_path))
    report = check_ge_feasibility(seq, SCANNERS['GE_UHP'])

    assert report.max_grad_mT_m >= 0
    assert report.max_slew_T_m_s >= 0
    assert report.max_b1_gauss >= 0
    assert report.peak_pns_percent >= 0
    assert report.acoustics_max_in_band >= 0


def test_check_ge_feasibility_noise_has_no_gradients():
    seq_path = OUTPUT_DIR / 'noise.seq'
    if not seq_path.exists():
        pytest.skip(f'{seq_path} not found; run `uv run python main.py` first')

    seq = pp.Sequence()
    seq.read(str(seq_path))
    report = check_ge_feasibility(seq, SCANNERS['GE_UHP'])

    assert report.max_grad_mT_m == 0
    assert report.max_slew_T_m_s == 0
    assert report.peak_pns_percent == 0
