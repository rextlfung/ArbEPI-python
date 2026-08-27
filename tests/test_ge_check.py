"""Self-consistency smoke test for ge/check.py (ge/pns.py + ge/acoustics.py
wired together) against real generated sequences in output/ (run
`uv run python main.py` first if missing).

Numerical correctness of the underlying algorithms is validated against
real MATLAB output separately (ge/validate_pns.py, ge/acoustics.py's
module docstring, and CLAUDE.md's PNS finding) -- this only checks the
pipeline runs and returns sane values on real sequences, since that
cross-check needs a local MATLAB install this suite can't assume.
"""

from pathlib import Path

import pypulseq as pp
import pytest

from ge.check import (
    PNS_FIRST_CONTROLLED_MODE_THRESHOLD,
    PNS_NORMAL_MODE_THRESHOLD,
    FeasibilityReport,
    check_ge_feasibility,
)
from scanners import SCANNERS

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


def test_pns_between_80_and_100_warns_but_does_not_block_ok():
    # ../PulCeq/matlab/+pge2/checksegment.m throws above 80% ("normal
    # mode"), but that's a GE-specific policy stricter than IEC
    # 60601-2-33:2022 itself (80-100% is "first level controlled operating
    # mode" -- permitted, not prohibited). This port only blocks .ok at the
    # 100% line; 80-100% is WARN (see ge/check.py's threshold comment).
    report = _clean_report(peak_pns_percent=PNS_NORMAL_MODE_THRESHOLD + 1)
    assert report.ok
    assert 'WARN' in report.summary()


def test_pns_over_100_blocks_ok():
    # >100% ("first controlled mode") is predicted to actually stimulate --
    # this line still hard-blocks .ok.
    report = _clean_report(peak_pns_percent=PNS_FIRST_CONTROLLED_MODE_THRESHOLD + 1)
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


def test_arbepi_default_params_peak_pns_under_normal_mode_limit(tmp_path):
    """Safety regression bound: a full-dims, 1-frame ArbEPI built with the
    default params (POPE asymmetric readout ramps + tuned blip slew, see
    params.py's slew section) must stay under GE's 80% normal-mode PNS
    limit. This is the check that was impossible before the POPE readout:
    the old symmetric-derate design measured ~84% (see CLAUDE.md's PNS
    section), and PNSwt = 0 before that disabled the check entirely."""
    import warnings
    from dataclasses import replace

    import numpy as np

    from ge.check import sample_gradients_tesla_per_m
    from ge.pns import pns
    from params import load_params
    from sampling.gen_sampling_masks import gen_sampling_masks
    from sequences.ArbEPI import generate_arbepi

    p = replace(load_params(output_dir=str(tmp_path)), Nframes=1, seed=0)
    omegas = gen_sampling_masks(p.R, p, rng=np.random.default_rng(p.seed))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        seq = generate_arbepi(omegas, p, seqname='pnscheck')

    gw, dt = sample_gradients_tesla_per_m(seq)
    pt, _ = pns(p.spec.rheobase / p.spec.alpha, p.spec.chronaxie, gw, dt, tuple(p.PNSwt))
    peak = float(np.max(pt))
    assert peak < PNS_NORMAL_MODE_THRESHOLD, (
        f'peak PNS {peak:.1f}% exceeds the {PNS_NORMAL_MODE_THRESHOLD}% normal-mode '
        f'limit -- the default readout/blip slews are no longer PNS-safe'
    )
