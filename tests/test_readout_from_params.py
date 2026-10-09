"""Item 198: lib/readout_from_params.py's acoustic-resonance-band avoidance."""

import pypulseq as pp
import pytest

import lib.readout_from_params as rfp
from lib.make_readout_grads import InfeasibleDwellError, make_readout_grads
from lib.readout_from_params import (
    ACOUSTIC_MARGIN_US,
    _echo_spacing_in_forbidden_band,
    blip_sys,
    find_min_feasible_dwell,
)
from params import load_params


def test_forbidden_band_edges_include_margin():
    # 'xrm' bands: x/y (410, 510), z (360, 440) -> union spans 360..510;
    # the margin extends it on both sides, inclusive at the edge.
    m = ACOUSTIC_MARGIN_US
    f = lambda us: _echo_spacing_in_forbidden_band(us, 'xrm')  # noqa: E731
    assert f(460)
    assert f(360 - m) and not f(360 - m - 0.5)
    assert f(510 + m) and not f(510 + m + 0.5)
    assert not f(300) and not f(600)


def test_forbidden_band_checks_union_of_all_axes():
    # 'xrm' z band (360, 440) reaches below x/y's (410, 510): 365 is only in z's.
    assert _echo_spacing_in_forbidden_band(365.0, 'xrm')
    # 'xrmw' has only a z band (330, 460): x-axis list is empty
    assert _echo_spacing_in_forbidden_band(400.0, 'xrmw')
    assert not _echo_spacing_in_forbidden_band(330 - ACOUSTIC_MARGIN_US - 1, 'xrmw')
    assert not _echo_spacing_in_forbidden_band(460 + ACOUSTIC_MARGIN_US + 1, 'xrmw')


def test_forbidden_band_coil_name_is_case_insensitive():
    assert _echo_spacing_in_forbidden_band(460.0, 'XRM')


def test_find_min_feasible_dwell_steps_past_a_forbidden_band(monkeypatch):
    p = load_params()
    dwell0 = find_min_feasible_dwell(2, 2, p)

    gamma = p.sys.gamma

    def echo_us(dwell):
        rg = make_readout_grads(
            2, 2, p.Nx, p.fov, dwell, blip_sys(p), p.crt,
            slew_rise=p.ro_slew_rise * gamma, slew_fall=p.ro_slew_fall * gamma,
        )
        return pp.calc_duration(rg.gro) * 1e6

    e0 = echo_us(dwell0)
    # Forbid a band exactly around the unconstrained choice's echo spacing.
    coil = p.spec.ge_coil.lower()
    monkeypatch.setitem(rfp._ESP_BANDS_US, coil, [[(e0 - 1.0, e0 + 1.0, 0.0)], [], []])
    dwell1 = find_min_feasible_dwell(2, 2, p)
    assert dwell1 > dwell0
    e1 = echo_us(dwell1)
    assert not (e0 - 1.0 - ACOUSTIC_MARGIN_US <= e1 <= e0 + 1.0 + ACOUSTIC_MARGIN_US)


def test_find_min_feasible_dwell_raises_when_everything_is_forbidden(monkeypatch):
    p = load_params()
    coil = p.spec.ge_coil.lower()
    monkeypatch.setitem(rfp._ESP_BANDS_US, coil, [[(0.0, 1e6, 0.0)], [], []])
    with pytest.raises(RuntimeError, match='No feasible ADC dwell'):
        find_min_feasible_dwell(2, 2, p, max_multiple=5)


def test_find_min_feasible_dwell_skips_only_dwell_dependent_failures(monkeypatch):
    """Review item 197: an InfeasibleDwellError (dwell-dependent geometry) moves
    on to the next dwell; any other AssertionError (dwell-independent, e.g.
    the blip/raster mismatch) propagates instead of being reported as 'no
    feasible dwell'."""
    p = load_params()
    real = rfp.make_readout_grads
    calls = []

    def flaky(*a, **k):
        calls.append(a[4])
        if len(calls) <= 2:
            raise InfeasibleDwellError('triangular')
        return real(*a, **k)

    monkeypatch.setattr(rfp, 'make_readout_grads', flaky)
    dwell = find_min_feasible_dwell(2, 2, p)
    assert len(calls) >= 3 and dwell == calls[-1]

    def broken(*a, **k):
        raise AssertionError('blip_duration must be an even multiple of crt')

    monkeypatch.setattr(rfp, 'make_readout_grads', broken)
    with pytest.raises(AssertionError, match='even multiple'):
        find_min_feasible_dwell(2, 2, p)
