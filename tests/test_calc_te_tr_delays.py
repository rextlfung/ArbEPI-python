"""Item 144: calc_te_tr_delays warns (never raises) and falls back to a
zero padding delay when the prescribed TE/TR is unachievable."""

import math
import warnings

import pytest

from lib.calc_te_tr_delays import calc_te_tr_delays
from lib.make_excitation_pulse import make_excitation_from_params
from lib.make_prephasers import make_prephasers
from lib.make_spoilers import make_spoilers
from lib.readout_from_params import derated_sys, make_readout_grads_from_params
from params import load_params


@pytest.fixture(scope='module')
def parts():
    p = load_params()
    sys = derated_sys(p)
    rf, gz_ss, gz_ssr, rfsat = make_excitation_from_params(p, sys)
    rg = make_readout_grads_from_params(2, 2, p)
    gx_pre, gy_pre, gz_pre = make_prephasers(p.Nx, p.Ny, p.Nz, p.fov, sys, p.crt)
    gx_s, gy_s, gz_s = make_spoilers(p.res, [p.spoil_cycles_max] * 3, sys, p.crt)
    return p, sys, (rf, rfsat, gz_ss, gz_ssr, gx_pre, gy_pre, gz_pre, rg, gx_s, gy_s, gz_s)


def _call(parts, TE, TR):
    p, sys, (rf, rfsat, gz_ss, gz_ssr, gx_pre, gy_pre, gz_pre, rg, gx_s, gy_s, gz_s) = parts
    return calc_te_tr_delays(
        rf, rfsat, gz_ss, gz_ssr, gx_pre, gy_pre, gz_pre, rg.gro, gx_s, gy_s, gz_s,
        p.ETL, TE, TR, sys, echo_offset=rg.echo_offset,
    )


def test_achievable_te_tr_does_not_warn_and_pads_to_raster(parts):
    sys = parts[1]
    _, _, min_te, min_tr = _call(parts, 1.0, 10.0)  # probe minima (warns; ignore)
    TE, TR = min_te + 1e-3, min_tr + 1e-3 + 2e-3
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        te_delay, tr_delay, _, _ = _call(parts, TE, TR)
    r = sys.block_duration_raster
    assert 0 < te_delay <= 1e-3 and 0 < tr_delay
    assert te_delay / r == pytest.approx(round(te_delay / r))
    assert math.isclose(min_te + te_delay, TE, abs_tol=r)


def test_unachievable_te_warns_and_falls_back_to_zero_delay(parts):
    with pytest.warns(UserWarning, match='Minimum achievable TE'):
        te_delay, _, min_te, _ = _call(parts, 1e-3, 10.0)
    assert te_delay == 0.0
    assert min_te > 1e-3


def test_unachievable_tr_warns_and_falls_back_to_zero_delay(parts):
    _, _, min_te, _ = _call(parts, 1.0, 10.0)
    with pytest.warns(UserWarning, match='Minimum achievable TR'):
        _, tr_delay, _, min_tr = _call(parts, min_te + 1e-3, 1e-3)
    assert tr_delay == 0.0
    assert min_tr > 1e-3
