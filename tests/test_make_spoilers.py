"""Item 142: make_spoilers' three channels share one duration -- the
minimum that fits the channel with the greatest area requirement."""

import pypulseq as pp
import pytest

from lib.make_spoilers import make_spoilers
from lib.readout_from_params import derated_sys
from params import load_params


@pytest.fixture(scope='module')
def setup():
    p = load_params()
    return p, derated_sys(p)


def _dur(g):
    return pp.calc_duration(g)


def test_isotropic_resolution_gives_equal_durations(setup):
    p, sys = setup
    gx, gy, gz = make_spoilers(p.res, [4.0] * 3, sys, p.crt)
    assert _dur(gx) == _dur(gy) == _dur(gz)


@pytest.mark.parametrize('res_scale', [(1, 1, 2), (1, 0.5, 1), (2, 1, 0.5)])
def test_anisotropic_resolution_shares_one_minimal_duration(setup, res_scale):
    p, sys = setup
    res = [p.res[0] * res_scale[0], p.res[1] * res_scale[1], p.res[2] * res_scale[2]]
    n_cyc = [4.0, 4.0, 4.0]
    gx, gy, gz = make_spoilers(res, n_cyc, sys, p.crt)
    durs = [_dur(g) for g in (gx, gy, gz)]
    assert max(durs) - min(durs) < 1e-12

    # Areas are as requested (cycles / res), whatever the duration.
    for g, r, n in zip((gx, gy, gz), res, n_cyc):
        assert g.area == pytest.approx(n / r, rel=1e-3)

    # The shared duration is the greatest-area channel's own minimum: it
    # matches that channel built alone (to within trap4ge's rounding), not a
    # longer one.
    ax = max(range(3), key=lambda i: n_cyc[i] / res[i])
    alone = make_spoilers(
        [res[ax]] * 3, [n_cyc[ax]] * 3, sys, p.crt
    )[ax]
    assert durs[0] == pytest.approx(_dur(alone), abs=2 * p.crt)
