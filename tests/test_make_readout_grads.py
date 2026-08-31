import pytest

from lib.readout_from_params import make_readout_grads_from_params
from params import load_params


def test_make_readout_grads_handles_etl_one():
    """mask2epi's ETL == 1 case returns (0.0, 0.0) for (max_ky_step,
    max_kz_step) -- 'no steps within a shot means no blips are needed'
    (lib/mask2epi.py). make_readout_grads used to normalize each blip to
    unit amplitude via `1 / max_*_step`, which raised ZeroDivisionError
    here instead of honoring that documented guardrail."""
    rg = make_readout_grads_from_params(0.0, 0.0, load_params())
    assert rg.Nfid > 0


def test_make_readout_grads_handles_one_axis_zero():
    """A mask with no steps on a single axis (e.g. Nz == 1) zeroes only
    that axis's max step -- must not crash, and the surviving axis's
    unit-step blip must still carry its real (nonzero) area: a unit ky
    step is 1 * deltak[1] of area (the blip is normalized to unit
    amplitude, i.e. area-per-step)."""
    p = load_params()
    deltak_y, deltak_z = 1 / p.fov[1], 1 / p.fov[2]

    rg_kz_zero = make_readout_grads_from_params(5.0, 0.0, p)
    assert rg_kz_zero.gy_blip.area == pytest.approx(deltak_y, rel=1e-6)
    assert rg_kz_zero.gz_blip.area == pytest.approx(0.0, abs=1e-12)

    rg_ky_zero = make_readout_grads_from_params(0.0, 5.0, p)
    assert rg_ky_zero.gy_blip.area == pytest.approx(0.0, abs=1e-12)
    assert rg_ky_zero.gz_blip.area == pytest.approx(deltak_z, rel=1e-6)
