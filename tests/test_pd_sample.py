import math

import numpy as np
import pytest

from sampling.pd_sample import _calib_rho, _rho_grid, pd_sample


def test_pd_sample_exact_count():
    rng = np.random.default_rng(0)
    ny, nx = 40, 30
    accel = 4
    mask = pd_sample([ny, nx], accel, rng, calib_frac=0.2, crop_corner=True, decay=1.4)
    assert mask.shape == (ny, nx)
    assert mask.sum() == math.floor(ny * nx / accel)


def test_pd_sample_calibration_region_fully_sampled():
    rng = np.random.default_rng(1)
    ny, nx = 50, 40
    accel = 5
    calib_frac = 0.2
    mask = pd_sample([ny, nx], accel, rng, calib_frac=calib_frac, crop_corner=True, decay=1.0)

    target_samples = math.floor(ny * nx / accel)
    rho = _rho_grid(ny, nx)
    rho_calib = _calib_rho(target_samples, nx, ny, calib_frac)
    assert mask[rho <= rho_calib].all()


def test_pd_sample_density_falls_off_from_center():
    rng = np.random.default_rng(2)
    ny, nx = 60, 60
    mask = pd_sample([ny, nx], 6, rng, calib_frac=0.0, crop_corner=True, decay=1.4)

    cy, cx = ny // 2, nx // 2
    inner = mask[cy - 10 : cy + 10, cx - 10 : cx + 10].mean()
    outer_ring = mask.mean()  # whole-image density is necessarily lower than center
    assert inner > outer_ring


@pytest.mark.parametrize('accel', [0.5, 1.0])
def test_pd_sample_rejects_invalid_accel(accel):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        pd_sample([20, 20], accel, rng)
