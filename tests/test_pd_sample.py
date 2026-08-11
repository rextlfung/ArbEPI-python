import math

import numpy as np
import pytest

from sampling.pd_sample import pd_sample


def test_pd_sample_exact_count():
    rng = np.random.default_rng(0)
    ny, nx = 40, 30
    accel = 4
    mask = pd_sample([ny, nx], accel, rng, calib=[6, 4], crop_corner=True, decay=1.4)
    assert mask.shape == (ny, nx)
    assert mask.sum() == math.floor(ny * nx / accel)


def test_pd_sample_calibration_region_fully_sampled():
    rng = np.random.default_rng(1)
    ny, nx = 50, 40
    calib = [8, 6]
    mask = pd_sample([ny, nx], 5, rng, calib=calib, crop_corner=True, decay=1.0)

    y0 = max(0, math.floor(ny / 2 - calib[0] / 2))
    y1 = min(ny, math.floor(ny / 2 + calib[0] / 2))
    x0 = max(0, math.floor(nx / 2 - calib[1] / 2))
    x1 = min(nx, math.floor(nx / 2 + calib[1] / 2))
    assert mask[y0:y1, x0:x1].all()


def test_pd_sample_density_falls_off_from_center():
    rng = np.random.default_rng(2)
    ny, nx = 60, 60
    mask = pd_sample([ny, nx], 6, rng, calib=[0, 0], crop_corner=True, decay=1.4)

    cy, cx = ny // 2, nx // 2
    inner = mask[cy - 10 : cy + 10, cx - 10 : cx + 10].mean()
    outer_ring = mask.mean()  # whole-image density is necessarily lower than center
    assert inner > outer_ring


@pytest.mark.parametrize('accel', [0.5, 1.0])
def test_pd_sample_rejects_invalid_accel(accel):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        pd_sample([20, 20], accel, rng)
