import math

import numpy as np
import pytest

from sampling.pd_sample import _aniso_radii, _calib_rho, _rho_grid, pd_sample


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


def test_pd_sample_aniso_preserves_count_and_calibration_region():
    # aniso is a reshaping of the exclusion ellipse, not a density change --
    # the exact-count/calibration-region contracts must hold at any aniso.
    ny, nx = 120, 60
    accel = 6
    calib_frac = 0.1
    mask = pd_sample(
        [ny, nx], accel, np.random.default_rng(0), calib_frac=calib_frac,
        crop_corner=True, decay=1.4, aniso=2.5,
    )
    target_samples = math.floor(ny * nx / accel)
    assert mask.sum() == target_samples

    rho = _rho_grid(ny, nx)
    rho_calib = _calib_rho(target_samples, nx, ny, calib_frac)
    assert mask[rho <= rho_calib].all()


@pytest.mark.parametrize('aniso', [0.5, 1.0, 2.5])
def test_aniso_radii_reshape_ellipse_without_changing_area(aniso):
    # Away from the >= 1 floor, radius_y*aniso and radius_x/aniso should
    # leave the product (and hence 2D point density) unchanged relative to
    # the aniso=1 baseline, while shifting the ratio by aniso**2.
    # slope/r chosen large enough that radius_x stays above its >= 1 floor
    # at every aniso tested here (including shrinking it by 1/2.5) -- the
    # floor only engages near k-space center, see the dedicated floor test
    # below.
    ny, nx = 240, 45
    r = np.array([1.0])
    slope = 30.0
    decay = 1.4

    rx1, ry1 = _aniso_radii(r, slope, nx, ny, decay, aniso=1.0)
    rx_a, ry_a = _aniso_radii(r, slope, nx, ny, decay, aniso=aniso)

    assert rx1 * ry1 == pytest.approx(rx_a * ry_a)
    assert (ry_a / rx_a) == pytest.approx((ry1 / rx1) * aniso**2)


def test_aniso_radii_hits_floor_near_center_at_production_scale():
    # Near k-space center (small r, small slope), radius_x is already close
    # to its >= 1 floor at aniso=1 -- large aniso clamps it there rather
    # than shrinking further, so the area-preservation in the test above
    # does NOT hold in this region (documented in pd_sample's `aniso`
    # docstring: the achieved anisotropy is weaker near center than at the
    # periphery).
    ny, nx = 240, 45
    r = np.array([0.5])
    slope = 10.0
    decay = 1.4

    rx1, ry1 = _aniso_radii(r, slope, nx, ny, decay, aniso=1.0)
    rx_a, ry_a = _aniso_radii(r, slope, nx, ny, decay, aniso=2.5)

    assert rx1 > 1.0  # not floored at aniso=1
    assert rx_a == 1.0  # floored once aniso shrinks it below 1
    assert rx1 * ry1 < rx_a * ry_a  # so the product grows instead of holding


def test_aniso_radii_biases_ny_axis_at_production_scale():
    # At this repo's real (Ny, Nz) = (240, 45), aniso > 1 should grow the
    # ky-axis (ny) radius and shrink the kz-axis (nx) radius relative to
    # the isotropic-in-mm aniso=1 baseline.
    ny, nx = 240, 45
    r = np.array([0.5])
    slope = 10.0
    decay = 1.4

    rx1, ry1 = _aniso_radii(r, slope, nx, ny, decay, aniso=1.0)
    rx2, ry2 = _aniso_radii(r, slope, nx, ny, decay, aniso=2.0)

    assert ry2 > ry1
    assert rx2 < rx1
