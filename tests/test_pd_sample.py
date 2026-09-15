import math

import numpy as np
import pytest

from sampling.pd_sample import _calib_mask_rect, _calib_side_frac, pd_sample


def test_pd_sample_exact_count():
    rng = np.random.default_rng(0)
    ny, nx = 40, 30
    accel = 4
    mask = pd_sample([ny, nx], accel, rng, calib_frac=0.2, crop_corner=True, decay=1.4)
    assert mask.shape == (ny, nx)
    assert mask.sum() == math.floor(ny * nx / accel)


def test_calib_mask_rect_matches_worked_example():
    # Pins the spec from the feature request: ny/2=5 stands in for ky_max=5,
    # nx/2=2 for kz_max=2 (same per-axis pixel-index normalization used
    # throughout this module), calib_frac=0.2 -> |ky| <= 1, |kz| <= 0.4.
    # At this resolution (one pixel = kz step of 1), a 0.4-wide kz half-width
    # only reaches the single center column, not its neighbors.
    mask = _calib_mask_rect(10, 4, 0.2)
    ys, xs = np.nonzero(mask)
    assert sorted(set(ys.tolist())) == [4, 5, 6]
    assert sorted(set(xs.tolist())) == [2]
    assert mask.sum() == 3


def test_calib_mask_rect_area_matches_calib_frac_squared():
    ny, nx = 200, 200
    calib_frac = 0.3
    mask = _calib_mask_rect(ny, nx, calib_frac)
    assert abs(mask.mean() - calib_frac**2) < 0.01


def test_calib_mask_rect_zero_frac_is_empty():
    assert not _calib_mask_rect(20, 16, 0.0).any()


def test_pd_sample_calibration_region_fully_sampled():
    rng = np.random.default_rng(1)
    ny, nx = 50, 40
    accel = 5
    calib_frac = 0.2
    mask = pd_sample([ny, nx], accel, rng, calib_frac=calib_frac, crop_corner=True, decay=1.0)

    target_samples = math.floor(ny * nx / accel)
    side_frac = _calib_side_frac(target_samples, nx, ny, calib_frac)
    calib_mask = _calib_mask_rect(ny, nx, side_frac)
    assert mask[calib_mask].all()


def test_calib_side_frac_scales_with_sample_budget_not_grid():
    """The whole point of _calib_side_frac (docs/review-findings.md item
    195): the calibration region should hold a constant *fraction of the
    sample budget* across acceleration factors, not a fixed fraction of
    k-space -- so side_frac, and hence the realized calib pixel count,
    must shrink as accel grows (target_samples shrinks) at a fixed
    calib_frac, unlike a plain calib_frac-as-side_frac mapping which
    would stay constant regardless of accel."""
    ny, nx = 270, 180
    calib_frac = 0.1

    small_budget = math.floor(ny * nx / 6)  # low accel -- large budget
    large_budget = math.floor(ny * nx / 94)  # high accel -- small budget, as in the 0.8mm config

    side_frac_low_accel = _calib_side_frac(small_budget, nx, ny, calib_frac)
    side_frac_high_accel = _calib_side_frac(large_budget, nx, ny, calib_frac)
    assert side_frac_high_accel < side_frac_low_accel

    calib_pixels_high_accel = int(_calib_mask_rect(ny, nx, side_frac_high_accel).sum())
    # The calibration region should be a roughly constant share of each
    # budget (allowing for pixel-grid rounding), not close to consuming
    # the entire (much smaller) high-accel budget the way a fixed-kmax-
    # fraction region did (487 of 520 samples at the real 0.8mm config).
    assert calib_pixels_high_accel < 0.3 * large_budget


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
