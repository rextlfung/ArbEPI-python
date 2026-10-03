import numpy as np
import pytest

pytest.importorskip('sigpy')  # preprocess.preprocess imports epi_gridding (sigpy)

from preprocess.preprocess import find_calib_region  # noqa: E402
from sample.pd_sample import _calib_mask_rect, _calib_side_frac, pd_sample  # noqa: E402


def test_pd_masks_give_exactly_the_pd_calibration_rectangle():
    ny, nz, R, calib_frac, nframes = 60, 40, 6, 0.2, 12
    rng = np.random.default_rng(0)
    omegas = np.stack(
        [pd_sample([ny, nz], R, rng, calib_frac=calib_frac) for _ in range(nframes)], axis=-1
    ).astype(bool)
    target = int(np.floor(ny * nz / R))
    rect = _calib_mask_rect(ny, nz, _calib_side_frac(target, nz, ny, calib_frac))
    ys, zs = np.nonzero(rect)

    region = find_calib_region(omegas)
    assert region is not None
    y_sl, z_sl = region
    assert (y_sl.start, y_sl.stop) == (ys.min(), ys.max() + 1)
    assert (z_sl.start, z_sl.stop) == (zs.min(), zs.max() + 1)
    assert omegas[y_sl, z_sl].all()


def test_single_frame_region_is_fully_sampled_and_contains_the_center():
    rng = np.random.default_rng(1)
    omegas = pd_sample([48, 32], 4, rng, calib_frac=0.2)[..., None].astype(bool)
    y_sl, z_sl = find_calib_region(omegas)
    assert omegas[y_sl, z_sl, 0].all()
    assert y_sl.start <= 24 < y_sl.stop and z_sl.start <= 16 < z_sl.stop


def test_caipi_lattice_has_no_calibration_region():
    ny, nz = 40, 30
    omegas = np.zeros((ny, nz, 5), dtype=bool)
    omegas[::2, ::3, :] = True  # regular undersampling, center included
    omegas[ny // 2, nz // 2, :] = True
    assert find_calib_region(omegas) is None


def test_unsampled_center_gives_none():
    omegas = np.ones((20, 20, 3), dtype=bool)
    omegas[10, 10, 1] = False
    assert find_calib_region(omegas) is None
