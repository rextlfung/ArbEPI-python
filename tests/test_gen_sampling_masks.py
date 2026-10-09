from dataclasses import replace

import numpy as np

from params import load_params
from sample.gen_sampling_masks import gen_sampling_masks


def _small_caipi_params():
    p = load_params()
    return replace(
        p,
        Ny=12,
        Nz=8,
        Nframes=3,
        sampling_method='caipi',
    )


def test_gen_sampling_masks_caipi_shape_and_determinism():
    params = _small_caipi_params()
    omegas1 = gen_sampling_masks(4, params)
    omegas2 = gen_sampling_masks(4, params)
    assert omegas1.shape == (12, 8, 3)
    assert omegas1.dtype == bool
    # caipi is deterministic regardless of rng/seeding policy
    assert np.array_equal(omegas1, omegas2)


def test_gen_sampling_masks_ticaipi_frames_differ():
    params = replace(_small_caipi_params(), sampling_method='ticaipi')
    omegas = gen_sampling_masks(4, params)
    assert not np.array_equal(omegas[:, :, 0], omegas[:, :, 1])


def test_gen_sampling_masks_seed_per_frame_reproducible():
    params = replace(_small_caipi_params(), Ny=20, Nz=16, sampling_method='pd', pd_calib_frac=0.1)
    omegas1 = gen_sampling_masks(4, params, seed_per_frame=True)
    omegas2 = gen_sampling_masks(4, params, seed_per_frame=True)
    assert np.array_equal(omegas1, omegas2)


def test_gen_sampling_masks_unknown_method_raises():
    params = replace(_small_caipi_params(), sampling_method='bogus')
    try:
        gen_sampling_masks(4, params)
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_gen_sampling_masks_rand_default_sigma_shape_count_and_reproducible():
    # Item 187: the 'rand' dispatch (incl. the rand_gaussian_sigma=None
    # default of [Ny, Nz] / 6) had no test through gen_sampling_masks.
    params = replace(
        _small_caipi_params(), Ny=20, Nz=16, sampling_method='rand', rand_gaussian_sigma=None
    )
    R = 4
    omegas = gen_sampling_masks(R, params, seed_per_frame=True)
    assert omegas.shape == (20, 16, 3)
    assert omegas.dtype == bool
    n = 20 * 16 // R
    assert all(omegas[:, :, f].sum() == n for f in range(3))
    assert np.array_equal(omegas, gen_sampling_masks(R, params, seed_per_frame=True))
    # explicit sigma equal to the default gives the identical masks
    explicit = replace(params, rand_gaussian_sigma=np.array([20, 16]) / 6)
    assert np.array_equal(omegas, gen_sampling_masks(R, explicit, seed_per_frame=True))
