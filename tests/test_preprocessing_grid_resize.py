import numpy as np
import pytest

from preprocessing.grid_resize import resize_to_epi_grid


def test_resize_to_epi_grid_identity_when_grids_match():
    rng = np.random.default_rng(0)
    vol = rng.standard_normal((10, 10, 6))
    out = resize_to_epi_grid(vol, (0.2, 0.2, 0.12), (0.2, 0.2, 0.12), (10, 10, 6), order=1)
    np.testing.assert_allclose(out, vol)


def test_resize_to_epi_grid_crops_z_and_upsamples_xyz():
    # Mirrors this repo's real deGRE (108^3 @ 2mm) -> EPI (240x240x45 @
    # 0.9mm) geometry, scaled down: same x/y FOV, deGRE z-FOV larger than
    # EPI's (so a symmetric z-crop applies), coarser resolution everywhere.
    Nx_src, Ny_src, Nz_src = 12, 12, 6
    fov_src = (0.216, 0.216, 0.048)
    fov = (0.216, 0.216, 0.0405)
    n_target = (24, 24, 12)

    xx, _, _ = np.meshgrid(
        np.linspace(-1, 1, Nx_src), np.linspace(-1, 1, Ny_src), np.linspace(-1, 1, Nz_src),
        indexing='ij',
    )
    vol = xx  # a smooth ramp -- interpolation should reproduce it closely

    out = resize_to_epi_grid(vol, fov_src, fov, n_target, order=3)

    assert out.shape == n_target
    # A linear ramp along x should survive cubic-spline resize/crop closely
    # (crop is z-only, x is untouched in extent, just resampled).
    xx_target, _, _ = np.meshgrid(
        np.linspace(-1, 1, n_target[0]), np.linspace(-1, 1, n_target[1]),
        np.linspace(-1, 1, n_target[2]), indexing='ij',
    )
    np.testing.assert_allclose(out, xx_target, atol=0.05)


def test_resize_to_epi_grid_preserves_trailing_axes():
    rng = np.random.default_rng(0)
    vol = rng.standard_normal((8, 8, 8, 3)) + 1j * rng.standard_normal((8, 8, 8, 3))
    out = resize_to_epi_grid(vol, (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (16, 16, 16), order=3)
    assert out.shape == (16, 16, 16, 3)
    assert np.iscomplexobj(out)


def test_resize_to_epi_grid_nearest_order_keeps_boolean_values():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    out = resize_to_epi_grid(mask.astype(np.float64), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1),
                              (20, 20, 20), order=0)
    assert set(np.unique(out)) <= {0.0, 1.0}


def test_resize_to_epi_grid_rejects_epi_fov_larger_than_source():
    vol = np.zeros((8, 8, 8))
    with pytest.raises(ValueError):
        resize_to_epi_grid(vol, (0.1, 0.1, 0.1), (0.1, 0.1, 0.2), (4, 4, 4))
