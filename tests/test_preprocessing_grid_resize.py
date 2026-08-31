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
    # (crop is z-only, x is untouched in extent, just resampled). Outermost
    # 2 x-voxels excluded: linspace(-1, 1, N) anchors index 0/N-1 at the
    # array's own pixel *centers* -- the grid_mode=False convention this
    # function no longer uses (see its module docstring, item 12) -- so the
    # edge voxels disagree with this ramp's own endpoint value by
    # construction under grid_mode=True (an upsample-past-the-edge
    # extrapolation artifact, see
    # test_resize_to_epi_grid_matches_analytic_ramp_at_voxel_centers for a
    # precise characterization); interior voxels are unaffected and still
    # match closely, which is what this test is actually checking.
    xx_target, _, _ = np.meshgrid(
        np.linspace(-1, 1, n_target[0]), np.linspace(-1, 1, n_target[1]),
        np.linspace(-1, 1, n_target[2]), indexing='ij',
    )
    np.testing.assert_allclose(out[2:-2], xx_target[2:-2], atol=0.05)


def test_resize_to_epi_grid_matches_analytic_ramp_at_voxel_centers():
    """Item 12 regression: the z-crop above already assumes N voxels tile
    the FOV edge-to-edge (voxel i spans [i/N, (i+1)/N) * FOV, no
    pixel-center offset); the resize step must use that same convention
    (grid_mode=True) rather than scipy's default pixel-center alignment, or
    the two steps disagree about where a voxel physically sits. Checked at
    this repo's real deGRE (108 @ 2mm) -> EPI (240 @ 0.9mm) x-axis scale, no
    z-crop involved, so this isolates the resize step's own alignment.
    """

    def center(i, n, fov_m):
        return (i + 0.5) / n * fov_m - fov_m / 2

    Nx_src, Nx_tgt, fov_x = 108, 240, 0.216
    xs = center(np.arange(Nx_src), Nx_src, fov_x)
    vol = np.broadcast_to(xs[:, None, None], (Nx_src, 4, 4)).copy()

    out = resize_to_epi_grid(vol, (fov_x, fov_x, fov_x), (fov_x, fov_x, fov_x),
                              (Nx_tgt, 4, 4), order=3)

    xt = center(np.arange(Nx_tgt), Nx_tgt, fov_x)
    expected = np.broadcast_to(xt[:, None, None], (Nx_tgt, 4, 4))

    err = np.abs(out - expected)
    # Interior voxels (3+ in from each edge): near-exact, < 0.05mm on a
    # 216mm FOV (measured max 0.041mm).
    assert err[3:-3].max() < 5e-5
    # Outermost few voxels: unavoidable upsample-past-the-edge
    # extrapolation (measured max 0.39mm) -- still an order of magnitude
    # tighter than the grid_mode=False convention this replaces (measured
    # ~0.63mm max / 0.27mm mean error at this exact scale before the fix).
    assert err.max() < 5e-4


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
