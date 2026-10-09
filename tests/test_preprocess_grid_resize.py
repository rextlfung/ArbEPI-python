import numpy as np
import pytest

from preprocess.grid_resize import resize_to_epi_grid


def _centers(n, fov_m):
    """Centered-FFT voxel positions: voxel n // 2 is isocenter (item 263)."""
    return (np.arange(n) - n // 2) / n * fov_m


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

    xs = _centers(Nx_src, fov_src[0]) / fov_src[0]
    vol = np.broadcast_to(xs[:, None, None], (Nx_src, Ny_src, Nz_src)).copy()  # smooth ramp

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
    xt = _centers(n_target[0], fov[0]) / fov[0]
    expected = np.broadcast_to(xt[:, None, None], n_target)
    np.testing.assert_allclose(out[2:-2], expected[2:-2], atol=0.05)


def test_resize_to_epi_grid_matches_analytic_ramp_at_voxel_centers():
    """Item 12 regression: the z-crop above already assumes N voxels tile
    the FOV edge-to-edge (voxel i spans [i/N, (i+1)/N) * FOV, no
    pixel-center offset); the resize step must use that same convention
    (grid_mode=True) rather than scipy's default pixel-center alignment, or
    the two steps disagree about where a voxel physically sits. Checked at
    this repo's real deGRE (108 @ 2mm) -> EPI (240 @ 0.9mm) x-axis scale, no
    z-crop involved, so this isolates the resize step's own alignment.
    """

    Nx_src, Nx_tgt, fov_x = 108, 240, 0.216
    xs = _centers(Nx_src, fov_x)
    vol = np.broadcast_to(xs[:, None, None], (Nx_src, 4, 4)).copy()

    out = resize_to_epi_grid(vol, (fov_x, fov_x, fov_x), (fov_x, fov_x, fov_x),
                              (Nx_tgt, 4, 4), order=3)

    xt = _centers(Nx_tgt, fov_x)
    expected = np.broadcast_to(xt[:, None, None], (Nx_tgt, 4, 4))

    err = np.abs(out - expected)
    # Interior voxels (3+ in from each edge): near-exact, < 0.05mm on a
    # 216mm FOV (measured max 0.041mm).
    assert err[3:-3].max() < 1e-4
    # Outermost few voxels: unavoidable upsample-past-the-edge
    # extrapolation (measured max 0.39mm) -- still an order of magnitude
    # tighter than the grid_mode=False convention this replaces (measured
    # ~0.63mm max / 0.27mm mean error at this exact scale before the fix).
    assert err.max() < 1e-3  # measured 0.95 mm at the one outermost voxel the even-N source does not cover


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


@pytest.mark.parametrize('fov', [(0.1, 0.1, 0.2), (0.2, 0.1, 0.1), (0.1, 0.2, 0.1)])
def test_resize_to_epi_grid_rejects_epi_fov_larger_than_source(fov):
    vol = np.zeros((8, 8, 8))
    with pytest.raises(ValueError):
        resize_to_epi_grid(vol, (0.1, 0.1, 0.1), fov, (4, 4, 4))


def test_resize_to_epi_grid_crops_every_axis_to_the_physical_epi_grid():
    """deGRE FOV larger than the EPI's on all three axes, by amounts that are
    not whole deGRE voxels: a 3 mm deGRE covering 219 x 225 x 153 mm onto the
    default 2.4 mm 90 x 90 x 60 EPI grid (216 x 216 x 144 mm). A linear
    function of physical position must land at each EPI voxel's physical
    center."""
    n_src, fov_src = (73, 75, 51), (0.219, 0.225, 0.153)
    n_tgt, fov = (90, 90, 60), (0.216, 0.216, 0.144)
    xs, ys, zs = (_centers(n, f) for n, f in zip(n_src, fov_src))
    vol = xs[:, None, None] + 2 * ys[None, :, None] - 3 * zs[None, None, :]

    # Linear interpolation, so the check isolates the coordinate mapping: it
    # reproduces a linear function exactly wherever the target voxel center
    # lies inside the source's outermost voxel centers, which holds here.
    # (A cubic spline adds ~0.07 mm of edge ringing in the outermost couple
    # of voxels, the same boundary effect the ramp test above bounds.)
    out = resize_to_epi_grid(vol, fov_src, fov, n_tgt, order=1)

    xt, yt, zt = (_centers(n, f) for n, f in zip(n_tgt, fov))
    expected = xt[:, None, None] + 2 * yt[None, :, None] - 3 * zt[None, None, :]
    assert np.abs(out - expected).max() < 1e-12


def test_resize_to_epi_grid_z_crop_is_not_rounded_to_whole_voxels():
    """Review item 258: the z crop used to keep whole deGRE voxels, so a FOV
    difference of 1.5 voxels per side (153 vs 144 mm at 3 mm) shifted the
    maps by half a voxel (1.5 mm). Now each EPI slice samples its own
    physical position."""
    zs = _centers(51, 0.153)
    vol = np.broadcast_to(zs[None, None, :], (4, 4, 51)).copy()
    out = resize_to_epi_grid(vol, (0.1, 0.1, 0.153), (0.1, 0.1, 0.144), (4, 4, 60), order=1)
    np.testing.assert_allclose(out[0, 0], _centers(60, 0.144), atol=1e-12)


def test_resize_to_epi_grid_zero_pad_z_fills_outer_slices_with_zero():
    """docs/review-findings.md item 196: when the target z-FOV exceeds the
    source's, zero_pad_z=True must zero-fill the outer target-grid slices
    the source has no real coverage for, rather than raising (the
    zero_pad_z=False default's behavior, still covered by the test above).
    """
    Nx_src, Ny_src, Nz_src = 8, 8, 20
    fov_src = (0.1, 0.1, 0.1)  # 10cm z, dz=5mm
    # Target z-FOV 12cm -> 1cm (2 x 5mm dz) of unreal coverage total, 1
    # target slice zeroed per side at Nz=20 (dz_target=6mm; 0.5cm/side /
    # 6mm rounds up to 1 whole slice zeroed per side, matching the
    # conservative ceil/floor policy).
    fov = (0.1, 0.1, 0.12)
    n_target = (Nx_src, Ny_src, 20)

    vol = np.full((Nx_src, Ny_src, Nz_src), 3.0)
    out = resize_to_epi_grid(vol, fov_src, fov, n_target, order=1, zero_pad_z=True)

    assert out.shape == n_target
    np.testing.assert_array_equal(out[:, :, 0], 0.0)
    np.testing.assert_array_equal(out[:, :, -1], 0.0)
    # Interior: real data, close to the constant source value (order=1
    # linear interpolation of a constant field reproduces it exactly away
    # from the zero boundary).
    np.testing.assert_allclose(out[:, :, 2:-2], 3.0, atol=1e-9)


def test_resize_to_epi_grid_zero_pad_z_matches_real_5p4mm_config():
    """Pins the exact inner/outer split for this session's real case:
    deGRE's fixed 144mm z-FOV (72 @ 2mm) vs. the 5.4mm-resolution EPI
    variant's own 145.8mm z-FOV (27 @ 5.4mm) -- 1 slice zeroed (the top one) out
    of 27, matching the value used to build that dataset's smaps cache."""
    fov_gre = (0.216, 0.216, 0.144)
    fov_epi = (0.216, 0.216, 0.1458)
    n_target = (40, 40, 27)

    vol = np.ones((108, 108, 72))
    out = resize_to_epi_grid(vol, fov_gre, fov_epi, n_target, order=3, zero_pad_z=True)

    assert out.shape == n_target
    zero_slices = [z for z in range(27) if not np.any(out[:, :, z])]
    # the even-N source slab is asymmetric about isocenter ([-73, +71] mm),
    # so only the top slice (to +72.9 mm) lacks coverage (item 263)
    assert zero_slices == [26]


def test_resize_to_epi_grid_zero_pad_z_places_inner_slices_at_their_physical_positions():
    """Review item 258: the zero_pad_z path used to zoom the whole source slab
    onto the inner target slices (144 mm onto 25 x 5.4 = 135 mm at the real
    5.4 mm config), compressing the maps 6% in z -- up to 4.5 mm off at the
    slab edges. Each covered slice must sample its own physical position."""
    zs = _centers(72, 0.144)
    vol = np.broadcast_to(zs[None, None, :], (4, 4, 72)).copy()
    out = resize_to_epi_grid(vol, (0.1, 0.1, 0.144), (0.1, 0.1, 0.1458), (4, 4, 27),
                             order=1, zero_pad_z=True)
    np.testing.assert_allclose(out[0, 0, :-1], _centers(27, 0.1458)[:-1], atol=1e-12)
    np.testing.assert_array_equal(out[:, :, -1], 0.0)


def test_resize_to_epi_grid_puts_voxel_n_over_2_at_isocenter_on_both_grids():
    """Review item 263: a point at a given physical position must land at
    index N // 2 + position / d on each grid, for even and odd N (a 72 x 72 x
    51 deGRE onto a 90 x 90 x 60 EPI, the default protocol)."""
    n_src, fov_src = (72, 72, 51), (0.216, 0.216, 0.153)
    n_tgt, fov = (90, 90, 60), (0.216, 0.216, 0.144)
    pos = np.array([10.0, -7.0, 5.0]) * 1e-3
    grids = [_centers(n, f) for n, f in zip(n_src, fov_src)]
    # smooth bump centered at pos (Gaussian, sigma 8 mm)
    r2 = sum(((g - p) ** 2)[tuple(slice(None) if i == a else None for i in range(3))]
             for a, (g, p) in enumerate(zip(grids, pos)))
    vol = np.exp(-r2 / (2 * 8e-3**2))
    out = resize_to_epi_grid(vol, fov_src, fov, n_tgt, order=3)
    idx = np.indices(n_tgt)
    com = np.array([(idx[a] * out**2).sum() / (out**2).sum() for a in range(3)])
    expected = np.array(n_tgt) // 2 + pos / (np.array(fov) / n_tgt)
    off_mm = (com - expected) * np.array(fov) / n_tgt * 1e3
    assert np.abs(off_mm).max() < 0.1
