import numpy as np
import pytest
from scipy import ndimage

pytest.importorskip("sigpy")

from preprocessing.grid_resize import resize_to_epi_grid  # noqa: E402
from preprocessing.smaps import estimate_smaps, process_smaps  # noqa: E402


def _gaussian_coil_sens(nx, ny, center, sigma):
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    r2 = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
    return np.exp(-r2 / (2 * sigma**2)).astype(complex)


def test_estimate_smaps_shapes_and_object_support():
    nx, ny, ncoils = 48, 48, 4
    centers = [(10, 10), (10, 38), (38, 10), (38, 38)]
    coil_sens = np.stack(
        [_gaussian_coil_sens(nx, ny, c, sigma=30) for c in centers], axis=-1
    )  # [nx, ny, ncoils]

    obj = np.zeros((nx, ny), dtype=complex)
    obj[12:36, 12:36] = 1.0  # square object in the center

    img = obj[:, :, None] * coil_sens  # [nx, ny, ncoils]
    ksp = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    ksp = ksp[:, :, None, :]  # add a singleton z axis: [nx, ny, 1, ncoils]

    # cal_size=None disables the calibration-region crop (see
    # test_estimate_smaps_crops_to_cal_size below for that behavior) --
    # this test wants output shape to match input shape directly.
    smaps, emap = estimate_smaps(ksp, calib_width=24, crop=0.8, cal_size=None)

    assert smaps.shape == (nx, ny, 1, ncoils)
    assert emap.shape == (nx, ny, 1)
    # Eigenvalue should be high inside the object, low in the background
    # corners far from it.
    assert emap[24, 24, 0] > emap[0, 0, 0]


def test_estimate_smaps_crops_to_cal_size():
    # Real 3D GRE volumes (e.g. 108^3) are far too large to pass directly
    # to sigpy's EspiritCalib -- it allocates a coil-covariance array sized
    # to its *entire* input k-space's spatial shape, not just calib_width
    # (confirmed against real project data: this thrashed 14GB+ of memory
    # and never completed in over an hour). estimate_smaps must crop to
    # cal_size before calibrating, regardless of the input volume's size.
    n, ncoils = 32, 4
    xs, ys, zs = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    centers = [(8, 8, 8), (8, 24, 8), (24, 8, 24), (24, 24, 24)]
    sens = np.stack(
        [np.exp(-((xs - c[0]) ** 2 + (ys - c[1]) ** 2 + (zs - c[2]) ** 2) / (2 * 12**2))
         for c in centers],
        axis=-1,
    ).astype(complex)

    obj = np.zeros((n, n, n), dtype=complex)
    obj[10:22, 10:22, 10:22] = 1.0

    img = obj[:, :, :, None] * sens
    axes = (0, 1, 2)
    ksp = np.fft.ifftshift(np.fft.fftn(np.fft.ifftshift(img, axes=axes), axes=axes), axes=axes)

    cal_size = 16
    smaps, emap = estimate_smaps(ksp, calib_width=cal_size, crop=0.8, cal_size=cal_size)

    assert smaps.shape == (cal_size, cal_size, cal_size, ncoils)
    assert emap.shape == (cal_size, cal_size, cal_size)


def test_process_smaps_mask_crop_resize_normalize():
    Nx_gre, Ny_gre, Nz_gre, ncoils = 20, 20, 12, 3
    rng = np.random.default_rng(0)
    smaps_raw = rng.standard_normal((Nx_gre, Ny_gre, Nz_gre, ncoils)) + 1j * rng.standard_normal(
        (Nx_gre, Ny_gre, Nz_gre, ncoils)
    )

    emap = np.zeros((Nx_gre, Ny_gre, Nz_gre))
    emap[4:16, 4:16, 2:10] = 1.0  # "object" region has high eigenvalue

    fov_gre = (0.216, 0.216, 0.216)
    fov = (0.216, 0.216, 0.108)  # half the z-FOV -> expect a symmetric z-crop
    n_target = (16, 16, 8)

    smaps = process_smaps(smaps_raw, emap, fov_gre, fov, n_target, threshold_mask=0.5)

    assert smaps.shape == (16, 16, 8, ncoils)

    # RSS across coils should be ~1 wherever the resized volume isn't
    # background (background maps to exactly 0 pre-normalization, which
    # process_smaps guards against dividing by, leaving it at 0).
    rss = np.sqrt(np.sum(np.abs(smaps) ** 2, axis=-1))
    nonzero = rss > 1e-6
    assert nonzero.any()
    np.testing.assert_allclose(rss[nonzero], 1.0, atol=1e-6)


def test_process_smaps_background_is_exactly_zero_after_resize():
    # Regression: cubic-spline resize (order=3) is a global IIR prefilter,
    # so masking smaps_raw *before* the resize alone leaves a halo of small
    # (~1e-6 to 1e-9) nonzero leaked values just outside the object on the
    # target grid -- invisible until the RSS normalization below (or
    # recon/reconstruct.py's own re-normalization on load) divides by that
    # same tiny value and rescales it straight back up to full unit
    # magnitude, silently erasing the mask everywhere except voxels that
    # happen to be exact-zero. process_smaps must re-apply a hard,
    # nearest-neighbor-resized mask *after* the resize so background stays
    # exactly zero regardless.
    Nx_gre, Ny_gre, Nz_gre, ncoils = 20, 20, 20, 3
    rng = np.random.default_rng(0)
    smaps_raw = rng.standard_normal((Nx_gre, Ny_gre, Nz_gre, ncoils)) + 1j * rng.standard_normal(
        (Nx_gre, Ny_gre, Nz_gre, ncoils)
    )

    emap = np.zeros((Nx_gre, Ny_gre, Nz_gre))
    emap[8:12, 8:12, 8:12] = 1.0  # small "object" cube in the center

    fov = (0.2, 0.2, 0.2)  # same FOV both sides -- no z-crop, isolates the resize
    n_target = (40, 40, 40)  # 2x upsample -- enough to trigger spline leakage

    smaps = process_smaps(smaps_raw, emap, fov, fov, n_target, threshold_mask=0.5)
    rss = np.sqrt(np.sum(np.abs(smaps) ** 2, axis=-1))

    # Independently derive the same hard, nearest-neighbor-resized mask
    # process_smaps itself now applies, to know exactly which target-grid
    # voxels must be background.
    target_mask = resize_to_epi_grid(
        (emap > 0.5).astype(np.float64), fov, fov, n_target, order=0
    ) > 0.5
    assert (~target_mask).any()
    np.testing.assert_array_equal(rss[~target_mask], 0.0)


def test_process_smaps_smoothing_reduces_roughness_but_keeps_mask_exact():
    # smooth_sigma_mm > 0 (the default) should measurably reduce
    # voxel-to-voxel roughness relative to smooth_sigma_mm=0 -- this is the
    # fix for the blocky/rippling texture ESPIRiT's cal_size-resolution
    # calibration leaves near the object edge after cubic-spline resize
    # (confirmed on real data to be present even with no masking applied at
    # all, so a genuine resolution artifact worth low-pass filtering, not
    # an interpolation or masking bug -- see process_smaps' docstring).
    # Smoothing must not compromise the exact-background-zero guarantee
    # from the previous fix, since the mask-normalized blur intentionally
    # smears slightly past the true boundary and must be cut back.
    Nx_gre, Ny_gre, Nz_gre, ncoils = 24, 24, 24, 3
    rng = np.random.default_rng(0)
    smaps_raw = rng.standard_normal((Nx_gre, Ny_gre, Nz_gre, ncoils)) + 1j * rng.standard_normal(
        (Nx_gre, Ny_gre, Nz_gre, ncoils)
    )

    emap = np.zeros((Nx_gre, Ny_gre, Nz_gre))
    emap[4:20, 4:20, 4:20] = 1.0

    fov = (0.18, 0.18, 0.18)
    n_target = (91, 91, 91)  # large upsample factor -- where blockiness shows up

    smaps_unsmoothed = process_smaps(
        smaps_raw, emap, fov, fov, n_target, threshold_mask=0.5, smooth_sigma_mm=0,
    )
    smaps_smoothed = process_smaps(
        smaps_raw, emap, fov, fov, n_target, threshold_mask=0.5, smooth_sigma_mm=6.0,
    )

    target_mask = resize_to_epi_grid(
        (emap > 0.5).astype(np.float64), fov, fov, n_target, order=0
    ) > 0.5

    def roughness(vol):
        # Mean absolute discrete Laplacian magnitude over interior mask
        # voxels -- a standard roughness/blockiness proxy.
        lap = sum(
            np.abs(np.roll(vol, 1, axis=ax) + np.roll(vol, -1, axis=ax) - 2 * vol)
            for ax in range(3)
        )
        interior = ndimage.binary_erosion(target_mask, iterations=2)
        return np.abs(lap[interior]).mean()

    rough_unsmoothed = roughness(np.abs(smaps_unsmoothed[..., 0]))
    rough_smoothed = roughness(np.abs(smaps_smoothed[..., 0]))
    assert rough_smoothed < 0.5 * rough_unsmoothed

    # Both must still keep exact-zero background.
    rss_smoothed = np.sqrt(np.sum(np.abs(smaps_smoothed) ** 2, axis=-1))
    np.testing.assert_array_equal(rss_smoothed[~target_mask], 0.0)


def test_process_smaps_rejects_epi_fov_larger_than_gre():
    smaps_raw = np.zeros((8, 8, 8, 2), dtype=complex)
    emap = np.zeros((8, 8, 8))
    try:
        process_smaps(smaps_raw, emap, (0.1, 0.1, 0.1), (0.1, 0.1, 0.2), (4, 4, 4), 0.5)
        raised = False
    except ValueError:
        raised = True
    assert raised
