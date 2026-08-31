import numpy as np
import pytest

pytest.importorskip("sigpy")

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


def test_process_smaps_rejects_epi_fov_larger_than_gre():
    smaps_raw = np.zeros((8, 8, 8, 2), dtype=complex)
    emap = np.zeros((8, 8, 8))
    try:
        process_smaps(smaps_raw, emap, (0.1, 0.1, 0.1), (0.1, 0.1, 0.2), (4, 4, 4), 0.5)
        raised = False
    except ValueError:
        raised = True
    assert raised
