import numpy as np

from preprocessing.cg_sense import _fftc, cg_sense


def _gaussian_coil_sens(nx, ny, center, sigma):
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    r2 = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
    return np.exp(-r2 / (2 * sigma**2)).astype(complex)


def test_cg_sense_recovers_known_image_fully_sampled():
    nx, ny = 16, 16
    centers = [(2, 2), (2, 13), (13, 2), (13, 13)]
    sens = np.stack([_gaussian_coil_sens(nx, ny, c, sigma=10) for c in centers], axis=-1)

    rng = np.random.default_rng(0)
    img_true = (rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny))).astype(complex)

    kdata = _fftc(sens * img_true[:, :, None], (0, 1))  # fully sampled, no zeros anywhere

    recon = cg_sense(kdata, sens, num_iter=100, tol=1e-12)[:, :, 0]

    # cg_sense solves the SENSE normal equations, not a plain inverse, so
    # compare against the actual least-squares solution rather than assuming
    # perfect recovery of img_true (Gaussian coil sensitivities overlapping
    # everywhere makes this well conditioned, but not exactly the identity).
    residual = np.abs(recon - img_true)
    assert residual.max() < 1e-3 * np.abs(img_true).max()


def test_cg_sense_output_shape_keeps_trailing_singleton():
    nx, ny, ncoils = 8, 8, 2
    sens = np.ones((nx, ny, ncoils), dtype=complex)
    # Nonzero data -- an all-zero mask degenerates to a 0/0 in the first
    # step's alpha (matches MATLAB's identical behavior on all-zero input,
    # not a porting bug), which isn't what this test is checking.
    kdata = np.ones((nx, ny, ncoils), dtype=complex)
    recon = cg_sense(kdata, sens, num_iter=1)
    assert recon.shape == (nx, ny, 1)


def test_cg_sense_zero_iterations_returns_zero_initial_guess():
    nx, ny, ncoils = 8, 8, 3
    sens = np.ones((nx, ny, ncoils), dtype=complex)
    kdata = np.ones((nx, ny, ncoils), dtype=complex)  # nonzero data, but no iterations run
    recon = cg_sense(kdata, sens, num_iter=0)
    np.testing.assert_array_equal(recon, 0)
