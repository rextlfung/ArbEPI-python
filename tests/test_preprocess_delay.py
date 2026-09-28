import numpy as np
import pytest

# preprocess.py imports epi_gridding.py at module scope, which needs sigpy
# (the optional `preprocessing` extra), so gate on it the same way
# tests/test_recon_*.py gates on torch/mirtorch.
pytest.importorskip("sigpy")

from preprocess.preprocess import select_best_delay  # noqa: E402


def test_select_best_delay_picks_zero_wrap_closest_to_zero_a2():
    report = {
        'delay': [-1.0, -0.5, 0.0, 0.5, 1.0],
        'a1': [0.0, 0.0, 0.0, 0.0, 0.0],
        'a2': [0.9, -0.2, 0.05, 0.3, 1.1],
        'wrap_count': [2, 0, 0, 0, 3],
    }
    best = select_best_delay(report)
    assert best == 0.0  # a2=0.05 is closest to zero among the zero-wrap candidates


def test_select_best_delay_falls_back_to_fewest_wraps_if_none_safe():
    report = {
        'delay': [-1.0, 0.0, 1.0],
        'a1': [0.0, 0.0, 0.0],
        'a2': [5.0, -5.0, 0.1],
        'wrap_count': [3, 1, 2],
    }
    best = select_best_delay(report)
    assert best == 0.0  # delay index 1 has the fewest wraps (1)


@pytest.mark.parametrize('nx', [63, 64])
def test_oephase_estimate_and_correction_share_a_pixel_frame(nx):
    """compute_oephase's estimate, applied by epiphasecorrect to the same
    gridded data, removes the odd/even mismatch -- at odd nx too, where the
    MATLAB fftshift/ifftshift spelling calibrate_delay.py used to have inline
    lands one pixel away from epiphasecorrect's frame (review item 251)."""
    import sigpy

    from preprocess.epi_gridding import rampsampepi2cart
    from preprocess.oephase import epiphasecorrect, getoephase
    from preprocess.preprocess import compute_oephase

    etl, nshots, ncoils, fov_cm = 8, 2, 3, 20.0
    kx = (np.arange(nx) - nx // 2) / fov_cm  # uniform samples: gridding is ~an exact resample
    xc = (np.arange(nx) - nx / 2 + 0.5) / nx  # getoephase's x coordinate
    profile = np.exp(-(((np.arange(nx) - nx / 2) / (nx / 5)) ** 2))
    rng = np.random.default_rng(0)
    coil = rng.standard_normal(ncoils) + 1j * rng.standard_normal(ncoils)
    img = profile[:, None, None, None] * coil[None, None, None, :] * np.ones((1, etl, nshots, 1))
    img[:, 1::2] *= np.exp(1j * (0.4 + 2.0 * xc))[:, None, None, None]
    # Ramp-sampled data: forward NUFFT of each echo's profile at the kx samples.
    ksp = np.moveaxis(sigpy.nufft(np.moveaxis(img, 0, -1), (kx * fov_cm)[:, None]), -1, 0)

    a, _ = compute_oephase(ksp, kx, kx, nx, fov_cm)
    corrected = epiphasecorrect(rampsampepi2cart(ksp, kx, kx, nx, fov_cm), a)

    # Re-estimate the odd/even mismatch on the corrected data, in
    # epiphasecorrect's own image frame: it must be gone.
    x_corr = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(corrected, axes=0), axis=0), axes=0)
    a_residual, _ = getoephase(np.mean(x_corr, axis=2))
    np.testing.assert_allclose(a_residual, 0, atol=1e-6)
