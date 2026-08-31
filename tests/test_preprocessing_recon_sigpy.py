import numpy as np
import pytest

pytest.importorskip("sigpy")
import sigpy.mri as mr  # noqa: E402

from preprocessing.recon_sigpy import wavelet_tv_recon  # noqa: E402


def _gaussian_coil_sens(nx, ny, center, sigma):
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    r2 = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
    return np.exp(-r2 / (2 * sigma**2)).astype(complex)


def _setup(nx=16, ny=16, nz=4):
    centers = [(3, 3), (3, 13), (13, 3), (13, 13)]
    sens_2d = np.stack([_gaussian_coil_sens(nx, ny, c, sigma=10) for c in centers], axis=0)
    sens = np.tile(sens_2d[:, :, :, None], (1, 1, 1, nz))  # [Ncoils, Nx, Ny, Nz]

    img_true = np.zeros((nx, ny, nz), dtype=complex)
    img_true[4:12, 4:12, :] = 1.0

    # Synthesize with the same Sense operator the recon uses, so the test
    # is insensitive to any FFT-normalization convention sigpy happens to
    # use internally (verified during development: an independently
    # implemented FFT for test data produced a large, purely scale-factor
    # mismatch against sigpy's own Sense operator).
    ksp_full = mr.linop.Sense(sens)(img_true)
    smaps = np.moveaxis(sens, 0, -1)  # [Nx, Ny, Nz, Ncoils], this port's convention
    return img_true, smaps, ksp_full


def test_wavelet_tv_recon_converges_to_truth_as_lambda_shrinks_fully_sampled():
    img_true, smaps, ksp_full = _setup()
    ksp = np.moveaxis(ksp_full, 0, -1)  # [Nx, Ny, Nz, Ncoils]

    errors = []
    for lamda in (1e-2, 1e-3, 1e-5):
        recon = wavelet_tv_recon(ksp, smaps, lamda, lamda, num_iter=150, max_power_iter=30)
        errors.append(np.abs(recon - img_true).max())

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-3


def test_wavelet_tv_recon_output_shape():
    img_true, smaps, ksp_full = _setup(nx=16, ny=16, nz=4)
    ksp = np.moveaxis(ksp_full, 0, -1)
    recon = wavelet_tv_recon(ksp, smaps, 1e-3, 1e-3, num_iter=10)
    assert recon.shape == img_true.shape


def test_wavelet_tv_recon_handles_undersampled_data_without_error():
    img_true, smaps, ksp_full = _setup()
    ksp = np.moveaxis(ksp_full, 0, -1)
    ksp[:, 1::2, :, :] = 0  # R=2 undersampling in ky

    recon = wavelet_tv_recon(ksp, smaps, 1e-4, 1e-4, num_iter=50)
    assert recon.shape == img_true.shape
    assert np.all(np.isfinite(recon))
