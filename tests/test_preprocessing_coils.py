import numpy as np
import pytest

from preprocessing.coils import (
    apply_coil_compression,
    apply_whitening,
    coil_compression_matrix,
    compute_coil_covariance,
    compute_whitening_matrix,
    select_nvcoils,
)


def _crandn(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _correlated_noise(rng, n_samples, ncoils):
    """Complex Gaussian noise passed through a fixed random mixing matrix,
    so channels are correlated with non-unit, non-equal variances."""
    A = _crandn(rng, ncoils, ncoils)
    white = _crandn(rng, n_samples, ncoils)
    return white @ A.T, A


def test_whitening_decorrelates_and_normalizes():
    rng = np.random.default_rng(0)
    ncoils = 8
    calib_noise, _ = _correlated_noise(rng, 20000, ncoils)
    W = compute_whitening_matrix(calib_noise)

    # preprocess.m always applies the noise-derived W back onto data drawn
    # from that same noise process, so whitening the calibration noise
    # itself is the realistic check here.
    whitened = apply_whitening(calib_noise, W)
    cov = (whitened.conj().T @ whitened) / whitened.shape[0]

    np.testing.assert_allclose(cov, np.eye(ncoils), atol=0.1)


def test_coil_compression_preserves_dominant_signal_energy():
    rng = np.random.default_rng(1)
    ncoils = 16
    n_true = 3
    n_samples = 5000

    # Signal effectively lives in a 3-dimensional coil subspace; embed it in
    # 16 channels via a random mixing matrix, plus a little full-rank noise.
    S = _crandn(rng, ncoils, n_true)
    latent = _crandn(rng, n_samples, n_true)
    signal = latent @ S.T
    noise = 1e-3 * _crandn(rng, n_samples, ncoils)
    data = signal + noise

    cov = compute_coil_covariance(data)
    nvcoils, nvcoils_energy = select_nvcoils(cov, energy_thresh=0.99, floor=1)
    assert nvcoils_energy == n_true
    assert nvcoils >= n_true

    V = coil_compression_matrix(cov, nvcoils)
    compressed = apply_coil_compression(data, V)
    assert compressed.shape == (n_samples, nvcoils)

    # Energy retained after compression should be close to total signal energy.
    assert np.sum(np.abs(compressed) ** 2) == pytest.approx(np.sum(np.abs(data) ** 2), rel=0.05)


def test_select_nvcoils_respects_floor():
    rng = np.random.default_rng(2)
    ncoils = 10
    # Nearly rank-1 data: energy threshold alone would pick ~1 component.
    S = rng.standard_normal((ncoils, 1)) + 1j * rng.standard_normal((ncoils, 1))
    latent = rng.standard_normal((2000, 1)) + 1j * rng.standard_normal((2000, 1))
    data = latent @ S.T + 1e-6 * rng.standard_normal((2000, ncoils))

    cov = compute_coil_covariance(data)
    nvcoils, nvcoils_energy = select_nvcoils(cov, energy_thresh=0.9, floor=4)
    assert nvcoils_energy == 1
    assert nvcoils == 4
