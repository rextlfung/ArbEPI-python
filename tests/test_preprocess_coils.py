import numpy as np
import pytest

from preprocess.coils import (
    align_gcc,
    apply_gcc_image,
    apply_gcc_kspace,
    apply_whitening,
    compute_whitening_matrix,
    gcc_calibration,
    gcc_compression,
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
    data = latent @ S.T + 1e-3 * _crandn(rng, n_samples, ncoils)

    # one x position: GCC reduces to a single compression matrix
    A0, evals = gcc_compression(data[None])
    nvcoils = select_nvcoils(evals, energy_thresh=0.99)
    assert nvcoils == n_true

    compressed = data @ A0[0, :nvcoils].T
    assert compressed.shape == (n_samples, nvcoils)
    assert np.sum(np.abs(compressed) ** 2) == pytest.approx(np.sum(np.abs(data) ** 2), rel=0.05)


def test_select_nvcoils_sums_energy_over_x_and_has_no_floor():
    # Two x positions: one with lots of energy in 2 components, one weak
    # position with flat (noise-like) energy. Summed over x, 2 components hold
    # > 99% -- the weak position doesn't force more, and nothing forces a floor.
    evals = np.array([[1000.0, 1000.0] + [0.01] * 6, [0.1] * 8])
    assert select_nvcoils(evals, 0.99) == 2
    assert select_nvcoils(evals[1], 0.99) == 8  # the weak row alone needs everything
    assert select_nvcoils(np.array([5.0, 0.0, 0.0]), 0.99) == 1


def _xvarying_hybrid(rng, nx=32, m=400, nc=12, rank=2):
    """Hybrid (x, sample, coil) data whose coil subspace rotates along x: at
    each x the signal has rank 2, but over all x it spans every coil."""
    data = np.empty((nx, m, nc), dtype=complex)
    for x in range(nx):
        basis = np.linalg.qr(_crandn(rng, nc, rank))[0]  # a different subspace per x
        data[x] = _crandn(rng, m, rank) @ basis.T
    return data + 1e-4 * _crandn(rng, nx, m, nc)


def test_gcc_keeps_what_global_pca_loses_when_sensitivities_vary_along_x():
    rng = np.random.default_rng(3)
    hyb = _xvarying_hybrid(rng)
    A0, evals = gcc_compression(hyb)
    nv = select_nvcoils(evals, 0.99)
    assert nv == 2
    kept_gcc = np.sum(np.abs(np.einsum('xvc,xmc->xmv', A0[:, :nv], hyb)) ** 2)
    # one global (PCA) matrix for all x: top eigenvectors of sum c c^H
    flat = hyb.reshape(-1, hyb.shape[-1])
    evecs = np.linalg.eigh(flat.T @ flat.conj())[1][:, ::-1]
    kept_pca = np.sum(np.abs(flat @ evecs[:, :nv].conj()) ** 2)
    total = np.sum(np.abs(hyb) ** 2)
    assert kept_gcc / total > 0.999
    assert kept_pca / total < 0.5


def test_align_gcc_keeps_each_subspace_and_smooths_along_x():
    rng = np.random.default_rng(4)
    # Smoothly varying subspace, but with a random unitary scramble per x
    # (which is what independent per-x eigendecompositions give you).
    nx, nc, nv = 20, 8, 3
    base = np.linalg.qr(_crandn(rng, nc, nc))[0]
    A = np.empty((nx, nv, nc), dtype=complex)
    for x in range(nx):
        theta = 0.05 * x
        rot = np.eye(nc, dtype=complex)
        rot[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        smooth = (base @ rot)[:, :nv].conj().T
        scramble = np.linalg.qr(_crandn(rng, nv, nv))[0]
        A[x] = scramble @ smooth
    aligned = align_gcc(A)
    proj = np.einsum('xvc,xvd->xcd', A.conj(), A)
    proj_aligned = np.einsum('xvc,xvd->xcd', aligned.conj(), aligned)
    np.testing.assert_allclose(proj_aligned, proj, atol=1e-10)
    jump = np.linalg.norm(np.diff(A, axis=0), axis=(1, 2)).mean()
    jump_aligned = np.linalg.norm(np.diff(aligned, axis=0), axis=(1, 2)).mean()
    assert jump_aligned < 0.2 * jump


def _fftc3(x):
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm='ortho'),
                           axes=axes)


def test_gcc_kspace_compression_matches_compressed_maps_forward_model():
    """The SENSE identity GCC relies on: with a (ky, kz) mask shared by every
    kx, compressing the data per x equals the forward model of the maps
    compressed with the same per-x matrices."""
    rng = np.random.default_rng(5)
    nx, ny, nz, nc, nv = 9, 8, 6, 5, 3  # odd nx on purpose
    rho = _crandn(rng, nx, ny, nz)
    smaps = _crandn(rng, nx, ny, nz, nc)
    mask = rng.random((ny, nz)) < 0.4
    A = np.stack([np.linalg.qr(_crandn(rng, nc, nv))[0].conj().T for _ in range(nx)])

    y = _fftc3(smaps * rho[..., None]) * mask[None, :, :, None]
    lhs = apply_gcc_kspace(y, A)
    rhs = _fftc3(apply_gcc_image(smaps, A) * rho[..., None]) * mask[None, :, :, None]
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)
    assert np.all(lhs[:, ~mask] == 0)  # unsampled locations stay exactly zero


def test_gcc_keeps_whitened_noise_at_unit_variance():
    rng = np.random.default_rng(6)
    nx, nc, nv = 16, 8, 4
    noise = _crandn(rng, nx, 50, 20, nc) / np.sqrt(2)  # unit variance per complex sample
    A = np.stack([np.linalg.qr(_crandn(rng, nc, nv))[0].conj().T for _ in range(nx)])
    assert np.mean(np.abs(apply_gcc_kspace(noise, A)) ** 2) == pytest.approx(1.0, rel=0.03)


def test_gcc_calibration_lands_on_the_target_readout_grid():
    """Cropping/zero-padding kx then inverse-FFTing puts the calibration data on
    the target grid: a point object at the source's center x stays at the
    target's center x."""
    rng = np.random.default_rng(7)
    nx_src, ny, nz, nc = 12, 8, 8, 3
    img = np.zeros((nx_src, ny, nz, nc), dtype=complex)
    img[nx_src // 2, :, :, :] = _crandn(rng, nc)
    ksp = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img, axes=(0, 1, 2)), axes=(0, 1, 2)),
                          axes=(0, 1, 2))
    for nx in (9, 12, 20):
        calib = gcc_calibration(ksp, nx, calib_size=4)
        energy_x = np.sum(np.abs(calib) ** 2, axis=(1, 2))
        assert calib.shape == (nx, 16, nc)
        assert np.argmax(energy_x) == nx // 2


def test_gcc_calibration_evaluates_a_larger_source_fov_at_the_target_x_positions():
    """A deGRE with a larger x-FOV has finer kx spacing than the EPI grid, so
    cropping kx to nx would put the wrong x positions on the target grid. A
    single kx sample (a complex exponential in x) must come out as that
    exponential evaluated at the target's own x positions."""
    rng = np.random.default_rng(8)
    nx_src, fov_src, nx, fov = 30, 0.25, 20, 0.2
    ny, nz, nc = 6, 6, 2
    m = nx_src // 2 + 5  # kx = 5 / fov_src = 20 cycles/m, inside the target band (+-50)
    ksp = np.zeros((nx_src, ny, nz, nc), dtype=complex)
    ksp[m] = _crandn(rng, ny, nz, nc)
    calib = gcc_calibration(ksp, nx, calib_size=4, fov_src=fov_src, fov=fov)
    x = (np.arange(nx) - nx // 2) * fov / nx
    blk = ksp[m, 1:5, 1:5].reshape(-1, nc)
    expected = np.exp(2j * np.pi * (5 / fov_src) * x)[:, None, None] * blk[None] / np.sqrt(nx)
    np.testing.assert_allclose(calib, expected, atol=1e-12)
