import numpy as np
import pytest

# run_rss.py imports recon_frames.py (which needs sigpy and nibabel, via
# smaps.py/nifti_io.py) and nifti_io.py directly at module scope.
pytest.importorskip("sigpy")
pytest.importorskip("nibabel")

from preprocessing.run_rss import _ift3, _rss_recon  # noqa: E402


def test_ift3_inverts_centered_forward_fft():
    rng = np.random.default_rng(0)
    nx, ny, nz, ncoils = 6, 8, 4, 3
    img_true = rng.standard_normal((nx, ny, nz, ncoils)) + 1j * rng.standard_normal(
        (nx, ny, nz, ncoils)
    )

    axes = (0, 1, 2)
    ksp = np.fft.ifftshift(np.fft.fftn(np.fft.ifftshift(img_true, axes=axes), axes=axes), axes=axes)

    recovered = _ift3(ksp)
    np.testing.assert_allclose(recovered, img_true, atol=1e-10)


def test_rss_recon_combines_coils_as_root_sum_of_squares():
    nx, ny, nz, ncoils = 4, 4, 4, 5
    rng = np.random.default_rng(1)
    img_true = rng.standard_normal((nx, ny, nz)) + 1j * rng.standard_normal((nx, ny, nz))
    coil_sens = rng.standard_normal(ncoils) + 1j * rng.standard_normal(ncoils)

    per_coil_img = img_true[:, :, :, None] * coil_sens[None, None, None, :]
    axes = (0, 1, 2)
    ksp = np.fft.ifftshift(
        np.fft.fftn(np.fft.ifftshift(per_coil_img, axes=axes), axes=axes), axes=axes
    )

    rss_img = _rss_recon(ksp, None)
    expected = np.abs(img_true) * np.sqrt(np.sum(np.abs(coil_sens) ** 2))

    np.testing.assert_allclose(rss_img, expected, atol=1e-10)
