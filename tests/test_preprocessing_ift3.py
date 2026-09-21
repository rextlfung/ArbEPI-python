import numpy as np
import pytest

# gre_diagnostics.py needs matplotlib and nibabel (via nifti_io.py) at module scope.
pytest.importorskip("matplotlib")
pytest.importorskip("nibabel")

from preprocessing.gre_diagnostics import _ift3  # noqa: E402


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
