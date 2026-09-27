import numpy as np
import pytest

# gre_diagnostics.py needs matplotlib and nibabel (via nifti_io.py) at module scope.
pytest.importorskip("matplotlib")
pytest.importorskip("nibabel")

from preprocess.gre_diagnostics import _ift3  # noqa: E402


def test_ift3_inverts_centered_forward_fft():
    rng = np.random.default_rng(0)
    # Odd sizes on purpose: the fftshift-on-both-sides spelling this used to
    # have only agrees with the standard centered pairing on even axes.
    nx, ny, nz, ncoils = 7, 8, 5, 3
    img_true = rng.standard_normal((nx, ny, nz, ncoils)) + 1j * rng.standard_normal(
        (nx, ny, nz, ncoils)
    )

    axes = (0, 1, 2)
    ksp = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img_true, axes=axes), axes=axes), axes=axes)

    recovered = _ift3(ksp)
    np.testing.assert_allclose(recovered, img_true, atol=1e-10)
