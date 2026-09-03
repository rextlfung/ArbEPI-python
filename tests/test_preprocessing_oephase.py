import numpy as np
import pytest

from preprocessing.oephase import epiphasecorrect, getoephase, smooth_custom


def test_smooth_custom_matches_reference_edge_truncated_average():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = smooth_custom(x, span=3)
    # Edge points average over a shrunk window (no padding).
    expected = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 4.5])
    np.testing.assert_allclose(y, expected)


def test_smooth_custom_rejects_even_span():
    with pytest.raises(ValueError):
        smooth_custom(np.arange(5.0), span=4)


def _synthetic_echo_train(nx, etl, ncoils, a0, a1, rng):
    """Flat-magnitude object, distinct-per-coil amplitude/phase (which must
    cancel out of getoephase's odd/even ratio), and an injected a0 + a1*x
    phase offset on every even echo relative to every odd echo -- exactly
    the quantity getoephase's returned `a` is meant to recover."""
    x_coord = (np.arange(nx) - nx / 2 + 0.5) / nx
    coil_sens = np.exp(1j * rng.uniform(-1, 1, ncoils)) * rng.uniform(0.5, 1.5, ncoils)

    obj = np.ones(nx, dtype=complex)  # flat-magnitude object
    x = np.empty((nx, etl, ncoils), dtype=complex)
    even_phase = np.exp(1j * (a0 + a1 * x_coord))
    for e in range(etl):
        base = obj * (even_phase if e % 2 == 1 else 1.0)  # MATLAB 1-based even -> Python odd index
        x[:, e, :] = base[:, None] * coil_sens[None, :]
    return x


def test_getoephase_recovers_injected_linear_phase():
    rng = np.random.default_rng(42)
    nx, etl, ncoils = 64, 20, 4
    a0_true, a1_true = 0.7, -1.3

    x = _synthetic_echo_train(nx, etl, ncoils, a0_true, a1_true, rng)
    a, th = getoephase(x)

    assert th.shape == (nx, etl // 2)
    np.testing.assert_allclose(a, [a0_true, a1_true], atol=1e-6)


def _img_to_kspace(x_img):
    # Standard centered-FFT pairing (ifftshift-in/fftshift-out), the exact
    # inverse of _kspace_to_img below -- matches epiphasecorrect's own
    # convention (see its docstring), so this round-trips correctly for
    # both even and odd nx, unlike the fftshift-on-both-sides spelling.
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x_img, axes=0), axis=0), axes=0)


def _kspace_to_img(d):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(d, axes=0), axis=0), axes=0)


@pytest.mark.parametrize('nx', [64, 63])
def test_epiphasecorrect_removes_odd_even_mismatch(nx):
    # epiphasecorrect operates on Cartesian *k-space* (it does its own
    # ifft/correct/fft round trip internally), unlike getoephase which
    # expects data already ifft'd to image space -- so the synthetic
    # image-space profile has to be forward-FFT'd before being handed in,
    # and the output FFT'd back to image space before re-checking with
    # getoephase. Parametrized over odd nx too (docs/review-findings.md
    # item 44): epiphasecorrect's ifftshift-in/fftshift-out convention
    # round-trips exactly regardless of parity, unlike the previous
    # fftshift-on-both-sides spelling, which only did for even nx.
    rng = np.random.default_rng(7)
    etl, ncoils = 20, 4
    a0_true, a1_true = 0.4, 0.9

    x_img = _synthetic_echo_train(nx, etl, ncoils, a0_true, a1_true, rng)
    d_kspace = _img_to_kspace(x_img)

    dc_kspace = epiphasecorrect(d_kspace, np.array([a0_true, a1_true]))
    dc_img = _kspace_to_img(dc_kspace)

    a_after, _ = getoephase(dc_img)
    np.testing.assert_allclose(a_after, [0.0, 0.0], atol=1e-6)
