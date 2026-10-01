import numpy as np
import pytest

from preprocess.oephase import epiphasecorrect, getoephase, smooth_custom


def test_smooth_custom_matches_reference_edge_truncated_average():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = smooth_custom(x, span=3)
    # Edge points average over a shrunk window (no padding).
    expected = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 4.5])
    np.testing.assert_allclose(y, expected)


def test_smooth_custom_rejects_even_span():
    with pytest.raises(ValueError):
        smooth_custom(np.arange(5.0), span=4)


def _synthetic_echo_train(nx, etl, ncoils, a0, a1, rng, decay=1.0):
    """Flat-magnitude object, distinct-per-coil amplitude/phase (which must
    cancel out of getoephase's odd/even ratio), and an injected a0 + a1*x
    phase offset on every even echo relative to every odd echo -- exactly
    the quantity getoephase's returned `a` is meant to recover. a0/a1 may be
    scalars or [etl//2] arrays (one value per echo pair); every echo's
    magnitude is scaled by decay**echo."""
    x_coord = (np.arange(nx) - nx / 2 + 0.5) / nx
    coil_sens = np.exp(1j * rng.uniform(-1, 1, ncoils)) * rng.uniform(0.5, 1.5, ncoils)
    a0 = np.broadcast_to(a0, (etl // 2,))
    a1 = np.broadcast_to(a1, (etl // 2,))

    obj = np.ones(nx, dtype=complex)  # flat-magnitude object
    x = np.empty((nx, etl, ncoils), dtype=complex)
    for e in range(etl):
        even_phase = np.exp(1j * (a0[e // 2] + a1[e // 2] * x_coord))
        # MATLAB 1-based even echo -> Python odd index
        base = obj * (even_phase if e % 2 == 1 else 1.0) * decay**e
        x[:, e, :] = base[:, None] * coil_sens[None, :]
    return x


def test_getoephase_recovers_injected_linear_phase():
    rng = np.random.default_rng(42)
    nx, etl, ncoils = 64, 20, 4
    a0_true, a1_true = 0.7, -1.3

    x = _synthetic_echo_train(nx, etl, ncoils, a0_true, a1_true, rng)
    a, th = getoephase(x)

    assert th.shape == (nx, etl // 2)
    assert a.shape == (etl // 2, 2)
    np.testing.assert_allclose(a, np.tile([a0_true, a1_true], (etl // 2, 1)), atol=1e-6)


def _pair_drift(etl, mid, slope):
    t = np.arange(etl // 2) - (etl // 2 - 1) / 2
    return mid + slope * t


def test_getoephase_recovers_echo_dependent_drift():
    """The odd/even phase drifts along the train on real data (review item
    260); getoephase's per-pair model recovers a linear drift in both terms."""
    rng = np.random.default_rng(3)
    nx, etl, ncoils = 64, 60, 4
    a0 = _pair_drift(etl, -0.33, -0.002)
    a1 = _pair_drift(etl, 0.15, -0.004)

    a, _ = getoephase(_synthetic_echo_train(nx, etl, ncoils, a0, a1, rng))
    np.testing.assert_allclose(a[:, 0], a0, atol=1e-6)
    np.testing.assert_allclose(a[:, 1], a1, atol=1e-6)

    # echo_order=0 fits one value for the whole train: the drift's mid-train value.
    a_const, _ = getoephase(_synthetic_echo_train(nx, etl, ncoils, a0, a1, rng), echo_order=0)
    np.testing.assert_allclose(a_const, np.tile([-0.33, 0.15], (etl // 2, 1)), atol=1e-6)


def test_getoephase_estimates_a_strongly_decayed_train():
    """Regression for review item 260: on 20260930ballfat's 1x radial run the
    last quarter of a 60-echo train fell below 10% of the first echo, the
    ported getoephase.m mask came out empty and np.linalg.lstsq silently
    returned a = [0, 0]. Here the last echo is ~1% of the first."""
    rng = np.random.default_rng(5)
    nx, etl, ncoils = 64, 60, 4
    x = _synthetic_echo_train(nx, etl, ncoils, -0.36, -0.25, rng, decay=0.925)
    assert np.abs(x[:, -15:]).max() < 0.1 * np.abs(x).max()

    a, _ = getoephase(x)
    np.testing.assert_allclose(a, np.tile([-0.36, -0.25], (etl // 2, 1)), atol=1e-6)


def test_getoephase_raises_without_signal():
    """No signal must be an error, not a silent a = 0 (review item 260)."""
    with pytest.raises(ValueError, match='cannot determine'):
        getoephase(np.zeros((64, 20, 4), dtype=complex))


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
    np.testing.assert_allclose(a_after, 0.0, atol=1e-6)


def test_epiphasecorrect_applies_a_per_echo_pair():
    """A drifting mismatch is removed by the per-pair a, and left by the
    mid-train constant alone; a mis-shaped a is rejected."""
    rng = np.random.default_rng(11)
    nx, etl, ncoils = 64, 40, 4
    a0 = _pair_drift(etl, 0.3, 0.01)
    a1 = _pair_drift(etl, -0.5, 0.02)
    a_true = np.stack([a0, a1], axis=1)
    d_kspace = _img_to_kspace(_synthetic_echo_train(nx, etl, ncoils, a0, a1, rng))

    a_after, _ = getoephase(_kspace_to_img(epiphasecorrect(d_kspace, a_true)))
    np.testing.assert_allclose(a_after, 0.0, atol=1e-6)

    a_left, _ = getoephase(_kspace_to_img(epiphasecorrect(d_kspace, a_true.mean(axis=0))))
    np.testing.assert_allclose(a_left, a_true - a_true.mean(axis=0), atol=1e-6)

    with pytest.raises(ValueError, match='shape'):
        epiphasecorrect(d_kspace, a_true[:-1])
