import numpy as np
import pytest

from preprocess.r2star import fit_r2star, resize_r2star_to_epi


def test_two_echo_fit_is_the_two_point_formula():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.5, 1.0, size=(4, 5, 3, 2))
    te = np.array([0.003, 0.0052])
    mask = np.ones(s.shape[:3], dtype=bool)
    expected = np.clip(np.log(s[..., 0] / s[..., 1]) / (te[1] - te[0]), 0, 200)
    np.testing.assert_allclose(fit_r2star(s, te, mask), expected, rtol=1e-5)


def test_multi_echo_fit_recovers_a_known_decay():
    te = np.linspace(0.003, 0.08, 8)
    r2_true = np.array([10.0, 25.0, 60.0])
    s = 3.0 * np.exp(-r2_true[:, None] * te[None, :])  # [3 voxels, 8 echoes]
    r2 = fit_r2star(s[None, None], te, np.ones((1, 1, 3), dtype=bool))
    np.testing.assert_allclose(r2[0, 0], r2_true, rtol=1e-5)


def test_masked_nonpositive_and_negative_estimates_become_zero():
    te = np.array([0.003, 0.005])
    s = np.array([[[[1.0, 0.9], [0.9, 1.0], [0.0, 0.5], [1.0, 0.5]]]])  # growth, zero, fine
    mask = np.array([[[True, True, True, False]]])
    r2 = fit_r2star(s, te, mask)
    assert r2[0, 0, 0] > 0
    assert r2[0, 0, 1] == 0  # signal grew: clipped
    assert r2[0, 0, 2] == 0  # non-positive echo
    assert r2[0, 0, 3] == 0  # outside the mask


def test_needs_two_echoes():
    with pytest.raises(ValueError):
        fit_r2star(np.ones((2, 2, 2, 1)), np.array([0.003]), np.ones((2, 2, 2), dtype=bool))


def test_resized_r2star_is_zero_outside_the_resized_mask():
    """Review item 222: the cubic resize rings past the mask edge; the result
    must be exactly zero where the resized mask is."""
    from preprocess.grid_resize import resize_to_epi_grid

    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[3:9, 3:9, 3:9] = True
    r2 = np.where(mask, 30.0, 0.0)
    fov = (0.12, 0.12, 0.12)
    n = (24, 24, 24)
    out = resize_r2star_to_epi(r2, mask, fov, fov, n)
    m = resize_to_epi_grid(mask.astype(float), fov, fov, n, order=0) > 0.5
    plain = np.clip(resize_to_epi_grid(r2 * mask, fov, fov, n, order=3), 0, None)
    assert np.any((plain > 0) & ~m)  # the spline does leak without the re-mask
    assert np.all(out[~m] == 0)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[m], plain[m], rtol=1e-5)
