"""analyze.design / analyze.glm: design matrix, AR(1) prewhitening, z, FDR."""

import numpy as np
import pytest
from scipy import stats

from analyze import glm
from analyze.design import (
    Condition,
    ExperimentParams,
    block_response,
    build_design_matrix,
    canonical_hrf,
    contrast_vector,
    dct_matrix,
    drift_basis,
    hrf_integral,
)

TR = 0.5
NT = 480  # 240 s


def _ar1(rng, shape, rho):
    e = rng.standard_normal(shape)
    out = np.empty_like(e)
    out[..., 0] = e[..., 0]
    for t in range(1, shape[-1]):
        out[..., t] = rho * out[..., t - 1] + np.sqrt(1 - rho**2) * e[..., t]
    return out


def test_hrf_integral_is_the_running_integral_of_the_hrf():
    t = np.arange(0, 40, 0.001)
    num = np.cumsum(canonical_hrf(t)) * 0.001
    assert np.abs(num - hrf_integral(t)).max() < 2e-3
    assert abs(hrf_integral(np.array([200.0]))[0] - 1) < 1e-9


def test_block_response_is_zero_before_the_first_block_and_decays_after():
    r = block_response(np.array([-5.0, 0.0, 15.0, 200.0]), [0.0], [20.0])
    assert r[0] == 0 and r[1] == 0 and r[2] > 0.5 and abs(r[3]) < 1e-6


def test_drift_basis_counts_cosines_longer_than_the_cutoff():
    # 240 s run, 128 s cutoff: floor(2 * 240 / 128) = 3 terms
    B = drift_basis(NT, TR, 128.0)
    assert B.shape == (NT, 3)
    C = dct_matrix(NT)
    assert np.allclose(B.T, C[1:4])
    assert np.allclose(C @ C.T, np.eye(NT), atol=1e-10)
    assert drift_basis(NT, TR, None).shape == (NT, 1)


def test_design_matrix_layout_and_contrast():
    p = ExperimentParams(TR, [Condition("a", (0, 40), (20,)), Condition("b", (20, 60), (20,))],
                         contrast=(1, -1))
    X, names = build_design_matrix(p, NT)
    assert names[:2] == ["a", "b"] and names[-1] == "constant"
    assert X.shape == (NT, 2 + 3 + 1)
    assert np.allclose(contrast_vector(p, names)[:2], [1, -1])
    assert not contrast_vector(p, names)[2:].any()
    # task regressors are what simulate_fmri would inject: the exact convolution
    t = (np.arange(NT) + 0.5) * TR
    assert np.allclose(X[:, 0], block_response(t, [0, 40], [20, 20]))


def test_condition_validates_durations():
    with pytest.raises(ValueError):
        Condition("x", (0, 10), (5, 5, 5))
    with pytest.raises(ValueError):
        Condition("x", (0,), (0,))


def test_t_to_z_matches_naive_form_and_does_not_saturate():
    t = np.array([-4.0, -0.5, 0.0, 2.0, 6.0])
    naive = np.sign(t) * stats.norm.isf(stats.t.sf(np.abs(t), 50))
    assert np.allclose(glm.t_to_z(t, 50), naive, atol=1e-9)
    z = glm.t_to_z(np.array([80.0]), 100)[0]  # F_t rounds to 1 in the naive form
    assert np.isfinite(z) and z > 8.3


def test_fdr_and_bonferroni():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(size=9000), rng.uniform(0, 1e-5, size=1000)])
    thr = glm.fdr_threshold(p, 0.05)
    assert 1e-5 * 0.5 < thr < 0.01
    assert (p <= thr).sum() >= 900
    assert glm.fdr_threshold(rng.uniform(0.5, 1, 100), 0.05) == 0.0
    assert glm.bonferroni_threshold(1000, 0.05) == 5e-5


def test_ols_matches_closed_form_and_known_beta():
    rng = np.random.default_rng(1)
    p = ExperimentParams.block(TR, [20, 60, 100, 140, 180], 20.0)
    X, names = build_design_matrix(p, NT)
    c = contrast_vector(p, names)
    beta = np.zeros((200, X.shape[1]))
    beta[:, 0] = 2.0
    beta[:, -1] = 100.0
    Y = beta @ X.T + rng.standard_normal((200, NT))
    r = glm.fit_glm(Y, X, c, ar1=False)
    assert r.rho == 0 and r.dof == NT - X.shape[1]
    assert abs(r.beta[:, 0].mean() - 2.0) < 0.1
    ref = np.linalg.lstsq(X, Y.T, rcond=None)[0].T
    assert np.allclose(r.beta, ref, atol=1e-4)
    assert np.allclose(r.z, glm.t_to_z(r.t, r.dof), atol=1e-4)


def test_ar1_recovers_rho_and_calibrates_the_null():
    rng = np.random.default_rng(2)
    rho = 0.4
    p = ExperimentParams.block(TR, [20, 60, 100, 140, 180], 20.0)
    X, names = build_design_matrix(p, NT)
    c = contrast_vector(p, names)
    Y = _ar1(rng, (3000, NT), rho)
    plain = glm.fit_glm(Y, X, c, ar1=False)
    white = glm.fit_glm(Y, X, c, ar1=True)
    assert abs(white.rho - rho) < 0.03
    # false positives at nominal 0.001 (two-sided): colored noise inflates them
    thr = glm.t_threshold(plain.dof, 0.001)
    fp_plain = (np.abs(plain.t) > thr).mean()
    fp_white = (np.abs(white.t) > thr).mean()
    assert fp_plain > 3 * 0.001
    assert fp_white < 0.0035
    # the null t is ~ unit variance after prewhitening
    assert abs(white.t.std() - 1) < 0.08 and plain.t.std() > 1.2


def test_ar1_on_white_noise_estimates_zero():
    rng = np.random.default_rng(3)
    p = ExperimentParams.block(TR, [20, 100], 20.0)
    X, names = build_design_matrix(p, NT)
    r = glm.fit_glm(rng.standard_normal((2000, NT)), X, contrast_vector(p, names))
    assert abs(r.rho) < 0.02


def test_drift_terms_remove_a_slow_drift_from_the_residual():
    rng = np.random.default_rng(4)
    t = np.arange(NT) * TR
    drift = 5 * np.cos(np.pi * t / 240 * 1.5)  # 160 s period: inside the basis
    Y = drift[None, :] + rng.standard_normal((300, NT))
    p_no = ExperimentParams.block(TR, [20, 100], 20.0, drift_cutoff_s=None)
    p_dct = ExperimentParams.block(TR, [20, 100], 20.0, drift_cutoff_s=128.0)
    res = []
    for p in (p_no, p_dct):
        X, names = build_design_matrix(p, NT)
        res.append(glm.fit_glm(Y, X, contrast_vector(p, names), ar1=False).sigma2.mean())
    assert res[1] < 0.5 * res[0]


def test_low_band_basis_is_the_alternative_not_a_companion():
    X = np.ones((NT, 1))
    with pytest.raises(ValueError):
        glm.fit_glm(np.zeros((2, NT)), X, np.array([1.0]), ar1=True, basis=glm.low_band(NT, TR))
    r = glm.fit_glm(np.random.default_rng(5).standard_normal((50, NT)), X, np.array([1.0]),
                    ar1=False, basis=glm.low_band(NT, TR, 0.15))
    assert r.dof == len(glm.low_band(NT, TR, 0.15)) - 1


def test_fit_glm_validates_shapes():
    with pytest.raises(ValueError):
        glm.fit_glm(np.zeros((3, 10)), np.ones((11, 1)), np.array([1.0]))
    with pytest.raises(ValueError):
        glm.fit_glm(np.zeros((3, 10)), np.ones((10, 1)), np.array([1.0, 0.0]))
