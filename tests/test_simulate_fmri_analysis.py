"""simulate_fmri/analysis.py: activation maps, ROC curves, the mixed-binomial
reliability model, aliasing."""

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('snake')

from simulate_fmri import analysis, task  # noqa: E402

TR, NT = 0.5, 400


def fake_truth(shape=(6, 5, 4), n_shots=2, factor=2):
    rng = np.random.default_rng(0)
    nt = 8
    roi = np.zeros((1, *shape), dtype=bool)
    roi[0, 1:3, 1:3, 1:3] = True
    amp_map = 0.03 * roi[0] + 0.004 * (np.arange(shape[0])[:, None, None] == 4)
    return analysis.Truth(
        x0=np.ones(shape), image_rest=np.ones(shape), brain=np.ones(shape, dtype=bool),
        tissues={'gm': np.ones(shape)}, roi_masks=roi, roi_names=['block_a'],
        waveforms=rng.standard_normal((1, nt)), amps=np.array([0.03]), amp_map=amp_map,
        canonical=rng.standard_normal((1, nt)), response=rng.standard_normal((1, nt * n_shots)),
        paradigm=np.ones((1, nt * n_shots)), mode_names=['block_a', 'cardiac'],
        mode_gain=rng.standard_normal((2, *shape)) * 0.01,
        mode_course=rng.standard_normal((2, nt * n_shots)),
        t1w=rng.random(tuple(factor * n for n in shape)).astype(np.float32), factor=factor,
        volume_tr=1.0, tr_shot=0.5, te=0.03,
    )


# ---------------------------------------------------------------------------
# activation maps
# ---------------------------------------------------------------------------


def test_t_map_recovers_an_injected_response_and_its_size():
    rng = np.random.default_rng(1)
    t = (np.arange(NT) + 0.5) * TR
    w = task.bold_response(t, 20, 20, NT * TR)
    beta_true = np.r_[np.zeros(300), np.full(100, 0.02)]
    series = 50 * (1 + beta_true[:, None] * w[None, :] + 0.01 * rng.standard_normal((400, NT)))
    for cutoff in (None, 0.15):
        beta, t_stat = analysis.t_map(series, w, TR, cutoff_hz=cutoff)
        assert np.median(beta[300:]) == pytest.approx(0.02, rel=0.03)  # a fraction of the mean
        assert abs(np.median(beta[:300])) < 2e-4
        thr = analysis.t_threshold(NT, TR, 0.001, cutoff)
        assert (t_stat[300:] > thr).all()
        # white noise: the test is calibrated, about 0.1% of null voxels pass
        assert (np.abs(t_stat[:300]) > thr).mean() < 0.02
    # the low band has fewer degrees of freedom, hence a higher threshold
    n_low = len(analysis.low_band(NT, TR, 0.15))
    assert n_low == int(np.floor(0.15 * 2 * NT * TR)) + 1
    assert analysis.t_threshold(NT, TR, 0.001, 0.15) > analysis.t_threshold(NT, TR, 0.001, None)


def test_to_volume_puts_values_back_in_place():
    mask = np.zeros((3, 4, 2), dtype=bool)
    mask[1, 2, 0] = mask[2, 3, 1] = True
    vol = analysis.to_volume(np.array([5.0, 7.0]), mask, fill=-1)
    assert vol[1, 2, 0] == 5 and vol[2, 3, 1] == 7 and (vol[~mask] == -1).all()


def test_true_signal_is_one_plus_the_modes():
    truth = fake_truth()
    mask = np.zeros(truth.x0.shape, dtype=bool)
    mask[2, 2, 2] = mask[4, 0, 1] = True
    s = analysis.true_signal(truth, mask)
    expected = 1 + truth.mode_gain[:, mask].T @ truth.mode_course
    np.testing.assert_allclose(s, expected)
    np.testing.assert_allclose(analysis.true_signal(truth, mask, per_frame=True),
                               expected.reshape(2, 8, 2).mean(axis=2))
    only = analysis.true_signal(truth, mask, modes=['cardiac'])
    np.testing.assert_allclose(only, 1 + truth.mode_gain[1][mask][:, None] * truth.mode_course[1])
    assert truth.n_frames == 8 and truth.n_shots == 2


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------


def test_roc_of_separated_overlapping_and_reversed_scores():
    active = np.r_[np.ones(50, bool), np.zeros(950, bool)]
    inactive = ~active
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(1000)
    fpr, tpr, thr, auc = analysis.roc(noise + 20 * active, active, inactive)
    assert auc == pytest.approx(1.0) and fpr[0] == tpr[0] == 0 and fpr[-1] == tpr[-1] == 1
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0) and np.all(np.diff(thr) < 0)
    assert analysis.roc(noise, active, inactive)[3] == pytest.approx(0.5, abs=0.1)
    assert analysis.roc(noise - 20 * active, active, inactive)[3] == pytest.approx(0.0)
    # shifted Gaussians: the area is Phi(d / sqrt 2)
    big = np.r_[np.ones(20000, bool), np.zeros(20000, bool)]
    score = rng.standard_normal(40000) + 1.0 * big
    assert analysis.roc(score, big, ~big)[3] == pytest.approx(0.7602, abs=0.01)
    # voxels in neither class are left out
    neither = np.zeros(1000, bool)
    neither[60:500] = True
    assert analysis.roc(noise + 20 * active, active, inactive & ~neither)[3] == pytest.approx(1.0)


def test_rates_are_the_roc_at_given_thresholds():
    rng = np.random.default_rng(3)
    active = np.r_[np.ones(300, bool), np.zeros(700, bool)]
    score = rng.standard_normal(1000) + 1.5 * active
    fpr, tpr = analysis.rates(score, active, ~active, [-10.0, 0.5, 10.0])
    assert fpr[0] == tpr[0] == 1 and fpr[2] == tpr[2] == 0
    assert tpr[1] == pytest.approx((score[active] >= 0.5).mean())
    assert fpr[1] == pytest.approx((score[~active] >= 0.5).mean())


def test_truth_classes_leave_partly_activated_voxels_out():
    truth = fake_truth()
    active, inactive = analysis.truth_classes(truth, truth.brain)
    assert active.sum() == 8 and not (active & inactive).any()
    # the plane with a 0.4% change is neither (5% of the 3% amplitude is 0.15%)
    assert (~active & ~inactive).sum() == 5 * 4
    assert inactive.sum() == truth.brain.sum() - 8 - 20


# ---------------------------------------------------------------------------
# the mixed-binomial model
# ---------------------------------------------------------------------------


def test_levels_and_reliability_map():
    scores = np.array([[0.0, 1.0, 2.5, 9.0], [3.0, 3.0, -1.0, 9.0]])
    np.testing.assert_array_equal(analysis.levels(scores, [1.0, 3.0]), [[0, 1, 1, 2], [2, 2, 0, 2]])
    np.testing.assert_array_equal(analysis.reliability_map(scores, 3.0), [1, 1, 0, 2])
    # active at threshold k  <=>  level >= k + 1
    lev = analysis.levels(scores, [1.0, 3.0])
    np.testing.assert_array_equal(lev >= 2, scores >= 3.0)
    with pytest.raises(ValueError, match='increase'):
        analysis.levels(scores, [3.0, 1.0])


@pytest.mark.parametrize('m', [4, 8])
def test_mixed_binomial_recovers_the_rates_at_one_threshold(m):
    rng = np.random.default_rng(4)
    n, lam, p_a, p_i = 60000, 0.06, 0.8, 0.03
    is_active = rng.random(n) < lam
    level = (rng.random((m, n)) < np.where(is_active, p_a, p_i)).astype(int)
    fit = analysis.fit_mixed_binomial(level)
    assert fit.lam == pytest.approx(lam, abs=0.006)
    assert fit.p_active[0] == pytest.approx(p_a, abs=0.02)
    assert fit.p_inactive[0] == pytest.approx(p_i, abs=0.003)
    assert fit.posterior.shape == (n,)
    # the posterior separates the classes about as well as the counts allow
    assert fit.posterior[is_active].mean() > 0.85 and fit.posterior[~is_active].mean() < 0.02
    # its likelihood is at least that of the true parameters
    counts = level.sum(axis=0)
    true_ll = np.log(lam * p_a**counts * (1 - p_a) ** (m - counts)
                     + (1 - lam) * p_i**counts * (1 - p_i) ** (m - counts)).sum()
    assert fit.loglik >= true_ll - 1e-6


def test_mixed_binomial_shares_lambda_between_thresholds():
    """Four thresholds on Gaussian scores: the fitted rates are the Gaussian
    tail probabilities, i.e. the ROC curve, found without the labels."""
    from scipy import stats

    rng = np.random.default_rng(5)
    m, n, lam, shift = 5, 80000, 0.08, 3.0
    is_active = rng.random(n) < lam
    scores = rng.standard_normal((m, n)) + shift * is_active
    thresholds = np.array([0.5, 1.5, 2.5, 3.5])
    fit = analysis.fit_mixed_binomial(analysis.levels(scores, thresholds), len(thresholds))
    assert fit.lam == pytest.approx(lam, abs=0.005)
    np.testing.assert_allclose(fit.p_inactive, stats.norm.sf(thresholds), atol=0.01)
    np.testing.assert_allclose(fit.p_active, stats.norm.sf(thresholds - shift), atol=0.02)
    assert np.all(np.diff(fit.p_active) <= 0) and np.all(np.diff(fit.p_inactive) <= 0)
    assert fit.q_active.sum() == pytest.approx(1) and fit.q_inactive.sum() == pytest.approx(1)
    np.testing.assert_allclose(fit.p_active, np.cumsum(fit.q_active[::-1])[::-1][1:])
    # and agree with what the labels give
    fpr, tpr = analysis.rates(scores.ravel(), np.tile(is_active, m), np.tile(~is_active, m),
                              thresholds)
    np.testing.assert_allclose(fit.p_active, tpr, atol=0.02)
    np.testing.assert_allclose(fit.p_inactive, fpr, atol=0.01)


# ---------------------------------------------------------------------------
# spectra, figures
# ---------------------------------------------------------------------------


def test_alias_folds_about_the_sampling_rate():
    fs = 1 / 0.506
    np.testing.assert_allclose(analysis.alias([0.025, 0.25, 0.9], fs), [0.025, 0.25, 0.9])
    assert analysis.alias(1.1, fs) == pytest.approx(fs - 1.1)  # cardiac: 0.876 Hz
    assert analysis.alias(2.2, fs) == pytest.approx(2.2 - fs)  # its harmonic: 0.224 Hz
    assert analysis.alias(fs, fs) == pytest.approx(0)
    # sampled directly: a 1.1 Hz cosine at 0.506 s is a 0.876 Hz cosine
    n = np.arange(200)
    np.testing.assert_allclose(np.cos(2 * np.pi * 1.1 * n * 0.506),
                               np.cos(2 * np.pi * analysis.alias(1.1, fs) * n * 0.506), atol=1e-9)
    # averaging over the frame attenuates it by |sinc|
    assert analysis.frame_average_gain(0.0, 0.506) == pytest.approx(1)
    assert analysis.frame_average_gain(1.1, 0.506) == pytest.approx(0.563, abs=0.002)
    assert analysis.frame_average_gain(fs, 0.506) == pytest.approx(0, abs=1e-9)


def test_overlay_draws_the_thresholded_map_on_the_anatomical_image():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    truth = fake_truth()
    stat = np.zeros(truth.x0.shape)
    stat[truth.roi_masks[0]] = 6.0
    fig = analysis.overlay(truth, stat, threshold=3.0, centers={'a': np.array([2, 2, 2])},
                           outline=truth.roi_masks[0], title='t')
    assert len(fig.axes) == 4  # three views and a colorbar
    images = fig.axes[2].get_images()  # axial: anatomy, positive map, negative map
    assert images[0].get_array().shape == (10, 12)  # the fine grid, rotated
    shown = [int((~np.ma.getmaskarray(im.get_array())).sum()) for im in images[1:]]
    assert shown == [2 * 2 * 4, 0]  # 2 x 2 voxels x 2^2 fine voxels; nothing negative
    plt.close(fig)
