"""simulate_fmri/task.py: the block paradigm and the BOLD response to it."""

import numpy as np
import pytest

pytest.importorskip('snake')

from simulate_fmri import task  # noqa: E402


def test_canonical_hrf_is_spms():
    """Unit area, peak at 5 s, undershoot around 15 s; the same curve as
    nilearn's spm_hrf."""
    t = np.arange(0, 60, 0.01)
    h = task.canonical_hrf(t)
    assert np.sum(h) * 0.01 == pytest.approx(1.0, abs=1e-3)
    assert t[h.argmax()] == pytest.approx(5.0, abs=0.02)
    assert 12 < t[h.argmin()] < 18 and h.min() < 0
    assert not task.canonical_hrf(np.array([-1.0, 0.0])).any()
    spm_hrf = pytest.importorskip('nilearn.glm.first_level.hemodynamic_models').spm_hrf
    ref = spm_hrf(1.0, oversampling=100, time_length=32.0)  # sums to 1 on a 0.01 s grid
    np.testing.assert_allclose(h[: len(ref)] * 0.01, ref, atol=5e-3 * ref.max())


def test_block_paradigm_is_plus_one_on_task_and_minus_one_at_rest():
    t = np.array([-3.0, 0.0, 19.99, 20.0, 39.99, 40.0, 300.0, 319.0])
    np.testing.assert_array_equal(
        task.block_paradigm(t, 20, 20, 320), [-1, 1, 1, -1, -1, 1, -1, -1])
    assert len(task.block_starts(20, 20, 320)) == 8
    # a limited number of cycles, and a later onset
    np.testing.assert_array_equal(
        task.block_paradigm(np.array([5.0, 12.0, 45.0, 52.0, 85.0]), 10, 30, 320, onset=10,
                            n_cycles=2), [-1, 1, -1, 1, -1])
    with pytest.raises(ValueError, match='task_s'):
        task.block_starts(0, 20, 320)


@pytest.mark.parametrize('delay', [0.0, 0.6])
def test_bold_response_is_the_hrf_convolved_with_the_paradigm(delay):
    """Against a brute-force convolution on a fine grid."""
    dt = 0.005
    t = np.arange(-40, 140, dt)
    p = task.block_paradigm(t, 20, 20, 120)
    h = task.canonical_hrf(np.arange(0, 40, dt))
    # p is -1 before the grid starts: convolve p + 1 (zero there) and subtract 1
    brute = np.convolve(p + 1, h)[: len(t)] * dt - 1
    query = np.arange(0, 120, 0.37)
    np.testing.assert_allclose(
        task.bold_response(query, 20, 20, 120, delay=delay),
        np.interp(query - delay, t, brute), atol=3e-3)


def test_bold_response_runs_from_rest_to_task_levels():
    t = np.arange(0, 320, 0.0506)
    w = task.bold_response(t, 20, 20, 320)
    assert w[0] == pytest.approx(-1.0)  # at rest until the first block acts
    assert abs(w.mean()) < 0.06  # equal task and rest: about zero mean
    # 20 s is not long enough to settle at +-1: the canonical response overshoots
    assert 1.2 < w.max() < 1.35 and -1.35 < w.min() < -1.2
    # a block long enough to settle does
    assert task.bold_response(np.array([200.0]), 300, 20, 320)[0] == pytest.approx(1.0, abs=1e-3)
    # a delay moves the response later; the paradigm itself does not move
    late = task.bold_response(t, 20, 20, 320, delay=0.6)
    np.testing.assert_allclose(late[100:], np.interp(t[100:] - 0.6, t, w), atol=1e-3)
    assert task.MOTOR_DELAY_S == 0.6 and task.VISUAL_DELAY_S == 0.0
