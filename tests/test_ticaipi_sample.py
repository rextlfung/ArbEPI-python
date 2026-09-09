import numpy as np
import pytest

from sampling.ticaipi_sample import ticaipi_sample


@pytest.mark.parametrize('N,R', [([12, 8], 4), ([24, 16], 6), ([90, 60], 6)])
def test_ticaipi_full_coverage_over_R_frames(N, R):
    union = np.zeros(N, dtype=bool)
    count = None
    for frame_idx in range(R):
        m = ticaipi_sample(N, R, frame_idx).astype(bool)
        if count is None:
            count = m.sum()
        assert m.sum() == count  # same density every frame
        union |= m
    assert union.all()


def test_ticaipi_cycles_with_period_R():
    N = [16, 12]
    R = 4
    m0 = ticaipi_sample(N, R, 0)
    m_period = ticaipi_sample(N, R, R)
    assert np.array_equal(m0, m_period)


def test_ticaipi_sample_does_not_raise_at_shipped_default_dims():
    # Regression test for docs/review-findings.md item 147: at the repo's
    # own shipped default (Ny, Nz, R) = (240, 45, 9), balanced_factors used
    # to pick a non-dividing (Ry, Rz) = (9, 1), tripping this module's own
    # divisibility guard (added by item 103). balanced_factors's fix
    # (restricting to dividing pairs) should keep this working.
    N, R = [240, 45], 9
    for frame_idx in range(3):
        ticaipi_sample(N, R, frame_idx)
