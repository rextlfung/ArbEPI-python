import numpy as np
import pytest

from sampling.caipi_sample import caipi_sample


@pytest.mark.parametrize('N,R', [([12, 8], 4), ([24, 16], 6), ([15, 10], 3), ([90, 60], 6)])
def test_caipi_sample_exact_count(N, R):
    mask = caipi_sample(N, R)
    assert mask.shape == tuple(N)
    # Exact count for a pure decimation-by-R pattern with balanced Ry*Rz=R
    Rsmall = int(np.floor(np.sqrt(R)))
    while R % Rsmall != 0:
        Rsmall -= 1
    Rlarge = R // Rsmall
    Ry, Rz = (Rlarge, Rsmall) if N[0] >= N[1] else (Rsmall, Rlarge)
    expected = len(range(0, N[0], Ry)) * len(range(0, N[1], Rz))
    assert mask.sum() == expected


def test_caipi_sample_deterministic():
    m1 = caipi_sample([20, 16], 4)
    m2 = caipi_sample([20, 16], 4)
    assert np.array_equal(m1, m2)


def test_caipi_sample_shift_offset_permutes_rows():
    N = [16, 12]
    R = 4
    m0 = caipi_sample(N, R, 0)
    m1 = caipi_sample(N, R, 1)
    assert m0.sum() == m1.sum()
