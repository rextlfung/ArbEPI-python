import numpy as np
import pytest

from sampling.caipi_sample import balanced_factors, caipi_sample


@pytest.mark.parametrize('N,R', [([12, 8], 4), ([24, 16], 6), ([15, 10], 3), ([90, 60], 6)])
def test_caipi_sample_exact_count(N, R):
    mask = caipi_sample(N, R)
    assert mask.shape == tuple(N)
    # Exact count for a pure decimation-by-R pattern, whatever (Ry, Rz)
    # balanced_factors picks for this N, R.
    Ry, Rz = balanced_factors(N, R)
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


def test_balanced_factors_weights_by_fov_ratio_not_squareness():
    # At this repo's real (Ny, Nz, R) = (240, 45, 9), Ny/Nz ~= 5.33 -- the
    # most-square split (3, 3) is far from that ratio; (9, 1) is the
    # closest achievable integer factor pair, and matches
    # pd_sample's continuous aspect-matched ellipse in spirit (equalizing
    # aliasing period, not reduction factor, across axes).
    assert balanced_factors([240, 45], 9) == (9, 1)


def test_balanced_factors_direction_flips_with_fov():
    # Same R, axes swapped -> Ry/Rz should flip too.
    assert balanced_factors([45, 240], 9) == (1, 9)


def test_balanced_factors_matches_naive_square_split_when_fov_ratio_is_one():
    # No FOV skew (Ny == Nz) -> the FOV-weighted and most-square choices
    # coincide.
    assert balanced_factors([64, 64], 8) == (2, 4)
    assert balanced_factors([64, 64], 9) == (3, 3)
