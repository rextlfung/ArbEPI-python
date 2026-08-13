import numpy as np
import pytest

from lib.mask2epi import _center_out, mask2epi_laminar, mask2epi_radial
from sampling.caipi_sample import caipi_sample


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
def test_center_out_is_permutation(n):
    out = _center_out(n)
    assert sorted(out.tolist()) == list(range(n))


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6, 7, 8])
def test_center_out_starts_near_center(n):
    out = _center_out(n)
    center = (n - 1) / 2
    # The first visited index should be within 1 of the true center (exact
    # for even n; off by one is an inherent artifact of the ported MATLAB
    # fftshift-based algorithm for odd n, verified against a hand-traced
    # MATLAB reference for n=5 and n=6).
    distances = np.abs(np.arange(n) - center)
    assert distances[out[0]] <= distances.min() + 1


def _check_common_invariants(schedule, parts, mask, ETL, Nshots):
    """Invariants shared by every mask2epi_* variant: complete/exclusive
    partitioning, valid coordinates, and each sample scheduled exactly
    once. Ordering-specific invariants (e.g. laminar's non-decreasing ky,
    radial's center-crossing echo index) are checked separately."""
    Ny, Nz = mask.shape

    # Every sampled point is assigned to exactly one shot.
    assert set(np.unique(parts[mask])) == set(range(1, Nshots + 1))
    assert (parts[~mask] == 0).all()

    for shot in range(Nshots):
        ys = schedule[shot, :, 0]
        zs = schedule[shot, :, 1]

        # All coordinates in range.
        assert (ys >= 0).all() and (ys < Ny).all()
        assert (zs >= 0).all() and (zs < Nz).all()

        # Every scheduled point was actually sampled and belongs to this shot.
        for e in range(ETL):
            assert mask[ys[e], zs[e]]
            assert parts[ys[e], zs[e]] == shot + 1

    # Every sample in the mask appears exactly once across all schedules.
    scheduled_points = {
        (int(schedule[s, e, 0]), int(schedule[s, e, 1]))
        for s in range(Nshots)
        for e in range(ETL)
    }
    mask_points = {(int(y), int(z)) for y, z in zip(*np.nonzero(mask))}
    assert scheduled_points == mask_points
    assert len(scheduled_points) == Nshots * ETL


def test_mask2epi_laminar_small_synthetic_mask():
    Ny, Nz = 8, 6
    ETL = 4
    Nshots = 2
    rng = np.random.default_rng(0)
    mask = np.zeros((Ny, Nz), dtype=bool)
    idx = rng.choice(Ny * Nz, size=Nshots * ETL, replace=False)
    mask.flat[idx] = True

    schedule, parts = mask2epi_laminar(mask, ETL, Nshots)
    _check_common_invariants(schedule, parts, mask, ETL, Nshots)
    for shot in range(Nshots):
        assert np.all(np.diff(schedule[shot, :, 0]) >= 0)


def test_mask2epi_laminar_caipi_mask():
    Ny, Nz = 24, 16
    R = 4
    ETL = 12
    mask = caipi_sample([Ny, Nz], R).astype(bool)
    Nshots = mask.sum() // ETL
    assert mask.sum() == Nshots * ETL

    schedule, parts = mask2epi_laminar(mask, ETL, int(Nshots))
    _check_common_invariants(schedule, parts, mask, ETL, int(Nshots))
    for shot in range(int(Nshots)):
        assert np.all(np.diff(schedule[shot, :, 0]) >= 0)


def test_mask2epi_laminar_rejects_wrong_sample_count():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    with pytest.raises(AssertionError):
        mask2epi_laminar(mask, ETL=2, Nshots=2)


def _assert_center_crossing_at_target(schedule, mask, ETL, Nshots):
    """For each shot, the scheduled sample closest to k-space center
    (measured independently of mask2epi_radial's own axis-projection
    logic, as a plain Euclidean distance) must land at the echo index
    calc_te_tr_delays.py's min_te treats as the nominal TE echo: min_te
    places TE at (ETL/2 - 0.5)*gro after the prep, and echo e (0-based)
    has its ADC center at (e + 0.5)*gro into the train, so e = ETL/2 - 1,
    i.e. (ETL - 1) // 2 (for odd ETL this falls between two echoes; either
    neighbor is off by half an echo spacing, so mask2epi_radial's own
    choice of which one is authoritative here, not re-derived)."""
    Ny, Nz = mask.shape
    cy, cz = Ny / 2, Nz / 2
    target = (ETL - 1) // 2
    for shot in range(Nshots):
        ys = schedule[shot, :, 0].astype(float)
        zs = schedule[shot, :, 1].astype(float)
        dist2 = (ys - cy) ** 2 + (zs - cz) ** 2
        # Ties (two samples equally close to center) are possible on
        # regular/symmetric masks; any tied-nearest point at `target` is
        # equally valid, so check membership in the minimizing set rather
        # than requiring the specific first-occurrence np.argmin index.
        assert dist2[target] == dist2.min()


def test_mask2epi_radial_small_synthetic_mask():
    Ny, Nz = 16, 12
    ETL = 8
    Nshots = 3
    rng = np.random.default_rng(0)
    mask = np.zeros((Ny, Nz), dtype=bool)
    idx = rng.choice(Ny * Nz, size=Nshots * ETL, replace=False)
    mask.flat[idx] = True

    schedule, parts = mask2epi_radial(mask, ETL, Nshots)
    _check_common_invariants(schedule, parts, mask, ETL, Nshots)
    _assert_center_crossing_at_target(schedule, mask, ETL, Nshots)


def test_mask2epi_radial_caipi_mask():
    Ny, Nz = 24, 16
    R = 4
    ETL = 12
    mask = caipi_sample([Ny, Nz], R).astype(bool)
    Nshots = int(mask.sum() // ETL)
    assert mask.sum() == Nshots * ETL

    schedule, parts = mask2epi_radial(mask, ETL, Nshots)
    _check_common_invariants(schedule, parts, mask, ETL, Nshots)
    _assert_center_crossing_at_target(schedule, mask, ETL, Nshots)


def test_mask2epi_radial_rejects_wrong_sample_count():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    with pytest.raises(AssertionError):
        mask2epi_radial(mask, ETL=2, Nshots=2)
