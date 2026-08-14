import itertools

import numpy as np
import pytest

from lib.mask2epi import _center_out, _weighted_step, mask2epi_laminar, mask2epi_radial
from sampling.caipi_sample import caipi_sample


def _max_weighted_step(schedule: np.ndarray, Nshots: int, deltak) -> float:
    worst = 0.0
    for shot in range(Nshots):
        ys = schedule[shot, :, 0].astype(float)
        zs = schedule[shot, :, 1].astype(float)
        for e in range(len(ys) - 1):
            worst = max(worst, _weighted_step(ys[e + 1] - ys[e], zs[e + 1] - zs[e], deltak))
    return worst


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


def _assert_endpoints_are_half_extremes(schedule, mask, ETL, Nshots):
    """Each shot's echo train must start and end at that half's own
    farthest-from-center point, not merely somewhere unconstrained -- i.e.
    the two-pass ordering optimization must only reorder the *interior* of
    each half, never bury the true outer point mid-train (see
    _order_half_anchored)."""
    Ny, Nz = mask.shape
    cy, cz = Ny / 2, Nz / 2
    target = (ETL - 1) // 2
    for shot in range(Nshots):
        ys = schedule[shot, :, 0].astype(float)
        zs = schedule[shot, :, 1].astype(float)
        dist = np.sqrt((ys - cy) ** 2 + (zs - cz) ** 2)
        if target > 0:
            assert dist[0] == pytest.approx(dist[:target].max())
        if target < ETL - 1:
            assert dist[-1] == pytest.approx(dist[target + 1 :].max())


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
    _assert_endpoints_are_half_extremes(schedule, mask, ETL, Nshots)


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
    _assert_endpoints_are_half_extremes(schedule, mask, ETL, Nshots)


def test_mask2epi_radial_rejects_wrong_sample_count():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    with pytest.raises(AssertionError):
        mask2epi_radial(mask, ETL=2, Nshots=2)


_LAMINAR_TEST_ROWS = np.array([0, 2, 5, 6])
_LAMINAR_TEST_ROW_Z = [
    np.array([0, 1, 8]),
    np.array([2, 9]),
    np.array([0, 7]),
    np.array([3]),
]


def _row_directions_objective(rows, row_z, deltak, directions, objective):
    """Reference (non-DP) evaluation of either objective for a given
    direction assignment -- used to check _dp_row_directions/_optimize_row_
    directions against brute force."""
    from lib.mask2epi import _weighted_step

    steps = []
    prev_exit = None
    prev_row = None
    for i, d in enumerate(directions):
        z_seq = row_z[i] if d == 'F' else row_z[i][::-1]
        for e in range(len(z_seq) - 1):
            steps.append(_weighted_step(0, z_seq[e + 1] - z_seq[e], deltak))
        if prev_exit is not None:
            steps.append(_weighted_step(rows[i] - prev_row, z_seq[0] - prev_exit, deltak))
        prev_exit = z_seq[-1]
        prev_row = rows[i]
    return max(steps) if objective == 'max' else sum(steps)


@pytest.mark.parametrize('objective', ['sum', 'max'])
def test_mask2epi_laminar_dp_row_directions_match_bruteforce(objective):
    """_dp_row_directions must be exact for either objective: brute-force
    every 2^n_rows direction combination on a small hand-built single-shot
    mask (several rows of uneven length/offset, deliberately laid out so
    direction choice matters) and confirm the DP finds the true optimum,
    not just a locally-greedy one."""
    from lib.mask2epi import _dp_row_directions

    rows, row_z, deltak = _LAMINAR_TEST_ROWS, _LAMINAR_TEST_ROW_Z, (1.0, 1.0)

    brute_best = min(
        _row_directions_objective(rows, row_z, deltak, combo, objective)
        for combo in itertools.product('FB', repeat=len(rows))
    )
    dp_directions = _dp_row_directions(rows, row_z, deltak, objective=objective)
    achieved = _row_directions_objective(rows, row_z, deltak, dp_directions, objective)
    assert achieved == pytest.approx(brute_best)


def test_mask2epi_laminar_hillclimb_never_worse_than_pass1_alone():
    """Pass 2 (_hillclimb_row_directions_max) only accepts strictly
    max-improving moves, so the two-pass pipeline's worst-case step must
    never exceed pass 1's (min-sum) result on its own."""
    from lib.mask2epi import _dp_row_directions, _hillclimb_row_directions_max

    rows, row_z, deltak = _LAMINAR_TEST_ROWS, _LAMINAR_TEST_ROW_Z, (1.0, 1.0)

    pass1_directions = _dp_row_directions(rows, row_z, deltak, objective='sum')
    pass1_max = _row_directions_objective(rows, row_z, deltak, pass1_directions, 'max')

    pass2_directions = _hillclimb_row_directions_max(rows, row_z, deltak, pass1_directions)
    pass2_max = _row_directions_objective(rows, row_z, deltak, pass2_directions, 'max')

    assert pass2_max <= pass1_max + 1e-9


def _naive_radial_schedule(mask, ETL, Nshots):
    """Reproduce the pre-optimization mask2epi_radial ordering (plain
    projection sort + center splice, no 2-opt) as a baseline to compare
    against."""
    import math

    Ny, Nz = mask.shape
    cy, cz = Ny / 2, Nz / 2
    ys, zs = np.nonzero(mask)
    theta = np.arctan2(zs - cz, ys - cy)
    theta_folded = np.mod(theta, np.pi)
    angle_order = np.argsort(theta_folded, kind='stable')

    schedule = np.zeros((Nshots, ETL, 2), dtype=int)
    target = (ETL - 1) // 2
    for shot in range(Nshots):
        idx = angle_order[shot * ETL : (shot + 1) * ETL]
        shot_y, shot_z = ys[idx], zs[idx]
        th = theta[idx]
        axis_angle = 0.5 * math.atan2(np.sin(2 * th).mean(), np.cos(2 * th).mean())
        proj = (shot_y - cy) * math.cos(axis_angle) + (shot_z - cz) * math.sin(axis_angle)
        proj_order = np.argsort(proj, kind='stable')
        shot_y, shot_z = shot_y[proj_order], shot_z[proj_order]
        dist2 = (shot_y - cy) ** 2 + (shot_z - cz) ** 2
        center_idx = int(np.argmin(dist2))
        if center_idx != target:
            y_c, z_c = shot_y[center_idx], shot_z[center_idx]
            shot_y = np.insert(np.delete(shot_y, center_idx), target, y_c)
            shot_z = np.insert(np.delete(shot_z, center_idx), target, z_c)
        schedule[shot, :, 0] = shot_y
        schedule[shot, :, 1] = shot_z
    return schedule


def test_sum_optimized_order_matches_bruteforce_small():
    """_sum_optimized_order (nearest-neighbor + sum-2opt) should find the
    true minimum-total-length open path, anchored at point 0, on a small
    point set small enough to brute-force (6 points -> 5! = 120 orderings
    of the non-anchor points)."""
    from lib.mask2epi import _pairwise_weighted_dist, _sum_optimized_order

    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 20, size=(6, 2))
    deltak = (1.0, 1.0)
    D = _pairwise_weighted_dist(coords, deltak)

    def total_length(order):
        return sum(D[order[i], order[i + 1]] for i in range(len(order) - 1))

    brute_best = min(
        total_length([0] + list(perm))
        for perm in itertools.permutations(range(1, 6))
    )
    order = _sum_optimized_order(coords, deltak)
    assert order[0] == 0
    assert total_length(order) == pytest.approx(brute_best)


def test_sum_optimized_order_fix_end_matches_bruteforce_small():
    """With fix_end=True, both point 0 and the last point must stay fixed
    -- brute-force over only the interior permutations (4 interior points
    of 6 total -> 4! = 24 orderings) and confirm the optimizer finds the
    true minimum among paths respecting both fixed endpoints."""
    from lib.mask2epi import _pairwise_weighted_dist, _sum_optimized_order

    rng = np.random.default_rng(4)
    coords = rng.uniform(0, 20, size=(6, 2))
    deltak = (1.0, 1.0)
    D = _pairwise_weighted_dist(coords, deltak)

    def total_length(order):
        return sum(D[order[i], order[i + 1]] for i in range(len(order) - 1))

    brute_best = min(
        total_length([0] + list(perm) + [5])
        for perm in itertools.permutations(range(1, 5))
    )
    order = _sum_optimized_order(coords, deltak, fix_end=True)
    assert order[0] == 0 and order[-1] == 5
    assert total_length(order) == pytest.approx(brute_best)


def test_bottleneck_2opt_never_worse_than_sum_optimized_order_alone():
    """Pass 2 (_bottleneck_2opt_order) only accepts strictly max-improving
    moves, so running it after pass 1 must never leave the worst-case step
    larger than pass 1's (min-sum) tour on its own."""
    from lib.mask2epi import (
        _bottleneck_2opt_order,
        _pairwise_weighted_dist,
        _sum_optimized_order,
    )

    rng = np.random.default_rng(3)
    deltak = (1.0, 1.3)
    for trial in range(20):
        coords = rng.uniform(0, 30, size=(15, 2))
        D = _pairwise_weighted_dist(coords, deltak)

        pass1_order = _sum_optimized_order(coords, deltak)
        pass1_max = D[pass1_order[:-1], pass1_order[1:]].max()

        pass1_coords = coords[pass1_order]
        pass2_order = _bottleneck_2opt_order(pass1_coords, deltak)
        pass2_coords = pass1_coords[pass2_order]
        D2 = _pairwise_weighted_dist(pass2_coords, deltak)
        pass2_max = np.array(
            [D2[i, i + 1] for i in range(len(pass2_coords) - 1)]
        ).max()

        assert pass2_max <= pass1_max + 1e-9


def test_mask2epi_radial_2opt_no_worse_than_plain_projection_sort():
    Ny, Nz = 24, 16
    R = 4
    ETL = 12
    mask = caipi_sample([Ny, Nz], R).astype(bool)
    Nshots = int(mask.sum() // ETL)
    assert mask.sum() == Nshots * ETL
    deltak = (1.0, 1.3)  # anisotropic weighting, matching real FOV usage

    schedule, _ = mask2epi_radial(mask, ETL, Nshots, deltak)
    naive_schedule = _naive_radial_schedule(mask, ETL, Nshots)

    optimized_max = _max_weighted_step(schedule, Nshots, deltak)
    naive_max = _max_weighted_step(naive_schedule, Nshots, deltak)
    assert optimized_max <= naive_max + 1e-9


def _has_any_crossing(ys, zs):
    from lib.mask2epi import _segments_cross

    n = len(ys)
    pts = list(zip(ys, zs))
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if i == 0 and j == n - 2:
                continue
            if _segments_cross(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                return True
    return False


def test_euclidean_uncross_refine_fixes_seam_crossing_across_pinned_center():
    """Regression test for a real bug: a crossing straddling a *pinned
    interior* position (the k-space-center sample) can be an exact tie in
    weighted-Euclidean length for the specific swap that fixes it (the two
    center-adjacent edges just exchange values), so stage 1's strict-
    improvement rule alone never takes it. Stage 2 (direct geometric
    crossing detection) must still remove it. Reproduces the shape of the
    crossing found by eyeballing a real run's --plot output: an interior
    segment on one side of the pinned center crossing the very next
    segment leading away from center on the other side."""
    from lib.mask2epi import _euclidean_uncross_refine, _segments_cross

    coords = np.array(
        [
            [0, 0],  # 0: pinned start
            [1, 5],  # 1: free
            [1, -5],  # 2: free
            [2, 0],  # 3: pinned center
            [0, 3],  # 4: free
            [3, 3],  # 5: pinned end
        ],
        dtype=float,
    )
    deltak = (1.0, 1.0)
    order = np.arange(6)
    pts = coords[order]
    assert _segments_cross(pts[1], pts[2], pts[3], pts[4])

    new_order = _euclidean_uncross_refine(
        coords, deltak, order, max_allowed=100.0, pinned=frozenset({0, 3, 5})
    )
    assert new_order[0] == 0 and new_order[3] == 3 and new_order[5] == 5
    assert not _has_any_crossing(coords[new_order, 0], coords[new_order, 1])


def test_mask2epi_radial_full_shot_crossings_no_worse_than_per_half_only():
    """The whole-shot uncrossing pass (pass 3) must never leave a shot with
    more self-crossings than the per-half-only (pass 1-2) result had, and
    must never regress the achieved worst-case weighted step -- checked
    across several random masks/seeds since crossings are a real-data
    phenomenon a single small synthetic case can't fully represent."""
    from lib.mask2epi import _order_half_anchored

    def per_half_only_schedule(mask, ETL, Nshots, deltak):
        import math

        Ny, Nz = mask.shape
        cy, cz = Ny / 2, Nz / 2
        ys, zs = np.nonzero(mask)
        theta = np.arctan2(zs - cz, ys - cy)
        theta_folded = np.mod(theta, np.pi)
        angle_order = np.argsort(theta_folded, kind='stable')
        schedule = np.zeros((Nshots, ETL, 2), dtype=int)
        target = (ETL - 1) // 2
        for shot in range(Nshots):
            idx = angle_order[shot * ETL : (shot + 1) * ETL]
            shot_y, shot_z = ys[idx], zs[idx]
            th = theta[idx]
            axis_angle = 0.5 * math.atan2(np.sin(2 * th).mean(), np.cos(2 * th).mean())
            proj = (shot_y - cy) * math.cos(axis_angle) + (shot_z - cz) * math.sin(axis_angle)
            proj_order = np.argsort(proj, kind='stable')
            shot_y, shot_z = shot_y[proj_order], shot_z[proj_order]
            dist2 = (shot_y - cy) ** 2 + (shot_z - cz) ** 2
            center_idx = int(np.argmin(dist2))
            center_coord = np.array([shot_y[center_idx], shot_z[center_idx]], dtype=float)
            rest_idx = np.delete(np.arange(ETL), center_idx)
            before_idx, after_idx = rest_idx[:target], rest_idx[target:]
            before_coords = np.column_stack(
                [shot_y[before_idx], shot_z[before_idx]]
            ).astype(float)
            after_coords = np.column_stack([shot_y[after_idx], shot_z[after_idx]]).astype(float)
            before_path = _order_half_anchored(
                center_coord, before_coords, cy, cz, deltak, near_first=False
            )
            after_path = _order_half_anchored(
                center_coord, after_coords, cy, cz, deltak, near_first=True
            )
            schedule[shot, :target, :] = before_path
            schedule[shot, target, :] = center_coord
            schedule[shot, target + 1 :, :] = after_path
        return schedule

    Ny, Nz, ETL, Nshots = 40, 24, 20, 6
    deltak = (1.0, 2.3)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        mask = np.zeros((Ny, Nz), dtype=bool)
        idx = rng.choice(Ny * Nz, size=Nshots * ETL, replace=False)
        mask.flat[idx] = True

        per_half = per_half_only_schedule(mask, ETL, Nshots, deltak)
        full, _ = mask2epi_radial(mask, ETL, Nshots, deltak)

        per_half_crossings = sum(
            _has_any_crossing(per_half[s, :, 0], per_half[s, :, 1]) for s in range(Nshots)
        )
        full_crossings = sum(
            _has_any_crossing(full[s, :, 0], full[s, :, 1]) for s in range(Nshots)
        )
        assert full_crossings <= per_half_crossings

        per_half_max = _max_weighted_step(per_half, Nshots, deltak)
        full_max = _max_weighted_step(full, Nshots, deltak)
        assert full_max <= per_half_max + 1e-9
