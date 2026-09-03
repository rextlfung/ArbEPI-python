"""mask2epi_laminar (ported from ../ArbEPI/lib/mask2epi.m) and
mask2epi_radial (this repo's own addition, no MATLAB counterpart) are two
interchangeable ways to partition a 2D (ky, kz) sampling mask into
`Nshots` EPI trajectories of length `ETL` each.

mask2epi_laminar: samples near ky = 0 are spread center-out across shots;
ky is non-decreasing within each echo train, and the kz sweep direction per
ky row is chosen by the two-pass ordering optimization described below, not
just greedily — each shot occupies its own non-overlapping band of the
visiting order, hence "laminar."

mask2epi_radial: each shot instead sweeps through k-space center as a
single spoke (an opposite pair of angular wedges about the mask's center),
so every shot samples near k-space center rather than only whichever shots
happen to include the central ky row. Shots are sequenced in a
golden-angle-like order (`_golden_angle_shot_order`), so that any prefix of
consecutively-acquired shots -- not just the complete set -- covers the
angular range close to uniformly. See its own docstring for the
partitioning/ordering details, including why ky non-decreasing is
deliberately given up here.

Both return `schedule`/`parts` in the same shapes and conventions; which to
use is a call-site decision, not something enforced in this module.

Pass 2 ordering optimization (both variants): given a shot's set of sampled
points, in what order should the echo train visit them? Two objectives are
both worth optimizing here, and neither one alone is enough:

- Total path length (ordinary, min-sum TSP) matters because a long,
  wandering path means many *unnecessarily large* individual steps spread
  throughout the train, not just a single worst-case outlier -- a smooth
  low-total-length tour tends to avoid these even where the bottleneck
  objective below wouldn't flag them.
- The single *worst-case* step matters separately, and more, because
  `make_readout_grads.py` sizes the y and z unit blips independently, each
  from the largest step seen on its own axis across the whole sequence --
  gradient/PNS feasibility is set by that one worst step, not the total.
  This is the bottleneck (min-max) Hamiltonian-path problem, which is a
  genuinely different (and, like ordinary TSP, NP-hard) objective: a
  min-sum-optimal tour can still have one large step that a min-sum
  algorithm has no reason to avoid, since one large edge barely affects a
  sum over dozens of edges.

So ordering is done in two passes: pass 1 constructs a tour minimizing
total weighted path length (`_sum_optimized_order` for radial,
`_dp_row_directions(..., objective='sum')` for laminar's row-direction
choice); pass 2 then locally refines that tour to reduce its worst single
step (`_bottleneck_2opt_order` for radial, `_hillclimb_row_directions_max`
for laminar), using pass 1's result as its starting point rather than
optimizing from scratch. Both passes use the same physically-weighted step
metric `max(|dy|*deltak[0], |dz|*deltak[1])` (Chebyshev distance in
k-space units, matching how the two blip axes are independently sized) --
raw index-distance would not reflect that a 1-index kz step and a 1-index
ky step can cost very different amounts of gradient area when the FOV is
anisotropic across (y, z).

Index convention: `schedule`'s (iy, iz) values are 0-based, matching
Python/numpy array indices into `mask`. `parts` keeps MATLAB's convention of
1-based shot labels with 0 meaning "unsampled" — it's a label/sentinel
array, not a coordinate used in downstream gradient-scaling arithmetic, so
there's no reason to renumber it. (Conversion of `schedule`'s indices to
1-based, for saving into `scan_info.mat`'s `schedules` array in a form
existing MATLAB-side reconstruction code can read unchanged, happens once at
the save boundary in sequences/ArbEPI.py — not here.)
"""

import bisect
import math
from typing import Sequence

import numpy as np


def max_blip_steps(schedules: np.ndarray) -> tuple[float, float]:
    """Largest consecutive-sample ky/kz step size across every frame/shot in
    `schedules` (Nframes x Nshots x ETL x 2), for sizing unit blips in
    make_readout_grads.py.

    ETL == 1 has no consecutive samples within a shot to diff (the step
    along that axis is empty), so np.diff(..., axis=2) would otherwise hand
    np.max a zero-size array and raise -- handled explicitly here rather
    than at each of this function's three call sites (sequences/ArbEPI.py,
    EPIcal.py, noise.py). No steps within a shot means no blips are needed.
    """
    if schedules.shape[2] <= 1:
        return 0.0, 0.0
    max_ky_step = np.max(np.abs(np.diff(schedules[..., 0], axis=2)))
    max_kz_step = np.max(np.abs(np.diff(schedules[..., 1], axis=2)))
    return max_ky_step, max_kz_step


def _center_out(n: int) -> np.ndarray:
    """0-based center-out visiting order of range(n).

    Mirrors MATLAB's `center_out` helper (flip(fftshift(1:N)) interleaved
    from both ends) applied to an identity sequence — since center_out only
    permutes positions, applying it to a 0-based identity array yields
    exactly (MATLAB's 1-based result - 1), elementwise, in the same order.

    Note: MATLAB's `fftshift` rotates left by `floor(N/2)`; numpy's
    `np.fft.fftshift` rotates right by `N // 2`. These coincide for even N
    but diverge for odd N, so `np.fft.fftshift` cannot be used here directly
    — the left rotation is reproduced explicitly via `np.roll(..., -(n//2))`.
    """
    fftshift_matlab = np.roll(np.arange(n), -(n // 2))
    tmp = np.flip(fftshift_matlab)
    out = np.empty(n, dtype=int)
    j = 0
    i = 0
    while i < math.ceil(n / 2):
        out[j] = tmp[i]
        j += 1
        if j >= n:
            break
        out[j] = tmp[-(i + 1)]
        j += 1
        i += 1
    return out


def _weighted_step(dy: float, dz: float, deltak: Sequence[float]) -> float:
    """Physically-weighted bottleneck step metric — see module docstring."""
    return max(abs(dy) * deltak[0], abs(dz) * deltak[1])


def _pairwise_weighted_dist(coords: np.ndarray, deltak: Sequence[float]) -> np.ndarray:
    """(m, m) matrix of `_weighted_step` between every pair of `coords`
    (m, 2) rows (y, z)."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.maximum(np.abs(diff[..., 0]) * deltak[0], np.abs(diff[..., 1]) * deltak[1])


def _mst_bottleneck(D: np.ndarray) -> float:
    """Minimum bottleneck spanning tree edge weight for distance matrix D --
    a valid lower bound on any Hamiltonian path's worst-case step (a
    Hamiltonian path is itself a spanning tree), via a standard Prim's
    construction: the largest edge Prim's is ever forced to add is
    provably minimal over all spanning trees."""
    m = D.shape[0]
    if m <= 1:
        return 0.0
    in_tree = np.zeros(m, dtype=bool)
    in_tree[0] = True
    min_edge = D[0].copy()
    bottleneck = 0.0
    for _ in range(m - 1):
        candidate = np.where(in_tree, np.inf, min_edge)
        j = int(np.argmin(candidate))
        bottleneck = max(bottleneck, candidate[j])
        in_tree[j] = True
        min_edge = np.minimum(min_edge, D[j])
    return bottleneck


def _nearest_neighbor_order(D: np.ndarray, fix_end: bool = False) -> np.ndarray:
    """Greedy nearest-neighbor open-path construction: starting from a
    fixed anchor at index 0, repeatedly visit whichever unvisited point is
    closest under distance matrix D. Pass-1 (min-sum TSP) initial tour --
    a standard, cheap starting point for `_sum_2opt_refine`.

    If `fix_end`, index `m - 1` is also treated as fixed -- excluded from
    greedy selection and appended last, rather than being picked whenever
    it happens to be nearest. Used when both path endpoints are pinned to
    specific points (see `mask2epi_radial`'s outer-anchor / center-anchor
    endpoints) and only the interior ordering should be optimized."""
    m = D.shape[0]
    last = m - 1 if fix_end and m > 1 else None
    visited = np.zeros(m, dtype=bool)
    visited[0] = True
    if last is not None:
        visited[last] = True
    order = [0]
    cur = 0
    n_free = m - 1 - (1 if last is not None else 0)
    for _ in range(n_free):
        dists = np.where(visited, np.inf, D[cur])
        nxt = int(np.argmin(dists))
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    if last is not None:
        order.append(last)
    return np.array(order)


def _sum_2opt_refine(
    D: np.ndarray, order: np.ndarray, max_passes: int = 40, fix_end: bool = False
) -> np.ndarray:
    """Best-improvement 2-opt local search minimizing *total* path length
    (ordinary min-sum TSP objective), keeping `order[0]` (and, if
    `fix_end`, `order[-1]`) fixed. Unlike the bottleneck objective, min-sum
    2-opt has no plateau problem -- almost every reversal changes the
    total by a nonzero amount, so plain steepest-descent on the delta
    converges cleanly."""
    order = order.copy()
    m = len(order)
    if m <= 2:
        return order
    j_upper = m - 1 if fix_end else m

    for _ in range(max_passes):
        best_delta = -1e-9
        best_move = None
        for i in range(1, m - 1):
            for j in range(i + 1, j_upper):
                removed = D[order[i - 1], order[i]]
                added = D[order[i - 1], order[j]]
                if j < m - 1:
                    removed += D[order[j], order[j + 1]]
                    added += D[order[i], order[j + 1]]
                delta = added - removed
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
        if best_move is None:
            break
        i, j = best_move
        order[i : j + 1] = order[i : j + 1][::-1]

    return order


def _sum_optimized_order(
    coords: np.ndarray, deltak: Sequence[float], max_passes: int = 40, fix_end: bool = False
) -> np.ndarray:
    """Pass 1: an open-path ordering of `coords` (m, 2), anchored at
    `coords[0]` (and, if `fix_end`, also at `coords[-1]`), approximately
    minimizing total weighted path length -- ordinary (min-sum) TSP, via
    nearest-neighbor construction + 2-opt refinement. Meant to be handed to
    `_bottleneck_2opt_order` (pass 2) as a warm start, not used as a final
    answer on its own -- see module docstring for why both objectives
    matter."""
    m = coords.shape[0]
    if m <= 2:
        return np.arange(m)
    D = _pairwise_weighted_dist(coords, deltak)
    order = _nearest_neighbor_order(D, fix_end=fix_end)
    return _sum_2opt_refine(D, order, max_passes, fix_end=fix_end)


def _bottleneck_2opt_order(
    coords: np.ndarray, deltak: Sequence[float], max_passes: int = 40, fix_end: bool = False
) -> np.ndarray:
    """Pass 2: open-path (not cycle) ordering of `coords` (m, 2) that
    approximately minimizes the worst-case single-step `_weighted_step`
    between consecutive points -- the bottleneck-TSP-path objective, not
    ordinary (min-sum) TSP. `coords[0]` is treated as a fixed anchor
    (always stays first in the returned order); if `fix_end`, `coords[-1]`
    is likewise fixed last. Every other row is free to be reordered.
    `coords` is expected to already be pass-1 (min-sum) ordered (e.g. via
    `_sum_optimized_order`, with the same `fix_end`) -- this function
    refines that starting tour rather than building one from scratch, so a
    smoother input tour gives it a better/faster-converging starting point.

    Best-improvement 2-opt: each pass evaluates every segment reversal
    (i, j) and applies the single one that most reduces the path's worst
    step, stopping when no reversal helps or `max_passes` is hit. A
    reversal only changes the two edges at its boundaries (all interior
    edges of the reversed segment are unchanged, since distance is
    symmetric), so the new worst step after a candidate move is
    `max(worst step over all *other* edges, the two new boundary edges)` --
    computed in O(1) per candidate via the top-3 largest edges (at most 2
    edges are ever excluded by a single move, so the third-largest edge is
    always still present), making a full O(m^2) pass over all candidate
    reversals cheap for the per-shot sizes this repo uses (m ~ ETL/2).

    Terminates early once the current worst step matches the minimum
    bottleneck spanning tree bound (`_mst_bottleneck`) -- at that point the
    result is provably optimal, since no Hamiltonian path can beat its own
    spanning tree's bottleneck.
    """
    m = coords.shape[0]
    order = np.arange(m)
    if m <= 2:
        return order

    D = _pairwise_weighted_dist(coords, deltak)
    bound = _mst_bottleneck(D)
    j_upper = m - 1 if fix_end else m

    for _ in range(max_passes):
        edges = D[order[:-1], order[1:]]
        current_max = edges.max()
        if current_max <= bound + 1e-9:
            break

        k = len(edges)
        top_idx = np.argsort(edges)[::-1][: min(3, k)]
        top_vals = edges[top_idx]
        top_idx = list(top_idx) + [-1] * (3 - len(top_idx))
        top_vals = list(top_vals) + [-np.inf] * (3 - len(top_vals))

        def max_excl(p: int, q: int, top_idx=top_idx, top_vals=top_vals) -> float:
            # top_idx/top_vals bound as default args (not read from the
            # enclosing closure) so this doesn't silently start reading a
            # later pass's rebound values if ever deferred past this pass
            # (ruff B023 -- currently safe since every call happens within
            # the same pass that defines it, but a real footgun otherwise).
            excl = {p, q}
            for idx, val in zip(top_idx, top_vals):
                if idx not in excl:
                    return val
            return -np.inf

        best_max = current_max
        best_move = None
        for i in range(1, m - 1):
            # j starts at i + 1, not i: a single-element "reversal" (j == i)
            # is a no-op (order[i:i+1][::-1] == order[i:i+1]) that can never
            # change the candidate max, so evaluating it wastes O(m)
            # candidates per pass for no benefit.
            for j in range(i + 1, j_upper):
                new_e1 = D[order[i - 1], order[j]]
                if j < m - 1:
                    new_e2 = D[order[i], order[j + 1]]
                    cand = max(max_excl(i - 1, j), new_e1, new_e2)
                else:
                    cand = max(max_excl(i - 1, i - 1), new_e1)
                if cand < best_max:
                    best_max = cand
                    best_move = (i, j)

        if best_move is None:
            break
        i, j = best_move
        order[i : j + 1] = order[i : j + 1][::-1]

    return order


def _segments_cross(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """True if open segments p1-p2 and p3-p4 properly cross (share no
    endpoint, standard orientation test)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
    d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def _crossing_matrix(pts: np.ndarray) -> np.ndarray:
    """(n_seg, n_seg) bool matrix (n_seg = m-1): entry [i, j] is True iff
    segments i and j of the open path `pts` visits properly cross AND
    j >= i + 2 (the same non-adjacency exclusion _count_crossings'
    docstring explains -- no special case for the first/last segment pair,
    since that exclusion is only valid for a *closed* tour). Returns a
    (0, 0) array for m < 4 (no possible crossing).

    Vectorized: a fixed O(m^2) all-pairs orientation test, broadcasting
    the four `ccw` evaluations _segments_cross uses rather than looping in
    Python -- shared core for _count_crossings (sum) and
    _euclidean_uncross_refine's first-crossing search (argmax), both of
    which used to call _segments_cross directly in a Python double loop:
    measured 289k such calls (0.47s/frame) at this repo's real hot-loop
    scale, with the crossing-search loop (not _count_crossings, whose own
    call count is negligible -- ~30 calls/frame) the actual dominant
    caller (see docs/review-findings.md item 57). Verified equivalent to
    the scalar loop on 5000+ random trials (small-integer grids with
    shared endpoints/collinear points, and continuous coordinates up to
    m=80)."""
    m = len(pts)
    if m < 4:
        return np.zeros((0, 0), dtype=bool)
    seg_a, seg_b = pts[:-1], pts[1:]  # (n_seg, 2) each
    n_seg = len(seg_a)

    def ccw(a, b, c):
        return (c[..., 1] - a[..., 1]) * (b[..., 0] - a[..., 0]) - (b[..., 1] - a[..., 1]) * (
            c[..., 0] - a[..., 0]
        )

    p1, p2 = seg_a[:, None, :], seg_b[:, None, :]  # broadcast over j (axis 1)
    p3, p4 = seg_a[None, :, :], seg_b[None, :, :]  # broadcast over i (axis 0)
    d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
    d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
    crosses = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))

    i_idx = np.arange(n_seg)[:, None]
    j_idx = np.arange(n_seg)[None, :]
    return crosses & (j_idx >= i_idx + 2)


def _count_crossings(pts: np.ndarray) -> int:
    """Number of properly-crossing (non-adjacent, non-endpoint-sharing)
    segment pairs along the open path `pts` visits in order -- see
    _crossing_matrix's docstring for the exclusion rule and the
    vectorization this relies on."""
    return int(np.count_nonzero(_crossing_matrix(pts)))


def _find_first_crossing(pts: np.ndarray) -> tuple[int, int] | None:
    """(i, j) of the first crossing pair in row-major (i ascending, then j
    ascending) order -- the same pair a `for i: for j: ... break` scalar
    search would find first -- or None if the path has no crossing.
    Vectorized via _crossing_matrix (see its docstring); `np.argmax` on a
    boolean array returns the index of the first True entry, which for a
    C-order-flattened (n_seg, n_seg) matrix is exactly the first (i, j) in
    that same row-major order."""
    crosses = _crossing_matrix(pts)
    if not crosses.any():
        return None
    n_seg = crosses.shape[1]
    flat = int(np.argmax(crosses.reshape(-1)))
    return divmod(flat, n_seg)


def _euclidean_uncross_refine(
    coords: np.ndarray,
    deltak: Sequence[float],
    order: np.ndarray,
    max_allowed: float,
    max_passes: int = 40,
    pinned: frozenset = frozenset(),
) -> np.ndarray:
    """Pass 3: a tie-breaking cleanup after pass 2 (`_bottleneck_2opt_order`).

    The weighted-Chebyshev metric passes 1-2 optimize (`max(|dy|*deltak[0],
    |dz|*deltak[1])`, matching how the hardware actually sizes blips) has a
    square unit ball, not a strictly convex one -- unlike Euclidean
    distance, its triangle inequality is very often an *equality* rather
    than a strict one. The classical argument that any two crossing edges
    in a tour can always be uncrossed to strictly reduce length relies on
    strict convexity, so it can fail under this metric: a move that would
    visibly remove a self-crossing can have exactly zero effect on both
    the weighted sum and the weighted max, and passes 1-2's strict-
    improvement-only acceptance never takes a zero-delta move, leaving the
    crossing in the final path even though nothing hardware-relevant
    favors keeping it.

    `pinned` is the set of positions in `order` that must never move --
    both endpoints and (when this is called on a whole shot rather than
    one half, see `mask2epi_radial`) the interior center-sample position.
    A reversal is only considered if no pinned position falls inside its
    span (reversing would displace it); a swap is only considered if
    neither of its two positions is itself pinned (the values at other,
    unpinned positions it references as neighbors are untouched).

    Stage 1 minimizes *weighted* Euclidean path length -- Euclidean
    distance computed in the same deltak-scaled coordinates as the
    bottleneck metric (`sqrt((dy*deltak[0])**2 + (dz*deltak[1])**2)`), not
    raw index distance. This still gives the strict convexity the
    uncrossing argument needs, while staying physically consistent with
    passes 1-2: raw (unweighted) index distance was tried first and does
    remove crossings, but since it ignores the ~5x y/z cost asymmetry this
    repo's anisotropic FOV creates, it will happily trade a small z step
    for a large y step to shave raw index distance, inflating the actual
    weighted total by ~15-20% empirically for little additional crossing
    removal over the weighted version -- the weighted metric gets the same
    crossing reduction at negligible weighted-total cost (see git history
    for the comparison). Two move types, both only accepted on a *strict*
    weighted-Euclidean improvement and only if every edge they touch stays
    within `max_allowed` under the (unscaled) bottleneck metric -- so
    neither can ever regress pass 2's achieved worst-case step:

    - Segment reversal (2-opt), as in the other passes -- changes the two
      boundary edges, leaves interior edges (and their weights) unchanged.
    - Pairwise exchange -- swaps two *non-adjacent* points, touching up to
      4 edges. Not a special case of segment reversal (reversing the span
      between them would also flip every point in between): a crossing
      where the two points that need to trade places are separated by
      other points already in good position is common, and plain 2-opt
      has no single move that fixes it without needlessly disturbing that
      interior order.

    Stage 2 handles the residual case stage 1's strict-improvement rule
    can miss: a swap across a pinned interior position (e.g. the two
    points on either side of the center sample) can leave both
    center-adjacent edges' *values* unchanged -- the crossing-removing
    swap is a true tie in weighted-Euclidean length, not an improvement,
    so stage 1's strict-improvement rule never takes it even though it
    strictly helps the actual goal (no self-crossings). (Stage 1's swap
    loop only excludes `i`/`j` themselves from `pinned`, not the span
    between them -- unlike the reversal loop, a swap doesn't displace
    anything strictly between `i` and `j`, so a pinned point there isn't
    at risk and doesn't need to block the move.) Stage 2 directly detects
    any remaining geometric crossing (`_find_first_crossing`) and greedily
    applies the first fixing reversal or swap found that respects
    `max_allowed` and doesn't increase weighted-Euclidean length beyond a
    small tolerance -- honest because crossing count, not
    weighted-Euclidean length, is the actual thing this pass exists to
    reduce.
    """
    order = order.copy()
    m = len(order)
    if m <= 2:
        return order

    Dw = _pairwise_weighted_dist(coords, deltak)
    diff = coords[:, None, :] - coords[None, :, :]
    De = np.sqrt((diff[..., 0] * deltak[0]) ** 2 + (diff[..., 1] * deltak[1]) ** 2)

    # Precomputed once (pinned is small and fixed for the whole call), so
    # span_has_pinned below is a single bisect lookup instead of building
    # and iterating a fresh generator on every call -- measured 108,530
    # calls (0.15s/frame) at this repo's real hot-loop scale (see
    # docs/review-findings.md item 57). |pinned| is tiny (endpoints plus
    # at most one interior center position), so the algorithmic win is
    # negligible; the real cost was per-call Python/generator overhead.
    pinned_sorted = sorted(pinned)

    def span_has_pinned(i: int, j: int) -> bool:
        pos = bisect.bisect_left(pinned_sorted, i)
        return pos < len(pinned_sorted) and pinned_sorted[pos] <= j

    # Stage 1: strict weighted-Euclidean descent.
    for _ in range(max_passes):
        best_delta = -1e-9
        best_move = None

        for i in range(1, m - 1):
            if i in pinned:
                continue
            for j in range(i + 1, m):
                if j in pinned or span_has_pinned(i, j):
                    continue
                w_e1 = Dw[order[i - 1], order[j]]
                if w_e1 > max_allowed + 1e-9:
                    continue
                removed = De[order[i - 1], order[i]]
                added = De[order[i - 1], order[j]]
                if j < m - 1:
                    w_e2 = Dw[order[i], order[j + 1]]
                    if w_e2 > max_allowed + 1e-9:
                        continue
                    removed += De[order[j], order[j + 1]]
                    added += De[order[i], order[j + 1]]
                delta = added - removed
                if delta < best_delta:
                    best_delta = delta
                    best_move = ('reverse', i, j)

        for i in range(1, m - 1):
            if i in pinned:
                continue
            for j in range(i + 2, m):
                # No span_has_pinned check here (unlike the reversal loop
                # above): a swap only touches positions i and j -- both
                # already checked individually -- and leaves every
                # position strictly between them untouched, so a pinned
                # point in that span is not displaced and doesn't block
                # the move.
                if j in pinned:
                    continue
                a, b = order[i], order[j]
                new_edge_weights = [Dw[order[i - 1], b], Dw[b, order[i + 1]], Dw[order[j - 1], a]]
                removed = De[order[i - 1], a] + De[a, order[i + 1]] + De[order[j - 1], b]
                added = De[order[i - 1], b] + De[b, order[i + 1]] + De[order[j - 1], a]
                if j < m - 1:
                    new_edge_weights.append(Dw[a, order[j + 1]])
                    removed += De[b, order[j + 1]]
                    added += De[a, order[j + 1]]
                if max(new_edge_weights) > max_allowed + 1e-9:
                    continue
                delta = added - removed
                if delta < best_delta:
                    best_delta = delta
                    best_move = ('swap', i, j)

        if best_move is None:
            break
        kind, i, j = best_move
        if kind == 'reverse':
            order[i : j + 1] = order[i : j + 1][::-1]
        else:
            order[i], order[j] = order[j], order[i]

    # Stage 2: directly target any remaining geometric crossing. Greedily
    # fixing one crossing can reintroduce a different one elsewhere (there's
    # no guarantee crossing count decreases monotonically move to move), so
    # this can otherwise cycle between the same two or more states forever
    # -- observed empirically: a real case where the only available
    # candidate for one crossing is exactly the swap that re-creates
    # another, oscillating for the full `max_passes` budget without ever
    # converging. `seen` makes any move that would revisit an already-seen
    # `order` state ineligible, so a cycle is detected and stage 2 stops
    # rather than spinning. That alone isn't enough, though: the greedy
    # single-crossing fix can also *wander* through a strictly-worse state
    # on its way to a cycle or a dead end -- e.g. disturbing a pinned-half
    # boundary to chase one crossing, without ever actually reaching a
    # crossing-free result. So the best (fewest-crossings) state seen is
    # tracked separately and returned at the end, defaulting to the
    # original (unmodified) order if no move ever strictly improves on it
    # -- Stage 2 should never hand back something worse than what Stage 1
    # already achieved.
    best_order = order.copy()
    best_n_cross = _count_crossings(coords[order])
    seen = {tuple(order.tolist())}
    for _ in range(max_passes):
        pts = coords[order]
        # Vectorized search (see _find_first_crossing's docstring) --
        # this loop, not _count_crossings, is the actual dominant caller
        # of the scalar orientation test at real hot-loop scale (item 57).
        crossing = _find_first_crossing(pts)
        if crossing is None:
            break

        i, j = crossing
        candidates = []
        if i + 1 not in pinned and j not in pinned and not span_has_pinned(i + 1, j):
            candidates.append(('reverse', i + 1, j))
        if i not in pinned and j not in pinned:
            candidates.append(('swap', i, j))
        if i + 1 not in pinned and j + 1 <= m - 1 and (j + 1) not in pinned:
            candidates.append(('swap', i + 1, j + 1))

        applied = False
        for kind, a, b in candidates:
            trial = order.copy()
            if kind == 'reverse':
                trial[a : b + 1] = trial[a : b + 1][::-1]
            else:
                trial[a], trial[b] = trial[b], trial[a]
            edges = Dw[trial[:-1], trial[1:]]
            if edges.max() > max_allowed + 1e-9:
                continue
            trial_key = tuple(trial.tolist())
            if trial_key in seen:
                continue
            order = trial
            seen.add(trial_key)
            applied = True
            n_cross = _count_crossings(coords[order])
            if n_cross < best_n_cross:
                best_n_cross = n_cross
                best_order = order.copy()
            break

        if not applied:
            break

        if best_n_cross == 0:
            break

    return best_order


def _dp_row_directions(
    rows: np.ndarray, row_z: list[np.ndarray], deltak: Sequence[float], objective: str = 'sum'
) -> list[str]:
    """Exact choice of each ky row's kz sweep direction ('F' = increasing,
    'B' = decreasing), minimizing either the total (`objective='sum'`,
    pass 1) or worst-case (`objective='max'`, used directly only by tests --
    `_hillclimb_row_directions_max` is pass 2's actual refinement, see
    below) weighted step across the whole shot.

    A row's own within-row step sizes (gaps between its sampled kz values)
    are the same multiset regardless of sweep direction -- reversing a
    sweep just reverses the order the same gaps are visited in -- so
    direction choice only affects the *inter-row* link step (this row's
    exit kz to the next row's entry kz), for both objectives. That makes
    this a small chain optimization: dp[i][d] = best achievable objective
    value for rows 0..i, given row i is traversed in direction d. Two
    states per row, O(#rows) overall -- exact within the "each row is a
    monotone kz sweep" family (itself optimal for the within-row steps, for
    either objective).
    """
    n_rows = len(rows)
    if n_rows == 1:
        return ['F']

    if objective == 'sum':
        combine = lambda a, b: a + b  # noqa: E731
        base = 0.0
    else:
        combine = max
        base = -np.inf

    dp = [{'F': base, 'B': base}]
    back: list[dict | None] = [None]
    for i in range(1, n_rows):
        cur = {}
        back_i = {}
        for d_cur in ('F', 'B'):
            entry_cur = row_z[i][0] if d_cur == 'F' else row_z[i][-1]
            best_val = np.inf
            best_prev = None
            for d_prev in ('F', 'B'):
                exit_prev = row_z[i - 1][-1] if d_prev == 'F' else row_z[i - 1][0]
                link = _weighted_step(rows[i] - rows[i - 1], entry_cur - exit_prev, deltak)
                cand = combine(dp[i - 1][d_prev], link)
                if cand < best_val:
                    best_val = cand
                    best_prev = d_prev
            cur[d_cur] = best_val
            back_i[d_cur] = best_prev
        dp.append(cur)
        back.append(back_i)

    directions = [None] * n_rows
    directions[-1] = min(('F', 'B'), key=lambda d: dp[-1][d])
    for i in range(n_rows - 1, 0, -1):
        directions[i - 1] = back[i][directions[i]]
    return directions


def _row_link_costs(
    rows: np.ndarray, row_z: list[np.ndarray], deltak: Sequence[float], directions: list[str]
) -> list[float]:
    """Inter-row link weighted steps for a given direction assignment (see
    `_dp_row_directions`) -- the only steps direction choice affects."""
    costs = []
    for i in range(1, len(rows)):
        exit_prev = row_z[i - 1][-1] if directions[i - 1] == 'F' else row_z[i - 1][0]
        entry_cur = row_z[i][0] if directions[i] == 'F' else row_z[i][-1]
        costs.append(_weighted_step(rows[i] - rows[i - 1], entry_cur - exit_prev, deltak))
    return costs


def _hillclimb_row_directions_max(
    rows: np.ndarray,
    row_z: list[np.ndarray],
    deltak: Sequence[float],
    directions: list[str],
    max_passes: int = 40,
) -> list[str]:
    """Pass 2 for laminar: starting from pass 1's (min-sum) `directions`,
    greedily flip single row directions to reduce the worst inter-row link
    step, accepting a flip only if it strictly improves the current worst
    case (never makes it worse -- a bottleneck 2-opt analogue, using
    single-direction-flip as the move instead of segment reversal, since
    the "path" here is really just a length-`n_rows` sequence of two-valued
    choices)."""
    n_rows = len(rows)
    if n_rows <= 1:
        return directions

    directions = list(directions)
    for _ in range(max_passes):
        current_max = max(_row_link_costs(rows, row_z, deltak, directions), default=0.0)
        best_max = current_max
        best_row = None
        for r in range(n_rows):
            directions[r] = 'B' if directions[r] == 'F' else 'F'
            candidate_max = max(_row_link_costs(rows, row_z, deltak, directions), default=0.0)
            if candidate_max < best_max - 1e-9:
                best_max = candidate_max
                best_row = r
            directions[r] = 'B' if directions[r] == 'F' else 'F'  # revert

        if best_row is None:
            break
        directions[best_row] = 'B' if directions[best_row] == 'F' else 'F'

    return directions


def _optimize_row_directions(
    rows: np.ndarray, row_z: list[np.ndarray], deltak: Sequence[float]
) -> list[str]:
    """Two-pass row-direction choice for a laminar shot: pass 1 minimizes
    total weighted path length, pass 2 then refines from there to reduce
    the worst-case step (see module docstring)."""
    directions = _dp_row_directions(rows, row_z, deltak, objective='sum')
    return _hillclimb_row_directions_max(rows, row_z, deltak, directions)


def mask2epi_laminar(
    mask: np.ndarray, ETL: int, Nshots: int, deltak: Sequence[float] = (1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    mask : (Ny, Nz) boolean/numeric 2D sampling mask.
    ETL : echo train length (samples per shot).
    Nshots : number of shots/excitations per volume.
    deltak : (deltak_y, deltak_z) k-space units per index step (1/fov), used
        to weight the pass-2 bottleneck ordering (see module docstring).
        Defaults to (1.0, 1.0) -- equal per-axis weighting -- when the
        caller doesn't have a physical FOV to hand.

    Returns
    -------
    schedule : (Nshots, ETL, 2) array; schedule[shot, echo, 0] = iy (0-based),
        schedule[shot, echo, 1] = iz (0-based).
    parts : (Ny, Nz) map of which shot (1..Nshots) each sampled point
        belongs to (0 = unsampled).
    """
    assert mask.ndim == 2, 'mask must be 2D (ky x kz).'
    assert ETL >= 1 and int(ETL) == ETL, 'ETL must be a positive integer.'
    assert Nshots >= 1 and int(Nshots) == Nshots, 'Nshots must be a positive integer.'
    n_samples = int(mask.sum())
    assert n_samples == Nshots * ETL, (
        f'Number of samples ({n_samples}) must equal Nshots*ETL ({Nshots * ETL}).'
    )

    Ny, Nz = mask.shape
    schedule = np.zeros((Nshots, ETL, 2), dtype=int)

    # First pass: partition samples into shots.
    parts = np.zeros((Ny, Nz), dtype=int)
    echo_count = 0
    part = 1
    center_out_iy = _center_out(Ny)
    for iz in range(Nz):
        for iy in center_out_iy:
            if mask[iy, iz]:
                assert part <= Nshots, 'Partitions exceeds number of shots.'
                parts[iy, iz] = part
                echo_count += 1
                if echo_count == ETL:
                    part += 1
                    echo_count = 0

    # Second pass: order each shot's echoes with non-decreasing ky and, for
    # the kz sweep direction per row, the two-pass min-sum-then-bottleneck
    # choice (see _optimize_row_directions).
    for shot in range(1, Nshots + 1):
        samples = parts == shot
        ys, zs = np.nonzero(samples)
        rows = np.unique(ys)
        row_z = [np.sort(zs[ys == iy]) for iy in rows]

        directions = _optimize_row_directions(rows, row_z, deltak)

        echo_idx = 0
        for i, iy in enumerate(rows):
            z_seq = row_z[i] if directions[i] == 'F' else row_z[i][::-1]
            for iz in z_seq:
                schedule[shot - 1, echo_idx, 0] = iy
                schedule[shot - 1, echo_idx, 1] = iz
                echo_idx += 1

    return schedule, parts


def _golden_angle_shot_order(wedge_centers: np.ndarray) -> np.ndarray:
    """Golden-angle-like temporal ordering of `mask2epi_radial`'s `Nshots`
    angular wedges (see that function's docstring for how the wedges
    themselves -- the spatial partition -- are formed; this only decides
    the order shots visit them in).

    Real golden-angle radial MRI (Winkelmann et al. 2007; see also
    https://pmc.ncbi.nlm.nih.gov/articles/PMC9189059/) picks each new
    spoke's angle freely, incrementing by the golden angle `180 deg / GR`
    (GR = the golden ratio) every view -- since consecutive multiples of an
    irrational angle never repeat and always fill the current largest gap,
    *any* prefix of consecutively-acquired spokes covers the angular range
    close to uniformly, not just the full set. This repo can't pick spoke
    angles freely: `mask2epi_radial`'s wedges are fixed in advance (exact
    equal-ETL-per-wedge counts, needed for the exact-per-shot sample-count
    constraint every mask2epi_* variant requires) before any ordering
    decision is made. So the golden-angle idea is applied one level up, to
    the discrete set of wedges rather than to a continuous angle: shot `t`'s
    *target* angle is still `t * golden_angle mod pi` exactly as in the
    continuous case, and is greedily matched to whichever not-yet-assigned
    wedge's own center angle is closest (circular distance mod pi) --
    consuming that wedge and moving to shot `t + 1`'s target. This
    preserves the same prefix-uniformity property approximately (exactly,
    in the limit of many narrow wedges): the first N shots' wedges are the
    ones closest to the first N terms of the continuous golden-angle
    sequence, which are themselves already close to evenly spread.

    Parameters
    ----------
    wedge_centers : (Nshots,) representative folded angle (radians,
        [0, pi)) of each wedge, indexed in the same order as the spatial
        partition (ascending by angle, i.e. wedge_centers is sorted).

    Returns
    -------
    order : (Nshots,) permutation of `range(Nshots)`; shot `t` is assigned
        wedge `order[t]`.
    """
    n = len(wedge_centers)
    golden_angle = np.pi * (math.sqrt(5) - 1) / 2  # 180 deg / GR, folded mod pi
    order = np.empty(n, dtype=int)
    available = list(range(n))
    for t in range(n):
        target = (t * golden_angle) % np.pi
        avail_centers = wedge_centers[available]
        d = np.abs(avail_centers - target)
        d = np.minimum(d, np.pi - d)
        pick = int(np.argmin(d))
        order[t] = available.pop(pick)
    return order


def _order_half_anchored(
    near_anchor: np.ndarray,
    far_side_coords: np.ndarray,
    cy: float,
    cz: float,
    deltak: Sequence[float],
    near_first: bool,
) -> np.ndarray:
    """Order one side of a radial shot's echo train: `near_anchor` (the
    k-space-center sample) sits at one end, and whichever point in
    `far_side_coords` is farthest from center by actual 2D distance (same
    rationale as `mask2epi_radial`'s own center-sample pick: projection
    alone can't see perpendicular offset) is pinned at the other end -- so
    the echo train genuinely sweeps from k-space periphery through center
    like an actual spoke, rather than potentially starting/ending near
    center by accident (which an unconstrained min-sum/bottleneck reorder
    has no reason to avoid, since it only optimizes step sizes, not where
    the path's endpoints land relative to center). Only the interior
    points are free for the two-pass min-sum-then-bottleneck optimization.

    Returns the ordered coordinates *excluding* `near_anchor` (the caller
    places the shared center sample explicitly at the shot's target echo
    index). `near_first`: True for the "after" half (center first, outer
    point last), False for the "before" half (outer point first, center
    last).
    """
    n = far_side_coords.shape[0]
    if n == 0:
        return far_side_coords

    dist2 = (far_side_coords[:, 0] - cy) ** 2 + (far_side_coords[:, 1] - cz) ** 2
    outer_idx = int(np.argmax(dist2))
    interior = np.delete(far_side_coords, outer_idx, axis=0)
    outer_coord = far_side_coords[outer_idx : outer_idx + 1]

    if near_first:
        init = np.vstack([near_anchor[None, :], interior, outer_coord])
    else:
        init = np.vstack([outer_coord, interior, near_anchor[None, :]])

    pass1 = init[_sum_optimized_order(init, deltak, fix_end=True)]
    pass2 = pass1[_bottleneck_2opt_order(pass1, deltak, fix_end=True)]

    return pass2[1:] if near_first else pass2[:-1]


def mask2epi_radial(
    mask: np.ndarray, ETL: int, Nshots: int, deltak: Sequence[float] = (1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Radial-like variant of `mask2epi_laminar`: each shot's echo train is
    a single spoke through k-space center (an opposite pair of angular
    wedges about the mask's center) instead of a raster of ky rows. `ky`
    non-decreasing is deliberately not preserved here. This is safe for the
    standard reference-scan-based Nyquist ghost correction this repo relies
    on (`sequences/EPIcal.py`'s blips-off scan): that correction is keyed
    to readout gradient polarity (alternating every echo regardless of
    trajectory shape), not to ky ordering or ky-adjacency between
    consecutive echoes — interleaved multi-shot EPI already violates
    ky-adjacency between odd/even echoes routinely, and is corrected the
    same reference-scan way.

    Parameters/returns match `mask2epi_laminar` exactly (including the new
    `deltak` parameter).

    Pass 1 — angular partitioning: every sampled point's angle about
    k-space center is folded into [0, pi) (`theta mod pi`), so a point and
    its 180-degree-opposite point land in the same bucket — this realizes
    "opposite wedge pair -> one shot" as a sort rather than a fixed set of
    2*Nshots pie wedges. Sorting all samples by folded angle and cutting
    into `Nshots` contiguous chunks of exactly `ETL` guarantees the
    exact-ETL-per-shot constraint directly, with no separate rebalancing
    step: fixed-width angular wedges can't hit an exact per-shot count for
    an arbitrary (e.g. Poisson-disc) mask, but equal-count contiguous
    chunks of angle-sorted samples always can, by construction. (This is a
    different "pass 1" from the ordering pass 1 described in the module
    docstring -- this one assigns points to shots, the ordering pass 1
    below sequences points within a shot.)

    Shot (temporal) order — golden-angle-like wedge sequencing: the
    angle-sorted chunks above form `Nshots` wedges indexed by ascending
    angle, but that spatial index is not used directly as the acquisition
    order. Instead `_golden_angle_shot_order` assigns wedges to shots so
    that shot `t`'s wedge has folded angle close to `t * golden_angle mod
    pi` (`golden_angle = 180 deg / GR`, the same low-discrepancy angle used
    in golden-angle radial MRI -- Winkelmann et al. 2007,
    https://pmc.ncbi.nlm.nih.gov/articles/PMC9189059/), greedily matching
    each target angle to whichever wedge is closest and not yet claimed.
    This gives the same practically useful property real golden-angle
    radial sampling has: *any* prefix of consecutively-acquired shots (not
    just the complete set) covers the angular range close to uniformly, so
    a downstream reconstruction that only has the first few shots of a
    frame available (e.g. sliding-window / retrospective temporal
    undersampling) still sees roughly uniform k-space coverage rather than
    a narrow angular wedge -- which is exactly what acquiring wedges in
    plain angular order would give.

    Splitting within a shot: each shot's points are projected onto that
    shot's own dominant spoke axis — found via the doubled-angle circular
    mean (`0.5 * atan2(mean(sin(2*theta)), mean(cos(2*theta)))`), which is
    robust to the pi-periodic wraparound `theta_folded` introduces (a plain
    mean of angles straddling the 0/pi boundary would cancel incorrectly)
    — sorted by that projection, then split *by count*, not by position
    relative to the center point: the point nearest k-space center (by
    actual 2D distance, found within this projection-sorted order) is
    removed, and the remaining `ETL - 1` projection-sorted points are cut
    into a "before" half (the first `target` of them) and an "after" half
    (the rest, `ETL - 1 - target`), matching `calc_te_tr_delays.py`'s
    nominal TE echo index `target = (ETL - 1) // 2` (see the derivation
    below). When the center point's own projection-sorted position isn't
    exactly `target`, this "before" half can include points that sit
    projection-wise on the far side of center -- a deliberate tradeoff
    that guarantees the exact `target`/`ETL - 1 - target` counts the fixed
    TE echo index needs, at the cost of "before"/"after" not being a
    literal geometric split around center. This projection sort only
    decides *which* points land in which half -- the actual
    visiting order within each half is decided by `_order_half_anchored`'s
    two-pass optimization (`_sum_optimized_order` for min-sum TSP,
    `_bottleneck_2opt_order` for bottleneck refinement -- see module
    docstring), run as an open path anchored at *both* ends: the center
    sample, and (by actual 2D distance, not projection -- same rationale
    as the center pick above) that half's own farthest point, so the echo
    train still sweeps from the k-space periphery through center like an
    actual spoke instead of potentially starting/ending near center by
    accident. Only the interior points between those two fixed ends are
    free for the two-pass optimization. A plain projection sort alone
    can't see that two points with similar projection have large
    perpendicular offset from the spoke axis -- both ordering passes fix
    this, pass 1 by minimizing total detour, pass 2 by directly targeting
    the worst remaining step.

    After both halves are assembled into the shot's full schedule, a third
    pass (`_euclidean_uncross_refine`, over the *whole* shot this time, not
    each half separately) cleans up self-crossings that per-half
    optimization structurally cannot see -- including ones straddling the
    seam where the two halves meet at the center sample. This is a real,
    separate failure mode from "worst-case step is high": passes 1-2
    optimize the weighted-Chebyshev metric, whose unit ball is a square,
    not strictly convex, so a move that visibly uncrosses the path can be
    an exact *tie* under that metric (zero effect on both weighted sum and
    weighted max) and never get taken by passes 1-2's strict-improvement
    acceptance rule, even though nothing hardware-relevant favors keeping
    the crossing. Pass 3 fixes this in two stages: first, minimize
    *weighted Euclidean* length (strictly convex, so the classical
    uncrossing argument holds) via 2-opt reversal and pairwise-exchange
    moves; second, for any crossing that still survives (typically only
    ones straddling the pinned center sample, where a fixing swap's two
    new edges happen to exactly equal the two they replace), directly
    detect it geometrically and apply whichever candidate fixing move is
    available. Every move in both stages is only accepted if it keeps
    every edge within this shot's already-achieved worst-case step -- pass
    3 can only clean up slack that step is blind to, never regress it (see
    `_euclidean_uncross_refine`'s own docstring for the full derivation).

    `calc_te_tr_delays.py`'s `min_te` places TE at `(ETL/2 - 0.5) * gro`
    after the prep sequence, i.e. at echo `e` (0-based) satisfying
    `e + 0.5 = ETL/2 - 0.5` (echo `e`'s ADC center sits at `(e + 0.5) *
    gro` into the train) — solving gives `e = ETL/2 - 1`, computed below as
    `(ETL - 1) // 2` (for odd `ETL` this lands between two echoes; either
    neighbor is off by half an echo spacing, which is as exact as the
    continuous-time TE definition supports). The true nearest-to-center
    sample is placed at that fixed index by construction (it's the pivot
    the two halves are built around), keeping every shot's actual
    k-space-center sample aligned with the sequence's nominal TE echo.
    """
    assert mask.ndim == 2, 'mask must be 2D (ky x kz).'
    assert ETL >= 1 and int(ETL) == ETL, 'ETL must be a positive integer.'
    assert Nshots >= 1 and int(Nshots) == Nshots, 'Nshots must be a positive integer.'
    n_samples = int(mask.sum())
    assert n_samples == Nshots * ETL, (
        f'Number of samples ({n_samples}) must equal Nshots*ETL ({Nshots * ETL}).'
    )

    Ny, Nz = mask.shape
    cy, cz = Ny / 2, Nz / 2

    ys, zs = np.nonzero(mask)
    dy = ys - cy
    dz = zs - cz
    theta = np.arctan2(dz, dy)  # (-pi, pi]
    theta_folded = np.mod(theta, np.pi)  # [0, pi); folds opposite points together

    angle_order = np.argsort(theta_folded, kind='stable')

    wedge_centers = np.array([
        theta_folded[angle_order[w * ETL:(w + 1) * ETL]].mean() for w in range(Nshots)
    ])
    shot_order = _golden_angle_shot_order(wedge_centers)

    parts = np.zeros((Ny, Nz), dtype=int)
    schedule = np.zeros((Nshots, ETL, 2), dtype=int)
    target = (ETL - 1) // 2

    for shot in range(Nshots):
        w = shot_order[shot]
        idx = angle_order[w * ETL:(w + 1) * ETL]
        shot_y, shot_z = ys[idx], zs[idx]
        parts[shot_y, shot_z] = shot + 1

        th = theta[idx]
        axis_angle = 0.5 * math.atan2(np.sin(2 * th).mean(), np.cos(2 * th).mean())
        proj = (shot_y - cy) * math.cos(axis_angle) + (shot_z - cz) * math.sin(axis_angle)

        proj_order = np.argsort(proj, kind='stable')
        shot_y, shot_z = shot_y[proj_order], shot_z[proj_order]

        # Nearest-to-center by actual 2D distance, not by |proj| -- a point
        # can have a near-zero projection onto the spoke axis while still
        # sitting well off that axis (large perpendicular offset), so
        # minimizing |proj| would not reliably pick the true closest point.
        dist2 = (shot_y - cy) ** 2 + (shot_z - cz) ** 2
        center_idx = int(np.argmin(dist2))
        center_coord = np.array([[shot_y[center_idx], shot_z[center_idx]]], dtype=float)

        rest_idx = np.delete(np.arange(ETL), center_idx)
        before_idx, after_idx = rest_idx[:target], rest_idx[target:]
        before_coords = np.column_stack([shot_y[before_idx], shot_z[before_idx]]).astype(float)
        after_coords = np.column_stack([shot_y[after_idx], shot_z[after_idx]]).astype(float)

        # Order each half as an open path with *both* ends pinned: the
        # center sample at one end, and (by actual 2D distance, same
        # rationale as the center pick above) that half's own farthest
        # point at the other -- only the interior points are free for the
        # two-pass optimization (see _order_half_anchored). Without this,
        # an unconstrained min-sum/bottleneck reorder has no notion that an
        # echo train "should" sweep from the k-space periphery through
        # center like an actual spoke, and can leave the true outer point
        # stranded mid-train instead of at the start/end.
        before_path = _order_half_anchored(
            center_coord[0], before_coords, cy, cz, deltak, near_first=False
        )
        after_path = _order_half_anchored(
            center_coord[0], after_coords, cy, cz, deltak, near_first=True
        )

        schedule[shot, :target, :] = before_path
        schedule[shot, target, :] = center_coord[0]
        schedule[shot, target + 1 :, :] = after_path

        # Pass 3: Euclidean uncrossing cleanup over the *whole* shot (not
        # just each half separately) -- some self-crossings straddle the
        # seam where the before- and after-halves meet at the center
        # sample, which per-half optimization alone can never see or fix.
        # Positions 0, target, and ETL - 1 are pinned (start, the actual
        # k-space-center sample, and end); capped at this shot's own
        # achieved worst-case step so it can only remove slack that step
        # is blind to, never regress it (see _euclidean_uncross_refine).
        shot_coords = schedule[shot, :, :].astype(float)
        shot_max = _pairwise_weighted_dist(shot_coords, deltak)[
            np.arange(ETL - 1), np.arange(1, ETL)
        ].max()
        uncrossed_order = _euclidean_uncross_refine(
            shot_coords, deltak, np.arange(ETL), shot_max,
            pinned=frozenset({0, target, ETL - 1}),
        )
        schedule[shot, :, :] = shot_coords[uncrossed_order].astype(int)

    return schedule, parts
