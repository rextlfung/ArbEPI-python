"""Variable-density Poisson-disc sampling, following ../ArbEPI/lib/pd_sample.m
(itself a MATLAB port of `sigpy.mri.poisson`, https://github.com/mikgroup/sigpy)
with two corrections found while debugging a "hangs forever" report against
the default full-scale params:

1. **Reseed-per-iteration.** The point-placement core must be reseeded to
   the *same* fixed value on every binary-search iteration -- this is what
   makes the achieved acceleration a stable, repeatable function of the
   density `slope` alone, which the bisection search's convergence
   assumption requires. Both ../ArbEPI/lib/pd_sample.m (its
   `poisson_disc_core` helper only reseeds when `seed ~= 0`, but the outer
   loop always calls it with `seed=0`, so it silently never reseeds) and
   this port's first version (a continuously-advancing `rng` across
   iterations rather than a fixed-per-call seed) got this wrong. Without
   it, the same `slope` value gives wildly different sample counts across
   iterations (measured: 54 to 1676 samples for the *identical* radius
   field on a 90x60 grid), so the search never converges and burns through
   every iteration of its budget on every single call.

2. **Bounded outer loop.** Real `sigpy.mri.poisson` (correctly reseeded,
   per point 1) still has an unbounded `while slope_min < slope_max` search
   with no iteration cap. On small/coarse grids the achievable sample count
   jumps in large discrete steps as `slope` varies, so no achievable slope
   may land within `tol` of the target -- confirmed this hangs forever
   (slope_min/slope_max converge to adjacent float64 values a single ULP
   apart, so the loop condition never goes false and neither candidate
   ever satisfies the tolerance check). The search below is instead capped
   at `max_search_iters`; since the exact target count is enforced by the
   prune/fill step regardless of how close the search got, a capped search
   costs nothing in correctness.

3. **Active-list cap.** `sigpy.mri.samp._poisson` also bounds its active
   list at `nx*ny` (`while nx*ny > num_actives > 0`); an earlier version of
   this port dropped that bound. Without it, once radius sits at its floor
   of 1 pixel (always true near the calibration/high-density center
   regardless of slope), the continuous-space ellipse collision check can
   keep accepting new float-valued candidates that are individually
   collision-free but land on or immediately adjacent to already-occupied
   *integer* cells once floored -- so the active list can grow far past
   `nx*ny` (measured 36,000+ on a 5,400-pixel grid, still climbing after
   420,000 outer iterations, corresponding to a single `pd_sample()` call
   taking 3.5+ minutes) instead of naturally saturating. The cap forces
   termination at the same point real sigpy does.

4. **Seed the growth front outside the calibration region.** The single
   initial active point used to be drawn uniformly over the whole grid.
   The calibration region starts pre-filled (every one of its cells
   already "occupied"), so a seed landing inside it collides on every one
   of its `max_attempts` tries -- nothing outside a fully-sampled region
   is ever within one exclusion radius of a point deep inside it -- the
   active list drops to zero on the very first outer-loop iteration, and
   `_poisson_disc_core` returns `calib_mask` completely unchanged. Because
   the same fixed seed is reused across every binary-search iteration
   (point 1 above), this isn't a one-off fluke: it silently kills genuine
   Poisson-disc placement for the *entire* call, and `pd_sample`'s
   exact-count step then hits the target count via uniform-random fill
   over the whole non-calibration budget instead of density-tapered
   placement -- confirmed by direct reproduction (a seed landing inside a
   ~13%-area calibration region returned `mask == calib_mask`, zero grown
   points). `_poisson_disc_core_jit` now rejection-samples the initial
   point against `calib_mask` so growth always starts outside it; this is
   guaranteed to terminate because `pd_sample` only ever calls it with
   `calib_mask.sum() <= target_samples < nx*ny` (checked before the
   binary-search loop), so at least one non-calibration cell always
   exists, but the loop is still capped defensively since the function is
   callable directly.

Even with all four fixes, `_poisson_disc_core`'s point-placement loop
itself is a tight, highly sequential (each new point depends on all prior
ones -- not vectorizable) loop that can run hundreds of thousands of
iterations for the worst-case radius/seed combinations above; in pure
Python this still took up to ~12s per `pd_sample()` call even after fixes
1-3 (point 4 added later, measured across 60 seeds at production scale).
It is JIT-compiled
with `numba` for this reason -- the same reason real sigpy JIT-compiles
its own equivalent function. `numba` is added as a narrow, single-function
dependency here (unlike the earlier attempt to depend on the `sigpy`
package wholesale, which was rejected both for its unrelated dependency
surface and its own multi-year-stale PyPI release).
"""

import math
from typing import Sequence

import numba as nb
import numpy as np


def _rho_grid(ny: int, nx: int) -> np.ndarray:
    """(ny, nx) grid of radial distance from center, normalized so the
    ellipse inscribed in the full (ny, nx) rectangle sits at rho == 1."""
    Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    yn = (Y - ny / 2) / (ny / 2)
    xn = (X - nx / 2) / (nx / 2)
    return np.sqrt(yn**2 + xn**2)


def _calib_mask_rect(ny: int, nx: int, side_frac: float) -> np.ndarray:
    """(ny, nx) boolean mask of a centered rectangle covering `side_frac`
    of each axis' own half-extent (kmax) independently: |ky| <= side_frac
    * ky_max and |kz| <= side_frac * kz_max, using the same per-axis
    normalization as `_rho_grid` (ky_max/kz_max correspond to ny/2, nx/2
    in pixel units, so this rectangle's pixel area is side_frac**2 of the
    full (ny, nx) grid). side_frac <= 0 -> no calibration region.

    Named `side_frac`, not `calib_frac`, because callers that want a
    calibration region sized as a fraction of the *sample budget*
    (`pd_sample`'s `calib_frac`, R-dependent) must first convert that
    fraction into this per-axis side fraction via `_calib_side_frac` --
    passing `calib_frac` straight through here instead makes the region's
    pixel area a fixed fraction of the whole grid, independent of R (the
    bug `_calib_side_frac` exists to fix -- see its docstring)."""
    if side_frac <= 0:
        return np.zeros((ny, nx), dtype=bool)
    Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    yn = np.abs((Y - ny / 2) / (ny / 2))
    xn = np.abs((X - nx / 2) / (nx / 2))
    return (yn <= side_frac) & (xn <= side_frac)


def _calib_side_frac(target_samples: int, nx: int, ny: int, calib_frac: float) -> float:
    """Per-axis side fraction (see `_calib_mask_rect`) whose rectangle
    pixel area equals `calib_frac * target_samples` -- i.e. the
    calibration region holds a *constant fraction of the R-dependent
    sample budget* regardless of acceleration, not a fixed fraction of
    k-space. side_frac**2 * nx*ny = calib_frac*target_samples =>
    side_frac = sqrt(calib_frac*target_samples / (nx*ny)).

    This restores the pre-existing "fraction of the sample budget"
    semantics (see docs/review-findings.md item 195) that a since-reverted
    change had replaced with a fixed side_frac = calib_frac (fraction of
    each axis' own kmax, independent of R): at high acceleration,
    target_samples shrinks a lot faster than the grid does, so a
    fixed-kmax-fraction region can end up holding nearly the *entire*
    sample budget (or exceeding it outright) -- confirmed on a real
    0.8mm/R~94 config, where a 0.1 kmax-fraction rectangle (487 pixels)
    left only 33 of 520 target samples for the actual variable-density
    region. Scaling side_frac with target_samples keeps the calibration
    region's own share of the budget constant across every resolution/R
    combination instead."""
    if calib_frac <= 0:
        return 0.0
    return min(math.sqrt(calib_frac * target_samples / (nx * ny)), 0.999)


@nb.njit(cache=True)
def _poisson_disc_core_jit(
    nx: int,
    ny: int,
    max_attempts: int,
    radius_x: np.ndarray,
    radius_y: np.ndarray,
    calib_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    mask = calib_mask.copy()

    # numba's np.random is its own internal generator, independent of both
    # numpy's legacy global state and the numpy.random.Generator instances
    # used elsewhere in this codebase -- seeding it here is local to this
    # call and doesn't touch any other RNG state.
    np.random.seed(seed)

    # Active list, preallocated at the nx*ny cap (see module docstring
    # point 3) and grown/shrunk via swap-with-last removal, like the
    # MATLAB original.
    pxs = np.empty(nx * ny, dtype=np.float64)
    pys = np.empty(nx * ny, dtype=np.float64)

    # Rejection-sample the initial seed outside the calibration region --
    # see module docstring point 4: a seed landing inside it can never
    # grow (everything nearby is already "occupied"), silently killing
    # placement for the whole call. Bounded defensively even though
    # `pd_sample`'s own pre-check guarantees at least one non-calibration
    # cell exists whenever this is reached via that path.
    x0 = np.random.randint(0, nx)
    y0 = np.random.randint(0, ny)
    attempts = 0
    while calib_mask[y0, x0] and attempts < nx * ny:
        x0 = np.random.randint(0, nx)
        y0 = np.random.randint(0, ny)
        attempts += 1
    pxs[0] = float(x0)
    pys[0] = float(y0)
    num_actives = 1

    while num_actives > 0 and num_actives < nx * ny:
        i = np.random.randint(0, num_actives)
        px = int(math.floor(pxs[i]))
        py = int(math.floor(pys[i]))
        rx = radius_x[py, px]
        ry = radius_y[py, px]

        done = False
        k = 0
        while not done and k < max_attempts:
            v = math.sqrt(np.random.random() * 3 + 1)
            t = 2 * math.pi * np.random.random()
            qx = px + v * rx * math.cos(t)
            qy = py + v * ry * math.sin(t)

            if 0 <= qx < nx and 0 <= qy < ny:
                sx = max(int(math.floor(qx - rx)), 0)
                ex = min(int(math.ceil(qx + rx)), nx - 1)
                sy = max(int(math.floor(qy - ry)), 0)
                ey = min(int(math.ceil(qy + ry)), ny - 1)

                # A plain nested loop compiles to tight native code under
                # numba -- unlike the pure-Python path this replaced,
                # there's no per-call interpreter/numpy-array-allocation
                # overhead to dodge by pre-filtering with `.any()`, so the
                # straightforward cell-by-cell scan is both simpler and
                # faster here.
                valid = True
                for xx in range(sx, ex + 1):
                    for yy in range(sy, ey + 1):
                        if mask[yy, xx] == 1:
                            if ((qx - xx) / radius_x[yy, xx]) ** 2 + ((qy - yy) / radius_y[yy, xx]) ** 2 < 1:
                                valid = False
                                break
                    if not valid:
                        break

                if valid:
                    done = True
                    pxs[num_actives] = qx
                    pys[num_actives] = qy
                    mask[int(math.floor(qy)), int(math.floor(qx))] = 1
                    num_actives += 1
            k += 1

        if not done:
            pxs[i] = pxs[num_actives - 1]
            pys[i] = pys[num_actives - 1]
            num_actives -= 1

    return mask


def _poisson_disc_core(
    nx: int,
    ny: int,
    max_attempts: int,
    radius_x: np.ndarray,
    radius_y: np.ndarray,
    calib_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    return _poisson_disc_core_jit(
        nx, ny, max_attempts, radius_x, radius_y, calib_mask.astype(np.float64), seed
    )


def pd_sample(
    img_shape: Sequence[int],
    accel: float,
    rng: np.random.Generator,
    calib_frac: float = 0.0,
    dtype: str = 'logical',
    crop_corner: bool = True,
    max_attempts: int = 30,
    tol: float = 0.1,
    decay: float = 1.0,
    max_search_iters: int = 50,
) -> np.ndarray:
    """
    Parameters
    ----------
    img_shape : (ny, nx)
    accel : target acceleration factor (> 1)
    rng : numpy random Generator. A single int seed is drawn from it up
        front and reused (fixed) across every binary-search iteration --
        see module docstring point 1; `rng` itself is used directly for
        the exact-count prune/fill step below.
    calib_frac : fraction of the target sample budget (`target_samples =
        floor(ny*nx/accel)`) to place in a centered, fully-sampled
        *rectangular* calibration region -- a constant share of the
        acquisition regardless of acceleration (see
        `_calib_side_frac`/docs/review-findings.md item 195). The
        rectangle's own per-axis half-width (`side_frac`, in the same
        ky_max/kz_max-normalized units `_calib_mask_rect` and `_rho_grid`
        use) is derived so its pixel area equals `calib_frac *
        target_samples`; pass that derived `side_frac`, not `calib_frac`
        itself, to `_calib_mask_rect`. 0 = no calibration region. The
        surrounding variable-density taper (`r` below) still transitions
        outward using the elliptical (Euclidean) `_rho_grid` metric at
        that same `side_frac` radius -- an ellipse inscribed in the
        calibration rectangle, touching it exactly on each axis -- rather
        than switching to the rectangle's own (Chebyshev) metric, so the
        rectangle's corners (outside that inscribed ellipse) are still
        forced fully sampled via `calib_mask` directly; they just don't
        drive the taper's own shape.
    dtype : 'logical', 'double', or 'complex'.
    crop_corner : whether to crop sampling corners (elliptical mask).
    max_attempts : max attempts to generate a point per active point.
    tol : tolerance for the binary-search loop on density.
    decay : density falloff exponent (1 = linear; > 1 = steeper toward center).
    max_search_iters : cap on binary-search iterations -- see module
        docstring point 2. The exact target count is enforced regardless
        of whether the search converges within this budget.

    Returns
    -------
    mask : (ny, nx) array with exactly floor(ny*nx/accel) samples.
    """
    if accel <= 1:
        raise ValueError(f'accel must be greater than 1, got {accel}')

    ny, nx = img_shape[0], img_shape[1]
    total_pixels = nx * ny
    target_samples = math.floor(total_pixels / accel)

    rho = _rho_grid(ny, nx)
    side_frac = _calib_side_frac(target_samples, nx, ny, calib_frac)
    calib_mask = _calib_mask_rect(ny, nx, side_frac)
    # Elliptical taper radius matching the rectangle's per-axis extent --
    # see pd_sample's calib_frac docstring for why the taper stays
    # elliptical rather than switching to the rectangle's own metric.
    rho_calib = min(max(side_frac, 0.0), 0.999)

    # The exact-count prune/fill step below can only remove non-calibration
    # samples, so if the calibration region alone already exceeds the target
    # budget, the "exactly floor(ny*nx/accel) samples" contract (see this
    # function's Returns docstring) can't be met -- fail loudly instead of
    # silently over-returning.
    n_calib = int(calib_mask.sum())
    if n_calib > target_samples:
        raise ValueError(
            f'Calibration region ({n_calib} samples) exceeds the target sample '
            f'budget ({target_samples} samples, from accel={accel}); lower '
            f'calib_frac or accel.'
        )

    r = np.maximum(rho - rho_calib, 0) / max(1 - rho_calib, 1e-6)

    # Binary search for the density slope. Target a slightly higher density
    # (lower accel) than requested, to ensure enough points to prune later.
    accel_search = accel * 0.95

    slope_max = max(nx, ny)
    slope_min = 0.0

    seed = int(rng.integers(0, 2**31 - 1))

    mask = None
    for _ in range(max_search_iters):
        slope = (slope_max + slope_min) / 2

        radius_x = np.maximum((1 + r ** (1 / decay) * slope) * nx / max(nx, ny), 1)
        radius_y = np.maximum((1 + r ** (1 / decay) * slope) * ny / max(nx, ny), 1)

        # Reseed to the same fixed value every iteration -- see module
        # docstring point 1.
        mask = _poisson_disc_core(nx, ny, max_attempts, radius_x, radius_y, calib_mask, seed)

        if crop_corner:
            mask = mask * (rho <= 1)

        num_samples = mask.sum()
        current_accel = total_pixels / num_samples

        if abs(current_accel - accel_search) < tol:
            break

        if current_accel < accel_search:
            slope_min = slope  # increase slope to reduce samples
        else:
            slope_max = slope  # decrease slope to increase samples

    # Enforce exact acceleration via random pruning/filling.
    mask = mask.astype(bool)
    current_samples = int(mask.sum())

    if current_samples > target_samples:
        num_to_remove = current_samples - target_samples
        candidates = np.flatnonzero(mask & ~calib_mask)
        if candidates.size > 0:
            perm = rng.permutation(candidates.size)
            remove_idx = candidates[perm[: min(num_to_remove, candidates.size)]]
            mask.flat[remove_idx] = False

    elif current_samples < target_samples:
        num_to_add = target_samples - current_samples
        candidates = np.flatnonzero(~mask)
        if candidates.size > 0:
            perm = rng.permutation(candidates.size)
            add_idx = candidates[perm[: min(num_to_add, candidates.size)]]
            mask.flat[add_idx] = True

    if dtype == 'complex':
        return mask.astype(complex)
    elif dtype == 'double':
        return mask.astype(float)
    else:
        return mask.astype(bool)
