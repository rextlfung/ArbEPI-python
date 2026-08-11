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

Even with all three fixes, `_poisson_disc_core`'s point-placement loop
itself is a tight, highly sequential (each new point depends on all prior
ones -- not vectorizable) loop that can run hundreds of thousands of
iterations for the worst-case radius/seed combinations above; in pure
Python this still took up to ~12s per `pd_sample()` call even after fixes
1-3 (measured across 60 seeds at production scale). It is JIT-compiled
with `numba` for this reason -- the same reason real sigpy JIT-compiles
its own equivalent function. `numba` is added as a narrow, single-function
dependency here (unlike the earlier attempt to depend on the `sigpy`
package wholesale, which was rejected both for its unrelated dependency
surface and its own multi-year-stale PyPI release).
"""

import math
from typing import Sequence, Tuple

import numba as nb
import numpy as np


def _calib_bounds(n: int, calib_n: float) -> Tuple[int, int]:
    """0-based [start, stop) bounds of a centered calibration region of
    width `calib_n` along a dimension of length `n`."""
    start = max(0, math.floor(n / 2 - calib_n / 2))
    stop = min(n, math.floor(n / 2 + calib_n / 2))
    return start, stop


@nb.njit(cache=True)
def _poisson_disc_core_jit(
    nx: int,
    ny: int,
    max_attempts: int,
    radius_x: np.ndarray,
    radius_y: np.ndarray,
    y_start: int,
    y_stop: int,
    x_start: int,
    x_stop: int,
    seed: int,
) -> np.ndarray:
    mask = np.zeros((ny, nx))
    mask[y_start:y_stop, x_start:x_stop] = 1

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
    pxs[0] = float(np.random.randint(0, nx))
    pys[0] = float(np.random.randint(0, ny))
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
    calib: Sequence[float],
    seed: int,
) -> np.ndarray:
    calib_y, calib_x = calib[0], calib[1]
    y_start, y_stop = _calib_bounds(ny, calib_y)
    x_start, x_stop = _calib_bounds(nx, calib_x)
    return _poisson_disc_core_jit(
        nx, ny, max_attempts, radius_x, radius_y, y_start, y_stop, x_start, x_stop, seed
    )


def pd_sample(
    img_shape: Sequence[int],
    accel: float,
    rng: np.random.Generator,
    calib: Sequence[float] = (0, 0),
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
    calib : (calib_y, calib_x) fully-sampled calibration region size.
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

    Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    x_dist = np.maximum(np.abs(X - nx / 2) - calib[1] / 2, 0)
    x_dist = x_dist / x_dist.max()

    y_dist = np.maximum(np.abs(Y - ny / 2) - calib[0] / 2, 0)
    y_dist = y_dist / y_dist.max()

    r = np.sqrt(x_dist**2 + y_dist**2)

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
        mask = _poisson_disc_core(nx, ny, max_attempts, radius_x, radius_y, calib, seed)

        if crop_corner:
            mask = mask * (r <= 1)

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

    calib_mask = np.zeros((ny, nx), dtype=bool)
    y_start, y_stop = _calib_bounds(ny, calib[0])
    x_start, x_stop = _calib_bounds(nx, calib[1])
    calib_mask[y_start:y_stop, x_start:x_stop] = True

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
