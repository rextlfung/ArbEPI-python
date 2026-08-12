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


def _calib_rho(target_samples: int, nx: int, ny: int, calib_frac: float) -> float:
    """Radius (in the normalized units of `_rho_grid`) of a centered,
    aspect-matched ellipse whose pixel area is `calib_frac * target_samples`."""
    if calib_frac <= 0:
        return 0.0
    rho_calib = math.sqrt(4 * calib_frac * target_samples / (math.pi * nx * ny))
    return min(rho_calib, 0.999)


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
    calib_frac : fraction of the target sample budget (floor(ny*nx/accel))
        to place in a fully-sampled calibration region: a centered ellipse,
        aspect-matched to (ny, nx), sized so its pixel area equals
        `calib_frac * target_samples`. 0 = no calibration region.
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
    rho_calib = _calib_rho(target_samples, nx, ny, calib_frac)
    calib_mask = rho <= rho_calib

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
