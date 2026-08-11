"""Ported from ../ArbEPI/lib/mask2epi.m — the core algorithm.

Partitions a 2D (ky, kz) sampling mask into `Nshots` EPI trajectories, each
of length `ETL`. Samples near ky = 0 are spread center-out across shots; ky
is non-decreasing within each echo train, and the kz sweep direction per ky
row is chosen greedily to minimize travel from the previous echo.

Index convention: `schedule`'s (iy, iz) values are 0-based, matching
Python/numpy array indices into `mask`. `parts` keeps MATLAB's convention of
1-based shot labels with 0 meaning "unsampled" — it's a label/sentinel
array, not a coordinate used in downstream gradient-scaling arithmetic, so
there's no reason to renumber it. (Conversion of `schedule`'s indices to
1-based, for saving `samp_locs.mat` in a form existing MATLAB-side
reconstruction code can read unchanged, happens once at the save boundary in
sequences/arbepi.py — not here.)
"""

import math

import numpy as np


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


def mask2epi(mask: np.ndarray, ETL: int, Nshots: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    mask : (Ny, Nz) boolean/numeric 2D sampling mask.
    ETL : echo train length (samples per shot).
    Nshots : number of shots/excitations per volume.

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

    # Second pass: order each shot's echoes with non-decreasing ky and a
    # greedy nearest-endpoint kz sweep direction.
    for shot in range(1, Nshots + 1):
        samples = parts == shot
        ys, zs = np.nonzero(samples)
        kz_min = int(zs.min())
        kz_max = int(zs.max())

        echo_idx = 0
        for iy in np.unique(ys):
            if echo_idx > 0:
                last_kz = schedule[shot - 1, echo_idx - 1, 1]
                if abs(last_kz - kz_min) < abs(last_kz - kz_max):
                    kz_range = range(kz_min, kz_max + 1)
                else:
                    kz_range = range(kz_max, kz_min - 1, -1)
            else:
                kz_range = range(kz_min, kz_max + 1)

            for iz in kz_range:
                if samples[iy, iz]:
                    schedule[shot - 1, echo_idx, 0] = iy
                    schedule[shot - 1, echo_idx, 1] = iz
                    echo_idx += 1

    return schedule, parts
