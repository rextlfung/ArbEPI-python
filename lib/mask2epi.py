"""mask2epi_laminar (ported from ../ArbEPI/lib/mask2epi.m) and
mask2epi_radial (this repo's own addition, no MATLAB counterpart) are two
interchangeable ways to partition a 2D (ky, kz) sampling mask into
`Nshots` EPI trajectories of length `ETL` each.

mask2epi_laminar: samples near ky = 0 are spread center-out across shots;
ky is non-decreasing within each echo train, and the kz sweep direction per
ky row is chosen greedily to minimize travel from the previous echo — each
shot occupies its own non-overlapping band of the visiting order, hence
"laminar."

mask2epi_radial: each shot instead sweeps through k-space center as a
single spoke (an opposite pair of angular wedges about the mask's center),
so every shot samples near k-space center rather than only whichever shots
happen to include the central ky row. See its own docstring for the
partitioning/ordering details, including why ky non-decreasing is
deliberately given up here.

Both return `schedule`/`parts` in the same shapes and conventions; which to
use is a call-site decision, not something enforced in this module.

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


def mask2epi_laminar(mask: np.ndarray, ETL: int, Nshots: int) -> tuple[np.ndarray, np.ndarray]:
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


def mask2epi_radial(mask: np.ndarray, ETL: int, Nshots: int) -> tuple[np.ndarray, np.ndarray]:
    """Radial-like variant of `mask2epi_laminar`: each shot's echo train is
    a single spoke through k-space center (an opposite pair of angular
    wedges about the mask's center) instead of a raster of ky rows. `ky`
    non-decreasing is deliberately not preserved here. This is safe for the
    standard reference-scan-based Nyquist ghost correction this repo relies
    on (`sequences/epical.py`'s blips-off scan): that correction is keyed
    to readout gradient polarity (alternating every echo regardless of
    trajectory shape), not to ky ordering or ky-adjacency between
    consecutive echoes — interleaved multi-shot EPI already violates
    ky-adjacency between odd/even echoes routinely, and is corrected the
    same reference-scan way.

    Parameters/returns match `mask2epi_laminar` exactly.

    Pass 1 — angular partitioning: every sampled point's angle about
    k-space center is folded into [0, pi) (`theta mod pi`), so a point and
    its 180-degree-opposite point land in the same bucket — this realizes
    "opposite wedge pair -> one shot" as a sort rather than a fixed set of
    2*Nshots pie wedges. Sorting all samples by folded angle and cutting
    into `Nshots` contiguous chunks of exactly `ETL` guarantees the
    exact-ETL-per-shot constraint directly, with no separate rebalancing
    step: fixed-width angular wedges can't hit an exact per-shot count for
    an arbitrary (e.g. Poisson-disc) mask, but equal-count contiguous
    chunks of angle-sorted samples always can, by construction.

    Pass 2 — ordering within a shot: each shot's points are projected onto
    that shot's own dominant spoke axis — found via the doubled-angle
    circular mean (`0.5 * atan2(mean(sin(2*theta)), mean(cos(2*theta)))`),
    which is robust to the pi-periodic wraparound `theta_folded`
    introduces (a plain mean of angles straddling the 0/pi boundary would
    cancel incorrectly) — and sorted by that signed projection. This is a
    monotonic sweep from one end of the spoke, through the region nearest
    center, to the other end: the radial analogue of
    `mask2epi_laminar`'s non-decreasing-ky sweep, and (since a shot's
    points already lie roughly along one line through center) it
    approximates the same "consecutive echoes are spatially close"
    property the old greedy kz-direction choice targeted, without needing
    a discrete-row structure to be greedy over.

    Finally, the sample nearest k-space center is spliced to a fixed echo
    index if it didn't already land there from the projection sort.
    `calc_te_tr_delays.py`'s `min_te` places TE at `(ETL/2 - 0.5) * gro`
    after the prep sequence, i.e. at echo `e` (0-based) satisfying
    `e + 0.5 = ETL/2 - 0.5` (echo `e`'s ADC center sits at `(e + 0.5) *
    gro` into the train) — solving gives `e = ETL/2 - 1`, computed below as
    `(ETL - 1) // 2` (for odd `ETL` this lands between two echoes; either
    neighbor is off by half an echo spacing, which is as exact as the
    continuous-time TE definition supports). Splicing the true
    nearest-to-center sample there keeps every shot's actual k-space-center
    sample aligned with the sequence's nominal TE echo — not just
    approximately close to it, which is what the projection sort alone
    would give. This splice is the only place a shot's ordering deviates
    from the monotonic projection sweep.
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

    parts = np.zeros((Ny, Nz), dtype=int)
    schedule = np.zeros((Nshots, ETL, 2), dtype=int)
    target = (ETL - 1) // 2

    for shot in range(Nshots):
        idx = angle_order[shot * ETL:(shot + 1) * ETL]
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
        if center_idx != target:
            y_c, z_c = shot_y[center_idx], shot_z[center_idx]
            shot_y = np.insert(np.delete(shot_y, center_idx), target, y_c)
            shot_z = np.insert(np.delete(shot_z, center_idx), target, z_c)

        schedule[shot, :, 0] = shot_y
        schedule[shot, :, 1] = shot_z

    return schedule, parts
