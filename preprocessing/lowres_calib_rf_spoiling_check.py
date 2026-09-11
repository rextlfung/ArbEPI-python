"""Tests for an RF/gradient-spoiling pseudo-steady-state (PSS) artifact in
lowres_calib_recon.py's output, merged in from a sibling investigation
("epi sequence setup" session, same ArbEPI-python repo, real 2026-09-10
phantom scans) that independently confirmed this mechanism on different
real acquisitions from the same sequence.

Mechanism (not re-derived here, taken from that investigation and
cross-checked against PMC13527254): this sequence's quadratic RF-spoiling
phase increment (`params.py`'s `rf_phase_0`, historically 117 degrees)
gives the continuously-incrementing RF phase counter an *exact* integer
period in RF excitations: `P_shots = 720 / gcd(2 * rf_phase_0_int, 720)`
(360 for the ideal continuous case, discretized to 720 here since the
phase itself is defined mod 360 but the *quadratic* term needs mod 720 to
close exactly -- see that session's report for the closed-form check,
117*40 = 4680 = 6.5*720 languages this loosely; the load-bearing fact is
just that P_shots=40 for rf_phase_0=117, confirmed both by derivation and
against real data on four independent sequence configs there).

RF-spoiled sequences never reach a true steady state -- only a periodic
"pseudo-steady-state" (PSS) transverse magnetization that repeats with
that same period in shot index. Normally this causes spatial ghosting
(different phase-encode lines within one image see different points in
the PSS cycle). Here it becomes *temporal* instead, because `rf_count`
never resets between frames (this repo's own confirmed lack of dummy
shots -- see CLAUDE.md/the shading-audit report) while every frame plays
the same number of shots: frame f's starting global shot count is
`f * Nshots`, so its position in the P_shots-cycle is `(f * Nshots) mod
P_shots` -- a periodic function of frame index with period
`P_frames = P_shots / gcd(Nshots, P_shots)`, *independent of which
specific (ky,kz) locations that frame happens to sample* (unlike this
repo's calibration mask, which does vary in echo-index assignment frame
to frame -- see lowres_calib_te_distribution_check.py -- this mechanism's
predicted period depends only on shot *count*, not on trajectory
content). A fixed-area gradient spoiler (this repo's
`lib/make_spoilers.py`, pre-fix) doesn't perturb this cycle at all, which
is the root cause identified upstream: a plain Tx/Rx RF-phase-match check
(the usual spoiling-correctness check) passes cleanly here and would not
catch it.

This module tests the P_frames=2 (even/odd frame) special case directly,
since both this repo's real datasets have Nshots=20 -> P_frames=2 exactly
(40 / gcd(20, 40) = 2) -- a plain even-vs-odd-frame comparison, more
statistically powerful than a general-lag autocorrelation on a short
(~30-frame) series. Detrends first (linear) to remove slow drift/warm-up
transients this repo's *other* confirmed mechanisms (T2*/off-resonance,
gradient-heating drift) already explain, then compares the two frame
parities both as an ROI-mean scalar (t-test + a nonparametric permutation
test, since 30 samples is too few to fully trust ttest_ind's normality
assumption) and per-voxel (to check whether the effect is a spatially
*uniform* whole-image gain modulation -- the PSS signature -- or
localized shading, which would point elsewhere).

The fix (RF phase increment 117 -> 115.4 degrees per Leupold, Weigel &
Baer, PLOS ONE 2025, plus a per-shot-randomized gradient spoiler in place
of the fixed-area design) is merged to `main` as PR #8
(`8448ff3`/`7af04d4`) but not yet present on this branch or in the
already-acquired 20260822ball data this module tests -- it can only be
adopted for future acquisitions.

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_calib_rf_spoiling_check <datdir> [datdir2 ...]
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import stats

from preprocessing.config import load_config, set_seq_paths
from preprocessing.matio import read_mat


def predicted_pss_period(n_shots: int, p_shots: int = 40) -> int:
    """Predicted period, in frames, of the RF/gradient-spoiling PSS
    artifact -- see module docstring. p_shots=40 is this repo's own
    rf_phase_0=117-degree value; pass a different p_shots if rf_phase_0
    ever changes (e.g. 720/gcd(2*115, 720)=72 for the upstream 115.4-degree
    fix, rounded to an integer degree)."""
    return p_shots // math.gcd(int(n_shots), p_shots)


def even_odd_alternation(
    img: np.ndarray, mask: np.ndarray, n_perm: int = 20000, seed: int = 0
) -> dict:
    """img: [Nx,Ny,Nz,Nt] magnitude, mask: [Nx,Ny,Nz] bool. Tests for a
    period-2 (even/odd frame) alternation in the ROI-mean signal, after
    linear detrending. Returns the scalar even-odd difference (t-test and
    permutation-test p-values) plus the per-voxel even-odd map, to check
    whether the effect is spatially uniform (the PSS signature) or
    localized."""
    rng = np.random.default_rng(seed)
    Nt = img.shape[-1]
    t = np.arange(Nt)

    roi_mean_t = img[mask].mean(axis=0)
    coeffs = np.polyfit(t, roi_mean_t, 1)
    resid = roi_mean_t - np.polyval(coeffs, t)

    even, odd = resid[0::2], resid[1::2]
    real_diff = even.mean() - odd.mean()
    tstat, p_ttest = stats.ttest_ind(even, odd)

    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(resid)
        perm_diffs[i] = shuffled[0::2].mean() - shuffled[1::2].mean()
    p_perm = float(np.mean(np.abs(perm_diffs) >= np.abs(real_diff)))

    even_v = img[..., 0::2].mean(axis=-1)
    odd_v = img[..., 1::2].mean(axis=-1)
    mean_img = img.mean(axis=-1)
    diff_v = (even_v - odd_v)[mask]
    rel_diff_pct = 100 * diff_v / mean_img[mask]

    return dict(
        roi_signal=roi_mean_t, roi_resid=resid, t=t,
        even_odd_diff_pct=100 * real_diff / roi_mean_t.mean(),
        tstat=tstat, p_ttest=p_ttest, p_perm=p_perm,
        per_voxel_diff_mean_pct=float(rel_diff_pct.mean()),
        per_voxel_diff_std_pct=float(rel_diff_pct.std()),
        even_v=even_v, odd_v=odd_v, mask=mask,
    )


def main(datdirs: list[str], seqname: str = 'ArbEPI', rf_phase_0_int: int = 117) -> None:
    p_shots = 720 // math.gcd(2 * rf_phase_0_int, 720)
    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))
        cfg = load_config(datdir=datdir, seqnames=[seqname])
        paths = set_seq_paths(cfg, seqname)
        raw = read_mat(paths.scan_info, ['schedules'])['schedules']
        _Nframes, n_shots, _ETL, _ = raw.shape

        period = predicted_pss_period(n_shots, p_shots)
        print(f'\n=== {label} ===')
        print(f'  Nshots/frame = {n_shots}, rf_phase_0 = {rf_phase_0_int} deg -> P_shots = {p_shots}, '
              f'predicted PSS period = {period} frame(s)')

        fn = os.path.join(datdir, 'recon', 'basic', f'{seqname}_recon_lowres_calib.nii.gz')
        img = np.asarray(nib.load(fn).dataobj)  # (Nx,Ny,Nz,Nt) magnitude
        mean_img = img.mean(axis=-1)
        mask = mean_img > 0.2 * mean_img.max()

        if period != 2:
            print(f'  predicted period is {period}, not 2 -- this module only implements the '
                  'even/odd-frame (period-2) special case; skipping the quantitative test.')
            continue

        result = even_odd_alternation(img, mask)
        print(f'  even-odd relative difference (ROI mean, detrended): {result["even_odd_diff_pct"]:.3f}%')
        print(f'  t-test: t={result["tstat"]:.3f}, p={result["p_ttest"]:.4f}  '
              f'|  permutation test: p={result["p_perm"]:.5f}')
        print(f'  per-voxel even-odd relative difference: mean={result["per_voxel_diff_mean_pct"]:.3f}%, '
              f'std={result["per_voxel_diff_std_pct"]:.3f}% '
              f'({"spatially uniform -- PSS-like" if result["per_voxel_diff_std_pct"] < result["per_voxel_diff_mean_pct"] else "spatially structured -- inconsistent with a uniform PSS gain effect"})')

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(result['t'], result['roi_signal'], 'o-', color='C0')
        axes[0].scatter(result['t'][0::2], result['roi_signal'][0::2], color='C1', zorder=3, label='even frames')
        axes[0].scatter(result['t'][1::2], result['roi_signal'][1::2], color='C2', zorder=3, label='odd frames')
        axes[0].set_xlabel('frame')
        axes[0].set_ylabel('ROI-mean signal (a.u.)')
        axes[0].set_title(f'{label}: even/odd = {result["even_odd_diff_pct"]:.2f}%, p_perm={result["p_perm"]:.4f}')
        axes[0].legend(fontsize=8)

        iz = img.shape[2] // 2
        diff_pct = 100 * (result['even_v'] - result['odd_v']) / (mean_img + np.finfo(np.float64).eps)
        vmax = np.percentile(np.abs(diff_pct[mask]), 99)
        im = axes[1].imshow(diff_pct[:, :, iz].T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1].set_title('even-odd relative difference (%)')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        plt.tight_layout()
        out_dir = os.path.join(datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)
        fn_out = os.path.join(out_dir, 'lowres_calib_rf_spoiling_check.png')
        plt.savefig(fn_out, dpi=130)
        print(f'  Wrote {fn_out}')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdirs', nargs='+')
    parser.add_argument('--seqname', default='ArbEPI')
    parser.add_argument('--rf-phase-0-int', type=int, default=117,
                         help='integer-degree RF phase increment used for the acquisition '
                              '(117 for pre-PR#8 data; the upstream fix uses 115.4, round as needed)')
    args = parser.parse_args()
    main(args.datdirs, args.seqname, args.rf_phase_0_int)
