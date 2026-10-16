#!/usr/bin/env python
"""One summary PNG per analysis/ folder, in analysis/figures/, plus <name>_metrics.json.

Activation is Benjamini-Hochberg FDR q < 0.05 (two-sided, over the brain mask) on the saved t map.
Three voxel sets: active (survivor, z > 0, red), counter-active (survivor, z < 0, blue) and
inactive (survivor of neither sign, black).

  left column    axial, coronal, sagittal z maps (FDR survivors only) over the temporal mean of
                 |X_recon|, axes in mm from the centre voxel, cut through the highest-z voxel; colorbar
  top middle     mean time series of the three sets (+- SE over voxels), task windows shaded
  middle row     cycle average of the three sets (+- SE over cycles), and the mean per-voxel
                 periodogram of each set (+- SE)
  third row      split-half reproducibility: z from the odd vs the even cycles (AR(1) restarted at each
                 cycle), and a guide
  bottom row     test-retest (Genovese, Noll & Eddy 1997): M interleaved replicates of the run, the
                 reliability-map histogram with the fitted mixed-binomial model, the ROC it implies,
                 and a guide
  right column   every statistic as a table; only brain-mask voxels are ever classified

Reads the saved NIfTIs (*_sum_t, *_sum_z, *_sum_psc, *_mask) and the reconstruction's series; the
full-run maps are never refit; the odd/even and replicate fits are done here (about a minute).
Run with .venv-recon:
    python analysis_figure.py [--root DIR] [--only SUBSTRING] [--force]
or call make_figure(analysis_dir) from the queue driver.
"""
import argparse
import glob
import json
import os
import re
import sys
import textwrap
from types import SimpleNamespace

REPO = os.environ.get('ARBEPI_REPO', '/home/rexfung/github/ArbEPI-python')
sys.path.insert(0, REPO)

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402

from analyze import glm  # noqa: E402
from analyze.design import ExperimentParams, build_design_matrix, drift_basis  # noqa: E402
from analyze.io import read_recon  # noqa: E402
from simulate_fmri.analysis import fit_mixed_binomial, levels  # noqa: E402

ROOT = '/StorageRAID/rexfung/20261006tap'
Q_FDR = 0.05
TR_REPLICATES = 4  # test-retest: interleaved replicates of the run
Z_GRID = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])  # thresholds of the mixed-binomial model
Z_HIST = 3.0  # threshold of the reliability-map histogram
Z_MAX = 8.0  # one colour range for every figure
LOWBAND_HZ = 0.15
RESP_BAND = (0.2, 0.4)  # Hz, usual breathing
CARDIAC_BAND = (0.8, 1.3)  # Hz, usual heart rate; drawn where it aliases to at this frame rate
SCALE_COLORS = {'scale0': 'tab:blue', 'scale1': 'tab:orange'}


def _zcmap(thr):
    """Transparent for |z| < thr; red -> yellow above, blue -> cyan below."""
    v = np.linspace(-Z_MAX, Z_MAX, 512)
    out = np.zeros((v.size, 4))
    pos, neg = v >= thr, v <= -thr
    f = np.clip((np.abs(v) - thr) / max(Z_MAX - thr, 1e-6), 0, 1)
    out[pos] = np.column_stack([np.ones(pos.sum()), 0.1 + 0.9 * f[pos], np.zeros(pos.sum()), np.ones(pos.sum())])
    out[neg] = np.column_stack([np.zeros(neg.sum()), 0.3 + 0.7 * f[neg], np.ones(neg.sum()), np.ones(neg.sum())])
    return ListedColormap(out)


def alias(f, fs):
    return abs(f - round(f / fs) * fs)


def periodogram(y, tr):
    """One-sided power density (percent^2 / Hz), Hann window; y (..., nt)."""
    nt = y.shape[-1]
    w = np.hanning(nt)
    P = np.abs(np.fft.rfft(y * w, axis=-1)) ** 2 * 2 * tr / np.sum(w ** 2)
    return np.fft.rfftfreq(nt, tr), P


def find_recon(root, run, tag):
    c = sorted(glob.glob(f'{root}/recon/sense_*_b0_{tag}/joined/{run}_recon.h5'))
    c = [f for f in c if os.path.basename(os.path.dirname(os.path.dirname(f))).endswith('_b0_' + tag)]
    if len(c) != 1:
        raise FileNotFoundError(f'{run} {tag}: {len(c)} reconstructions {c}')
    return c[0]


def recon_path(root, run, tag, summ):
    """The reconstruction an analysis folder was made from: the path its summary recorded, else a search by tag."""
    f = summ.get('file')
    return f if f and os.path.exists(f) else find_recon(root, run, tag)


def parse(folder):
    name = os.path.basename(os.path.normpath(folder))
    lowband = name.endswith('_lowband')
    base = name[:-len('_lowband')] if lowband else name
    m = re.match(r'^(\d+\.\d+mm_(?:radial|laminar))_(.+)$', base)
    return base, m.group(1), m.group(2), lowband


class Loaded:
    """One reconstruction, read once: the series of the sum (and scales), its mean, the design."""

    def __init__(self, h5, tr, onsets_fn, paradigm):
        self.rec = read_recon(h5, scales=('all',))
        if self.rec.n_scales < 2:  # a single-scale recon: scale0 would just repeat the sum
            self.rec.series = {'sum': self.rec.series['sum']}
        with h5py.File(h5) as f:
            self.pieces = json.loads(f.attrs['pieces']) if 'pieces' in f.attrs else []
        meta = json.load(open(h5[:-3] + '.json'))
        self.first_frame = meta['first_frame']
        self.tr = tr
        self.nt = self.rec.series['sum'].shape[-1]
        self.t0 = self.first_frame * tr
        onsets = onsets_fn(self.t0, self.nt * tr)
        self.params = ExperimentParams.block(tr=tr, onsets=onsets, duration=paradigm['on_s'], name='tap', n_discard=0)
        X, self.names = build_design_matrix(self.params, self.nt)
        self.X = X
        self.reg = X[:, 0]
        cut = self.params.drift_cutoff_s
        self.D = np.column_stack([np.ones(self.nt), drift_basis(self.nt, tr, cut)]) if cut else np.ones((self.nt, 1))
        self.Dpinv = np.linalg.pinv(self.D)
        self.t = self.t0 + (np.arange(self.nt) + 0.5) * tr
        self.paradigm = paradigm
        self._brain_spec = {}

    def detrend(self, y):
        return y - (y @ self.Dpinv.T) @ self.D.T


def fdr_survivors(t, dof, q=Q_FDR):
    """Boolean over t: Benjamini-Hochberg survivors of the two-sided t test."""
    p = glm.t_to_p(t, dof)
    thr = glm.fdr_threshold(p, q)
    return (p <= thr) if thr > 0 else np.zeros(p.shape, bool)


def cycle_average(L, y, paradigm):
    """Mean and SE over cycles of series y (nt,) by second of the task cycle; returns (centres, m, se, n_cycles)."""
    period = paradigm['period_s']
    ph = np.mod(L.t - paradigm['task_onset_s'], period)
    cyc = np.floor((L.t - paradigm['task_onset_s']) / period)
    nb = int(round(period))
    bins = np.minimum((ph / period * nb).astype(int), nb - 1)
    m, se = np.full(nb, np.nan), np.full(nb, np.nan)
    ok = (cyc >= 0) & (cyc < paradigm['n_cycles'])
    for bb in range(nb):
        v = y[(bins == bb) & ok]
        if v.size > 1:
            m[bb], se[bb] = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
    return np.arange(nb) + 0.5, m, se, len(np.unique(cyc[ok]))


def task_power(f, P, f0, band=(0.01, LOWBAND_HZ), near_hz=0.02):
    """Power at the task frequencies relative to non-task frequencies, for a mean periodogram P.

    Task power = sum over the bin of the frequency and its two neighbours (the Hann main lobe).
    Reference, per bin: 'local' = mean over bins within near_hz of it (task bins +-2 excluded);
    'band' = mean over `band` with every odd harmonic in it (+-2 bins) excluded. Ratios are
    task power / (3 x reference per-bin power); odd = harmonics 1, 3, 5 pooled."""
    df = f[1] - f[0]
    if not np.isfinite(P).all() or P.sum() <= 0:  # empty set (no voxels, or no power)
        return {k: float('nan') for k in ('ref_band_density', 'odd_over_local', 'odd_over_band',
                *[f'{n}_over_{r}' for n in ('f0', '3f0', '5f0') for r in ('local', 'band')])}
    odd = [h * f0 for h in (1, 3, 5)]
    k = [int(round(x / df)) for x in odd]
    idx = np.arange(f.size)
    excl = np.zeros(f.size, bool)
    for kk in k:
        excl |= np.abs(idx - kk) <= 2
    in_band = (f >= band[0]) & (f <= band[1]) & ~excl
    ref_band = float(P[in_band].mean())
    out = {'ref_band_density': ref_band}
    sig, loc = [], []
    for name, kk in zip(('f0', '3f0', '5f0'), k):
        s_ = float(P[kk - 1:kk + 2].sum())
        near = (np.abs(f - f[kk]) <= near_hz) & ~excl
        r_loc = float(P[near].mean())
        out[f'{name}_over_local'] = s_ / (3 * r_loc)
        out[f'{name}_over_band'] = s_ / (3 * ref_band)
        sig.append(s_)
        loc.append(3 * r_loc)
    out['odd_over_local'] = float(sum(sig) / sum(loc))
    out['odd_over_band'] = float(sum(sig) / (9 * ref_band))
    return out


def cycle_index(L, paradigm):
    """Task cycle (0-based) of every frame; the lead-in before the first onset counts with cycle 0
    and frames after the last cycle with the last one."""
    c = np.floor((L.t - paradigm['task_onset_s']) / paradigm['period_s']).astype(int)
    return np.clip(c, 0, paradigm['n_cycles'] - 1)


def _whiten_seg(a, rho, starts):
    """The AR(1) prewhitening filter along the last axis, restarted at each index of `starts`."""
    out = np.empty(a.shape, dtype=np.result_type(a.dtype, np.float32))
    out[..., 1:] = a[..., 1:] - rho * a[..., :-1]
    out[..., 0] = a[..., 0]
    out[..., starts] = np.sqrt(1 - rho ** 2) * a[..., starts]
    return out


def fit_glm_segmented(Y, X, c, starts, max_iter=5, chunk=20000):
    """analyze.glm.fit_glm with the AR(1) filter restarted at each segment start, for frames that are
    not all adjacent in time (a subset of cycles). Y (V, n), X (n, P), c (P,), starts: indices in 0..n
    where a new segment begins. Returns SimpleNamespace(t, z, dof, rho)."""
    X = np.asarray(X, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n, V = X.shape[0], Y.shape[0]
    dof = n - np.linalg.matrix_rank(X)
    lag_ok = np.ones(n - 1, bool)
    lag_ok[np.asarray(starts[1:], dtype=int) - 1] = False  # the pair (i-1, i) is not a lag-1 pair at a start
    rho = 0.0
    for _ in range(max_iter):
        pinv = np.linalg.pinv(_whiten_seg(X.T, rho, starts).T)
        num = den = 0.0
        for s in range(0, V, chunk):
            y = Y[s:s + chunk].astype(np.float64)
            r = y - (_whiten_seg(y, rho, starts) @ pinv.T) @ X.T
            r -= r.mean(axis=1, keepdims=True)
            r /= np.maximum(np.sqrt((r ** 2).mean(axis=1, keepdims=True)), 1e-30)
            num += float((r[:, 1:] * r[:, :-1])[:, lag_ok].sum())
            den += float((r ** 2).sum())
        new = num / den
        done = abs(new - rho) < 1e-3
        rho = new
        if done:
            break
    Xw = _whiten_seg(X.T, rho, starts).T
    XtXi = np.linalg.pinv(Xw.T @ Xw)
    var_c = float(c @ XtXi @ c)
    t = np.empty(V, np.float32)
    for s in range(0, V, chunk):
        y = _whiten_seg(Y[s:s + chunk].astype(np.float64), rho, starts)
        b = y @ Xw @ XtXi
        s2 = ((y - b @ Xw.T) ** 2).sum(axis=1) / dof
        t[s:s + chunk] = (b @ c) / np.maximum(np.sqrt(s2 * var_c), 1e-30)
    return SimpleNamespace(t=t, z=glm.t_to_z(t, dof).astype(np.float32), dof=float(dof), rho=float(rho))


def fit_cycle_groups(L, Y, paradigm, groups):
    """One GLM per group of cycles (each a set of 0-based cycle indices), on those frames only.
    The regressor and the drift columns are the full run's, restricted to the frames, so the HRF and
    the slow drift carry across the gaps; the AR(1) filter restarts at every cycle. Returns a list of
    SimpleNamespace(t, z, dof, rho)."""
    cyc = cycle_index(L, paradigm)
    out = []
    for g in groups:
        h = np.isin(cyc, list(g))
        idx = np.flatnonzero(h)
        starts = np.r_[0, np.flatnonzero(np.diff(idx) > 1) + 1]
        X = np.column_stack([L.reg[h], L.D[h]])
        c = np.zeros(X.shape[1])
        c[0] = 1.0
        out.append(fit_glm_segmented(Y[:, h], X, c, starts))
    return out


def make_figure(analysis_dir, loaded=None, out_dir=None, paradigm=None, force=False):
    root = os.path.dirname(os.path.dirname(os.path.normpath(analysis_dir)))
    base, run, tag, lowband = parse(analysis_dir)
    name = os.path.basename(os.path.normpath(analysis_dir))
    out_dir = out_dir or f'{os.path.dirname(os.path.normpath(analysis_dir))}/figures'
    os.makedirs(out_dir, exist_ok=True)
    fn = f'{out_dir}/{name}.png'
    if os.path.exists(fn) and not force:
        return fn
    paradigm = paradigm or json.load(open(f'{root}/paradigm.json'))
    summ = json.load(open(f'{analysis_dir}/{base}_summary.json'))
    tr = summ['tr']
    sm = summ['scales']['sum']
    zimg = nib.load(f'{analysis_dir}/{base}_sum_z.nii.gz')
    z = np.asarray(zimg.dataobj, dtype=np.float32)
    psc_map = np.asarray(nib.load(f'{analysis_dir}/{base}_sum_psc.nii.gz').dataobj, dtype=np.float32)
    brain = np.asarray(nib.load(f'{analysis_dir}/{base}_mask.nii.gz').dataobj) > 0

    def onsets_fn(t0, T):
        o = [paradigm['task_onset_s'] + k * paradigm['period_s'] - t0 for k in range(paradigm['n_cycles'])]
        return [x for x in o if x + paradigm['on_s'] > 0 and x < T]

    if loaded is None:
        loaded = Loaded(recon_path(root, run, tag, summ), tr, onsets_fn, paradigm)
    L = loaded
    vox = np.asarray(summ['voxel_size_mm'], dtype=float)

    # ---- the three sets, on the saved t map (brain-mask voxels only)
    t_vol = np.asarray(nib.load(f'{analysis_dir}/{base}_sum_t.nii.gz').dataobj, dtype=np.float32)
    surv = fdr_survivors(t_vol[brain], sm['dof'])
    zb_ = z[brain]
    sets = {'active': surv & (zb_ > 0), 'inactive': ~surv, 'counter-active': surv & (zb_ < 0)}
    n_set = {k: int(v.sum()) for k, v in sets.items()}
    act = sets['active']
    n_vox = int(brain.sum())
    z_fdr = float(np.abs(zb_[surv]).min()) if surv.any() else np.inf
    zthr_s = f'{z_fdr:.2f}' if np.isfinite(z_fdr) else 'none'
    zb = np.where(brain, z, -np.inf)
    pk = np.unravel_index(np.argmax(zb), z.shape)
    zpk = float(z[pk])

    under = L.rec.mean_signal
    vmax = np.percentile(under[brain], 99.5)
    base_v = under[brain].astype(np.float64)
    Y = L.rec.series['sum'][brain].astype(np.float32)  # (V, nt)
    psc = (100 * (Y - Y.mean(1, keepdims=True)) / base_v[:, None]).astype(np.float32)
    psc_vox = psc_map[brain]
    eff = {k: (float(psc_vox[sets[k]].mean()) if n_set[k] else np.nan) for k in ('active', 'counter-active')}

    # ---- odd vs even cycles (split half)
    ncyc_all = paradigm['n_cycles']
    (r1, r2) = fit_cycle_groups(L, Y, paradigm, [range(0, ncyc_all, 2), range(1, ncyc_all, 2)])
    s1, s2 = fdr_survivors(r1.t, r1.dof), fdr_survivors(r2.t, r2.dof)
    a1, a2 = s1 & (r1.z > 0), s2 & (r2.z > 0)
    n1, n2, nb_ = int(a1.sum()), int(a2.sum()), int((a1 & a2).sum())
    from scipy import stats as _st
    rep = {
        'z_pearson': float(np.corrcoef(r1.z, r2.z)[0, 1]),
        'z_spearman': float(_st.spearmanr(r1.z, r2.z)[0]),
        'n_active_odd': n1, 'n_active_even': n2, 'n_both': nb_,
        'dice': float(2 * nb_ / (n1 + n2)) if n1 + n2 else float('nan'),
        'dice_chance': float(2 * (n1 * n2 / n_vox) / (n1 + n2)) if n1 + n2 else float('nan'),
        'frac_full_active_in_both': float((a1 & a2 & act).sum() / act.sum()) if n_set['active'] else float('nan'),
        'z_fdr_odd': float(np.abs(r1.z[s1]).min()) if s1.any() else float('inf'),
        'z_fdr_even': float(np.abs(r2.z[s2]).min()) if s2.any() else float('inf'),
    }

    # ---- test-retest: M interleaved replicates, mixed-binomial model (Genovese, Noll & Eddy 1997)
    M = max(2, min(TR_REPLICATES, ncyc_all // 2))
    reps = fit_cycle_groups(L, Y, paradigm, [range(r, ncyc_all, M) for r in range(M)])
    scores = np.stack([r.z for r in reps])  # (M, V)
    mb = None
    try:
        mb = fit_mixed_binomial(levels(scores, Z_GRID), n_thresholds=len(Z_GRID))
    except Exception as e:  # e.g. nothing exceeds any threshold
        print(f'  mixed-binomial fit failed for {name}: {e!r}', flush=True)
    k0 = int(np.argmin(np.abs(Z_GRID - Z_HIST)))
    Rv = (scores >= Z_GRID[k0]).sum(axis=0)
    tr_stats = {'M': M, 'cycles_per_replicate': ncyc_all // M, 'thresholds_z': Z_GRID.tolist()}
    if mb is not None:
        crit = mb.lam * mb.p_active + (1 - mb.lam) * (1 - mb.p_inactive)
        kb = int(np.argmax(crit))
        order = np.argsort(mb.p_inactive)
        fpr_roc = np.r_[0.0, mb.p_inactive[order], 1.0]
        tpr_roc = np.r_[0.0, mb.p_active[order], 1.0]
        tr_stats.update(lam=float(mb.lam), n_est_active=float(mb.lam * n_vox), tpr=mb.p_active.tolist(),
                        fpr=mb.p_inactive.tolist(), best_z=float(Z_GRID[kb]), best_tpr=float(mb.p_active[kb]),
                        best_fpr=float(mb.p_inactive[kb]), auc=float(np.trapezoid(tpr_roc, fpr_roc)))

    # ---- one pass: mean +- SE over voxels of the series and of the periodogram, per set
    nt = L.nt
    S = {k: np.zeros(nt) for k in sets}
    SS = {k: np.zeros(nt) for k in sets}
    nf = nt // 2 + 1
    PS = {k: np.zeros(nf) for k in sets}
    PSS = {k: np.zeros(nf) for k in sets}
    f_p = None
    for s in range(0, psc.shape[0], 20000):
        ch = psc[s:s + 20000].astype(np.float64)
        f_p, Pc = periodogram(L.detrend(ch), L.tr)
        for k, m in sets.items():
            mm = m[s:s + 20000]
            if mm.any():
                S[k] += ch[mm].sum(0)
                SS[k] += (ch[mm] ** 2).sum(0)
                PS[k] += Pc[mm].sum(0)
                PSS[k] += (Pc[mm] ** 2).sum(0)

    def mean_se(sum_, sumsq, n):
        if n == 0:
            return np.full_like(sum_, np.nan), np.full_like(sum_, np.nan)
        m = sum_ / n
        var = np.clip(sumsq / n - m ** 2, 0, None) * n / max(n - 1, 1)
        return m, np.sqrt(var / n)

    ts = {k: mean_se(S[k], SS[k], n_set[k]) for k in sets}
    ps = {k: mean_se(PS[k], PSS[k], n_set[k]) for k in sets}
    f0 = 1 / paradigm['period_s']
    tp = {k: task_power(f_p, ps[k][0], f0) for k in sets}
    COL = {'active': 'tab:red', 'inactive': 'k', 'counter-active': 'tab:blue'}

    def lab(k):
        return f'{k} (n = {n_set[k]})'

    # ---- layout: slices | time series, cycle average + spectrum, split half, test-retest | table
    fig = plt.figure(figsize=(21, 18), constrained_layout=True)
    gs = fig.add_gridspec(5, 4, width_ratios=[1.0, 1.1, 1.1, 0.9], height_ratios=[1.0, 0.67, 0.75, 0.07, 0.8])
    cmap = _zcmap(z_fdr)
    norm = Normalize(-Z_MAX, Z_MAX)

    # --- left column: axial, coronal, sagittal, colorbar; mm from the centre voxel (N // 2) at 0
    i, j, k = pk

    def extent(n, d):
        lo = (-(n // 2) - 0.5) * d
        return lo, lo + n * d

    survm = np.zeros(z.shape, bool)
    survm[brain] = surv
    shape = under.shape
    views = [('axial', (0, 1), (i, j), k, 'x', 'y'),
             ('coronal', (0, 2), (i, k), j, 'x', 'z'),
             ('sagittal', (1, 2), (j, k), i, 'y', 'z')]
    for row, (title, (da_, db_), (a, b), fixed, la, lb) in enumerate(views):
        sl = [slice(None)] * 3
        sl[3 - da_ - db_] = fixed  # the dim not shown
        u, zz, sv_ = under[tuple(sl)], z[tuple(sl)], survm[tuple(sl)]
        xl, xh = extent(shape[da_], vox[da_])
        yl, yh = extent(shape[db_], vox[db_])
        ax = fig.add_subplot(gs[row, 0])
        ext = [xl, xh, yl, yh]
        ax.imshow(u.T, cmap='gray', vmin=0, vmax=vmax, origin='lower', extent=ext, interpolation='nearest')
        ax.imshow(np.ma.masked_where(~sv_.T, zz.T), cmap=cmap, norm=norm, origin='lower',
                  extent=ext, interpolation='nearest')
        ax.axvline((a - shape[da_] // 2) * vox[da_], color='c', lw=0.6, alpha=0.8)
        ax.axhline((b - shape[db_] // 2) * vox[db_], color='c', lw=0.6, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f'{la} (mm from centre)')
        ax.set_ylabel(f'{lb} (mm from centre)')
    cax = fig.add_subplot(gs[3, 0])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cb.set_label(f'z, FDR q < {Q_FDR:g} (|z| > {zthr_s}); slices through the highest-z voxel', fontsize=8)

    # --- time series of the three sets (mean +- SE over voxels)
    ax = fig.add_subplot(gs[0, 1:3])
    on = paradigm['on_s']
    for kk in range(paradigm['n_cycles']):
        s = paradigm['task_onset_s'] + kk * paradigm['period_s']
        ax.axvspan(s, s + on, color='gold', alpha=0.25, lw=0)
    for kname in ('inactive', 'counter-active', 'active'):
        m, se = ts[kname]
        ax.fill_between(L.t, m - se, m + se, color=COL[kname], alpha=0.3, lw=0)
        ax.plot(L.t, m, color=COL[kname], lw=1.0, label=lab(kname))
    ax.set_xlim(L.t[0] - tr, L.t[-1] + tr)
    ax.set_xlabel('time since sequence start (s)')
    ax.set_ylabel('signal change (%)')
    ax.legend(loc='upper right', ncol=3, fontsize=8, framealpha=0.9)
    ax.set_title('mean time series of each voxel set (± SE over voxels); shaded = task windows', fontsize=10)

    # --- cycle average (mean +- SE over cycles of each set's mean series)
    ax = fig.add_subplot(gs[1, 1])
    period = paradigm['period_s']
    ax.axvspan(0, on, color='gold', alpha=0.25, lw=0)
    n_cyc = 0
    for kname in ('inactive', 'counter-active', 'active'):
        c, m_, se_, n_cyc = cycle_average(L, L.detrend(np.nan_to_num(ts[kname][0])), paradigm)
        ax.fill_between(c, m_ - se_, m_ + se_, color=COL[kname], alpha=0.3, lw=0)
        ax.plot(c, m_, color=COL[kname], lw=1.4, label=kname)
    ax.set_xlim(0, period)
    ax.set_xlabel('time in task cycle (s)')
    ax.set_ylabel('signal change (%), drift removed')
    ax.legend(fontsize=7, loc='best')
    ax.set_title(f'cycle average (mean ± SE over {n_cyc} cycles)', fontsize=10)

    # --- spectrum (mean +- SE over voxels of the per-voxel periodogram)
    ax = fig.add_subplot(gs[1, 2])
    fs = 1 / tr
    ax.axvspan(*RESP_BAND, color='tab:green', alpha=0.15, lw=0, label='respiration 0.2-0.4 Hz')
    ca = sorted(alias(x, fs) for x in np.linspace(CARDIAC_BAND[0], CARDIAC_BAND[1], 50))
    ax.axvspan(ca[0], ca[-1], color='tab:red', alpha=0.10, lw=0,
               label=f'cardiac 0.8-1.3 Hz as sampled ({ca[0]:.2f}-{ca[-1]:.2f} Hz)')
    for h, fh in enumerate([f0, 3 * f0, 5 * f0]):
        ax.axvline(fh, color='gold', lw=2 if h == 0 else 1, alpha=0.9, label='task fundamental, odd harmonics' if h == 0 else None)
    ax.axvline(LOWBAND_HZ, color='gray', ls='--', lw=1, label=f'{LOWBAND_HZ:g} Hz')
    keep = f_p >= 0.01  # above the drift terms, so the empty lowest bins do not set the axis
    lo_hi = []
    for kname in ('inactive', 'counter-active', 'active'):
        m, se = ps[kname]
        if n_set[kname] == 0:
            continue
        ax.fill_between(f_p[keep], np.clip(m - se, 1e-12, None)[keep], (m + se)[keep], color=COL[kname], alpha=0.3, lw=0)
        ax.plot(f_p[keep], m[keep], color=COL[kname], lw=1.0, label=kname)
        lo_hi += [float(np.nanmin(m[keep])), float(np.nanmax(m[keep]))]
    ax.set_yscale('log')
    if lo_hi:
        ax.set_ylim(min(lo_hi) / 2, max(lo_hi) * 2)
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('power density (%$^2$/Hz), mean over voxels')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.set_title(f'spectrum (Hann, drift removed; resolution {1 / (L.nt * tr):.4f} Hz; ± SE over voxels)', fontsize=10)

    # --- split-half scatter (odd vs even cycles) and how to read it
    ax = fig.add_subplot(gs[2, 1])
    lim = 10
    hb = ax.hexbin(np.clip(r1.z, -lim, lim), np.clip(r2.z, -lim, lim), gridsize=70, bins='log', mincnt=1,
                   cmap='viridis', extent=(-lim, lim, -lim, lim))
    for zt, kind in ((rep['z_fdr_odd'], 'x'), (rep['z_fdr_even'], 'y')):
        if np.isfinite(zt):
            for sgn in (-1, 1):
                (ax.axvline if kind == 'x' else ax.axhline)(sgn * zt, color='r', lw=0.8, ls='--')
    ax.set_xlabel('z from the odd cycles only (1, 3, 5, 7)')
    ax.set_ylabel('z from the even cycles only (2, 4, 6, 8)')
    ax.set_title('split-half reproducibility: z of every brain voxel', fontsize=10)
    fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.01).set_label('voxels per bin (log)', fontsize=8)

    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    paras = [
        'HOW TO READ THE SPLIT-HALF PANEL',
        'The GLM is fit twice: on the odd cycles (1, 3, 5, 7) and on the even cycles (2, 4, 6, 8), with the '
        'full-run HRF regressor and drift terms and the AR(1) filter restarted at every cycle. Each hexagon '
        'groups brain voxels: x is their z from the odd fit, y from the even fit; colour is the voxel count (log).',
        'Noise alone gives a round cloud centred on 0 (correlation 0). A stable effect pulls voxels onto '
        'the diagonal; the upper-right lobe responds in both halves with the same sign.',
        'Red dashed lines are each half\'s own FDR threshold: beyond both in one corner = active in both '
        'halves; beyond one only = active in one half (a fragile call).',
        'NUMBERS (TABLE, RIGHT)',
        'Pearson / Spearman: agreement over all voxels (Spearman is not dominated by the strongest few). '
        'Dice = 2 x both / (odd + even), the overlap of the two active sets; compare it with the chance '
        'value, which is what random sets of those sizes give. "Full-run active in both" is the share of '
        'whole-run active voxels that each half confirms alone: low means single-voxel calls are fragile '
        'even when set averages are stable.',
        'WHY ODD/EVEN, AND ITS CAVEAT: interleaving puts both halves across the whole run, so drift, '
        'habituation and the recon pieces affect them equally. But neighbouring cycles share slow noise '
        'and HRF tails, and both halves come from one joint reconstruction, so read the agreement as an '
        'upper bound on test-retest reliability.',
    ]
    ax.text(0.0, 1.0, '\n\n'.join(textwrap.fill(t, width=74) for t in paras), va='top', ha='left',
            fontsize=8, transform=ax.transAxes, linespacing=1.25)

    # --- test-retest: reliability histogram and estimated ROC
    ax = fig.add_subplot(gs[4, 1])
    ks = np.arange(M + 1)
    obs = np.bincount(Rv, minlength=M + 1)[:M + 1]
    ax.bar(ks, np.maximum(obs, 0.5), color='0.75', edgecolor='k', lw=0.6, label='observed', bottom=0)
    if mb is not None:
        exp = n_vox * (mb.lam * _st.binom.pmf(ks, M, mb.p_active[k0]) + (1 - mb.lam) * _st.binom.pmf(ks, M, mb.p_inactive[k0]))
        ax.plot(ks, np.maximum(exp, 1e-3), 'o-', color='tab:orange', label='fitted mixture')
        ax.plot(ks, n_vox * mb.lam * _st.binom.pmf(ks, M, mb.p_active[k0]), '--', color='tab:red', lw=1, label='truly active part')
        ax.plot(ks, n_vox * (1 - mb.lam) * _st.binom.pmf(ks, M, mb.p_inactive[k0]), '--', color='k', lw=1, label='truly inactive part')
    ax.set_yscale('log')
    ax.set_ylim(0.5, n_vox * 3)
    ax.set_xticks(ks)
    ax.set_xlabel(f'R: replicates (of {M}) in which the voxel has z > {Z_GRID[k0]:g}')
    ax.set_ylabel('voxels')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title(f'reliability map histogram (M = {M} replicates of {ncyc_all // M} cycles)', fontsize=10)

    ax = fig.add_subplot(gs[4, 2])
    ax.plot([0, 1], [0, 1], color='0.7', lw=1, ls=':')
    if mb is not None:
        ax.plot(fpr_roc, tpr_roc, color='tab:purple', lw=1.2)
        ax.plot(mb.p_inactive, mb.p_active, 'o', color='tab:purple', ms=5)
        for kk_, zt in enumerate(Z_GRID):
            ax.annotate(f'{zt:g}', (mb.p_inactive[kk_], mb.p_active[kk_]), textcoords='offset points', xytext=(5, -9), fontsize=7)
        ax.plot(mb.p_inactive[kb], mb.p_active[kb], 'o', ms=12, mfc='none', mec='k', ls='--',
                label=f'best of the grid: z > {Z_GRID[kb]:g}')
        ax.legend(fontsize=7, loc='lower right')
    else:
        ax.text(0.5, 0.5, 'mixed-binomial fit failed', ha='center', transform=ax.transAxes)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('false-positive rate p_I (estimated)')
    ax.set_ylabel('true-positive rate p_A')
    ax.set_title('ROC estimated from the replicates (labels = z threshold)', fontsize=10)

    ax = fig.add_subplot(gs[4, 0])
    ax.axis('off')
    paras = [
        'HOW TO READ THE TEST-RETEST PANELS',
        f'(Genovese, Noll & Eddy 1997.) The run is cut into M = {M} replicates, each {ncyc_all // M} cycles taken '
        f'{M} cycles apart, and the GLM is fit to each. At each z threshold a voxel is called active or not '
        'in each replicate. Without ground truth the model estimates it: a voxel is truly active with '
        'probability lambda; if truly active it is called active in a replicate with probability p_A (true-positive '
        'rate), if not with p_I (false-positive rate), independently across replicates. All thresholds are '
        'fit together, sharing lambda.',
        f'Left: voxels by R, the number of replicates in which they exceed z = {Z_GRID[k0]:g}. Bars are observed; '
        'the orange line is the fitted mixture, red and black its truly active and inactive parts. Good fit = '
        'the model describes the data. Voxels piled at R = M with a flat middle are reliably detected; many at '
        'R = 1-2 are calls that do not repeat.',
        'Right: each point is one threshold at (p_I, p_A) estimated from the fit. Up and to the left is better; '
        'the dotted diagonal is chance; the circle maximises lambda p_A + (1 - lambda)(1 - p_I) over the grid.',
        'Caveats: the model assumes independent, identically distributed voxels and replicates. Spatial '
        'clustering and the shared reconstruction break that, and each replicate has only '
        f'{ncyc_all // M} cycles of power, so absolute rates are indicative.',
    ]
    ax.text(0.0, 1.0, '\n\n'.join(textwrap.fill(t, width=80) for t in paras), va='top', ha='left',
            fontsize=8, transform=ax.transAxes, linespacing=1.25)

    # --- right column: statistics table
    ax = fig.add_subplot(gs[:, 3])
    ax.axis('off')

    def ratio(v):
        return 'n/a' if not np.isfinite(v) else f'{v:.1f}x'

    def num(v, spec):
        return 'n/a' if v is None or not np.isfinite(v) else format(v, spec)

    mm_c = [(d - sh // 2) * v for d, sh, v in zip(pk, shape, vox)]
    rows = [
        ('H', 'Activation (FDR, two-sided, brain mask)'),
        ('R', 'q', f'{Q_FDR:g}'),
        ('R', '|z| threshold', zthr_s),
        ('R', 'brain-mask voxels', f'{n_vox}'),
        ('R', 'active (z > 0)', f'{n_set["active"]}'),
        ('R', 'counter-active (z < 0)', f'{n_set["counter-active"]}'),
        ('R', 'inactive', f'{n_set["inactive"]}'),
        ('R', 'mean task effect, active', num(eff['active'], '.2f') + ' %'),
        ('R', 'mean task effect, counter', num(eff['counter-active'], '.2f') + ' %'),
        ('H', 'Full-run GLM'),
        ('R', 'AR(1) rho' if summ['ar1'] else 'prewhitening', f'{sm["rho"]:.3f}' if summ['ar1'] else 'none (low-band)'),
        ('R', 'dof', f'{sm["dof"]:.0f}'),
        ('R', 'TR / frames', f'{tr:.4f} s / {L.nt}'),
        ('R', 'task period / f0', f'{paradigm["period_s"]:g} s / {f0:.4f} Hz'),
        ('R', 'highest z (voxel)', f'{zpk:.2f}'),
        ('R', 'at (x, y, z) mm', f'({mm_c[0]:.0f}, {mm_c[1]:.0f}, {mm_c[2]:.0f})'),
        ('H', 'Split half (odd vs even cycles)'),
        ('R', 'dof (odd / even)', f'{r1.dof:.0f} / {r2.dof:.0f}'),
        ('R', 'rho (odd / even)', f'{r1.rho:.2f} / {r2.rho:.2f}'),
        ('R', '|z| threshold (odd / even)', f'{rep["z_fdr_odd"]:.2f} / {rep["z_fdr_even"]:.2f}'),
        ('R', 'z correlation, Pearson', f'{rep["z_pearson"]:.2f}'),
        ('R', 'z correlation, Spearman', f'{rep["z_spearman"]:.2f}'),
        ('R', 'active: odd / even', f'{n1} / {n2}'),
        ('R', 'active in both', f'{nb_}'),
        ('R', 'Dice (chance)', f'{rep["dice"]:.2f} ({rep["dice_chance"]:.3f})'),
        ('R', 'full-run active in both', num(100 * rep['frac_full_active_in_both'], '.0f') + ' %'),
        ('H', f'Test-retest ({M} replicates of {ncyc_all // M} cycles)'),
        ('R', 'lambda (truly active share)', num(tr_stats.get('lam'), '.4f')),
        ('R', 'estimated active voxels', num(tr_stats.get('n_est_active'), '.0f')),
        ('R', 'best z on the grid', num(tr_stats.get('best_z'), 'g')),
        ('R', 'p_A / p_I there', f'{num(tr_stats.get("best_tpr"), ".2f")} / {num(tr_stats.get("best_fpr"), ".3f")}'),
        ('R', 'area under estimated ROC', num(tr_stats.get('auc'), '.2f')),
        ('H', 'Task / non-task power'),
        ('C', ['', 'active', 'inactive', 'counter']),
        ('C', ['f0 vs neighbours'] + [ratio(tp[k]['f0_over_local']) for k in sets]),
        ('C', ['f0 vs 0.01-0.15 Hz'] + [ratio(tp[k]['f0_over_band']) for k in sets]),
        ('C', ['odd harm. vs neighbours'] + [ratio(tp[k]['odd_over_local']) for k in sets]),
        ('C', ['odd harm. vs 0.01-0.15 Hz'] + [ratio(tp[k]['odd_over_band']) for k in sets]),
    ]
    step = 1.0 / (len(rows) + 2)
    y = 1.0 - 0.5 * step
    for r in rows:
        if r[0] == 'H':
            y -= 0.4 * step
            ax.text(0.0, y, r[1], fontsize=10, fontweight='bold', va='center', transform=ax.transAxes)
            ax.plot([0, 1], [y - 0.45 * step] * 2, color='0.5', lw=0.6, transform=ax.transAxes, clip_on=False)
        elif r[0] == 'R':
            ax.text(0.0, y, r[1], fontsize=9, va='center', transform=ax.transAxes)
            ax.text(1.0, y, r[2], fontsize=9, va='center', ha='right', transform=ax.transAxes)
        else:
            ax.text(0.0, y, r[1][0], fontsize=8.5, va='center', transform=ax.transAxes)
            for xx, v in zip((0.62, 0.79, 1.0), r[1][1:]):
                ax.text(xx, y, v, fontsize=8.5, va='center', ha='right', transform=ax.transAxes)
        y -= step

    fig.suptitle(name, fontsize=12)
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    metrics = {'name': name, 'q_fdr': Q_FDR, 'z_fdr': z_fdr, 'n_active': n_set['active'],
               'n_counter_active': n_set['counter-active'], 'n_inactive': n_set['inactive'], 'n_mask': n_vox,
               'mean_task_effect_active_pct': eff['active'],
               'mean_task_effect_counter_active_pct': eff['counter-active'], 'split_half_odd_even': rep,
               'test_retest': tr_stats,
               'task_power': {k.replace('-', '_'): v for k, v in tp.items()}, 'task_f0_hz': f0}
    with open(f'{out_dir}/{name}_metrics.json', 'w') as fh:
        json.dump(metrics, fh, indent=1, default=float)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--only', default='')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    paradigm = json.load(open(f'{a.root}/paradigm.json'))
    dirs = sorted(d for d in glob.glob(f'{a.root}/analysis/*') if os.path.isdir(d) and not d.endswith('figures'))
    dirs = [d for d in dirs if a.only in os.path.basename(d)
            and glob.glob(f'{d}/*_summary.json')]  # skip archived/, lowpass_check/, ...
    cache = {}
    for d in dirs:
        base, run, tag, lowband = parse(d)
        summ = json.load(open(f'{d}/{base}_summary.json'))
        h5 = recon_path(a.root, run, tag, summ)
        try:
            if h5 not in cache:
                cache.clear()

                def onsets_fn(t0, T, paradigm=paradigm):
                    o = [paradigm['task_onset_s'] + k * paradigm['period_s'] - t0 for k in range(paradigm['n_cycles'])]
                    return [x for x in o if x + paradigm['on_s'] > 0 and x < T]

                cache[h5] = Loaded(h5, summ['tr'], onsets_fn, paradigm)
            print(make_figure(d, loaded=cache[h5], paradigm=paradigm, force=a.force), flush=True)
        except Exception as e:  # keep going: one broken folder should not stop the rest
            print(f'FAILED {d}: {e!r}', flush=True)


if __name__ == '__main__':
    main()
