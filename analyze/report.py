#!/usr/bin/env python
"""One Markdown report per analysis/ folder, with each plot as its own PNG, in analysis/reports/<name>/.

    report.md, activation_{1..5}.png, timeseries.png, cycle_average.png, spectrum.png, split_half.png,
    reliability_histogram.png, roc.png, metrics.json     (and analysis/reports/README.md indexing them)

Sections, in order: (1) activation maps: z of the Benjamini-Hochberg FDR (q < 0.05, two-sided, brain mask)
survivors over the T1-weighted dual-echo GRE (on the EPI grid), orthogonal views through the five strongest
distinct peaks; (2) time and frequency domain: active (red), counter-active (blue) and inactive (black) voxel
sets as time series, cycle average and spectrum, and task / non-task power ratios; (3) reproducibility:
odd vs even cycle split half, and test-retest with the mixed-binomial model and estimated ROC (Genovese,
Noll & Eddy 1997).

Reads the saved NIfTIs (*_sum_t, *_sum_z, *_sum_psc, *_mask), the reconstruction's series and the GRE in
recon/<run>_preprocessed.h5; the full-run maps are never refit; the split-half and replicate fits are done
here (about a minute). Run with .venv-recon:
    python -m analyze.report [--root DIR] [--only SUBSTRING] [--force]
"""
import argparse
import glob
import json
import os
import re
import sys
from types import SimpleNamespace

REPO = os.environ.get('ARBEPI_REPO', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


N_PEAKS = 5  # ortho views, one per activation peak
PEAK_SEP_MM = 15.0  # peaks closer than this to a stronger one belong to the same cluster
DPI = 130
COL = {'active': 'tab:red', 'inactive': 'k', 'counter-active': 'tab:blue'}


def load_underlay(gre_h5):
    """The deGRE's first-echo magnitude (TE = 1 / f_fat, the T1-weighted contrast of the dual-echo GRE), which
    preprocessing resamples onto the EPI grid, so it overlays the activation maps voxel for voxel."""
    with h5py.File(gre_h5) as f:
        return np.asarray(f['degre/img_echoes'][..., 0], dtype=np.float32)


def top_peaks(z, brain, vox, n=N_PEAKS, sep_mm=PEAK_SEP_MM):
    """Indices (i, j, k) of the n highest-z brain voxels, each at least sep_mm from every stronger one picked
    (the next-highest voxel of a cluster is its neighbour and adds no new activation pattern)."""
    zb = np.where(brain & (z > 0), z, -np.inf)
    order = np.argsort(zb.ravel())[::-1][:200000]
    picked = []
    for o in order:
        if not np.isfinite(zb.flat[o]):
            break
        p = np.array(np.unravel_index(o, z.shape))
        if all(np.linalg.norm((p - q) * vox) >= sep_mm for q in picked):
            picked.append(p)
            if len(picked) == n:
                break
    return [tuple(int(x) for x in p) for p in picked]


def _extent(n, d):
    lo = (-(n // 2) - 0.5) * d
    return lo, lo + n * d


def plot_ortho(fn, under, z, survm, brain, peak, vox, z_fdr, title):
    """Axial, coronal and sagittal slices through `peak`: FDR survivors' z over the GRE, mm from the centre voxel."""
    shape = under.shape
    vmax = float(np.percentile(under[brain], 99.5))
    cmap, norm = _zcmap(z_fdr), Normalize(-Z_MAX, Z_MAX)
    views = [('axial', (0, 1), 'x', 'y'), ('coronal', (0, 2), 'x', 'z'), ('sagittal', (1, 2), 'y', 'z')]
    widths = [shape[a] * vox[a] for (_, (a, _b), _, _) in views]
    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=widths, height_ratios=[1, 0.05])
    for col, (name, (da, db), la, lb) in enumerate(views):
        sl = [slice(None)] * 3
        sl[3 - da - db] = peak[3 - da - db]
        sl = tuple(sl)
        xl, xh = _extent(shape[da], vox[da])
        yl, yh = _extent(shape[db], vox[db])
        ext = [xl, xh, yl, yh]
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(under[sl].T, cmap='gray', vmin=0, vmax=vmax, origin='lower', extent=ext, interpolation='nearest')
        ax.imshow(np.ma.masked_where(~survm[sl].T, z[sl].T), cmap=cmap, norm=norm, origin='lower', extent=ext,
                  interpolation='nearest', alpha=0.85)
        ax.axvline((peak[da] - shape[da] // 2) * vox[da], color='c', lw=0.6, alpha=0.8)
        ax.axhline((peak[db] - shape[db] // 2) * vox[db], color='c', lw=0.6, alpha=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(f'{la} (mm from centre)')
        ax.set_ylabel(f'{lb} (mm from centre)')
    cax = fig.add_subplot(gs[1, :])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cb.set_label(f'z of FDR survivors (|z| > {z_fdr:.2f}) over the GRE', fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)


def plot_timeseries(fn, L, ts, n_set, paradigm):
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    for kk in range(paradigm['n_cycles']):
        s = paradigm['task_onset_s'] + kk * paradigm['period_s']
        ax.axvspan(s, s + paradigm['on_s'], color='gold', alpha=0.25, lw=0)
    for k in ('inactive', 'counter-active', 'active'):
        m, se = ts[k]
        ax.fill_between(L.t, m - se, m + se, color=COL[k], alpha=0.3, lw=0)
        ax.plot(L.t, m, color=COL[k], lw=1.0, label=f'{k} (n = {n_set[k]})')
    ax.set_xlim(L.t[0] - L.tr, L.t[-1] + L.tr)
    ax.set_xlabel('time since sequence start (s)')
    ax.set_ylabel('signal change (%)')
    ax.legend(loc='upper right', ncol=3, fontsize=8, framealpha=0.9)
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)


def plot_cycle_average(fn, L, ts, paradigm):
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax.axvspan(0, paradigm['on_s'], color='gold', alpha=0.25, lw=0)
    n_cyc = 0
    for k in ('inactive', 'counter-active', 'active'):
        c, m, se, n_cyc = cycle_average(L, L.detrend(np.nan_to_num(ts[k][0])), paradigm)
        ax.fill_between(c, m - se, m + se, color=COL[k], alpha=0.3, lw=0)
        ax.plot(c, m, color=COL[k], lw=1.4, label=k)
    ax.set_xlim(0, paradigm['period_s'])
    ax.set_xlabel('time in task cycle (s)')
    ax.set_ylabel('signal change (%), drift removed')
    ax.legend(fontsize=8, loc='best')
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)
    return n_cyc


def plot_spectrum(fn, f_p, ps, n_set, f0, tr, nt):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    fs = 1 / tr
    ax.axvspan(*RESP_BAND, color='tab:green', alpha=0.15, lw=0, label='respiration 0.2-0.4 Hz')
    ca = sorted(alias(x, fs) for x in np.linspace(CARDIAC_BAND[0], CARDIAC_BAND[1], 50))
    ax.axvspan(ca[0], ca[-1], color='tab:red', alpha=0.10, lw=0,
               label=f'cardiac 0.8-1.3 Hz as sampled ({ca[0]:.2f}-{ca[-1]:.2f} Hz)')
    for h, fh in enumerate([f0, 3 * f0, 5 * f0]):
        ax.axvline(fh, color='gold', lw=2 if h == 0 else 1, alpha=0.9,
                   label='task fundamental, odd harmonics' if h == 0 else None)
    ax.axvline(LOWBAND_HZ, color='gray', ls='--', lw=1, label=f'{LOWBAND_HZ:g} Hz')
    keep = f_p >= 0.01  # above the drift terms, so the empty lowest bins do not set the axis
    lo_hi = []
    for k in ('inactive', 'counter-active', 'active'):
        m, se = ps[k]
        if n_set[k] == 0:
            continue
        ax.fill_between(f_p[keep], np.clip(m - se, 1e-12, None)[keep], (m + se)[keep], color=COL[k], alpha=0.3, lw=0)
        ax.plot(f_p[keep], m[keep], color=COL[k], lw=1.0, label=k)
        lo_hi += [float(np.nanmin(m[keep])), float(np.nanmax(m[keep]))]
    ax.set_yscale('log')
    if lo_hi:
        ax.set_ylim(min(lo_hi) / 2, max(lo_hi) * 2)
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('power density (%$^2$/Hz), mean over voxels')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)


def plot_split_half(fn, r1, r2, rep, odd, even):
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    lim = 10
    hb = ax.hexbin(np.clip(r1.z, -lim, lim), np.clip(r2.z, -lim, lim), gridsize=70, bins='log', mincnt=1,
                   cmap='viridis', extent=(-lim, lim, -lim, lim))
    for zt, kind in ((rep['z_fdr_odd'], 'x'), (rep['z_fdr_even'], 'y')):
        if np.isfinite(zt):
            for sgn in (-1, 1):
                (ax.axvline if kind == 'x' else ax.axhline)(sgn * zt, color='r', lw=0.8, ls='--')
    ax.set_xlabel(f'z from the odd cycles only ({odd})')
    ax.set_ylabel(f'z from the even cycles only ({even})')
    fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.01).set_label('voxels per bin (log)', fontsize=8)
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)


def plot_reliability(fn, Rv, mb, M, k0, n_vox):
    from scipy import stats as st
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ks = np.arange(M + 1)
    obs = np.bincount(Rv, minlength=M + 1)[:M + 1]
    ax.bar(ks, np.maximum(obs, 0.5), color='0.75', edgecolor='k', lw=0.6, label='observed')
    if mb is not None:
        pa, pi = mb.p_active[k0], mb.p_inactive[k0]
        exp = n_vox * (mb.lam * st.binom.pmf(ks, M, pa) + (1 - mb.lam) * st.binom.pmf(ks, M, pi))
        ax.plot(ks, np.maximum(exp, 1e-3), 'o-', color='tab:orange', label='fitted mixture')
        ax.plot(ks, n_vox * mb.lam * st.binom.pmf(ks, M, pa), '--', color='tab:red', lw=1, label='truly active part')
        ax.plot(ks, n_vox * (1 - mb.lam) * st.binom.pmf(ks, M, pi), '--', color='k', lw=1, label='truly inactive part')
    ax.set_yscale('log')
    ax.set_ylim(0.5, n_vox * 3)
    ax.set_xticks(ks)
    ax.set_xlabel(f'R: replicates (of {M}) in which the voxel has z > {Z_GRID[k0]:g}')
    ax.set_ylabel('voxels')
    ax.legend(fontsize=8, loc='upper right')
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)


def plot_roc(fn, mb, kb, fpr_roc, tpr_roc):
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color='0.7', lw=1, ls=':')
    if mb is not None:
        ax.plot(fpr_roc, tpr_roc, color='tab:purple', lw=1.2)
        ax.plot(mb.p_inactive, mb.p_active, 'o', color='tab:purple', ms=5)
        for i, zt in enumerate(Z_GRID):
            ax.annotate(f'{zt:g}', (mb.p_inactive[i], mb.p_active[i]), textcoords='offset points', xytext=(5, -9),
                        fontsize=7)
        ax.plot(mb.p_inactive[kb], mb.p_active[kb], 'o', ms=12, mfc='none', mec='k', ls='--',
                label=f'best of the grid: z > {Z_GRID[kb]:g}')
        ax.legend(fontsize=8, loc='lower right')
    else:
        ax.text(0.5, 0.5, 'mixed-binomial fit failed', ha='center', transform=ax.transAxes)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('false-positive rate p_I (estimated)')
    ax.set_ylabel('true-positive rate p_A (estimated)')
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)


def _num(v, spec):
    return 'n/a' if v is None or not np.isfinite(v) else format(v, spec)


def _ratio(v):
    return 'n/a' if not np.isfinite(v) else f'{v:.1f}x'


def _md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def make_report(analysis_dir, loaded=None, out_root=None, paradigm=None, force=False, h5=None,
                gre_h5=None, base=None, title=None, note=None):
    """Write <out_root>/<name>/report.md with its PNGs and metrics.json; returns the report path.

    A folder named like the 20261006tap analyses (<run>_<tag>) needs only analysis_dir. Otherwise pass h5 (the
    reconstruction), gre_h5 (a preprocessed file holding degre/img_echoes), base (the file prefix of the
    analysis folder's NIfTIs and summary) and optionally title and note (a paragraph under the title)."""
    from scipy import stats as st
    root = os.path.dirname(os.path.dirname(os.path.normpath(analysis_dir)))
    base_parsed = base is None
    if base is None:
        base, run, tag, lowband = parse(analysis_dir)
        h5 = h5 or None
        gre_h5 = gre_h5 or f'{root}/recon/{run}_preprocessed.h5'
    name = os.path.basename(os.path.normpath(analysis_dir))
    out_root = out_root or f'{os.path.dirname(os.path.normpath(analysis_dir))}/reports'
    out = f'{out_root}/{name}'
    os.makedirs(out, exist_ok=True)
    report_fn = f'{out}/report.md'
    if os.path.exists(report_fn) and not force:
        return report_fn
    paradigm = paradigm or json.load(open(f'{root}/paradigm.json'))
    summ = json.load(open(f'{analysis_dir}/{base}_summary.json'))
    tr = summ['tr']
    sm = summ['scales']['sum']
    z = np.asarray(nib.load(f'{analysis_dir}/{base}_sum_z.nii.gz').dataobj, dtype=np.float32)
    t_vol = np.asarray(nib.load(f'{analysis_dir}/{base}_sum_t.nii.gz').dataobj, dtype=np.float32)
    psc_map = np.asarray(nib.load(f'{analysis_dir}/{base}_sum_psc.nii.gz').dataobj, dtype=np.float32)
    brain = np.asarray(nib.load(f'{analysis_dir}/{base}_mask.nii.gz').dataobj) > 0

    def onsets_fn(t0, T):
        o = [paradigm['task_onset_s'] + k * paradigm['period_s'] - t0 for k in range(paradigm['n_cycles'])]
        return [x for x in o if x + paradigm['on_s'] > 0 and x < T]

    h5 = h5 or recon_path(root, run, tag, summ)
    L = loaded or Loaded(h5, tr, onsets_fn, paradigm)
    vox = np.asarray(summ['voxel_size_mm'], dtype=float)
    shape = z.shape

    # ---- the three sets, on the saved t map (brain-mask voxels only)
    surv = fdr_survivors(t_vol[brain], sm['dof'])
    zb_ = z[brain]
    sets = {'active': surv & (zb_ > 0), 'inactive': ~surv, 'counter-active': surv & (zb_ < 0)}
    n_set = {k: int(v.sum()) for k, v in sets.items()}
    n_vox = int(brain.sum())
    z_fdr = float(np.abs(zb_[surv]).min()) if surv.any() else np.inf
    zthr_s = f'{z_fdr:.2f}' if np.isfinite(z_fdr) else 'none'
    survm = np.zeros(shape, bool)
    survm[brain] = surv

    under = L.rec.mean_signal
    Y = L.rec.series['sum'][brain].astype(np.float32)
    base_v = under[brain].astype(np.float64)
    psc = (100 * (Y - Y.mean(1, keepdims=True)) / base_v[:, None]).astype(np.float32)
    psc_vox = psc_map[brain]
    eff = {k: (float(psc_vox[sets[k]].mean()) if n_set[k] else np.nan) for k in ('active', 'counter-active')}

    # ---- 1. activation maps over the T1-weighted GRE, at the strongest distinct peaks
    gre = load_underlay(gre_h5)
    if gre.shape != shape:
        raise ValueError(f'GRE {gre.shape} is not on the EPI grid {shape}')
    peaks = top_peaks(z, brain, vox)
    peak_rows = []
    for n, pk in enumerate(peaks, 1):
        mm = [(d - sh // 2) * v for d, sh, v in zip(pk, shape, vox)]
        peak_rows.append(dict(rank=n, voxel=list(pk), mm=mm, z=float(z[pk]), psc=float(psc_map[pk]),
                              survives=bool(survm[pk])))
        plot_ortho(f'{out}/activation_{n}.png', gre, z, survm, brain, pk, vox, z_fdr,
                   f'peak {n}: z = {z[pk]:.2f} at (x, y, z) = ({mm[0]:.0f}, {mm[1]:.0f}, {mm[2]:.0f}) mm')

    # ---- 3a. odd vs even cycles (split half)
    ncyc_all = paradigm['n_cycles']
    (r1, r2) = fit_cycle_groups(L, Y, paradigm, [range(0, ncyc_all, 2), range(1, ncyc_all, 2)])
    s1, s2 = fdr_survivors(r1.t, r1.dof), fdr_survivors(r2.t, r2.dof)
    a1, a2 = s1 & (r1.z > 0), s2 & (r2.z > 0)
    n1, n2, nb_ = int(a1.sum()), int(a2.sum()), int((a1 & a2).sum())
    act = sets['active']
    rep = {
        'z_pearson': float(np.corrcoef(r1.z, r2.z)[0, 1]),
        'z_spearman': float(st.spearmanr(r1.z, r2.z)[0]),
        'n_active_odd': n1, 'n_active_even': n2, 'n_both': nb_,
        'dice': float(2 * nb_ / (n1 + n2)) if n1 + n2 else float('nan'),
        'dice_chance': float(2 * (n1 * n2 / n_vox) / (n1 + n2)) if n1 + n2 else float('nan'),
        'frac_full_active_in_both': float((a1 & a2 & act).sum() / act.sum()) if n_set['active'] else float('nan'),
        'z_fdr_odd': float(np.abs(r1.z[s1]).min()) if s1.any() else float('inf'),
        'z_fdr_even': float(np.abs(r2.z[s2]).min()) if s2.any() else float('inf'),
    }

    # ---- 3b. test-retest: M interleaved replicates, mixed-binomial model (Genovese, Noll & Eddy 1997)
    M = max(2, min(TR_REPLICATES, ncyc_all // 2))
    reps = fit_cycle_groups(L, Y, paradigm, [range(r, ncyc_all, M) for r in range(M)])
    scores = np.stack([r.z for r in reps])
    mb = None
    try:
        mb = fit_mixed_binomial(levels(scores, Z_GRID), n_thresholds=len(Z_GRID))
    except Exception as e:  # e.g. nothing exceeds any threshold
        print(f'  mixed-binomial fit failed for {name}: {e!r}', flush=True)
    k0 = int(np.argmin(np.abs(Z_GRID - Z_HIST)))
    Rv = (scores >= Z_GRID[k0]).sum(axis=0)
    tr_stats = {'M': M, 'cycles_per_replicate': ncyc_all // M, 'thresholds_z': Z_GRID.tolist()}
    kb, fpr_roc, tpr_roc = 0, None, None
    if mb is not None:
        crit = mb.lam * mb.p_active + (1 - mb.lam) * (1 - mb.p_inactive)
        kb = int(np.argmax(crit))
        order = np.argsort(mb.p_inactive)
        fpr_roc = np.r_[0.0, mb.p_inactive[order], 1.0]
        tpr_roc = np.r_[0.0, mb.p_active[order], 1.0]
        tr_stats.update(lam=float(mb.lam), n_est_active=float(mb.lam * n_vox), tpr=mb.p_active.tolist(),
                        fpr=mb.p_inactive.tolist(), best_z=float(Z_GRID[kb]), best_tpr=float(mb.p_active[kb]),
                        best_fpr=float(mb.p_inactive[kb]), auc=float(np.trapezoid(tpr_roc, fpr_roc)))

    # ---- 2. one pass: mean +- SE over voxels of the series and of the periodogram, per set
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
            mm_ = m[s:s + 20000]
            if mm_.any():
                S[k] += ch[mm_].sum(0)
                SS[k] += (ch[mm_] ** 2).sum(0)
                PS[k] += Pc[mm_].sum(0)
                PSS[k] += (Pc[mm_] ** 2).sum(0)

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

    plot_timeseries(f'{out}/timeseries.png', L, ts, n_set, paradigm)
    n_cyc = plot_cycle_average(f'{out}/cycle_average.png', L, ts, paradigm)
    plot_spectrum(f'{out}/spectrum.png', f_p, ps, n_set, f0, tr, nt)
    odd_s = ', '.join(str(c + 1) for c in range(0, ncyc_all, 2))
    even_s = ', '.join(str(c + 1) for c in range(1, ncyc_all, 2))
    plot_split_half(f'{out}/split_half.png', r1, r2, rep, odd_s, even_s)
    plot_reliability(f'{out}/reliability_histogram.png', Rv, mb, M, k0, n_vox)
    plot_roc(f'{out}/roc.png', mb, kb, fpr_roc, tpr_roc)

    # ---- the report
    md = []
    md.append(f'# Functional analysis: {title or name}\n')
    if note:
        md.append(note + '\n')
    md.append((f'{run.replace("_", " ")}, regularization `{tag}`, ' if base_parsed else '') +
              f'reconstruction `{h5}`. '
              f'{L.nt} frames at TR {tr:.4f} s; block task of {paradigm["on_s"]:g} s on / '
              f'{paradigm["period_s"] - paradigm["on_s"]:g} s off, {ncyc_all} cycles, first onset at '
              f'{paradigm["task_onset_s"]:g} s. Only voxels inside the brain mask ({n_vox}) are ever classified.\n')
    md.append(_md_table(['', 'value'], [
        ['activation criterion', f'Benjamini-Hochberg FDR q < {Q_FDR:g}, two-sided, over the brain mask'],
        ['\\|z\\| threshold at that q', zthr_s],
        ['active (survivor, z > 0)', n_set['active']],
        ['counter-active (survivor, z < 0)', n_set['counter-active']],
        ['inactive (not a survivor)', n_set['inactive']],
        ['mean task effect, active / counter-active', f'{_num(eff["active"], ".2f")} % / '
                                                      f'{_num(eff["counter-active"], ".2f")} %'],
        ['AR(1) rho / dof', (f'{sm["rho"]:.3f}' if summ['ar1'] else 'none') + f' / {sm["dof"]:.0f}'],
    ]) + '\n')

    md.append('## 1. Activation maps\n')
    md.append(f'z of the FDR survivors (red/yellow z > 0, blue/cyan z < 0), overlaid on the first-echo magnitude of the '
              'dual-echo GRE (T1-weighted), which preprocessing resamples onto the EPI grid, so the overlay is voxel '
              'for voxel; residual misregistration (B0 distortion of the EPI, head motion between the GRE and the '
              'EPI) is not corrected. Axes are mm from the centre voxel. The five views are cut through the five '
              f'highest-z voxels that are at least {PEAK_SEP_MM:g} mm apart (otherwise the next-highest voxels are '
              'neighbours of the first peak and show the same cluster). Cyan crosshairs mark the peak voxel.\n')
    md.append(_md_table(['peak', 'voxel (i, j, k)', '(x, y, z) mm', 'z', 'task effect (%)', 'FDR survivor'],
                        [[r['rank'], tuple(r['voxel']), '({:.0f}, {:.0f}, {:.0f})'.format(*r['mm']),
                          f'{r["z"]:.2f}', f'{r["psc"]:.2f}', 'yes' if r['survives'] else 'no']
                         for r in peak_rows]) + '\n')
    for r in peak_rows:
        md.append(f'![peak {r["rank"]}](activation_{r["rank"]}.png)\n')

    md.append('## 2. Time and frequency domain\n')
    md.append('Voxels are divided into three sets by the full-run FDR result: **active** (red), **counter-active** '
              '(blue; significant with the opposite sign) and **inactive** (black; everything else in the brain '
              'mask). Each curve is the mean over the voxels of the set. Active and counter-active voxels were '
              'selected because they fit the task regressor, so their agreement with the task is partly built in '
              '(selection bias); the inactive set is the unselected reference. Percent change is relative to each '
              'voxel\'s temporal mean.\n')
    md.append('### Time series\n')
    md.append('Shaded bands: task windows. Curve bands: ± SE over the voxels of the set.\n')
    md.append('![time series](timeseries.png)\n')
    md.append('### Cycle average\n')
    md.append(f'Each set\'s mean series, drift removed, folded on the {paradigm["period_s"]:g} s task cycle; band: '
              f'± SE over the {n_cyc} cycles (a different error from the voxel SE above). Gold: task on.\n')
    md.append('![cycle average](cycle_average.png)\n')
    md.append('### Spectrum\n')
    md.append(f'Mean over voxels of each voxel\'s Hann periodogram (drift removed; resolution '
              f'{1 / (L.nt * tr):.4f} Hz); band: ± SE over voxels. Gold lines: the task fundamental '
              f'{f0:.4f} Hz and its odd harmonics (a 50 % duty square wave has no even ones). Green: usual '
              f'breathing. Red: where the usual heart rate lands after aliasing by the {1 / tr:.2f} Hz frame rate. '
              f'The axis starts at 0.01 Hz, above the drift terms.\n')
    md.append('![spectrum](spectrum.png)\n')
    md.append('### Task power relative to non-task power\n')
    md.append('Task power is the sum of the f0, 3 f0, 5 f0 bins (each ± 1 bin, the Hann main lobe). The reference is the '
              'mean power per bin either in the neighbouring bins (± 0.02 Hz, task bins excluded) or over '
              '0.01-0.15 Hz (task bins excluded). A ratio of 1 is no more power than the noise around it.\n')
    md.append(_md_table(['', 'active', 'inactive', 'counter-active'], [
        ['f0 vs neighbours'] + [_ratio(tp[k]['f0_over_local']) for k in sets],
        ['f0 vs 0.01-0.15 Hz'] + [_ratio(tp[k]['f0_over_band']) for k in sets],
        ['odd harmonics vs neighbours'] + [_ratio(tp[k]['odd_over_local']) for k in sets],
        ['odd harmonics vs 0.01-0.15 Hz'] + [_ratio(tp[k]['odd_over_band']) for k in sets]]) + '\n')

    md.append('## 3. Reproducibility\n')
    md.append('### Split half: odd vs even cycles\n')
    md.append(f'The GLM is fit twice, on the odd cycles ({odd_s}) and on the even cycles ({even_s}), with the '
              'full-run HRF regressor and drift terms restricted to each half\'s frames and the AR(1) filter restarted '
              'at every cycle. Interleaving puts both halves across the whole run, so drift, habituation and the '
              'reconstruction pieces affect them equally; but neighbouring cycles share slow noise and HRF tails, and '
              'both halves come from one joint reconstruction, so read the agreement as an upper bound on '
              'test-retest reliability.\n')
    md.append('Each hexagon groups brain voxels by their z from the odd (x) and even (y) fits. Noise alone gives a '
              'round cloud centred on 0; a stable effect pulls voxels onto the diagonal. Red dashed lines are each '
              'half\'s own FDR threshold: beyond both in the upper-right corner is active in both halves, beyond '
              'one only is a fragile call.\n')
    md.append('![split half](split_half.png)\n')
    md.append(_md_table(['', 'value'], [
        ['dof (odd / even)', f'{r1.dof:.0f} / {r2.dof:.0f}'],
        ['AR(1) rho (odd / even)', f'{r1.rho:.2f} / {r2.rho:.2f}'],
        ['\\|z\\| threshold (odd / even)', f'{rep["z_fdr_odd"]:.2f} / {rep["z_fdr_even"]:.2f}'],
        ['z correlation, Pearson', f'{rep["z_pearson"]:.2f}'],
        ['z correlation, Spearman', f'{rep["z_spearman"]:.2f}'],
        ['active: odd / even', f'{n1} / {n2}'],
        ['active in both', nb_],
        ['Dice (chance level)', f'{rep["dice"]:.2f} ({rep["dice_chance"]:.3f})'],
        ['full-run active confirmed by both halves', _num(100 * rep['frac_full_active_in_both'], '.0f') + ' %']]) +
        '\n')
    md.append('Pearson and Spearman measure agreement over all voxels (Spearman is not dominated by the strongest '
              'few). Dice = 2 × both / (odd + even) is the overlap of the two active sets; the value in brackets is '
              'what random sets of those sizes would give. The last row is the share of whole-run active voxels that '
              'each half confirms alone: low means single-voxel calls are fragile even when set averages are '
              'stable.\n')
    md.append('### Test-retest reliability and ROC\n')
    md.append(f'(Genovese, Noll & Eddy, MRM 1997;38:497.) The run is cut into M = {M} replicates of '
              f'{ncyc_all // M} cycles, taken {M} cycles apart, and the GLM is fit to each. At each z threshold a voxel is '
              'called active or not in each replicate. Without ground truth, a mixed-binomial model estimates it: '
              'a voxel is truly active with probability λ; if truly active it is called active in a replicate with '
              'probability p_A (true-positive rate), if not with p_I (false-positive rate), independently across '
              'replicates. All thresholds are fit together, sharing λ.\n')
    md.append(f'**Reliability map histogram.** Voxels by R, the number of replicates in which they exceed '
              f'z = {Z_GRID[k0]:g}. Bars are observed; the orange line is the fitted mixture, red and black its truly '
              'active and inactive parts. Voxels piled at R = M with a flat middle are reliably detected; many at '
              'R = 1-2 are calls that do not repeat.\n')
    md.append('![reliability histogram](reliability_histogram.png)\n')
    md.append('**Estimated ROC.** Each point is one threshold at (p_I, p_A) from the fit (labels: z threshold). Up and '
              'to the left is better; the dotted diagonal is chance; the circle maximises '
              'λ p_A + (1 − λ)(1 − p_I) over the grid.\n')
    md.append('![ROC](roc.png)\n')
    md.append(_md_table(['', 'value'], [
        ['λ (truly active share)', _num(tr_stats.get('lam'), '.4f')],
        ['estimated truly active voxels', _num(tr_stats.get('n_est_active'), '.0f')],
        ['best z on the grid', _num(tr_stats.get('best_z'), 'g')],
        ['p_A / p_I there', f'{_num(tr_stats.get("best_tpr"), ".2f")} / {_num(tr_stats.get("best_fpr"), ".3f")}'],
        ['area under the estimated ROC', _num(tr_stats.get('auc'), '.2f')]]) + '\n')
    md.append('Caveats: the model assumes independent, identically distributed voxels and replicates. Spatial '
              'clustering and the shared reconstruction break that, and each replicate has only '
              f'{ncyc_all // M} cycles of power, so absolute rates are indicative.\n')
    with open(report_fn, 'w') as fh:
        fh.write('\n'.join(md))

    metrics = {'name': name, 'q_fdr': Q_FDR, 'z_fdr': z_fdr, 'n_active': n_set['active'],
               'n_counter_active': n_set['counter-active'], 'n_inactive': n_set['inactive'], 'n_mask': n_vox,
               'mean_task_effect_active_pct': eff['active'],
               'mean_task_effect_counter_active_pct': eff['counter-active'], 'peaks': peak_rows,
               'split_half_odd_even': rep, 'test_retest': tr_stats,
               'task_power': {k.replace('-', '_'): v for k, v in tp.items()}, 'task_f0_hz': f0}
    with open(f'{out}/metrics.json', 'w') as fh:
        json.dump(metrics, fh, indent=1, default=float)
    return report_fn


def write_index(out_root):
    """reports/README.md: one row per report with the headline numbers."""
    rows = []
    for d in sorted(glob.glob(f'{out_root}/*/metrics.json')):
        m = json.load(open(d))
        sh, trt = m['split_half_odd_even'], m['test_retest']
        rows.append([f'[{m["name"]}]({m["name"]}/report.md)', m['n_active'], m['n_counter_active'],
                     f'{m["z_fdr"]:.2f}', f'{sh["z_pearson"]:.2f}', f'{sh["dice"]:.2f}',
                     _num(trt.get('auc'), '.2f')])
    with open(f'{out_root}/README.md', 'w') as fh:
        fh.write('# Functional analysis reports\n\n' + _md_table(
            ['report', 'active', 'counter-active', '\\|z\\| FDR', 'odd/even z corr', 'Dice', 'ROC area'], rows) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--only', default='')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    paradigm = json.load(open(f'{a.root}/paradigm.json'))
    dirs = sorted(d for d in glob.glob(f'{a.root}/analysis/*') if os.path.isdir(d))
    dirs = [d for d in dirs if a.only in os.path.basename(d)
            and glob.glob(f'{d}/*_summary.json')]  # skip archived/, figures/, reports/, ...
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
            print(make_report(d, loaded=cache[h5], paradigm=paradigm, force=a.force), flush=True)
        except Exception as e:  # keep going: one broken folder should not stop the rest
            print(f'FAILED {d}: {e!r}', flush=True)
    write_index(f'{a.root}/analysis/reports')


if __name__ == '__main__':
    main()
