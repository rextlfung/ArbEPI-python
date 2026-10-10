"""Side-by-side comparison of reconstructions (or of one reconstruction's
scales) of the same session: activation maps on a shared mask and design, and
the time courses behind them.

Everything is drawn from `Maps` (analyze.run.analyze_recon with
keep_series=True). Maps are compared on one threshold, so a panel that shows
less shows less, not a different cutoff:

- `plot_maps`: axial slices of the contrast statistic over the mean image, one
  column per series.
- `plot_timecourses`: the series of the peak voxel and the mean of the top n%
  of voxels (selected on the first series, or on each), with the task blocks
  shaded, and the power spectrum of the latter with the task's fundamental
  marked.
- `summary_table`: peak statistic, effect, and survivors, as text.
- `compare_recons`: the whole thing for a set of files.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from analyze import glm
from analyze.design import ExperimentParams
from analyze.run import Maps, analyze_recon


def _threshold(first: Maps, stat: str, threshold: float | None, p: float = 0.001) -> float:
    if threshold is not None:
        return float(threshold)
    return glm.t_threshold(first.dof, p) if stat == "t" else float(stats.norm.isf(p / 2))


def _slice_indices(brain: NDArray, n: int) -> list[int]:
    z = np.flatnonzero(brain.any(axis=(0, 1)))
    if z.size == 0:
        return list(np.linspace(0, brain.shape[2] - 1, n).astype(int))
    return sorted(set(np.linspace(z[0], z[-1], n + 2)[1:-1].round().astype(int)))


def plot_maps(maps: Mapping[str, Maps], stat: str = "t", threshold: float | None = None,
              n_slices: int = 6, slices: list[int] | None = None, vmax: float | None = None,
              fn: str | None = None):
    """Rows: series; columns: axial slices. Voxels below |threshold| are not
    colored. threshold default: p < 0.001 (two-sided, uncorrected) of the first
    series' degrees of freedom. Returns the figure."""
    import matplotlib

    if fn:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    labels = list(maps)
    first = maps[labels[0]]
    thr = _threshold(first, stat, threshold)
    sl = slices or _slice_indices(first.brain, n_slices)
    vols = {k: getattr(m, stat) for k, m in maps.items()}
    vmax = vmax or max(float(np.abs(vols[k][m.brain]).max()) for k, m in maps.items())
    under = first.underlay
    ulim = (0.0, float(np.percentile(under[first.brain], 99)) if first.brain.any() else 1.0)
    fig, axs = plt.subplots(len(labels), len(sl), figsize=(2.2 * len(sl) + 1.2, 2.2 * len(labels)),
                            squeeze=False, constrained_layout=True)
    im = None
    for r, k in enumerate(labels):
        v = vols[k]
        for c, zi in enumerate(sl):
            ax = axs[r, c]
            ax.imshow(under[:, :, zi].T, cmap="gray", vmin=ulim[0], vmax=ulim[1], origin="lower")
            ov = np.ma.masked_where(np.abs(v[:, :, zi]) < thr, v[:, :, zi]).T
            im = ax.imshow(ov, cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z = {zi}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{k}\nmax |{stat}| {np.abs(v[maps[k].brain]).max():.1f}",
                              fontsize=9)
    fig.suptitle(f"{stat}-maps, |{stat}| > {thr:.2f}", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axs, shrink=0.8, label=stat)
    if fn:
        fig.savefig(fn, dpi=150)
    return fig


def _select(m: Maps, top_percent: float, stat: str = "t") -> NDArray[np.bool_]:
    """Boolean over m's masked voxels: the top_percent % with the largest
    positive statistic."""
    v = getattr(m, stat)[m.brain]
    k = max(1, int(round(v.size * top_percent / 100)))
    sel = np.zeros(v.size, bool)
    sel[np.argsort(v)[-k:]] = True
    return sel


def plot_timecourses(maps: Mapping[str, Maps], params: ExperimentParams, top_percent: float = 1.0,
                     stat: str = "t", peak_source: str = "first", fn: str | None = None):
    """Peak voxel, top-n% mean and its power spectrum, one line per series.

    peak_source: 'first' selects the voxels on the first series (so the lines
    are the same voxels, in different reconstructions); 'per_series' selects on
    each series' own map. Needs maps from keep_series=True. Percent change is
    relative to the summed image's mean signal, so series are on one axis."""
    import matplotlib

    if fn:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    labels = list(maps)
    if any(maps[k].series_psc is None for k in labels):
        raise ValueError("maps need series (analyze_recon(..., keep_series=True))")
    nt = maps[labels[0]].series_psc.shape[1]
    t = (np.arange(nt) + 0.5) * params.tr
    sel0 = _select(maps[labels[0]], top_percent, stat)
    peak0 = int(np.argmax(getattr(maps[labels[0]], stat)[maps[labels[0]].brain]))
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
    for k in labels:
        m = maps[k]
        own = peak_source == "per_series"
        sel = _select(m, top_percent, stat) if own else sel0
        peak = int(np.argmax(getattr(m, stat)[m.brain])) if own else peak0
        axs[0].plot(t, m.series_psc[peak], label=k, lw=1)
        avg = m.series_psc[sel].mean(axis=0)
        axs[1].plot(t, avg, label=k, lw=1)
        f = np.fft.rfftfreq(nt, params.tr)
        axs[2].plot(f[1:], np.abs(np.fft.rfft(avg - avg.mean()))[1:] / nt, label=k, lw=1)
    for cond in params.conditions:
        for o, d in zip(cond.onsets, cond.durations):
            for ax in axs[:2]:
                ax.axvspan(o, o + d, color="0.85", lw=0, zorder=0)
    on = np.array(params.conditions[0].onsets)
    if on.size > 1:
        axs[2].axvline(1 / float(np.median(np.diff(on))), color="k", ls=":", lw=1,
                       label="task fundamental")
    axs[0].set_title(f"peak voxel ({'each series' if peak_source == 'per_series' else labels[0]})",
                     fontsize=10)
    axs[1].set_title(f"mean of the top {top_percent:g}% of voxels", fontsize=10)
    axs[2].set_title("power spectrum of the top-n% mean", fontsize=10)
    axs[0].set_ylabel("% change")
    axs[1].set_ylabel("% change")
    axs[1].set_xlabel("time (s)")
    axs[2].set_xlabel("frequency (Hz)")
    axs[2].set_ylabel("amplitude")
    axs[0].legend(fontsize=8, ncol=min(len(labels) + 1, 4))
    axs[2].legend(fontsize=8)
    if fn:
        fig.savefig(fn, dpi=150)
    return fig


def summary_table(maps: Mapping[str, Maps], top_percent: float = 1.0, stat: str = "t") -> str:
    """One line per series: peak statistic, effect in the top n%, survivors."""
    rows = [f"{'series':<24}{'dof':>6}{'rho':>7}{'max ' + stat:>9}{'min ' + stat:>9}"
            f"{'top%':>7}{'psc':>8}{'FDR n':>8}{'/voxels':>9}"]
    for k, m in maps.items():
        v = getattr(m, stat)[m.brain]
        sel = _select(m, top_percent, stat)
        psc = float(m.psc[m.brain][sel].mean())
        rows.append(f"{k:<24}{m.dof:>6.0f}{m.rho:>7.3f}{v.max():>9.2f}{v.min():>9.2f}"
                    f"{top_percent:>7g}{psc:>8.2f}{m.n_fdr:>8d}{m.n_voxels:>9d}")
    return "\n".join(rows)


def compare_recons(recons: Mapping[str, str], params: ExperimentParams, scales: tuple = ("sum",),
                   brain="auto", out_dir: str | None = None, ar1: bool = True,
                   top_percent: float = 1.0, stat: str = "t", threshold: float | None = None,
                   peak_source: str = "first", voxel_size_mm=None) -> dict[str, Maps]:
    """Analyze every file in `recons` ({label: path}) with the same design and
    one brain mask (computed from the first file unless given), then draw the
    comparison. A multi-scale file contributes one series per entry of
    `scales`, labelled '<label>' for 'sum' and '<label>/scale<i>'. Returns
    {series label: Maps}; with out_dir, writes maps.png, timecourses.png and
    summary.txt there."""
    out: dict[str, Maps] = {}
    for i, (label, path) in enumerate(recons.items()):  # the first file's mask is shared
        res = analyze_recon(path, params, scales=scales, brain=brain, ar1=ar1,
                            tag=label, voxel_size_mm=voxel_size_mm, keep_series=True)
        if i == 0:
            brain = next(iter(res.values())).brain
        for s, m in res.items():
            out[label if s == "sum" else f"{label}/{s}"] = m
    text = summary_table(out, top_percent, stat)
    print(text)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plot_maps(out, stat, threshold, fn=os.path.join(out_dir, "maps.png"))
        plot_timecourses(out, params, top_percent, stat, peak_source,
                         fn=os.path.join(out_dir, "timecourses.png"))
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(text + "\n")
    return out


def main(argv=None) -> None:
    import argparse

    from analyze.design import Condition

    ap = argparse.ArgumentParser(prog="python -m analyze.compare",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--recon", nargs="+", required=True, metavar="LABEL=PATH",
                    help="reconstructions to compare, as label=<name>_recon.h5")
    ap.add_argument("--tr", type=float, required=True)
    ap.add_argument("--onsets", type=float, nargs="+", required=True)
    ap.add_argument("--duration", type=float, nargs="+", required=True)
    ap.add_argument("--n-discard", type=int, default=0)
    ap.add_argument("--drift-cutoff", type=float, default=128.0)
    ap.add_argument("--no-ar1", action="store_true")
    ap.add_argument("--scales", nargs="+", default=["sum"],
                    help="'sum', 'all' or component indices, for every file")
    ap.add_argument("--mask", default="auto")
    ap.add_argument("--stat", choices=["t", "z"], default="t")
    ap.add_argument("--threshold", type=float, help="default: p < 0.001 uncorrected")
    ap.add_argument("--top-percent", type=float, default=1.0)
    ap.add_argument("--peak-source", choices=["first", "per_series"], default="first")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recons = dict(r.split("=", 1) for r in a.recon)
    params = ExperimentParams(a.tr, [Condition("task", tuple(a.onsets), tuple(a.duration))],
                              n_discard=a.n_discard, drift_cutoff_s=a.drift_cutoff or None)
    scales = tuple(s if s in ("sum", "all") else int(s) for s in a.scales)
    compare_recons(recons, params, scales=scales, brain=a.mask, out_dir=a.out, ar1=not a.no_ar1,
                   top_percent=a.top_percent, stat=a.stat, threshold=a.threshold,
                   peak_source=a.peak_source)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
