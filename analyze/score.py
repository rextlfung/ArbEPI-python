"""Score a reconstruction against a known truth.

Compares a reconstruction (the `<name>_recon.h5` of recon/sense.py) with the
`truth` group of a truth file, on magnitude images after one global
least-squares scale (recon outputs are not quantitative; one scale for all
frames, so it can't hide fluctuations). Reports the reconstruction error,
fluctuation in and out of the temporal band, each region's activation
amplitude, t-score, time-course correlation and leakage
into its surroundings, the false-positive fraction away from the regions, and
edge sharpness.

The truth file holds, in `truth/`: `x0` (the noise-free mean image),
`roi_masks` (regions, x, y, z), `waveforms` (regions, frames; zero-mean, peak
1) and the attributes `roi_names`, `amp` (peak fractional change; a per-region
`amps` is preferred when present); and the file attribute `volume_tr`. Three
producers write it: simulate_fmri/from_real.py (known activations injected
into a real fully sampled run), and simulate_fmri/session.py and ideal.py
(simulated sessions).

    python -m analyze.score <truth.h5> <recon.h5> [--json out.json] [--png out.png]
        [--title T] [--roi i]
"""

import argparse
import json

import h5py
import numpy as np
from scipy import ndimage

from analyze.design import dct_matrix
from analyze.glm import fit_glm


def object_masks(x0_abs: np.ndarray, thresh: float = 0.4):
    """(object, interior): the largest connected component above thresh x the
    99th percentile, and that eroded by 2 voxels. simulate_fmri.from_real places ROIs in the 0.4
    interior (brighter voxels); score measures over the 0.1 one (the whole
    object: on the 20260930ballfat ball, half of it is below 0.4)."""
    lab, _ = ndimage.label(x0_abs > thresh * np.percentile(x0_abs, 99))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    obj = lab == np.argmax(sizes)
    return obj, ndimage.binary_erosion(obj, iterations=2)


def _dct(nt: int) -> np.ndarray:
    return dct_matrix(nt)


def _glm(s: np.ndarray, w: np.ndarray):
    """Fit s (V, nt) = c0 + c1 t + beta w by ordinary least squares; return
    (beta, t-stat), each (V,). The fit is analyze.glm's."""
    nt = s.shape[1]
    X = np.stack([np.ones(nt), np.linspace(-1, 1, nt), w], 1)
    r = fit_glm(s, X, np.array([0.0, 0.0, 1.0]), ar1=False)
    return r.beta[:, 2], r.t


def score(fn_truth: str, fn_recon: str, cutoff_hz: float = 0.15,
          object_thresh: float = 0.1) -> dict:
    with h5py.File(fn_truth, "r") as f:
        tr = float(f.attrs["volume_tr"])
        g = f["truth"]
        x0 = np.abs(g["x0"][()])
        rois, waves = g["roi_masks"][()], g["waveforms"][()]
        names = list(g.attrs["roi_names"])
        amp = float(g.attrs["amp"])
        # one peak fractional change per ROI when the truth has them (simulated
        # sessions: regions differ in tissue mix), else the shared one
        amps = np.broadcast_to(np.asarray(g.attrs.get("amps", amp), float), (len(rois),))
    obj, interior = object_masks(x0, object_thresh)
    with h5py.File(fn_recon, "r") as f:
        X = np.abs(f["X_recon"][()]).astype(np.float64)
    nt = X.shape[-1]
    T = np.repeat(x0[..., None], nt, -1)
    for m, w, a in zip(rois, waves, amps):
        T[m] *= 1 + a * w[None, :]
    alpha = (X[obj] * T[obj]).sum() / (X[obj] ** 2).sum()
    M = alpha * X

    out = dict(alpha=float(alpha), object_thresh=object_thresh)
    mean_err = M[obj].mean(1) - x0[obj]
    out["nrmse_mean_pct"] = 100 * float(np.linalg.norm(mean_err) / np.linalg.norm(x0[obj]))
    out["nrmse_frame_pct"] = 100 * float(np.median(
        np.linalg.norm(M[obj] - T[obj], axis=0) / np.linalg.norm(T[obj], axis=0)))

    near_roi = ndimage.binary_dilation(rois.any(0), iterations=3)
    bg = interior & ~near_roi
    R = (M[bg] - T[bg]) / x0[bg][:, None]  # relative residual, (V, nt)
    Rc = R @ _dct(nt).T
    keep = int((np.arange(nt) / (2 * nt * tr) <= cutoff_hz).sum())
    out["fluct_pct"] = 100 * float(np.median(R.std(1)))
    out["fluct_inband_pct"] = 100 * float(np.median(np.sqrt((Rc[:, 1:keep] ** 2).sum(1) / nt)))
    out["fluct_outband_pct"] = 100 * float(np.median(np.sqrt((Rc[:, keep:] ** 2).sum(1) / nt)))

    for name, m, w, a in zip(names, rois, waves, amps):
        s = M[m] / x0[m][:, None]
        beta, tstat = _glm(s, w)
        shell = (ndimage.binary_dilation(m, iterations=4)
                 & ~ndimage.binary_dilation(m, iterations=1) & interior & ~(rois.any(0) & ~m))
        beta_shell, _ = _glm(M[shell] / x0[shell][:, None], w)
        out[f"{name}_amp_ratio"] = float(np.median(beta) / a)
        out[f"{name}_t"] = float(np.median(tstat))
        roi_mean = s.mean(0)
        out[f"{name}_corr"] = (float(np.corrcoef(roi_mean, w)[0, 1])
                               if roi_mean.std() > 1e-12 else 0.0)
        out[f"{name}_leak_ratio"] = float(np.median(beta_shell) / a)
    _, t_bg = _glm(M[bg] / x0[bg][:, None], waves[0])
    out["false_pos_frac_t3.29"] = float(np.mean(np.abs(t_bg) > 3.29))

    edge = ndimage.binary_dilation(obj, iterations=1) & ~ndimage.binary_erosion(obj, iterations=2)
    gm = np.linalg.norm(np.stack(np.gradient(M.mean(-1))), axis=0)
    gt = np.linalg.norm(np.stack(np.gradient(x0)), axis=0)
    out["edge_sharpness_ratio"] = float(gm[edge].mean() / gt[edge].mean())
    return out


def panel(fn_truth: str, fn_recon: str, fn_png: str, title: str = "", roi: int = 0) -> None:
    """Mean image, temporal std (relative) and the t-map of ROI `roi`'s
    regressor, through the center of that ROI (0: block_r3 in a built
    testbed), next to the truth's mean image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(fn_truth, "r") as f:
        g = f["truth"]
        x0 = np.abs(g["x0"][()])
        rois, waves = g["roi_masks"][()], g["waveforms"][()]
    obj, _ = object_masks(x0, 0.1)
    with h5py.File(fn_recon, "r") as f:
        X = np.abs(f["X_recon"][()])
    alpha = (X[obj].mean(1) * x0[obj]).sum() / (X[obj].mean(1) ** 2).sum()
    M = alpha * X
    mean = M.mean(-1)
    std = M.std(-1) / np.maximum(x0, 1e-6 * x0.max()) * obj
    _, tmap = _glm((M / np.maximum(x0, 1e-6 * x0.max())[..., None]).reshape(-1, X.shape[-1]),
                   waves[roi])
    tmap = tmap.reshape(x0.shape) * obj
    c = np.round(np.argwhere(rois[roi]).mean(0)).astype(int)
    vmax = np.percentile(x0[obj], 99.5)
    rows = [("truth mean", x0, dict(cmap="gray", vmin=0, vmax=vmax)),
            ("recon mean", mean, dict(cmap="gray", vmin=0, vmax=vmax)),
            ("temporal std / truth", std, dict(cmap="magma", vmin=0, vmax=0.08)),
            ("t (block regressor)", tmap, dict(cmap="RdBu_r", vmin=-8, vmax=8))]
    fig, ax = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    for i, (lab, vol, kw) in enumerate(rows):
        for j, sl in enumerate([vol[c[0]], vol[:, c[1]], vol[:, :, c[2]]]):
            im = ax[i, j].imshow(np.rot90(sl), **kw)
            ax[i, j].axis("off")
        ax[i, 0].set_title(lab, loc="left")
        fig.colorbar(im, ax=ax[i, 2], fraction=0.046)
    for m in rois:
        for j, sl in enumerate([m[c[0]], m[:, c[1]], m[:, :, c[2]]]):
            if sl.any():
                ax[3, j].contour(np.rot90(sl), levels=[0.5], colors="k", linewidths=0.5)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fn_png, dpi=110)
    plt.close(fig)


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("truth", help="file with a `truth` group (see above)")
    p.add_argument("recon", help="<name>_recon.h5 from recon.sense")
    p.add_argument("--json", default=None)
    p.add_argument("--png", default=None)
    p.add_argument("--title", default="")
    p.add_argument("--roi", type=int, default=0,
                   help="which ROI the --png panel shows (index into roi_names)")
    a = p.parse_args()
    out = score(a.truth, a.recon)
    print(json.dumps(out, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
    if a.png:
        panel(a.truth, a.recon, a.png, a.title, a.roi)


if __name__ == "__main__":
    _cli()
