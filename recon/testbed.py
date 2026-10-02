"""Known-truth testbed for comparing reconstructions, and its scoring.

build: synthesize an undersampled dynamic acquisition from a fully sampled
static one. The object x0 is the B0-SENSE CG reconstruction of the mean of
the fully sampled frames (all but the steady-state frame 0). Each synthetic
frame t is

    y_t = M_t A_B0 x_true(t) + n_t,   x_true(t) = x0 (1 + amp sum_r w_r(t) m_r)

with M_t the sampling mask of frame t of another (undersampled) run, A_B0 the
B0-SENSE operator (the same one the recon uses), with each sample's echo time
taken from the fully sampled run (timing='full') or from the undersampled run
itself (timing='masks': the real per-frame timing of that acquisition, which
puts each (ky, kz) at a different point of the echo train), n_t fresh complex white
noise of unit variance (the whitened data's noise level), and m_r, w_r the
regions and waveforms of a few injected "activations":

    block_r3, block_r1.5   10 s on / 10 s off boxcar convolved with SPM's
                           canonical HRF (in band: < 0.15 Hz), radius 3 / 1.5 voxels
    sin0.10                0.10 Hz sinusoid, in band
    sin0.25                0.25 Hz sinusoid, out of band (a temporal penalty
                           above 0.15 Hz should remove it)

each zero-mean with peak |w| = 1, so amp is the peak fractional change
(amp = 0: a static truth, every frame the fully sampled mean). Variants:
mask_period=K reuses the first K masks cyclically (frame t gets mask t mod K),
so the aliasing of anything static repeats with period K; source='measured'
uses the fully sampled run's measured mean k-space under each mask instead of
the model (y_t = M_t k_mean + fresh noise topping each sample up to unit
variance; amp must be 0 and timing 'full'), so the real data's model mismatch
is in it but no frame-to-frame change of the object or acquisition is. The
output is a <name>_preprocessed.h5 that recon/sense.py reconstructs as is,
plus a 'truth' group. It is an inverse crime by construction (same operator
and B0 segmentation both ways), so it measures sampling, noise and prior
effects, not model mismatch; confirm conclusions on the real data.

score: compare a reconstruction (.h5 from recon/sense.py) with the truth, on
magnitude images after one global least-squares scale (recon outputs are not
quantitative; one scale for all frames, so it can't hide fluctuations).

    UV_PROJECT_ENVIRONMENT=.venv-recon uv run --extra recon python -m recon.testbed \\
        build <full_preprocessed.h5> <masks_preprocessed.h5> <out_preprocessed.h5> \\
        --volume-tr 0.506 [--nt 80]
    ... python -m recon.testbed score <testbed.h5> <recon.h5> [--json out.json] [--png out.png]
"""

import argparse
import json
import math
import os

import h5py
import numpy as np
from scipy import ndimage

ROIS = (  # name, radius (voxels), waveform
    ("block_r3", 3.0, "block"),
    ("block_r1.5", 1.5, "block"),
    ("sin0.10", 3.0, "sin0.10"),
    ("sin0.25", 3.0, "sin0.25"),
)


def canonical_hrf(tr_s: float, duration_s: float = 32.0) -> np.ndarray:
    """SPM's canonical double-gamma HRF (peak 6 s, undershoot 16 s, ratio 1/6),
    sampled every tr_s, unit sum."""
    t = np.arange(0, duration_s, tr_s)
    h = (t**5 * np.exp(-t) / math.gamma(6)) - (t**15 * np.exp(-t) / math.gamma(16)) / 6
    return h / h.sum()


def waveform(kind: str, nt: int, tr_s: float) -> np.ndarray:
    """Zero-mean, peak |w| = 1."""
    t = np.arange(nt) * tr_s
    if kind == "block":
        box = ((t // 10.0) % 2 == 1).astype(float)  # off 10 s, on 10 s, ...
        w = np.convolve(box, canonical_hrf(tr_s))[:nt]
    else:
        w = np.sin(2 * np.pi * float(kind[3:]) * t)
    w = w - w.mean()
    return w / np.abs(w).max()


def object_masks(x0_abs: np.ndarray, thresh: float = 0.4):
    """(object, interior): the largest connected component above thresh x the
    99th percentile, and that eroded by 2 voxels. build places ROIs in the 0.4
    interior (brighter voxels); score measures over the 0.1 one (the whole
    object: on the 20260930ballfat ball, half of it is below 0.4)."""
    lab, _ = ndimage.label(x0_abs > thresh * np.percentile(x0_abs, 99))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    obj = lab == np.argmax(sizes)
    return obj, ndimage.binary_erosion(obj, iterations=2)


def place_rois(interior: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Spherical ROIs (ROIS' radii) inside `interior`, spread out by
    farthest-point sampling from a random start, each fully inside."""
    grid = np.indices(interior.shape).reshape(3, -1).T
    out, centers = [], []
    for _, radius, _ in ROIS:
        ok = ndimage.binary_erosion(interior, iterations=int(math.ceil(radius)) + 2)
        cand = grid[ok.ravel()]
        if centers:
            d = np.min(np.linalg.norm(cand[:, None, :] - np.array(centers)[None], axis=2), axis=1)
            c = cand[np.argmax(d)]
        else:
            c = cand[rng.integers(len(cand))]
        centers.append(c)
        out.append(np.linalg.norm(grid - c, axis=1).reshape(interior.shape) <= radius)
    return out


def build_testbed(
    fn_full: str, fn_masks: str, fn_out: str, volume_tr_s: float, nt: int = 80,
    mask_start: int = 1, amp: float = 0.02, seed: int = 0, device=None, L_b0: int = 32,
    nbins_b0: int = 128, cg_iters: int = 150, timing: str = "full",
    source: str = "model", mask_period: int | None = None,
) -> None:
    import torch

    from recon.operators import build_sense_b0
    from recon.solvers import cg
    from recon.utils import load_normalized_smaps, resolve_device

    device = resolve_device(device)
    rng = np.random.default_rng(seed)
    with h5py.File(fn_full, "r") as f:
        d = f["ksp_epi_zf"]
        Nx, Ny, Nz, Nc, nf = d.shape
        noise_var = float(f.attrs.get("noise_var", 1.0))
        k_mean = sum(np.asarray(d[..., t], dtype=np.complex64) for t in range(1, nf)) / (nf - 1)
        k_mean /= np.sqrt(noise_var)
        b0 = f["b0_map"][()].astype(np.float32)
        et1 = f["echo_times"][:, :, 1].astype(np.float32)
        fov = tuple(f.attrs["fov"])
    if timing not in ("full", "masks"):
        raise ValueError(f"timing={timing!r}, expected 'full' or 'masks'")
    if source not in ("model", "measured"):
        raise ValueError(f"source={source!r}, expected 'model' or 'measured'")
    if source == "measured" and (amp != 0 or timing != "full"):
        raise ValueError("source='measured' needs amp=0 and timing='full' (it is the "
                         "fully sampled run's own data, acquired with its timing)")
    n_masks = mask_period or nt
    with h5py.File(fn_masks, "r") as f:
        om = f["omegas"][:, :, mask_start : mask_start + n_masks]
        et_masks = (f["echo_times"][:, :, mask_start : mask_start + n_masks].astype(np.float32)
                    if timing == "masks" else None)
    assert om.shape[-1] == n_masks, f"{fn_masks} has fewer than {mask_start + n_masks} frames"
    cyc = np.arange(nt) % n_masks
    om = om[..., cyc]
    et_masks = et_masks[..., cyc] if et_masks is not None else None
    print(f"testbed: object from the mean of {nf - 1} frames of {fn_full}; masks: frames "
          f"{mask_start}-{mask_start + n_masks - 1} of {fn_masks}"
          + (f", cycled with period {mask_period}" if mask_period else "")
          + f" (R {om[..., 0].size / om[..., 0].sum():.1f}); data: {source}")

    _, smaps_chw = load_normalized_smaps(fn_full, device)
    b0_t = torch.from_numpy(b0).to(device)

    # the object: B0-SENSE CG of the fully sampled mean
    full = torch.ones(Nx, Ny, Nz, 1, dtype=torch.bool, device=device)
    A1 = build_sense_b0(smaps_chw, full, b0_t, torch.from_numpy(et1[..., None]).to(device),
                        L=L_b0, nbins=nbins_b0)
    y1 = torch.from_numpy(k_mean).to(device).reshape(-1, Nc)[A1.A[0].idx][..., None]
    x0, res = cg(A1, y1, (Nx, Ny, Nz, 1), num_iter=cg_iters, tol=1e-6)
    x0 = x0[..., 0]
    print(f"  object: CG {len(res) - 1} iterations, relative residual {res[-1]:.2e}")
    del A1, y1

    x0_abs = x0.abs().cpu().numpy()
    obj, interior = object_masks(x0_abs)
    rois = place_rois(interior, rng)
    waves = np.stack([waveform(kind, nt, volume_tr_s) for _, _, kind in ROIS])
    mod = np.ones((Nx, Ny, Nz, nt), np.float32)
    for m, w in zip(rois, waves):
        mod[m] += amp * w[None, :].astype(np.float32)
    x_true = x0[..., None] * torch.from_numpy(mod).to(device)

    omega = torch.from_numpy(om).to(device).unsqueeze(0).expand(Nx, -1, -1, -1)
    et_np = et_masks if timing == "masks" else np.repeat(et1[..., None], nt, axis=-1)
    idx = [torch.nonzero(omega[..., t].reshape(-1)).squeeze(-1) for t in range(nt)]
    g = torch.Generator(device=device).manual_seed(seed)
    if source == "model":
        A = build_sense_b0(smaps_chw, omega, b0_t, torch.from_numpy(et_np).to(device),
                           L=L_b0, nbins=nbins_b0)
        y = A.apply(x_true)  # (K, Nc, nt)
        noise_std = 1.0
    else:  # the measured mean (noise variance 1/(nf-1)), topped up to unit variance
        kflat = torch.from_numpy(k_mean).to(device).reshape(-1, Nc)
        y = torch.stack([kflat[i] for i in idx], -1)
        noise_std = math.sqrt(1 - 1 / (nf - 1))
    y = y + noise_std * torch.complex(torch.randn(y.shape, generator=g, device=device),
                                      torch.randn(y.shape, generator=g, device=device)) / math.sqrt(2)

    os.makedirs(os.path.dirname(os.path.abspath(fn_out)), exist_ok=True)
    with h5py.File(fn_out, "w") as f:
        ds = f.create_dataset("ksp_epi_zf", (Nx, Ny, Nz, Nc, nt), np.complex64,
                              chunks=(Nx, Ny, Nz, Nc, 1))
        for t in range(nt):
            frame = torch.zeros(Nx * Ny * Nz, Nc, dtype=torch.complex64, device=device)
            frame[idx[t]] = y[:, :, t]
            ds[..., t] = frame.reshape(Nx, Ny, Nz, Nc).cpu().numpy()
        f["omegas"] = om
        f["echo_times"] = et_np
        with h5py.File(fn_full, "r") as src:
            f["smaps"] = src["smaps"][()]
        f["b0_map"] = b0
        f.attrs.update(fov=fov, noise_var=1.0, whitened=True, volume_tr=volume_tr_s,
                       testbed_full=os.path.abspath(fn_full),
                       testbed_masks=os.path.abspath(fn_masks),
                       testbed_mask_start=mask_start, testbed_seed=seed, testbed_amp=amp,
                       testbed_timing=timing, testbed_source=source,
                       testbed_mask_period=mask_period or 0)
        t_ = f.create_group("truth")
        t_["x0"] = x0.cpu().numpy()
        t_["object"] = obj
        t_["interior"] = interior
        t_["roi_masks"] = np.stack(rois)
        t_["waveforms"] = waves
        t_.attrs["roi_names"] = [r[0] for r in ROIS]
        t_.attrs["roi_kinds"] = [r[2] for r in ROIS]
        t_.attrs["amp"] = amp
    print(f"  wrote {fn_out}: {nt} frames, ROIs "
          + ", ".join(f"{n} ({int(m.sum())} vox)" for (n, _, _), m in zip(ROIS, rois)))


# ---------------------------------------------------------------- scoring


def _dct(nt: int) -> np.ndarray:
    t = np.arange(nt)
    C = np.cos(np.pi * t[:, None] * (t[None, :] + 0.5) / nt) * np.sqrt(2.0 / nt)
    C[0] /= np.sqrt(2.0)
    return C


def _glm(s: np.ndarray, w: np.ndarray, basis: np.ndarray | None = None):
    """Fit s (V, nt) = c0 + c1 t + beta w; return (beta, t-stat), each (V,).

    basis (k, nt), orthonormal rows: fit in that subspace instead (data and
    regressors projected onto it, k - 3 degrees of freedom). With the DCT
    components up to the cutoff, this is the GLM on low-pass-filtered series --
    the fair comparison for recons whose residual is itself band-limited (a
    temporal penalty leaves it smooth, and the plain t assumes independent
    frames, overstating it by up to sqrt(nt / k))."""
    nt = s.shape[1]
    t = np.linspace(-1, 1, nt)
    X = np.stack([np.ones(nt), t, w], 1)
    if basis is not None:
        s, X, nt = s @ basis.T, basis @ X, basis.shape[0]
    XtXi = np.linalg.inv(X.T @ X)
    B = s @ X @ XtXi  # (V, 3)
    res = s - B @ X.T
    sigma2 = (res**2).sum(1) / (nt - 3)
    se = np.sqrt(sigma2 * XtXi[2, 2])
    return B[:, 2], B[:, 2] / np.maximum(se, 1e-30)


def score(fn_testbed: str, fn_recon: str, cutoff_hz: float = 0.15,
          object_thresh: float = 0.1) -> dict:
    with h5py.File(fn_testbed, "r") as f:
        tr = float(f.attrs["volume_tr"])
        g = f["truth"]
        x0 = np.abs(g["x0"][()])
        rois, waves = g["roi_masks"][()], g["waveforms"][()]
        names = list(g.attrs["roi_names"])
        amp = float(g.attrs["amp"])
    obj, interior = object_masks(x0, object_thresh)
    with h5py.File(fn_recon, "r") as f:
        X = np.abs(f["X_recon"][()]).astype(np.float64)
    nt = X.shape[-1]
    T = np.repeat(x0[..., None], nt, -1)
    for m, w in zip(rois, waves):
        T[m] *= 1 + amp * w[None, :]
    alpha = (X[obj] * T[obj]).sum() / (X[obj] ** 2).sum()
    M = alpha * X

    out = dict(alpha=float(alpha), object_thresh=object_thresh)
    mean_err = M[obj].mean(1) - x0[obj]
    out["nrmse_mean_pct"] = 100 * float(np.linalg.norm(mean_err) / np.linalg.norm(x0[obj]))
    out["nrmse_frame_pct"] = 100 * float(np.median(
        np.linalg.norm(M[obj] - T[obj], axis=0) / np.linalg.norm(T[obj], axis=0)))

    static = amp == 0
    near_roi = ndimage.binary_dilation(rois.any(0), iterations=3)
    bg = interior if static else interior & ~near_roi
    R = (M[bg] - T[bg]) / x0[bg][:, None]  # relative residual, (V, nt)
    Rc = R @ _dct(nt).T
    keep = int((np.arange(nt) / (2 * nt * tr) <= cutoff_hz).sum())
    out["fluct_pct"] = 100 * float(np.median(R.std(1)))
    out["fluct_inband_pct"] = 100 * float(np.median(np.sqrt((Rc[:, 1:keep] ** 2).sum(1) / nt)))
    out["fluct_outband_pct"] = 100 * float(np.median(np.sqrt((Rc[:, keep:] ** 2).sum(1) / nt)))

    lowband = _dct(nt)[:keep]  # the GLM on low-pass-filtered series (<= cutoff_hz)
    out["frame_err_max_pct"] = 100 * float(np.max(
        np.linalg.norm(M[obj] - T[obj], axis=0) / np.linalg.norm(T[obj], axis=0)))
    if static:  # no activation: ROI/GLM metrics don't apply
        names, rois, waves = [], rois[:0], waves[:0]
    for name, m, w in zip(names, rois, waves):
        s = M[m] / x0[m][:, None]
        beta, tstat = _glm(s, w)
        if name.startswith("block") or name == "sin0.10":  # in-band waveforms
            out[f"{name}_t_lowband"] = float(np.median(_glm(s, w, lowband)[1]))
        shell = (ndimage.binary_dilation(m, iterations=4)
                 & ~ndimage.binary_dilation(m, iterations=1) & interior & ~(rois.any(0) & ~m))
        beta_shell, _ = _glm(M[shell] / x0[shell][:, None], w)
        out[f"{name}_amp_ratio"] = float(np.median(beta) / amp)
        out[f"{name}_t"] = float(np.median(tstat))
        roi_mean = s.mean(0)
        out[f"{name}_corr"] = (float(np.corrcoef(roi_mean, w)[0, 1])
                               if roi_mean.std() > 1e-12 else 0.0)
        out[f"{name}_leak_ratio"] = float(np.median(beta_shell) / amp)
    if not static:
        _, t_bg = _glm(M[bg] / x0[bg][:, None], waves[0])
        out["false_pos_frac_t3.29"] = float(np.mean(np.abs(t_bg) > 3.29))
        # the same in the low band, against its own t threshold (two-sided p < 0.001)
        from scipy import stats

        _, t_bg_lb = _glm(M[bg] / x0[bg][:, None], waves[0], lowband)
        thr = float(stats.t.ppf(1 - 0.0005, keep - 3))
        out["false_pos_frac_lowband"] = float(np.mean(np.abs(t_bg_lb) > thr))

    edge = ndimage.binary_dilation(obj, iterations=1) & ~ndimage.binary_erosion(obj, iterations=2)
    gm = np.linalg.norm(np.stack(np.gradient(M.mean(-1))), axis=0)
    gt = np.linalg.norm(np.stack(np.gradient(x0)), axis=0)
    out["edge_sharpness_ratio"] = float(gm[edge].mean() / gt[edge].mean())
    return out


def panel(fn_testbed: str, fn_recon: str, fn_png: str, title: str = "") -> None:
    """Mean image, temporal std (relative) and block-regressor t-map, through
    the center of block_r3, next to the truth's mean image. For a static
    testbed (amp 0), the middle frame instead of the t-map, through the
    center of the object."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(fn_testbed, "r") as f:
        g = f["truth"]
        x0 = np.abs(g["x0"][()])
        rois, waves = g["roi_masks"][()], g["waveforms"][()]
        static = float(g.attrs["amp"]) == 0
    obj, _ = object_masks(x0, 0.1)
    with h5py.File(fn_recon, "r") as f:
        X = np.abs(f["X_recon"][()])
    alpha = (X[obj].mean(1) * x0[obj]).sum() / (X[obj].mean(1) ** 2).sum()
    M = alpha * X
    mean = M.mean(-1)
    std = M.std(-1) / np.maximum(x0, 1e-6 * x0.max()) * obj
    vmax = np.percentile(x0[obj], 99.5)
    rows = [("truth mean", x0, dict(cmap="gray", vmin=0, vmax=vmax)),
            ("recon mean", mean, dict(cmap="gray", vmin=0, vmax=vmax)),
            ("temporal std / truth", std, dict(cmap="magma", vmin=0, vmax=0.08))]
    if static:
        c = np.round(np.argwhere(obj).mean(0)).astype(int)
        rows.append((f"recon frame {X.shape[-1] // 2}", M[..., X.shape[-1] // 2],
                     dict(cmap="gray", vmin=0, vmax=vmax)))
    else:
        _, tmap = _glm((M / np.maximum(x0, 1e-6 * x0.max())[..., None]).reshape(
            -1, X.shape[-1]), waves[0])
        c = np.round(np.argwhere(rois[0]).mean(0)).astype(int)
        rows.append(("t (block regressor)", tmap.reshape(x0.shape) * obj,
                     dict(cmap="RdBu_r", vmin=-8, vmax=8)))
    fig, ax = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    for i, (lab, vol, kw) in enumerate(rows):
        for j, sl in enumerate([vol[c[0]], vol[:, c[1]], vol[:, :, c[2]]]):
            im = ax[i, j].imshow(np.rot90(sl), **kw)
            ax[i, j].axis("off")
        ax[i, 0].set_title(lab, loc="left")
        fig.colorbar(im, ax=ax[i, 2], fraction=0.046)
    for m in ([] if static else rois):
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
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("full")
    b.add_argument("masks")
    b.add_argument("out")
    b.add_argument("--volume-tr", type=float, required=True)
    b.add_argument("--nt", type=int, default=80)
    b.add_argument("--mask-start", type=int, default=1)
    b.add_argument("--amp", type=float, default=0.02)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device", default=None)
    b.add_argument("--timing", choices=["full", "masks"], default="full",
                   help="echo times from the fully sampled run or from the masks' run")
    b.add_argument("--source", choices=["model", "measured"], default="model",
                   help="synthesize with the forward model, or use the measured k-space")
    b.add_argument("--mask-period", type=int, default=None,
                   help="cycle the first K masks (frame t gets mask t mod K)")
    s = sub.add_parser("score")
    s.add_argument("testbed")
    s.add_argument("recon")
    s.add_argument("--json", default=None)
    s.add_argument("--png", default=None)
    s.add_argument("--title", default="")
    a = p.parse_args()
    if a.cmd == "build":
        build_testbed(a.full, a.masks, a.out, a.volume_tr, nt=a.nt, mask_start=a.mask_start,
                      amp=a.amp, seed=a.seed, device=a.device, timing=a.timing,
                      source=a.source, mask_period=a.mask_period)
        return
    out = score(a.testbed, a.recon)
    print(json.dumps(out, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
    if a.png:
        panel(a.testbed, a.recon, a.png, a.title)


if __name__ == "__main__":
    _cli()
