"""Known-truth testbed built from a real, fully sampled acquisition.

Synthesizes an undersampled dynamic acquisition from a fully sampled static
one. The object x0 is the B0-SENSE CG reconstruction of the mean of
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

each zero-mean with peak |w| = 1, so amp is the peak fractional change. The
output is a <name>_preprocessed.h5 that recon/sense.py reconstructs as is,
plus a 'truth' group in the layout analyze/score.py scores against. It is an
inverse crime by construction (same operator and B0 segmentation both ways),
so it measures sampling, noise and prior effects, not model mismatch; confirm
conclusions on the real data.

Unlike session.py and ideal.py it simulates no scanner: the object, coils and
field are the real ones, only the activation and the noise are synthetic.

    UV_PROJECT_ENVIRONMENT=.venv-simulate uv run --extra simulate --extra recon \\
        python -m simulate_fmri.from_real \\
        <full_preprocessed.h5> <masks_preprocessed.h5> <out_preprocessed.h5> \\
        --volume-tr 0.506 [--nt 80]
    ... python -m analyze.score <out_preprocessed.h5> <recon.h5> [--json out.json] [--png out.png]
"""

import argparse
import math
import os

import h5py
import numpy as np
from scipy import ndimage

from analyze.score import object_masks

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
    with h5py.File(fn_masks, "r") as f:
        om = f["omegas"][:, :, mask_start : mask_start + nt]
        et_masks = (f["echo_times"][:, :, mask_start : mask_start + nt].astype(np.float32)
                    if timing == "masks" else None)
    assert om.shape[-1] == nt, f"{fn_masks} has fewer than {mask_start + nt} frames"
    print(f"testbed: object from the mean of {nf - 1} frames of {fn_full}; masks: frames "
          f"{mask_start}-{mask_start + nt - 1} of {fn_masks} "
          f"(R {om[..., 0].size / om[..., 0].sum():.1f})")

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
    et = torch.from_numpy(et_np).to(device)
    A = build_sense_b0(smaps_chw, omega, b0_t, et, L=L_b0, nbins=nbins_b0)
    y = A.apply(x_true)  # (K, Nc, nt)
    g = torch.Generator(device=device).manual_seed(seed)
    y = y + torch.complex(torch.randn(y.shape, generator=g, device=device),
                          torch.randn(y.shape, generator=g, device=device)) / math.sqrt(2)

    os.makedirs(os.path.dirname(os.path.abspath(fn_out)), exist_ok=True)
    with h5py.File(fn_out, "w") as f:
        ds = f.create_dataset("ksp_epi_zf", (Nx, Ny, Nz, Nc, nt), np.complex64,
                              chunks=(Nx, Ny, Nz, Nc, 1))
        for t in range(nt):
            frame = torch.zeros(Nx * Ny * Nz, Nc, dtype=torch.complex64, device=device)
            frame[A.A[t].idx] = y[:, :, t]
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
                       testbed_timing=timing)
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


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("full", help="fully sampled run's <name>_preprocessed.h5")
    p.add_argument("masks", help="undersampled run's <name>_preprocessed.h5 (the masks)")
    p.add_argument("out", help="<name>_preprocessed.h5 to write")
    p.add_argument("--volume-tr", type=float, required=True)
    p.add_argument("--nt", type=int, default=80)
    p.add_argument("--mask-start", type=int, default=1)
    p.add_argument("--amp", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--timing", choices=["full", "masks"], default="full",
                   help="echo times from the fully sampled run or from the masks' run")
    a = p.parse_args()
    build_testbed(a.full, a.masks, a.out, a.volume_tr, nt=a.nt, mask_start=a.mask_start,
                  amp=a.amp, seed=a.seed, device=a.device, timing=a.timing)


if __name__ == "__main__":
    _cli()
