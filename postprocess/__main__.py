"""python -m postprocess: motion-correct a reconstruction, optionally brain-mask it.

    python -m postprocess --recon <name>_recon.h5 --out <name>_mc_recon.h5 \\
        [--brain-mask-from <seq>_preprocessed.h5]

Writes the corrected series in the same layout `analyze/` reads (`X_recon`, and
`X` per-scale components when present, all resampled with one set of transforms),
the transforms and motion parameters, and `<out minus .h5>.par` (rx ry rz [rad],
tx ty tz [mm] per frame). With --brain-mask-from, the corrected series is then
zeroed outside a brain mask estimated on the preprocessed file's T1-weighted deGRE
(postprocess/brainmask.py), registered to the series' mean first; the mask is stored as
`brain_mask` and written as `<out minus .h5>_brain_mask.nii.gz`.
"""

from __future__ import annotations

import argparse
import os
import shutil

import h5py
import numpy as np

from analyze.io import read_frames, voxel_size_from_sidecar
from analyze.mask import brain_mask
from postprocess.brainmask import apply_brain_mask, load_gre, mask_shift_mm, register_mask
from postprocess.motion import apply_transforms, correct_motion


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m postprocess", description=__doc__.split("\n\n")[0])
    ap.add_argument("--recon", required=True, help="<name>_recon.h5 from recon.sense")
    ap.add_argument("--out", required=True, help="output .h5")
    ap.add_argument("--method", default="auto", choices=["auto", "fsl", "python"])
    ap.add_argument("--ref", default="middle", help="'middle', 'mean' or a frame index")
    ap.add_argument("--n-discard", type=int, default=0, help="leading frames to drop")
    ap.add_argument("--order", type=int, default=3, help="resampling spline order")
    ap.add_argument("--voxel-size", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
                    help="mm; default from the recon's .json sidecar")
    ap.add_argument("--brain-mask-from", metavar="PREPROCESSED.h5",
                    help="mask the corrected series with a BET mask of this file's deGRE echo 1")
    ap.add_argument("--no-mask-registration", action="store_true",
                    help="use the deGRE mask as is, without registering the deGRE to the EPI mean")
    ap.add_argument("--bet-frac", type=float, default=0.5, help="BET fractional intensity threshold")
    a = ap.parse_args(argv)

    vox = tuple(a.voxel_size) if a.voxel_size else voxel_size_from_sidecar(a.recon)
    if vox is None:
        raise SystemExit("no voxel size: no .json sidecar next to the recon, pass --voxel-size")
    ref = a.ref if a.ref in ("middle", "mean") else int(a.ref)
    with h5py.File(a.recon, "r") as f:
        X_recon = read_frames(f["X_recon"], slice(a.n_discard, None))
        X = read_frames(f["X"], slice(a.n_discard, None)) if "X" in f and f["X"].ndim == 5 else None

    last = [0]

    def progress(i, n):
        if i == n or i - last[0] >= max(1, n // 10):
            last[0] = i
            print(f"  registered {i}/{n}", flush=True)

    res = correct_motion(X_recon, vox, ref=ref, method=a.method, order=a.order, progress=progress)
    fd = res.framewise_displacement_mm
    print(f"{res.method}: max |t| {np.abs(res.params[:, 3:]).max():.2f} mm, "
          f"max |r| {np.degrees(np.abs(res.params[:, :3]).max()):.2f} deg, "
          f"mean FD {fd.mean():.3f} mm")
    series = res.series
    comps = None
    if X is not None:
        comps = np.stack([apply_transforms(X[..., s], res.matrices, vox, a.order)
                          for s in range(X.shape[4])], axis=4)
    mask = None
    if a.brain_mask_from:
        gre, mvox = load_gre(a.brain_mask_from)
        if gre.shape != series.shape[:3]:
            raise SystemExit(f"deGRE {gre.shape} is not on the EPI grid {series.shape[:3]}")
        mask = brain_mask(gre, mvox, frac=a.bet_frac)
        xfm = None
        if not a.no_mask_registration:
            n0 = int(mask.sum())
            mask, xfm = register_mask(gre, mask, np.abs(series).mean(axis=3), vox)
            shift = mask_shift_mm(xfm, mask, vox)
            print(f"deGRE -> EPI registration: mask moved {shift:.2f} mm, mask {n0} -> {int(mask.sum())} voxels")
        series = apply_brain_mask(series, mask)
        comps = apply_brain_mask(comps, mask) if comps is not None else None
        print(f"brain mask: {mask.sum() * np.prod(mvox) / 1e3:.0f} cm3 from {a.brain_mask_from}")
    with h5py.File(a.out, "w") as f:
        f.create_dataset("X_recon", data=series)
        if comps is not None:
            f.create_dataset("X", data=comps)
        if mask is not None:
            f.create_dataset("brain_mask", data=mask.astype(np.uint8))
            if xfm is not None:
                f.create_dataset("brain_mask_xfm", data=xfm)  # FSL matrix (mm), deGRE -> EPI mean
            f.attrs["brain_mask_source"] = os.path.abspath(a.brain_mask_from)
        f.create_dataset("motion/matrices", data=res.matrices)
        f.create_dataset("motion/params", data=res.params)
        f.create_dataset("motion/ref", data=res.ref)
        f.attrs["motion_method"] = res.method
        f.attrs["n_discard"] = a.n_discard
        f.attrs["source"] = os.path.abspath(a.recon)
    if mask is not None:
        import nibabel as nib

        nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.diag([*vox, 1.0])),
                 os.path.splitext(a.out)[0] + "_brain_mask.nii.gz")
    np.savetxt(os.path.splitext(a.out)[0] + ".par", res.params, fmt="%.6f")
    side = os.path.splitext(a.recon)[0] + ".json"
    if os.path.exists(side):  # keep the fov for analyze's voxel size
        shutil.copy(side, os.path.splitext(a.out)[0] + ".json")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
