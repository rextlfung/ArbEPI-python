"""python -m postprocess: motion-correct a reconstruction.

    python -m postprocess --recon <name>_recon.h5 --out <name>_mc_recon.h5

Writes the corrected series in the same layout `analyze/` reads (`X_recon`, and
`X` per-scale components when present, all resampled with one set of transforms),
the transforms and motion parameters, and `<out minus .h5>.par` (rx ry rz [rad],
tx ty tz [mm] per frame).
"""

from __future__ import annotations

import argparse
import os
import shutil

import h5py
import numpy as np

from analyze.io import _read_t, voxel_size_from_sidecar
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
    a = ap.parse_args(argv)

    vox = tuple(a.voxel_size) if a.voxel_size else voxel_size_from_sidecar(a.recon)
    if vox is None:
        raise SystemExit("no voxel size: no .json sidecar next to the recon, pass --voxel-size")
    ref = a.ref if a.ref in ("middle", "mean") else int(a.ref)
    with h5py.File(a.recon, "r") as f:
        X_recon = _read_t(f["X_recon"], slice(a.n_discard, None))
        X = _read_t(f["X"], slice(a.n_discard, None)) if "X" in f and f["X"].ndim == 5 else None

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
    with h5py.File(a.out, "w") as f:
        f.create_dataset("X_recon", data=res.series)
        if X is not None:
            comps = [apply_transforms(X[..., s], res.matrices, vox, a.order)
                     for s in range(X.shape[4])]
            f.create_dataset("X", data=np.stack(comps, axis=4))
        f.create_dataset("motion/matrices", data=res.matrices)
        f.create_dataset("motion/params", data=res.params)
        f.create_dataset("motion/ref", data=res.ref)
        f.attrs["motion_method"] = res.method
        f.attrs["n_discard"] = a.n_discard
        f.attrs["source"] = os.path.abspath(a.recon)
    np.savetxt(os.path.splitext(a.out)[0] + ".par", res.params, fmt="%.6f")
    side = os.path.splitext(a.recon)[0] + ".json"
    if os.path.exists(side):  # keep the fov for analyze's voxel size
        shutil.copy(side, os.path.splitext(a.out)[0] + ".json")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
