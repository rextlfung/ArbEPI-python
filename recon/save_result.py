"""Save a ReconResult to disk -- the "natural next piece" CLAUDE.md's
recon/ section flagged as missing: run_recon() itself only returns a
ReconResult in memory, this is what actually persists one.

Writes two files per result: `<fn_base>.nii.gz` + `.json` (magnitude image
+ metadata sidecar, via preprocessing/nifti_io.py's save_recon_nifti --
same format/convention every other reconstructed image in this pipeline
uses) and `<fn_base>.h5` (full-precision complex X_recon/X plus the solver
convergence trace, plain numpy-order h5py -- no MATLAB consumer, matching
this repo's own .h5-not-.mat convention for internal artifacts).
"""

import h5py
import numpy as np

from preprocessing.nifti_io import save_recon_nifti
from recon.reconstruct import ReconResult


def save_result(
    fn_base: str, result: ReconResult, fov: tuple[float, float, float], **extra_attrs
) -> None:
    # Raw complex data first, deliberately: a run_recon() call can take tens
    # of minutes, and save_recon_nifti (below) needs a plain numpy array,
    # not a CUDA tensor -- getting that boundary wrong once already lost a
    # completed real reconstruction (see git history / session notes), so
    # the full-precision .h5 -- needing no such conversion care beyond the
    # explicit .cpu().numpy() already here -- goes to disk before anything
    # else gets a chance to fail.
    X_recon_np = result.X_recon.detach().cpu().numpy()
    with h5py.File(f"{fn_base}.h5", "w") as f:
        f.create_dataset("X_recon", data=X_recon_np)
        f.create_dataset("X", data=result.X.detach().cpu().numpy())
        f.create_dataset("omega", data=result.omega.detach().cpu().numpy())
        f.create_dataset("dc_costs", data=np.asarray(result.dc_costs))
        f.create_dataset("reg_costs", data=np.asarray(result.reg_costs))
        f.create_dataset("restarts", data=np.asarray(result.restarts, dtype=bool))
        f.create_dataset("rel_changes", data=np.asarray(result.rel_changes))
        f.attrs["R"] = result.R
        f.attrs["sigma1A"] = result.sigma1A
        f.attrs["L"] = result.L
        f.attrs["runtime_s"] = result.runtime_s
    print(f"Wrote {fn_base}.h5")

    save_recon_nifti(
        fn_base,
        X_recon_np,
        fov=fov,
        R=result.R,
        sigma1A=result.sigma1A,
        L=result.L,
        lambdas=result.lambdas,
        runtime_s=result.runtime_s,
        n_iters=len(result.dc_costs) - 1,
        final_dc_cost=result.dc_costs[-1],
        final_reg_cost=result.reg_costs[-1],
        **result.meta,
        **extra_attrs,
    )
    print(f"Wrote {fn_base}.nii.gz + .json")
