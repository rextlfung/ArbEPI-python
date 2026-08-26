"""Validate recon/reconstruct.py against real ../mslr-recon (Julia/MIRT.jl)
output, field by field. Not a pytest test -- like seq2ge/validate_against_
matlab.py, this depends on real reference output that isn't committed to
this repo (machine-specific acquisition data + a completed mslr-recon run).

Usage (from repo root, using the .venv-recon environment):
    .venv-recon/bin/python -m recon.validate_against_mslr <julia_reconstruct.mat>

The reference .mat is produced by ../mslr-recon's scripts/reconstruct.jl
(e.g. via experiments/20260822ball.jl -- run_recon's own `matwrite` call).
Every reconstruction parameter (fn_ksp/fn_smaps by convention -- see below,
patch_sizes, strides, sigma1A, lambda_global, conv_tol, niters, mom) is read
directly from the reference file rather than re-specified, so this always
replicates exactly what the reference run used, and fn_ksp/fn_smaps are
derived from the reference .mat's own directory layout
(<recon_dir>/mslr/<subdir>/<name>.mat -> <recon_dir>/ArbEPI_epi_zf.h5 +
<recon_dir>/smaps_ArbEPI_sigpy.h5 -- matching experiments/20260822ball.jl's
own `datasets` table), unless overridden with --ksp/--smaps.

Validated results (2026-08-25, RTX A6000, 20260822ball_radial dataset,
Nx,Ny,Nz,Nvc,Nt=240,240,45,18,30, R~9):

  config  iters  dc reldiff  reg reldiff  X_recon reldiff  Pearson r     py/julia s
  L       55     5.9e-7      3.0e-6       1.6e-5           0.9999999998  309/405
  G       56     1.6e-6      8.1e-6       3.8e-5           0.9999999989  96/134
  G+L     101    1.6e-6      2.1e-4       2.1e-5           0.9999999997  597/785

All three configs converge to the same iteration count as the Julia run
(confirming pogm_restart's early-stopping logic matches exactly) and match
to float32 summation-order noise -- the same class of ~1-ULP difference this
repo's seq2ge/ port already documents against real MATLAB output. Python
also runs consistently faster despite a simpler (fully-batched, not
hand-tuned) SVD/FFT implementation.
"""

import argparse
import os
import sys

import h5py
import numpy as np

from recon.reconstruct import run_recon


def _read_julia_mat(path: str) -> dict:
    """MAT.jl's matwrite output (v7.3, HDF5-backed): arrays are stored
    axis-reversed on disk like hdf5storage's Python-side v7.3 writer (see
    preprocessing/matio.py) -- reverse with .transpose(). Complex arrays are
    stored as a {real, imag} compound dtype rather than natively."""
    scalar_keys = ("R", "sigma1A", "L", "Nscales", "lambda_global", "Niters", "conv_tol")
    array_keys = ("dc_costs", "reg_costs", "lambdas")
    out = {}
    with h5py.File(path, "r") as f:
        for key in scalar_keys:
            out[key] = f[key][()].item()
        for key in array_keys:
            out[key] = np.asarray(f[key][()]).ravel()
        out["mom"] = f["mom"][()].tobytes().decode("utf-16-le").rstrip("\x00")
        raw = f["X_recon"][()]
        out["X_recon"] = (raw["real"] + 1j * raw["imag"]).astype(np.complex64).transpose()
        out["patch_sizes"] = [tuple(int(v) for v in f[ref][()]) for ref in f["patch_sizes"][()]]
        out["strides"] = [tuple(int(v) for v in f[ref][()]) for ref in f["strides"][()]]
    return out


def validate(fn_ksp: str, fn_smaps: str, fn_julia_mat: str) -> bool:
    ref = _read_julia_mat(fn_julia_mat)

    result = run_recon(
        fn_ksp=fn_ksp,
        fn_smaps=fn_smaps,
        patch_sizes=ref["patch_sizes"],
        strides=ref["strides"],
        niters=int(ref["Niters"]),
        sigma1A=float(ref["sigma1A"]),
        mom=ref["mom"],
        conv_tol=float(ref["conv_tol"]),
        lambda_global=float(ref["lambda_global"]),
    )

    ok = True

    def check(name, got, want, rtol=1e-4):
        nonlocal ok
        rel = abs(got - want) / (abs(want) + 1e-30)
        status = "OK  " if rel <= rtol else "FAIL"
        ok &= rel <= rtol
        print(f"{status} {name}: python={got} julia={want} rel_diff={rel:.3e}")

    check("R", result.R, float(ref["R"]))
    check("sigma1A", result.sigma1A, float(ref["sigma1A"]))
    check("L", result.L, float(ref["L"]))
    check("n_iters (incl iter0)", len(result.dc_costs), len(ref["dc_costs"]), rtol=0)

    n = min(len(result.dc_costs), len(ref["dc_costs"]))
    dc_p, dc_j = np.array(result.dc_costs[:n]), np.array(ref["dc_costs"][:n])
    reg_p, reg_j = np.array(result.reg_costs[:n]), np.array(ref["reg_costs"][:n])
    check("dc_cost[-1]", dc_p[-1], dc_j[-1])
    # Multi-scale reg_cost (Nscales>1, summing nuclear norms across scales
    # including a giant whole-volume SVD) accumulates more floating-point
    # noise than a single-scale reg_cost -- measured ~1-2e-4 on both real
    # G+L runs (radial and laminar) vs ~1e-6 for single-scale L/G, so this
    # check alone gets a looser tolerance rather than loosening every check.
    check("reg_cost[-1]", reg_p[-1], reg_j[-1], rtol=5e-4)
    dc_max_rel = float((np.abs(dc_p - dc_j) / np.abs(dc_j)).max())
    reg_max_rel = float((np.abs(reg_p - reg_j) / np.abs(reg_j)).max())
    print(f"     dc_cost max rel diff over all iters: {dc_max_rel:.3e}")
    print(f"     reg_cost max rel diff over all iters: {reg_max_rel:.3e}")
    ok &= dc_max_rel < 1e-3 and reg_max_rel < 1e-2

    Xj, Xp = ref["X_recon"], result.X_recon.cpu().numpy()
    rel_l2 = float(np.linalg.norm(Xj - Xp) / np.linalg.norm(Xj))
    pearson_r = float(np.corrcoef(np.abs(Xj).ravel(), np.abs(Xp).ravel())[0, 1])
    print(f"     X_recon relative L2 error: {rel_l2:.3e}")
    print(f"     X_recon Pearson r (magnitude): {pearson_r:.10f}")
    ok &= rel_l2 < 1e-2 and pearson_r > 0.999

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fn_julia_mat")
    parser.add_argument("--ksp", default=None)
    parser.add_argument("--smaps", default=None)
    args = parser.parse_args()

    # <recon_dir>/mslr/<subdir>/<name>.mat -> <recon_dir>/ArbEPI_epi_zf.h5,
    # matching experiments/20260822ball.jl's own `datasets` table.
    recon_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.fn_julia_mat)))
    fn_ksp = args.ksp or os.path.join(recon_dir, "ArbEPI_epi_zf.h5")
    fn_smaps = args.smaps or os.path.join(recon_dir, "smaps_ArbEPI_sigpy.h5")

    ok = validate(fn_ksp, fn_smaps, args.fn_julia_mat)
    sys.exit(0 if ok else 1)
