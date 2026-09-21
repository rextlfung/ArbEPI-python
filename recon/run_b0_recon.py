"""One-off driver: reconstruct real acquisitions with full time-segmented
B0 correction (recon/operators_b0.py), replicating the existing G+L
(multi-scale) config already validated against ../mslr-recon
(recon/analysis/validate_against_mslr.py) for the *uncorrected* case, and saving
results under <datdir>/recon/mslr_b0/G+L_L<L_b0>/ (recon/reconstruct.py) --
one directory per L (matches the convention already on disk from the
L=6/10/16 runs made during the sweep below).

--r2star additionally layers T2*/T1 amplitude-decay correction on top of
the B0 phase correction (recon/operators_b0.py's r2star_map/t_ref_s
generalization -- see that function's docstring for the ψ(r) = i*2*pi*Δf(r)
- R2*(r) complex-field derivation and its sign-convention divergence from
the worktree-lowres-calib-recon branch's adjoint-only calib script). R2* is
estimated from the same dual-echo deGRE data already used for the B0 map
(preprocessing/r2star_map.py's two-point log-ratio) and referenced to the
nominal-TE echo's acquisition time (scan_info.mat's
schedules[0,0,(ETL-1)//2,2] -- the same value
recon/lowres_calib/lowres_calib_recon_b0complex.py reads, read the same way here rather
than re-derived). Output moves to <datdir>/recon/mslr_b0complex/G+L_L<L_b0>/
so a --r2star run never collides with a plain B0-only run at the same L.

sigma1A is not reused from the uncorrected reference: the B0-corrected
operator's spectral norm has no known closed form (mri_exp_approx's B
weights are a least-squares fit, not guaranteed unit-norm/orthogonal -- see
operators_b0.py), so it's measured here via power iteration
(estimate_spectral_norm) before the real reconstruction runs, on the actual
per-dataset smaps/omega/b0map/echo_times rather than assumed.

L (segment count) defaults to 32 -- see operators_b0.py's module docstring
for the real-scale sweep (recon/analysis/sweep_time_segments.py) that settled it;
--L still lets it be overridden per run without a code change.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.run_b0_recon <datdir> <name>
e.g.
    .venv-recon/bin/python -m recon.run_b0_recon \\
        /StorageRAID/rexfung/20260822ball_laminar laminar
"""

import argparse
import os

import h5py
import numpy as np
import torch

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.matio import read_mat
from preprocessing.r2star_map import estimate_r2star_map_epi_grid
from recon.analysis.validate_against_mslr import _read_julia_mat
from recon.operators_b0 import build_encoding_operator_b0, estimate_spectral_norm
from recon.reconstruct import (
    _load_array,
    _load_echo_times,
    _load_normalized_smaps,
    run_recon,
    save_result,
)


def _load_omega(fn_ksp: str, Nx: int) -> np.ndarray:
    """(Nx,Ny,Nz,Nt) sampling mask, broadcast across the readout axis.

    Mirrors recon/reconstruct.py's own _load_omega: prefers the
    authoritative 'omegas' dataset preprocess.py writes (a few hundred KB)
    over inferring the mask from which k-space values happen to be exactly
    zero on a single coil -- a real acquired sample can round to exactly
    0+0j after phase correction, silently becoming "not acquired" on
    whichever coil is checked. Reading 'omegas' directly also avoids a
    second full ~11 GB / ~23s read of ksp_epi_zf (run_recon() below already
    loads it once); only the fallback path (for a recon file written before
    'omegas' existed) pays that cost.
    """
    with h5py.File(fn_ksp, "r") as f:
        has_omegas = "omegas" in f
        if has_omegas:
            omegas_yzt = np.asarray(f["omegas"][()])
    if has_omegas:
        return np.broadcast_to(omegas_yzt[None], (Nx,) + omegas_yzt.shape).astype(bool)

    print(
        f"  '{fn_ksp}' has no 'omegas' dataset (written before preprocess.py added it) -- "
        "falling back to inferring the sampling mask from exact-zero k-space values."
    )
    ksp0_coil0 = _load_array(fn_ksp, "ksp_epi_zf")[:, :, :, 0, :]
    return ksp0_coil0 != 0


def _nominal_te_s(scan_info_path: str, etl: int) -> float:
    """The prescribed-TE echo's acquisition time (seconds since RF
    excitation), frame/shot-invariant by construction (see CLAUDE.md's
    mask2epi_radial paragraph) -- read directly from scan_info.mat rather
    than re-derived, matching
    recon/lowres_calib/lowres_calib_recon_b0complex.py's own nominal_te_s on the
    worktree-lowres-calib-recon branch."""
    schedules = read_mat(scan_info_path, ["schedules"])["schedules"]  # (Nframes,Nshots,ETL,3)
    return float(schedules[0, 0, (etl - 1) // 2, 2])


def main(
    datdir: str, name: str, L_b0: int = 32, nbins_b0: int = 128, device: str = "cuda",
    r2star: bool = False,
) -> None:
    device_t = torch.device(device)
    recon_dir = os.path.join(datdir, "recon")
    fn_ksp = os.path.join(recon_dir, "ArbEPI_epi_zf.h5")
    fn_smaps = os.path.join(recon_dir, "smaps_ArbEPI_sigpy.h5")
    fn_b0map = os.path.join(recon_dir, "ArbEPI_b0map.h5")
    fn_ref = os.path.join(recon_dir, "mslr", "G+L", f"{name}_recon.mat")

    ref = _read_julia_mat(fn_ref)
    print(
        f"Reference config (from {fn_ref}): Nscales={ref['Nscales']}, "
        f"patch_sizes={ref['patch_sizes']}, strides={ref['strides']}, "
        f"lambda_global={ref['lambda_global']}, niters={ref['Niters']}, "
        f"conv_tol={ref['conv_tol']}, mom={ref['mom']}"
    )

    cfg = load_config(datdir=datdir, seqnames=["ArbEPI"])
    paths = set_seq_paths(cfg, "ArbEPI")
    sp = load_seq_params(paths)

    print("Loading smaps/omega/B0 map/echo_times for spectral-norm estimation...")
    smaps, smaps_chw = _load_normalized_smaps(fn_smaps, device_t)
    Nx, Ny, Nz, _Nvc = smaps.shape

    omega = torch.from_numpy(_load_omega(fn_ksp, Nx)).to(device_t)
    b0map_hz = torch.from_numpy(_load_array(fn_b0map, "b0map_hz").astype(np.float32)).to(device_t)
    echo_times_yz = _load_echo_times(fn_ksp, device_t)

    r2star_map, t_ref_s = None, 0.0
    if r2star:
        t_ref_s = _nominal_te_s(paths.scan_info, sp.ETL)
        print(
            f"  Estimating R2* map from dual-echo deGRE data "
            f"(TE_nominal={t_ref_s * 1000:.3f} ms)..."
        )
        r2star_map = torch.from_numpy(
            estimate_r2star_map_epi_grid(datdir, "ArbEPI", sp.fov_degre, sp.fov, (Nx, Ny, Nz))
        ).to(device_t)

    corrected_label = "B0+R2*-corrected" if r2star else "B0-corrected"
    print(
        f"Estimating sigma1(A) for the {corrected_label} operator "
        f"(L={L_b0}, nbins={nbins_b0})..."
    )
    A = build_encoding_operator_b0(
        smaps_chw, omega, b0map_hz, echo_times_yz, L=L_b0, nbins=nbins_b0,
        r2star_map=r2star_map, t_ref_s=t_ref_s,
    )
    Nt = omega.shape[-1]  # A is the full BlockDiagonal over every frame, size_in=(Nx,Ny,Nz,Nt)
    x0 = torch.randn(Nx, Ny, Nz, Nt, dtype=torch.complex64, device=device_t)
    sigma1A = estimate_spectral_norm(A, x0)
    print(
        f"  sigma1A ({corrected_label}) = {sigma1A:.6f}  "
        f"(uncorrected reference: {ref['sigma1A']:.6f})"
    )
    del A, smaps, smaps_chw, omega, b0map_hz, echo_times_yz, x0
    if device_t.type == "cuda":
        torch.cuda.empty_cache()

    out_subdir = "mslr_b0complex" if r2star else "mslr_b0"
    out_dir = os.path.join(recon_dir, out_subdir, f"G+L_L{L_b0}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nRunning {corrected_label} G+L reconstruction for {name}...")
    result = run_recon(
        fn_ksp=fn_ksp,
        fn_smaps=fn_smaps,
        patch_sizes=ref["patch_sizes"],
        strides=ref["strides"],
        niters=int(ref["Niters"]),
        sigma1A=sigma1A,
        mom=ref["mom"],
        conv_tol=float(ref["conv_tol"]),
        lambda_global=float(ref["lambda_global"]),
        device=device,
        fn_b0map=fn_b0map,
        L_b0=L_b0,
        nbins_b0=nbins_b0,
        r2star_map=r2star_map,
        t_ref_s=t_ref_s,
    )

    fn_out = os.path.join(out_dir, f"{name}_recon")
    save_result(
        fn_out,
        result,
        fov=sp.fov,
        seqname="ArbEPI",
        dataset=name,
        L_b0=L_b0,
        nbins_b0=nbins_b0,
        uncorrected_sigma1A_reference=float(ref["sigma1A"]),
        r2star=r2star,
        t_ref_s=t_ref_s,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datdir")
    parser.add_argument("name", help="'laminar' or 'radial' -- matches mslr/G+L/<name>_recon.mat")
    parser.add_argument("--L", type=int, default=32, dest="L_b0")
    parser.add_argument("--nbins", type=int, default=128, dest="nbins_b0")
    parser.add_argument(
        "--r2star", action="store_true",
        help="also correct T2*/T1 amplitude decay (see operators_b0.py's "
             "r2star_map/t_ref_s docstring)",
    )
    args = parser.parse_args()
    main(args.datdir, args.name, L_b0=args.L_b0, nbins_b0=args.nbins_b0, r2star=args.r2star)
