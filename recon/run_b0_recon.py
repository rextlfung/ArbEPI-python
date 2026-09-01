"""One-off driver: reconstruct real acquisitions with full time-segmented
B0 correction (recon/operators_b0.py), replicating the existing G+L
(multi-scale) config already validated against ../mslr-recon
(recon/validate_against_mslr.py) for the *uncorrected* case, and saving
results under <datdir>/recon/mslr_b0/G+L_L<L_b0>/ (recon/save_result.py) --
one directory per L, since L is under active investigation, not settled.
(Matches the convention already on disk from the L=6/10/16 runs.)

sigma1A is not reused from the uncorrected reference: the B0-corrected
operator's spectral norm has no known closed form (mri_exp_approx's B
weights are a least-squares fit, not guaranteed unit-norm/orthogonal -- see
operators_b0.py), so it's measured here via power iteration
(estimate_spectral_norm) before the real reconstruction runs, on the actual
per-dataset smaps/omega/b0map/echo_times rather than assumed.

L (segment count) is a fixed engineering default (6, matching mirtorch's
own Gmri default), not swept against a real min-max error bound the way
Fable's staged plan recommends -- see CLAUDE.md's recon/ section for that
open item; --L lets it be overridden per run without a code change.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.run_b0_recon <datdir> <name>
e.g.
    .venv-recon/bin/python -m recon.run_b0_recon \\
        /StorageRAID/rexfung/20260822ball_laminar laminar
"""

import argparse
import os

import numpy as np
import torch

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from recon.operators_b0 import build_encoding_operator_b0, estimate_spectral_norm
from recon.reconstruct import _load_array, run_recon
from recon.save_result import save_result
from recon.validate_against_mslr import _read_julia_mat


def _load_omega(fn_ksp: str) -> np.ndarray:
    """omega (coil 0's sampled-or-not pattern) from the full ksp_epi_zf
    load. A first attempt tried slicing out just coil 0 per chunk to avoid
    loading all 18 coils, but that slice breaks h5py's contiguous-chunk
    fast path (chunks span all coils within one frame -- see
    recon/reconstruct.py's _load_array docstring) and measured ~4x *slower*
    in wall-clock (91s vs ~23s) despite touching less final data -- so this
    just reuses _load_array's already-optimized full chunked read and
    discards everything but coil 0 afterward, matching run_recon()'s own
    approach. This does mean ksp_epi_zf gets loaded twice across this
    driver script (here, and again inside run_recon() itself) -- a real
    but bounded, one-off cost, not worth a bigger refactor of run_recon()
    to expose its internal load for reuse."""
    ksp0_coil0 = _load_array(fn_ksp, "ksp_epi_zf")[:, :, :, 0, :]
    return ksp0_coil0 != 0


def main(datdir: str, name: str, L_b0: int = 6, nbins_b0: int = 128, device: str = "cuda") -> None:
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
    smaps_raw = torch.from_numpy(_load_array(fn_smaps, "smaps").astype(np.complex64)).to(device_t)
    smaps_rss = smaps_raw.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    smaps = smaps_raw / (smaps_rss + torch.finfo(torch.float32).eps)
    smaps_chw = smaps.permute(3, 0, 1, 2).contiguous()
    Nx, Ny, Nz, _Nvc = smaps.shape

    omega = torch.from_numpy(_load_omega(fn_ksp)).to(device_t)
    b0map_hz = torch.from_numpy(_load_array(fn_b0map, "b0map_hz").astype(np.float32)).to(device_t)
    echo_times_2d = torch.from_numpy(_load_array(fn_ksp, "echo_times").astype(np.float32))
    echo_times_s = echo_times_2d.to(device_t).unsqueeze(0).expand(Nx, -1, -1, -1).contiguous()

    print(f"Estimating sigma1(A) for the B0-corrected operator (L={L_b0}, nbins={nbins_b0})...")
    A = build_encoding_operator_b0(smaps_chw, omega, b0map_hz, echo_times_s, L=L_b0, nbins=nbins_b0)
    Nt = omega.shape[-1]  # A is the full BlockDiagonal over every frame, size_in=(Nx,Ny,Nz,Nt)
    x0 = torch.randn(Nx, Ny, Nz, Nt, dtype=torch.complex64, device=device_t)
    sigma1A = estimate_spectral_norm(A, x0, niter=30)
    print(
        f"  sigma1A (B0-corrected) = {sigma1A:.6f}  "
        f"(uncorrected reference: {ref['sigma1A']:.6f})"
    )
    del A, smaps_raw, smaps, smaps_chw, omega, b0map_hz, echo_times_2d, echo_times_s, x0
    if device_t.type == "cuda":
        torch.cuda.empty_cache()

    out_dir = os.path.join(recon_dir, "mslr_b0", f"G+L_L{L_b0}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nRunning B0-corrected G+L reconstruction for {name}...")
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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datdir")
    parser.add_argument("name", help="'laminar' or 'radial' -- matches mslr/G+L/<name>_recon.mat")
    parser.add_argument("--L", type=int, default=6, dest="L_b0")
    parser.add_argument("--nbins", type=int, default=128, dest="nbins_b0")
    args = parser.parse_args()
    main(args.datdir, args.name, L_b0=args.L_b0, nbins_b0=args.nbins_b0)
