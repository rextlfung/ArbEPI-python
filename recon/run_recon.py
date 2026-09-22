"""Real-data reconstruction drivers on this repo's own zero-filled k-space
(.venv-recon), all sharing the B0-corrected (or plain) GatheredSense operator
and differing in the solver. Subcommands, each with the flags of the
standalone script it replaced:

    mslr-ref    MSLR G+L with B0 (and optional R2*) correction, parameters read from a ../mslr-recon reference .mat
    mslr-local  MSLR with a single local-low-rank scale, B0-corrected by default
    cg          unregularized CG-SENSE, B0-corrected by default

    .venv-recon/bin/python -m recon.run_recon {mslr-ref,mslr-local,cg} <datdir> <name|seqname> ...

The sections below are the original module docstrings, kept verbatim.

Formerly recon/run_b0_recon.py
------------------------------
One-off driver: reconstruct real acquisitions with full time-segmented
B0 correction (recon/operators.py), replicating the existing G+L
(multi-scale) config already validated against ../mslr-recon
(recon/analysis.py) for the *uncorrected* case, and saving
results under <datdir>/recon/mslr_b0/G+L_L<L_b0>/ (recon/mslr.py) --
one directory per L (matches the convention already on disk from the
L=6/10/16 runs made during the sweep below).

--r2star additionally layers T2*/T1 amplitude-decay correction on top of
the B0 phase correction (recon/operators.py's r2star_map/t_ref_s
generalization -- see that function's docstring for the ψ(r) = i*2*pi*Δf(r)
- R2*(r) complex-field derivation and its sign-convention divergence from
the worktree-lowres-calib-recon branch's adjoint-only calib script). R2* is
estimated from the same dual-echo deGRE data already used for the B0 map
(preprocessing/r2star_map.py's two-point log-ratio) and referenced to the
nominal-TE echo's acquisition time (scan_info.mat's
schedules[0,0,(ETL-1)//2,2] -- the same value
recon/lowres_calib.py reads, read the same way here rather
than re-derived). Output moves to <datdir>/recon/mslr_b0complex/G+L_L<L_b0>/
so a --r2star run never collides with a plain B0-only run at the same L.

sigma1A is not reused from the uncorrected reference: the B0-corrected
operator's spectral norm has no known closed form (mri_exp_approx's B
weights are a least-squares fit, not guaranteed unit-norm/orthogonal -- see
operators.py), so it's measured here via power iteration
(estimate_spectral_norm) before the real reconstruction runs, on the actual
per-dataset smaps/omega/b0map/echo_times rather than assumed.

L (segment count) defaults to 32 -- see operators.py's module docstring
for the real-scale sweep (recon/analysis.py) that settled it;
--L still lets it be overridden per run without a code change.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.run_recon mslr-ref <datdir> <name>
e.g.
    .venv-recon/bin/python -m recon.run_recon mslr-ref \\
        /StorageRAID/rexfung/20260822ball_laminar laminar


Formerly recon/run_mslr_local.py
--------------------------------
One-off driver: MSLR reconstruction with a single local-low-rank scale,
for this repo's own <seqname>_epi_zf.h5 / smaps_<seqname>_sigpy.h5 naming
convention (unlike run_recon.py / analysis.py, both
hardcoded to the ArbEPI_epi_zf.h5 name and, for the latter, a Julia
reference .mat that doesn't exist for a dataset that was never run through
../mslr-recon).

Motivation: recon/lowres_calib.py's fully-sampled-calibration-region
reconstruction is a fast diagnostic that uses only a few hundred of a
dataset's many thousand k-space samples -- deliberately, for speed, per its
own module docstring -- so its thermal-noise floor is far higher than the
full accelerated reconstruction's. Comparing recon/lowres_temporal_
stability.py's output on this driver's full, high-SNR reconstruction
against the calibration-region-only numbers already measured tests whether
that low-res diagnostic's own noise floor, not a real B0/T2*/spoiling
artifact, was inflating the fluctuation/drift figures measured so far.

patch_sizes/strides default to a single local scale ([(6,6,6)], [(3,3,3)]),
not the G+L multi-scale config recon/analysis.py's Julia
reference uses -- an explicit choice for this investigation (local-only
avoids the giant whole-volume SVD's added runtime/memory at the higher
accelerations here, R up to ~93.5, and isolates locally-low-rank spatial
regularization from a global-rank prior). No Julia reference exists for
this patch config on this dataset, so niters/conv_tol/mom fall back to
run_recon's own defaults rather than being read from one.

B0 correction defaults to on (`--no-b0` reverts to the original plain
operator) -- an uncorrected reconstruction shows real geometric distortion
in the phase-encode direction (classic uncorrected-off-resonance EPI
warping), confirmed by comparing this driver's plain output against
run_recon.py's B0-corrected one on the same data: the latter's object is
round, the former's is not. sigma1A has no closed form for either operator
(see operators.py's estimate_spectral_norm docstring) -- this driver
measures it once via power iteration before the real reconstruction runs,
the same pattern run_recon.py uses.

lambda_global (the Ong & Lustig regularization scale multiplying
_reg_weights' closed-form lambda_k) defaults to this dataset's own
acceleration factor R, not run_recon's flat default of 1.0 -- an explicit
choice (2026-09-18) after comparing reconstructions across this repo's
4-resolution/4-R sweep: _reg_weights' formula has no data-amplitude
normalization, so a fixed lambda_global applies disproportionately more
shrinkage to a lower-signal-amplitude (higher-resolution/higher-R, smaller
voxel volume) acquisition than a higher-amplitude one -- scaling by R
compensates for that so different resolutions in the same sweep get
comparably-effective regularization, rather than reusing the R~9
production-scale value analysis.py's own reference happened
to be tuned at. Override with --lambda-global to pin a specific value
instead (e.g. for reproducing a fixed-lambda comparison).

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.run_recon mslr-local <datdir> <seqname> [--device cuda]
e.g.
    .venv-recon/bin/python -m recon.run_recon mslr-local \\
        /StorageRAID/rexfung/20260915ball 1_1x_5.4mm


Formerly recon/cg_sense_b0.py
-----------------------------
B0-informed CG-SENSE: unregularized conjugate-gradient SENSE
reconstruction using the time-segmented B0-corrected encoding operator
(recon/operators.py's GatheredSenseB0/build_encoding_operator_b0),
solved via literal conjugate gradient (Pruessmann et al.) rather than
recon/mslr.py's multi-scale-low-rank POGM solve.

Ports preprocessing/cg_sense.py's exact CG algorithm (CG on the normal
equations E^H E x = E^H y) onto mirtorch's LinearMap interface
(A.apply/A.adjoint) instead of that module's explicit FFT+mask+smaps
closures, so it works with any encoding operator sharing that contract --
here, the B0-corrected one -- not just the plain uncorrected SENSE operator
preprocessing/cg_sense.py was written for.

Unlike recon/mslr.py's run_recon (POGM, needs sigma1A -- the
operator's spectral norm -- for its step size), CG is self-scaling and
needs no such estimate: skip recon/operators.py's estimate_spectral_norm
entirely here.

Reconstructs the whole (Nx,Ny,Nz,Nt) volume in one CG run rather than
looping per frame: build_encoding_operator_b0 returns a BlockDiagonal
operator (independent per-frame GatheredSenseB0 blocks, see
recon/operators.py's build_encoding_operator docstring), so A^H A has no
cross-frame coupling -- solving jointly over the stacked tensor is
mathematically identical to solving each frame's CG independently, just
one Python-level loop instead of Nt, and matches recon/mslr.py's own
convention of treating the whole (Nx,Ny,Nz,Nt) tensor as one state array.

`num_iter` defaults to 150, not 20 -- the original default converged well at
R=1 (residual 8.6e-4 by iteration 20) but left real undersampled data
nowhere near converged: R=6 residual was still 3.3e-3 at iteration 20 (CG's
per-iteration residual decay rate is measurably slower at higher R, exactly
as SENSE-CG conditioning theory predicts -- Pruessmann et al. 2001).
Verified directly on 2_6x_2.4mm real data (2026-09-18): 150 iterations
brings the residual to 9.95e-4 (vs 3.3e-3 at 20) and the recovered image's
median/peak intensity rise substantially (median +24%, peak +7x) -- the
missing signal at 20 iterations was under-convergence, not a model defect
(see the same day's operator round-trip investigation). 150 is a compromise
across this repo's R=1-93.5 sweep: cheap for R=1 (converges in ~20 anyway)
while getting R=6 most of the way to full convergence; higher-R acquisitions
in the same sweep may still need more -- check the saved `residuals` array
and raise --num-iter if the tail hasn't flattened.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.run_recon cg <datdir> <seqname>
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.matio import read_mat
from preprocessing.nifti_io import save_recon_nifti
from preprocessing.r2star_map import estimate_r2star_map_epi_grid
from recon.analysis import _read_julia_mat
from recon.mslr import (
    _load_array,
    _load_echo_times,
    _load_normalized_smaps,
    _load_omega,
    run_recon,
    save_result,
)
from recon.operators import (
    build_encoding_operator,
    build_encoding_operator_b0,
    estimate_spectral_norm,
    gather_ksp,
)


def _load_omega_broadcast(fn_ksp: str, Nx: int) -> np.ndarray:
    """(Nx,Ny,Nz,Nt) sampling mask, broadcast across the readout axis.

    Mirrors recon/mslr.py's own _load_omega: prefers the
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
    recon/lowres_calib.py's own nominal_te_s on the
    worktree-lowres-calib-recon branch."""
    schedules = read_mat(scan_info_path, ["schedules"])["schedules"]  # (Nframes,Nshots,ETL,3)
    return float(schedules[0, 0, (etl - 1) // 2, 2])


def main_mslr_ref(
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

    omega = torch.from_numpy(_load_omega_broadcast(fn_ksp, Nx)).to(device_t)
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

def _cli_mslr_ref() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datdir")
    parser.add_argument("name", help="'laminar' or 'radial' -- matches mslr/G+L/<name>_recon.mat")
    parser.add_argument("--L", type=int, default=32, dest="L_b0")
    parser.add_argument("--nbins", type=int, default=128, dest="nbins_b0")
    parser.add_argument(
        "--r2star", action="store_true",
        help="also correct T2*/T1 amplitude decay (see operators.py's "
             "r2star_map/t_ref_s docstring)",
    )
    args = parser.parse_args()
    main_mslr_ref(args.datdir, args.name, L_b0=args.L_b0, nbins_b0=args.nbins_b0, r2star=args.r2star)

def main_mslr_local(
    datdir: str, seqname: str, device: str = 'cuda',
    patch_size: tuple[int, int, int] = (6, 6, 6),
    stride: tuple[int, int, int] = (3, 3, 3),
    b0_correct: bool = True,
    L_b0: int = 32,
    nbins_b0: int = 128,
    lambda_global: float | None = None,
    conv_tol: float = 1e-5,
    r2star: bool = False,
    zero_pad_z: bool = False,
) -> None:
    """r2star: layer T2*/T1 amplitude-decay correction on top of the B0
    phase correction (see run_cgsense_b0's docstring for the same pattern
    -- R2* estimated from the dual-echo deGRE data, referenced to the
    nominal-TE echo's acquisition time). Requires b0_correct=True. Output
    moves to mslr_local_b0complex/ so a --r2star run never collides with a
    plain B0-only run.

    zero_pad_z: forwarded to estimate_r2star_map_epi_grid's deGRE-grid ->
    EPI-grid resize (only relevant with r2star=True) -- set True when this
    seqname's own z-FOV exceeds deGRE's fixed z-FOV (e.g. 1_1x_5.4mm's
    145.8mm vs. deGRE's 144mm).

    patch_size/stride/conv_tol are plain passthroughs to run_recon() (same
    defaults run_recon() itself uses) -- per-experiment choices (e.g.
    physical-size patches, disabling early stopping) belong in the
    calling script, not hardcoded here; see _cli_mslr_local's --patch-size/
    --stride/--conv-tol."""
    assert b0_correct or not r2star, 'main_mslr_local: r2star requires b0_correct=True'
    device_t = torch.device(device)
    recon_dir = os.path.join(datdir, 'recon')
    fn_ksp = os.path.join(recon_dir, f'{seqname}_epi_zf.h5')
    fn_smaps = os.path.join(recon_dir, f'smaps_{seqname}_sigpy.h5')
    fn_b0map = os.path.join(recon_dir, f'{seqname}_b0map.h5') if b0_correct else None

    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    sp = load_seq_params(paths)

    print(f'Loading sensitivity maps ({fn_smaps}) for spectral-norm estimation...')
    smaps, smaps_chw = _load_normalized_smaps(fn_smaps, device_t)
    print(f'  Sensitivity maps: {tuple(smaps.shape)}')

    # omega is needed only to build A for the power-iteration spectral-norm
    # estimate below (and to measure R for the default lambda_global) --
    # run_recon() re-derives it itself from fn_ksp when it actually runs, so
    # this doesn't need to match that call's Nt exactly, just the spatial
    # grid, which smaps already fixes.
    with h5py.File(fn_ksp, 'r') as f:
        omegas_yzt = f['omegas'][()]  # (Ny, Nz, Nt)
    Nx = smaps.shape[0]
    omega = torch.from_numpy(
        np.broadcast_to(omegas_yzt[None], (Nx,) + omegas_yzt.shape).astype(bool)
    ).to(device_t)

    R = (omegas_yzt.shape[0] * omegas_yzt.shape[1]) / omegas_yzt[..., 0].sum()
    if lambda_global is None:
        lambda_global = float(R)
    print(f'  Acceleration factor R ~ {R:.2f}  ->  lambda_global = {lambda_global:.4f}')

    corrected_label = 'B0+R2*-corrected' if r2star else ('B0-corrected' if b0_correct else 'plain (uncorrected)')
    print(f'Estimating sigma1(A) for the {corrected_label} operator via power iteration...')
    r2star_map, t_ref_s = None, 0.0
    if b0_correct:
        with h5py.File(fn_b0map, 'r') as f:
            b0map_hz = torch.from_numpy(f['b0map_hz'][()].astype(np.float32)).to(device_t)
        echo_times_yz = _load_echo_times(fn_ksp, device_t)
        if r2star:
            t_ref_s = _nominal_te_s(paths.scan_info, sp.ETL)
            print(f'  Estimating R2* map from dual-echo deGRE data (TE_nominal={t_ref_s * 1000:.3f} ms)...')
            r2star_map = torch.from_numpy(
                estimate_r2star_map_epi_grid(
                    datdir, seqname, sp.fov_degre, sp.fov, tuple(smaps.shape[:3]), zero_pad_z=zero_pad_z,
                )
            ).to(device_t)
        A = build_encoding_operator_b0(
            smaps_chw, omega, b0map_hz, echo_times_yz, L=L_b0, nbins=nbins_b0,
            r2star_map=r2star_map, t_ref_s=t_ref_s,
        )
        del b0map_hz, echo_times_yz
    else:
        A = build_encoding_operator(smaps_chw, omega)
    Nt = omega.shape[-1]
    Nx_, Ny, Nz = smaps.shape[:3]
    x0 = torch.randn(Nx_, Ny, Nz, Nt, dtype=torch.complex64, device=device_t)
    sigma1A = estimate_spectral_norm(A, x0)
    print(f'  sigma1A = {sigma1A:.6f}')
    del A, smaps, smaps_chw, omega, x0
    if device_t.type == 'cuda':
        torch.cuda.empty_cache()

    if r2star:
        out_subdir = 'mslr_local_b0complex'
    elif b0_correct:
        out_subdir = 'mslr_local_b0'
    else:
        out_subdir = 'mslr_local'
    out_dir = os.path.join(recon_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\nRunning {corrected_label} local-low-rank reconstruction for {seqname} '
          f'(patch_size={patch_size}, stride={stride}, lambda_global={lambda_global:.4f})...')
    result = run_recon(
        fn_ksp=fn_ksp,
        fn_smaps=fn_smaps,
        patch_sizes=[patch_size],
        strides=[stride],
        sigma1A=sigma1A,
        device=device,
        conv_tol=conv_tol,
        lambda_global=lambda_global,
        fn_b0map=fn_b0map,
        L_b0=L_b0,
        nbins_b0=nbins_b0,
        normalize_noise=True,
        r2star_map=r2star_map,
        t_ref_s=t_ref_s,
    )

    fn_out = os.path.join(out_dir, f'{seqname}_recon')
    save_result(
        fn_out, result, fov=sp.fov, seqname=seqname,
        patch_size=list(patch_size), stride=list(stride),
        b0_corrected=b0_correct, L_b0=L_b0 if b0_correct else None,
        nbins_b0=nbins_b0 if b0_correct else None,
        r2star_corrected=r2star,
    )
    print(f'Wrote {fn_out}.h5 + .nii.gz + .json')

def _cli_mslr_local() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datdir')
    parser.add_argument('seqname')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-b0', action='store_true', help='use the plain uncorrected operator')
    parser.add_argument('--L', type=int, default=32, dest='L_b0')
    parser.add_argument('--nbins', type=int, default=128, dest='nbins_b0')
    parser.add_argument(
        '--lambda-global', type=float, default=None,
        help='override the default lambda_global=R (this dataset\'s own acceleration factor)',
    )
    parser.add_argument('--r2star', action='store_true')
    parser.add_argument(
        '--zero-pad-z', action='store_true',
        help='(with --r2star) zero-pad the deGRE-grid R2* estimate where the EPI z-FOV exceeds '
             "deGRE's own -- see estimate_r2star_map_epi_grid's docstring",
    )
    parser.add_argument('--patch-size', type=int, nargs=3, default=(6, 6, 6), metavar=('PX', 'PY', 'PZ'))
    parser.add_argument('--stride', type=int, nargs=3, default=(3, 3, 3), metavar=('SX', 'SY', 'SZ'))
    parser.add_argument(
        '--conv-tol', type=float, default=1e-5,
        help='POGM early-stop threshold; <= 0 disables early stopping (always runs to niters)',
    )
    args = parser.parse_args()
    if args.r2star and args.no_b0:
        parser.error('--r2star requires b0 correction (omit --no-b0)')
    main_mslr_local(
        args.datdir, args.seqname, device=args.device, b0_correct=not args.no_b0,
        patch_size=tuple(args.patch_size), stride=tuple(args.stride),
        L_b0=args.L_b0, nbins_b0=args.nbins_b0, lambda_global=args.lambda_global,
        conv_tol=args.conv_tol, r2star=args.r2star, zero_pad_z=args.zero_pad_z,
    )

def cg_sense_solve(
    A, ksp: torch.Tensor, shape: tuple[int, int, int, int], num_iter: int = 20,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, list[float]]:
    """CG on E^H E x = E^H y (E = A here), ported from
    preprocessing/cg_sense.py's exact algorithm onto A.apply/A.adjoint.

    A: an operator with .apply(image)->k-space, .adjoint(k-space)->image,
    the mirtorch LinearMap contract build_encoding_operator{,_b0} share.
    ksp: (K,Nc,Nt) gathered k-space (see recon/operators.py's gather_ksp).
    shape: (Nx,Ny,Nz,Nt), the image-domain state shape A.adjoint returns.

    Returns (X, residuals): X is (Nx,Ny,Nz,Nt) complex64, residuals is
    len(iterations)+1 relative residual norms (residuals[0] = 1.0 by
    construction, matching preprocessing/cg_sense.py's rel_res definition
    sqrt(rsnew/rs0)) -- for a convergence-trace sidecar, same spirit as
    ReconResult's rel_changes.
    """
    B = A.adjoint(ksp)  # (Nx,Ny,Nz,Nt)
    X = torch.zeros_like(B)
    R = B.clone()
    P = R.clone()
    rsold = torch.sum(torch.abs(R) ** 2).item()
    rs0 = rsold
    residuals = [1.0]

    if rs0 == 0:
        return X, residuals

    for _ in range(num_iter):
        AP = A.apply(P)
        AHAP = A.adjoint(AP)
        alpha = rsold / torch.sum(torch.real(torch.conj(P) * AHAP)).item()
        X = X + alpha * P
        R = R - alpha * AHAP
        rsnew = torch.sum(torch.abs(R) ** 2).item()
        rel_res = (rsnew / rs0) ** 0.5
        residuals.append(rel_res)
        if rel_res < tol:
            break
        beta = rsnew / rsold
        P = R + beta * P
        rsold = rsnew

    return X, residuals


def run_cgsense_b0(
    datdir: str,
    seqname: str,
    num_iter: int = 150,
    tol: float = 1e-6,
    L_b0: int = 32,
    nbins_b0: int = 128,
    device: str = 'cuda',
    r2star: bool = False,
    fn_b0map: str | None = None,
) -> None:
    """fn_b0map: defaults to <datdir>/recon/<seqname>_b0map.h5
    (preprocessing/run_b0map.py's output) -- pass explicitly only to
    override. Set to '' (empty string, not None) to force the *plain*
    (uncorrected) SENSE operator instead -- useful for an uncorrected
    baseline comparison against the B0-informed result."""
    device_t = torch.device(device)
    recon_dir = os.path.join(datdir, 'recon')
    if fn_b0map is None:
        fn_b0map = os.path.join(recon_dir, f'{seqname}_b0map.h5')
    fn_ksp = os.path.join(recon_dir, f'{seqname}_epi_zf.h5')
    fn_smaps = os.path.join(recon_dir, f'smaps_{seqname}_sigpy.h5')

    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    sp = load_seq_params(paths)

    print('Loading sensitivity maps...')
    smaps, smaps_chw = _load_normalized_smaps(fn_smaps, device_t)
    Nx, Ny, Nz, Nvc = smaps.shape
    print(f'  Sensitivity maps: {tuple(smaps.shape)}')

    print('Loading k-space...')
    ksp0 = torch.from_numpy(_load_array(fn_ksp, 'ksp_epi_zf').astype(np.complex64)).to(device_t)
    Nx_, Ny_, Nz_, Nvc_, Nt = ksp0.shape
    assert (Nx_, Ny_, Nz_, Nvc_) == (Nx, Ny, Nz, Nvc), (
        f'smaps shape {(Nx, Ny, Nz, Nvc)} does not match k-space dims {(Nx_, Ny_, Nz_, Nvc_)}'
    )

    omega = _load_omega(fn_ksp, Nx, Ny, Nz, Nt, ksp0)
    R = (Nx * Ny * Nz) / omega[:, :, :, 0].sum().item()
    print(f'Acceleration factor R ~ {R:.2f}')

    use_b0 = bool(fn_b0map)
    if use_b0:
        print(f'Loading B0 field map from {fn_b0map} (L={L_b0}, nbins={nbins_b0})...')
        b0map_hz = torch.from_numpy(_load_array(fn_b0map, 'b0map_hz').astype(np.float32)).to(device_t)
        assert tuple(b0map_hz.shape) == (Nx, Ny, Nz), (
            f'b0map_hz shape {tuple(b0map_hz.shape)} does not match k-space dims ({Nx},{Ny},{Nz})'
        )
        echo_times_yz = _load_echo_times(fn_ksp, device_t)
        r2star_map, t_ref_s = None, 0.0
        if r2star:
            t_ref_s = _nominal_te_s(paths.scan_info, sp.ETL)
            print(f'  Estimating R2* map from dual-echo deGRE data (TE_nominal={t_ref_s * 1000:.3f} ms)...')
            r2star_map = torch.from_numpy(
                estimate_r2star_map_epi_grid(datdir, seqname, sp.fov_degre, sp.fov, (Nx, Ny, Nz))
            ).to(device_t)
        print('Building B0-corrected encoding operator...')
        A = build_encoding_operator_b0(
            smaps_chw, omega, b0map_hz, echo_times_yz, L=L_b0, nbins=nbins_b0,
            r2star_map=r2star_map, t_ref_s=t_ref_s,
        )
        label = 'B0+R2*-corrected' if r2star else 'B0-corrected'
    else:
        assert not r2star, 'run_cgsense_b0: r2star requires a B0 map (fn_b0map)'
        print('Building plain (uncorrected) encoding operator...')
        A = build_encoding_operator(smaps_chw, omega)
        label = 'uncorrected'

    ksp = gather_ksp(ksp0, A)  # (K,Nc,Nt)
    del ksp0
    if device_t.type == 'cuda':
        torch.cuda.empty_cache()

    print(f'\nRunning {label} CG-SENSE ({num_iter} iterations max, tol={tol})...')
    t_start = time.time()
    X, residuals = cg_sense_solve(A, ksp, (Nx, Ny, Nz, Nt), num_iter=num_iter, tol=tol)
    runtime_s = time.time() - t_start
    n_iters = len(residuals) - 1
    print(f'Done in {runtime_s:.1f} s ({n_iters} iterations, final relative residual '
          f'{residuals[-1]:.3e}).')

    out_dir = os.path.join(recon_dir, 'cgsense_b0' if use_b0 else 'cgsense')
    os.makedirs(out_dir, exist_ok=True)
    fn_base = os.path.join(out_dir, f'{seqname}_recon')

    X_np = X.detach().cpu().numpy()
    with h5py.File(f'{fn_base}.h5', 'w') as f:
        f.create_dataset('X_recon', data=X_np)
        f.create_dataset('omega', data=omega.detach().cpu().numpy())
        f.create_dataset('residuals', data=np.asarray(residuals))
        f.attrs['R'] = R
        f.attrs['runtime_s'] = runtime_s
        f.attrs['b0_corrected'] = use_b0
        f.attrs['r2star_corrected'] = r2star
    print(f'Wrote {fn_base}.h5')

    save_recon_nifti(
        fn_base, X_np, fov=sp.fov, seqname=seqname, R=R, runtime_s=runtime_s,
        n_iters=n_iters, final_residual=residuals[-1], num_iter=num_iter, tol=tol,
        b0_corrected=use_b0, r2star_corrected=r2star, L_b0=L_b0 if use_b0 else None,
        nbins_b0=nbins_b0 if use_b0 else None,
    )
    print(f'Wrote {fn_base}.nii.gz + .json')

def _cli_cg() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datdir')
    parser.add_argument('seqname')
    parser.add_argument('--num-iter', type=int, default=150)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--L-b0', type=int, default=32)
    parser.add_argument('--nbins-b0', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--r2star', action='store_true')
    parser.add_argument('--no-b0', action='store_true', help='use the plain uncorrected operator')
    args = parser.parse_args()
    run_cgsense_b0(
        args.datdir, args.seqname, num_iter=args.num_iter, tol=args.tol,
        L_b0=args.L_b0, nbins_b0=args.nbins_b0, device=args.device, r2star=args.r2star,
        fn_b0map='' if args.no_b0 else None,
    )


_COMMANDS = {
    'mslr-ref': _cli_mslr_ref,
    'mslr-local': _cli_mslr_local,
    'cg': _cli_cg,
}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        sys.exit('usage: python -m recon.run_recon {' + ','.join(_COMMANDS) + '} ...')
    _cmd = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {_cmd}'
    _COMMANDS[_cmd]()
