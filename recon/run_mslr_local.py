"""One-off driver: MSLR reconstruction with a single local-low-rank scale,
for this repo's own <seqname>_epi_zf.h5 / smaps_<seqname>_sigpy.h5 naming
convention (unlike run_b0_recon.py / validate_against_mslr.py, both
hardcoded to the ArbEPI_epi_zf.h5 name and, for the latter, a Julia
reference .mat that doesn't exist for a dataset that was never run through
../mslr-recon).

Motivation: recon/lowres_calib/lowres_calib_recon.py's fully-sampled-calibration-region
reconstruction is a fast diagnostic that uses only a few hundred of a
dataset's many thousand k-space samples -- deliberately, for speed, per its
own module docstring -- so its thermal-noise floor is far higher than the
full accelerated reconstruction's. Comparing recon/lowres_temporal_
stability.py's output on this driver's full, high-SNR reconstruction
against the calibration-region-only numbers already measured tests whether
that low-res diagnostic's own noise floor, not a real B0/T2*/spoiling
artifact, was inflating the fluctuation/drift figures measured so far.

patch_sizes/strides default to a single local scale ([(6,6,6)], [(3,3,3)]),
not the G+L multi-scale config recon/analysis/validate_against_mslr.py's Julia
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
cg_sense_b0.py's B0-corrected one on the same data: the latter's object is
round, the former's is not. sigma1A has no closed form for either operator
(see operators_b0.py's estimate_spectral_norm docstring) -- this driver
measures it once via power iteration before the real reconstruction runs,
the same pattern run_b0_recon.py uses.

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
production-scale value validate_against_mslr.py's own reference happened
to be tuned at. Override with --lambda-global to pin a specific value
instead (e.g. for reproducing a fixed-lambda comparison).

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.run_mslr_local <datdir> <seqname> [--device cuda]
e.g.
    .venv-recon/bin/python -m recon.run_mslr_local \\
        /StorageRAID/rexfung/20260915ball 1_1x_5.4mm
"""

import argparse
import os

import h5py
import numpy as np
import torch

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from recon.operators import build_encoding_operator
from recon.operators_b0 import build_encoding_operator_b0, estimate_spectral_norm
from recon.reconstruct import _load_echo_times, _load_normalized_smaps, run_recon, save_result


def main(
    datdir: str, seqname: str, device: str = 'cuda',
    patch_size: tuple[int, int, int] = (6, 6, 6),
    stride: tuple[int, int, int] = (3, 3, 3),
    b0_correct: bool = True,
    L_b0: int = 32,
    nbins_b0: int = 128,
    lambda_global: float | None = None,
) -> None:
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

    corrected_label = 'B0-corrected' if b0_correct else 'plain (uncorrected)'
    print(f'Estimating sigma1(A) for the {corrected_label} operator via power iteration...')
    if b0_correct:
        with h5py.File(fn_b0map, 'r') as f:
            b0map_hz = torch.from_numpy(f['b0map_hz'][()].astype(np.float32)).to(device_t)
        echo_times_yz = _load_echo_times(fn_ksp, device_t)
        A = build_encoding_operator_b0(
            smaps_chw, omega, b0map_hz, echo_times_yz, L=L_b0, nbins=nbins_b0,
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

    out_dir = os.path.join(recon_dir, 'mslr_local_b0' if b0_correct else 'mslr_local')
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
        lambda_global=lambda_global,
        fn_b0map=fn_b0map,
        L_b0=L_b0,
        nbins_b0=nbins_b0,
    )

    fn_out = os.path.join(out_dir, f'{seqname}_recon')
    save_result(
        fn_out, result, fov=sp.fov, seqname=seqname,
        patch_size=list(patch_size), stride=list(stride),
        b0_corrected=b0_correct, L_b0=L_b0 if b0_correct else None,
        nbins_b0=nbins_b0 if b0_correct else None,
    )
    print(f'Wrote {fn_out}.h5 + .nii.gz + .json')


if __name__ == '__main__':
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
    args = parser.parse_args()
    main(
        args.datdir, args.seqname, device=args.device, b0_correct=not args.no_b0,
        L_b0=args.L_b0, nbins_b0=args.nbins_b0, lambda_global=args.lambda_global,
    )
