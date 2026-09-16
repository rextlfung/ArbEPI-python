"""B0-informed CG-SENSE: unregularized conjugate-gradient SENSE
reconstruction using the time-segmented B0-corrected encoding operator
(recon/operators_b0.py's GatheredSenseB0/build_encoding_operator_b0),
solved via literal conjugate gradient (Pruessmann et al.) rather than
recon/reconstruct.py's multi-scale-low-rank POGM solve.

Ports preprocessing/cg_sense.py's exact CG algorithm (CG on the normal
equations E^H E x = E^H y) onto mirtorch's LinearMap interface
(A.apply/A.adjoint) instead of that module's explicit FFT+mask+smaps
closures, so it works with any encoding operator sharing that contract --
here, the B0-corrected one -- not just the plain uncorrected SENSE operator
preprocessing/cg_sense.py was written for.

Unlike recon/reconstruct.py's run_recon (POGM, needs sigma1A -- the
operator's spectral norm -- for its step size), CG is self-scaling and
needs no such estimate: skip recon/operators_b0.py's estimate_spectral_norm
entirely here.

Reconstructs the whole (Nx,Ny,Nz,Nt) volume in one CG run rather than
looping per frame: build_encoding_operator_b0 returns a BlockDiagonal
operator (independent per-frame GatheredSenseB0 blocks, see
recon/operators.py's build_encoding_operator docstring), so A^H A has no
cross-frame coupling -- solving jointly over the stacked tensor is
mathematically identical to solving each frame's CG independently, just
one Python-level loop instead of Nt, and matches recon/reconstruct.py's own
convention of treating the whole (Nx,Ny,Nz,Nt) tensor as one state array.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.cg_sense_b0 <datdir> <seqname>
"""

import argparse
import os
import time

import h5py
import numpy as np
import torch

from preprocessing.nifti_io import save_recon_nifti
from preprocessing.r2star_map import estimate_r2star_map_epi_grid
from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.matio import read_mat
from recon.operators import build_encoding_operator, gather_ksp
from recon.operators_b0 import build_encoding_operator_b0
from recon.reconstruct import (
    _load_array,
    _load_echo_times,
    _load_normalized_smaps,
    _load_omega,
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


def _nominal_te_s(scan_info_path: str, etl: int) -> float:
    """Same as recon/run_b0_recon.py's own helper -- the prescribed-TE
    echo's acquisition time, frame/shot-invariant (see CLAUDE.md's
    mask2epi_radial paragraph)."""
    schedules = read_mat(scan_info_path, ['schedules'])['schedules']  # (Nframes,Nshots,ETL,3)
    return float(schedules[0, 0, (etl - 1) // 2, 2])


def run_cgsense_b0(
    datdir: str,
    seqname: str,
    num_iter: int = 20,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datdir')
    parser.add_argument('seqname')
    parser.add_argument('--num-iter', type=int, default=20)
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
