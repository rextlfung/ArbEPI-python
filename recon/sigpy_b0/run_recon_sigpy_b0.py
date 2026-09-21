"""Stage 2 batch driver: B0-corrected combined L1-wavelet + TV regularized
reconstruction (recon/sigpy_b0/recon_sigpy_b0.py), for this repo's own
<seqname>_epi_zf.h5 / smaps_<seqname>_sigpy.h5 naming convention -- the
regularized-recovery counterpart to recon/cg_sense_b0.py (unregularized) and
recon/run_mslr_local.py (nuclear-norm regularized), all three sharing the
same B0-corrected GatheredSenseB0 encoding operator but differing in how
(or whether) they constrain the ill-posed part of the R>1 problem.

Unlike recon_frames.py's other Stage-2 drivers (run_rss.py, run_cg_sense.py,
run_recon_sigpy.py -- all .venv-preprocessing, sigpy-only), this needs
torch/mirtorch for GatheredSenseB0 -- runs in .venv-recon (sigpy installed
there specifically for this bridge; see recon/sigpy_b0/sigpy_torch_bridge.py's
module docstring for why cupy was not also added). Reimplements the
per-frame batch loop directly (rather than importing recon_frames.recon_frames)
since that loop's whole job -- calling recon_fn(data, smaps) per frame -- has
to change shape anyway: this driver's per-frame k-space is the *gathered*
(K,Nc) representation GatheredSenseB0 expects, not recon_frames.py's dense
zero-filled [Nx,Ny,Nz,Nc] per frame.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.sigpy_b0.run_recon_sigpy_b0 <datdir> <seqname> \\
        [--lamb-l1 0.005] [--lamb-tv 0.005] [--num-iter 100] [--frames 0,1,2]
"""

import argparse
import os
import time

import h5py
import numpy as np
import torch

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.nifti_io import save_recon_nifti
from recon.operators import gather_ksp
from recon.operators_b0 import build_encoding_operator_b0, check_operator_unitary
from recon.reconstruct import _load_array, _load_echo_times, _load_normalized_smaps, _load_omega
from recon.sigpy_b0.recon_sigpy_b0 import wavelet_tv_recon_b0
from recon.sigpy_b0.sigpy_torch_bridge import TorchLinopBridge


def main(
    datdir: str, seqname: str,
    lamb_l1: float = 0.005, lamb_tv: float = 0.005, num_iter: int = 100,
    wave_name: str = 'db4', max_power_iter: int = 30,
    L_b0: int = 32, nbins_b0: int = 128, device: str = 'cuda',
    frames: list[int] | None = None, normalize_operator: bool = True,
) -> None:
    """frames: reconstruct only these frame indices (for quick validation on
    real data before committing to a full multi-hour batch); None
    reconstructs every frame.

    normalize_operator: rescale each frame's operator by 1/sigma1(A) before
    solving (see the code below for the derivation) so lamb_l1/lamb_tv=0.005
    -- recon_sigpy.py's own defaults -- carry their original meaning; pass
    False to use the raw (non-unitary) operator instead, e.g. to reproduce
    an earlier un-normalized run for comparison."""
    device_t = torch.device(device)
    recon_dir = os.path.join(datdir, 'recon')
    fn_ksp = os.path.join(recon_dir, f'{seqname}_epi_zf.h5')
    fn_smaps = os.path.join(recon_dir, f'smaps_{seqname}_sigpy.h5')
    fn_b0map = os.path.join(recon_dir, f'{seqname}_b0map.h5')

    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    sp_params = load_seq_params(paths)

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

    print(f'Loading B0 field map from {fn_b0map} (L={L_b0}, nbins={nbins_b0})...')
    b0map_hz = torch.from_numpy(_load_array(fn_b0map, 'b0map_hz').astype(np.float32)).to(device_t)
    echo_times_yz = _load_echo_times(fn_ksp, device_t)

    print('Building B0-corrected encoding operator (shared across frames)...')
    A_full = build_encoding_operator_b0(
        smaps_chw, omega, b0map_hz, echo_times_yz, L=L_b0, nbins=nbins_b0,
    )
    # One check for the whole batch (not per frame -- see sigpy_torch_bridge.py's
    # module docstring): lamb_l1/lamb_tv above are tuned assuming a unitary
    # operator, and GatheredSenseB0 is not guaranteed unitary by construction.
    x0 = torch.randn(Nx, Ny, Nz, dtype=torch.complex64, device=device_t)
    sigma1 = check_operator_unitary(A_full.A[0], x0, name=f'{seqname} B0-corrected operator')
    print(f'  sigma1(A) = {sigma1:.4f} (1.0 = unitary; see warning above if not)')
    del x0

    # Rescale every frame's operator by 1/sigma1 so it's ~unitary (mirtorch's
    # `Multiply`, i.e. `a * A`, scales forward and adjoint by the same real
    # scalar -- (a*A)(x) = a*A(x), (a*A)^H(y) = a*A^H(y) for real a -- so this
    # is exactly A/sigma1, not a hack). Restores the assumption
    # recon_sigpy.py's lamb_l1/lamb_tv defaults were tuned/validated under
    # (a genuinely unitary sigpy.mri.linop.Sense) -- see
    # check_operator_unitary's docstring for why a non-unitary operator
    # otherwise needs a much larger lambda for the same effective
    # regularization strength. Since A_new = A_orig/sigma1 while `y` is
    # unchanged, wavelet_tv_recon_b0's own output solves for sigma1*x_true
    # (it already un-does its internal O(1) y-rescaling) -- divide by sigma1
    # once more here to recover the correctly-scaled image.
    normalize = sigma1 if normalize_operator else 1.0

    ksp = gather_ksp(ksp0, A_full)  # (K,Nc,Nt)
    del ksp0
    if device_t.type == 'cuda':
        torch.cuda.empty_cache()

    frame_idxs = frames if frames is not None else list(range(Nt))
    img = np.zeros((Nx, Ny, Nz, len(frame_idxs)), dtype=np.complex64)

    print(f'\nReconstructing {len(frame_idxs)}/{Nt} frame(s) '
          f'(lamb_l1={lamb_l1}, lamb_tv={lamb_tv}, num_iter={num_iter}, '
          f'operator normalized by 1/{normalize:.4f})...')
    t_start = time.time()
    for out_i, it in enumerate(frame_idxs):
        t_frame = time.time()
        A_torch = (1.0 / normalize) * A_full.A[it] if normalize_operator else A_full.A[it]
        A_sigpy = TorchLinopBridge(A_torch, device_t)
        y = ksp[:, :, it].detach().cpu().numpy()
        img[..., out_i] = wavelet_tv_recon_b0(
            A_sigpy, y, lamb_l1, lamb_tv, num_iter,
            wave_name=wave_name, max_power_iter=max_power_iter,
        ) / normalize
        print(f'  frame {it} ({out_i + 1}/{len(frame_idxs)}) done in {time.time() - t_frame:.1f}s')
    runtime_s = time.time() - t_start
    print(f'Wall-clock: {runtime_s:.1f}s ({runtime_s / len(frame_idxs):.1f}s/frame)')

    out_dir = os.path.join(recon_dir, 'cs_b0')
    os.makedirs(out_dir, exist_ok=True)
    frames_tag = 'all' if frames is None else '-'.join(map(str, frame_idxs))
    fn_out = os.path.join(out_dir, f'{seqname}_recon_frames{frames_tag}')

    with h5py.File(f'{fn_out}.h5', 'w') as f:
        f.create_dataset('X_recon', data=img)
        f.attrs['R'] = R
        f.attrs['runtime_s'] = runtime_s
        f.attrs['frame_idxs'] = np.asarray(frame_idxs)
    print(f'Wrote {fn_out}.h5')

    save_recon_nifti(
        fn_out, img, fov=sp_params.fov, seqname=seqname, R=R, runtime_s=runtime_s,
        lamb_l1=lamb_l1, lamb_tv=lamb_tv, num_iter=num_iter, wave_name=wave_name,
        L_b0=L_b0, nbins_b0=nbins_b0, frame_idxs=frame_idxs,
    )
    print(f'Wrote {fn_out}.nii.gz + .json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datdir')
    parser.add_argument('seqname')
    parser.add_argument('--lamb-l1', type=float, default=0.005)
    parser.add_argument('--lamb-tv', type=float, default=0.005)
    parser.add_argument('--num-iter', type=int, default=100)
    parser.add_argument('--wave-name', default='db4')
    parser.add_argument('--max-power-iter', type=int, default=30)
    parser.add_argument('--L', type=int, default=32, dest='L_b0')
    parser.add_argument('--nbins', type=int, default=128, dest='nbins_b0')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--frames', default=None,
        help='comma-separated frame indices to reconstruct (default: all frames)',
    )
    parser.add_argument(
        '--no-normalize', action='store_true',
        help='use the raw (non-unitary) operator instead of rescaling by 1/sigma1(A)',
    )
    args = parser.parse_args()
    frames = [int(x) for x in args.frames.split(',')] if args.frames else None
    main(
        args.datdir, args.seqname, lamb_l1=args.lamb_l1, lamb_tv=args.lamb_tv,
        num_iter=args.num_iter, wave_name=args.wave_name, max_power_iter=args.max_power_iter,
        L_b0=args.L_b0, nbins_b0=args.nbins_b0, device=args.device, frames=frames,
        normalize_operator=not args.no_normalize,
    )
