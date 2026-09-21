"""B0-corrected combined L1-wavelet + TV reconstruction: the torch B0 encoding
operator bridged into sigpy's solver (.venv-recon, which has both). Contains
TorchLinopBridge, wavelet_tv_recon_b0 and the batch driver; run as
`.venv-recon/bin/python -m recon.sigpy_b0 <datdir> <seqname> ...`.

The sections below are the original module docstrings, kept verbatim.

Formerly recon/sigpy_b0/sigpy_torch_bridge.py
---------------------------------------------
Bridges one mirtorch LinearMap (recon/operators.py, recon/operators.py)
into a pair of sigpy.linop.Linop objects, so sigpy's already-tested
regularization/solver machinery (recon/sigpy_recon.py's Wavelet +
FiniteDifference + prox.Stack + PrimalDualHybridGradient pattern) can be
reused unchanged with this repo's own B0-corrected encoding operator in
place of sigpy.mri.linop.Sense -- rather than reimplementing PDHG, wavelet
transforms, or TV proximal operators from scratch in torch.

Only sigpy (pure Python + numpy) needs to exist in .venv-recon for this --
not cupy -- so each Linop.apply()/.adjoint() call round-trips its argument
through host memory: numpy -> torch (GPU) -> mirtorch forward/adjoint ->
numpy. This costs one small host<->device copy per call (a few MB for this
repo's per-frame image/k-space sizes), negligible next to the actual
FFT + smaps + (for the B0-corrected operator) L-segment loop compute it
wraps -- not the bottleneck, so not worth adding a cupy dependency to
avoid it.

Before reusing sigpy regularization/solver defaults (e.g.
sigpy_recon.py's lamb_l1/lamb_tv) with a torch_op bridged through here,
call recon/operators.py's check_operator_unitary(torch_op, x0) once --
those defaults were tuned against sigpy.mri.linop.Sense, which is
genuinely unitary; a non-unitary torch_op (any B0-corrected
GatheredSenseB0, see that function's docstring) needs its own retuning,
not a transplanted lambda. This bridge class itself does not call it
automatically (a per-frame batch driver would otherwise pay a full power
iteration once per frame) -- callers building a new operator/driver
combination should check once, e.g. against frame 0's operator, before
looping.


Formerly recon/sigpy_b0/recon_sigpy_b0.py
-----------------------------------------
Combined L1-wavelet + total-variation regularized SENSE reconstruction,
B0-corrected -- sigpy_recon.py's exact regularization/solver pattern
(Wavelet + FiniteDifference stacked, prox.Stack of two L1Regs, solved via
PrimalDualHybridGradient), with sigpy_recon.py's plain sigpy.mri.linop.Sense
replaced by recon/operators.py's time-segmented GatheredSenseB0, bridged
into sigpy via recon/sigpy_b0.py.

Motivation: run_recon.py's unregularized CG-SENSE shows real semi-
convergence at this repo's undersampling factors (R=6+) -- the well-posed
part of the image converges within ~20-30 iterations, but unconstrained
high-spatial-frequency content (a sharp object edge, worst-conditioned at
higher R) grows essentially without bound the longer CG runs, with no
natural stopping point (verified 2026-09-18: tracked individual edge
voxels through 150 CG iterations on real 2_6x_2.4mm data -- they grow
~50x from iteration 10 to 150 while the object interior is flat by
iteration ~30). A proximal-regularized solve constrains exactly that
unconstrained high-frequency content via the TV/wavelet sparsity prior,
rather than trading under-recovery for unbounded edge amplification.

Bridge validated (2026-09-18) on real 1_1x_5.4mm smaps/B0-map data before
any real-data use here: TorchLinopBridge's forward/adjoint satisfy
<Ax,y> == <x,A^Hy> to 1e-6 relative precision, and a synthetic ground-truth
recovery (complex phantom -> A -> PDHG+wavelet+TV, lambda shrinking toward
0) converges toward the true image as max_iter increases (0.141 -> 0.109 ->
0.094 -> 0.089 relative L2 error at 50/150/400/800 iterations) -- slower
than sigpy_recon.py's own plain-operator verification (which reaches ~0
by ~100 iterations), consistent with GatheredSenseB0's L=32-segment loop
making the operator more expensive/less well-conditioned per iteration
than the plain FFT+smaps Sense operator, not a bridge defect (adjoint
self-consistency alone already rules out a wiring bug).

y here is the GATHERED (K,Nc) k-space at this frame's sampled locations
(recon/operators.py's convention throughout this repo), not a dense
zero-filled [Nx,Ny,Nz,Nc] array -- unlike sigpy_recon.py's wavelet_tv_recon,
which takes the dense array and infers its own sampling mask from where it's
exactly zero. A_sigpy's ishape (image domain) is what G/proxg are built
against; its oshape must match y's shape exactly.


Formerly recon/sigpy_b0/run_recon_sigpy_b0.py
---------------------------------------------
Stage 2 batch driver: B0-corrected combined L1-wavelet + TV regularized
reconstruction (recon/sigpy_b0.py), for this repo's own
<seqname>_epi_zf.h5 / smaps_<seqname>_sigpy.h5 naming convention -- the
regularized-recovery counterpart to recon/run_recon.py (unregularized) and
recon/run_recon.py (nuclear-norm regularized), all three sharing the
same B0-corrected GatheredSenseB0 encoding operator but differing in how
(or whether) they constrain the ill-posed part of the R>1 problem.

Unlike sigpy_recon.py's other Stage-2 drivers (sigpy_recon.py, sigpy_recon.py,
sigpy_recon.py -- all .venv-preprocessing, sigpy-only), this needs
torch/mirtorch for GatheredSenseB0 -- runs in .venv-recon (sigpy installed
there specifically for this bridge; see recon/sigpy_b0.py's
module docstring for why cupy was not also added). Reimplements the
per-frame batch loop directly (rather than importing recon_frames.recon_frames)
since that loop's whole job -- calling recon_fn(data, smaps) per frame -- has
to change shape anyway: this driver's per-frame k-space is the *gathered*
(K,Nc) representation GatheredSenseB0 expects, not sigpy_recon.py's dense
zero-filled [Nx,Ny,Nz,Nc] per frame.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.sigpy_b0 <datdir> <seqname> \\
        [--lamb-l1 0.005] [--lamb-tv 0.005] [--num-iter 100] [--frames 0,1,2]
"""

import argparse
import os
import time

import h5py
import numpy as np
import sigpy as sp
import torch
from mirtorch.linear.linearmaps import LinearMap

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.nifti_io import save_recon_nifti
from recon.operators import build_encoding_operator_b0, check_operator_unitary, gather_ksp
from recon.reconstruct import _load_array, _load_echo_times, _load_normalized_smaps, _load_omega


def _to_torch(x: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class TorchLinopBridge(sp.linop.Linop):
    """Forward half of the bridge: sigpy Linop wrapping torch_op.apply().

    torch_op: a mirtorch LinearMap (e.g. GatheredSenseB0) with size_in/
    size_out already fixed (one frame's operator, not the BlockDiagonal
    over every frame). device: where torch_op's own tensors (smaps,
    c_phasors, ...) already live -- inputs are moved there for the call
    and outputs are always returned to host (numpy), matching every other
    sigpy Linop's convention of operating on whatever array it's given.
    """

    def __init__(self, torch_op: LinearMap, device: torch.device):
        self.torch_op = torch_op
        self.device = device
        super().__init__(list(torch_op.size_out), list(torch_op.size_in))

    def _apply(self, input: np.ndarray) -> np.ndarray:
        x = _to_torch(input, torch.complex64, self.device)
        y = self.torch_op.apply(x)
        return _to_numpy(y)

    def _adjoint_linop(self):
        return TorchLinopBridgeAdjoint(self.torch_op, self.device, fwd=self)


class TorchLinopBridgeAdjoint(sp.linop.Linop):
    """Adjoint half: sigpy Linop wrapping torch_op.adjoint(). Never
    constructed directly -- returned by TorchLinopBridge._adjoint_linop()
    (and vice versa via its own _adjoint_linop() below), matching sigpy's
    own FFT/IFFT pairing convention (recon/sigpy_b0.py's module
    docstring)."""

    def __init__(self, torch_op: LinearMap, device: torch.device, fwd: TorchLinopBridge):
        self.torch_op = torch_op
        self.device = device
        self._fwd = fwd
        super().__init__(list(torch_op.size_in), list(torch_op.size_out))

    def _apply(self, input: np.ndarray) -> np.ndarray:
        y = _to_torch(input, torch.complex64, self.device)
        x = self.torch_op.adjoint(y)
        return _to_numpy(x)

    def _adjoint_linop(self):
        return self._fwd

def wavelet_tv_recon_b0(
    A_sigpy: sp.linop.Linop,
    y: np.ndarray,
    lamb_l1: float,
    lamb_tv: float,
    num_iter: int,
    wave_name: str = 'db4',
    max_power_iter: int = 30,
) -> np.ndarray:
    """[Nx, Ny, Nz] complex image for one frame.

    A_sigpy: bridged GatheredSenseB0 for this one frame (recon/
    sigpy_b0.py's TorchLinopBridge), ishape=(Nx,Ny,Nz),
    oshape=(K,Nc). y: (K,Nc) complex, this frame's gathered k-space,
    same sample ordering as A_sigpy's underlying torch_op.idx (see
    recon/operators.py's gather_ksp).
    """
    img_shape = tuple(A_sigpy.ishape)

    # Same O(1)-rescaling sigpy_recon.py uses and documents the necessity
    # of (see its module docstring) -- lamb_l1/lamb_tv are tuned for O(1)
    # data; real scanner-unit k-space (|y| ~ 1e4-1e5) would otherwise make
    # them negligible, silently degrading to an unregularized (and, at
    # this repo's real R, ill-posed) least-squares solve -- exactly the
    # semi-convergence pathology this function exists to avoid.
    scale = 1.0 / np.percentile(np.abs(y[y != 0]), 99)
    y_scaled = y * scale

    W = sp.linop.Wavelet(img_shape, wave_name=wave_name)
    Grad = sp.linop.FiniteDifference(img_shape)
    G = sp.linop.Vstack([W, Grad])
    proxg = sp.prox.Stack([sp.prox.L1Reg(W.oshape, lamb_l1), sp.prox.L1Reg(Grad.oshape, lamb_tv)])

    app = sp.app.LinearLeastSquares(
        A_sigpy, y_scaled, proxg=proxg, G=G,
        solver='PrimalDualHybridGradient',
        max_iter=num_iter, max_power_iter=max_power_iter,
        show_pbar=False,
    )
    return app.run() / scale

def main_run(
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
    -- sigpy_recon.py's own defaults -- carry their original meaning; pass
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
    # One check for the whole batch (not per frame -- see sigpy_b0.py's
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
    # sigpy_recon.py's lamb_l1/lamb_tv defaults were tuned/validated under
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

def _cli_run() -> None:
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
    main_run(
        args.datdir, args.seqname, lamb_l1=args.lamb_l1, lamb_tv=args.lamb_tv,
        num_iter=args.num_iter, wave_name=args.wave_name, max_power_iter=args.max_power_iter,
        L_b0=args.L_b0, nbins_b0=args.nbins_b0, device=args.device, frames=frames,
        normalize_operator=not args.no_normalize,
    )


if __name__ == '__main__':
    _cli_run()
