"""Combined L1-wavelet + total-variation regularized SENSE reconstruction,
B0-corrected -- recon_sigpy.py's exact regularization/solver pattern
(Wavelet + FiniteDifference stacked, prox.Stack of two L1Regs, solved via
PrimalDualHybridGradient), with recon_sigpy.py's plain sigpy.mri.linop.Sense
replaced by recon/operators_b0.py's time-segmented GatheredSenseB0, bridged
into sigpy via recon/sigpy_b0/sigpy_torch_bridge.py.

Motivation: cg_sense_b0.py's unregularized CG-SENSE shows real semi-
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
than recon_sigpy.py's own plain-operator verification (which reaches ~0
by ~100 iterations), consistent with GatheredSenseB0's L=32-segment loop
making the operator more expensive/less well-conditioned per iteration
than the plain FFT+smaps Sense operator, not a bridge defect (adjoint
self-consistency alone already rules out a wiring bug).

y here is the GATHERED (K,Nc) k-space at this frame's sampled locations
(recon/operators.py's convention throughout this repo), not a dense
zero-filled [Nx,Ny,Nz,Nc] array -- unlike recon_sigpy.py's wavelet_tv_recon,
which takes the dense array and infers its own sampling mask from where it's
exactly zero. A_sigpy's ishape (image domain) is what G/proxg are built
against; its oshape must match y's shape exactly.
"""

import numpy as np
import sigpy as sp


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
    sigpy_torch_bridge.py's TorchLinopBridge), ishape=(Nx,Ny,Nz),
    oshape=(K,Nc). y: (K,Nc) complex, this frame's gathered k-space,
    same sample ordering as A_sigpy's underlying torch_op.idx (see
    recon/operators.py's gather_ksp).
    """
    img_shape = tuple(A_sigpy.ishape)

    # Same O(1)-rescaling recon_sigpy.py uses and documents the necessity
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
