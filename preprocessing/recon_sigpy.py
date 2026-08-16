"""Combined L1-wavelet + total-variation regularized SENSE reconstruction,
replacing BART's `pics -R W:7:0:lamb_l1 -R T:7:0:lamb_tv -i N -S` (run_bart.m)
with sigpy (see CLAUDE.md for why BART was dropped in favor of sigpy).

Both regularizers are combined via sigpy's standard multi-regularizer
pattern for sigpy.app.LinearLeastSquares: a stacked operator
`G = Vstack([Wavelet, FiniteDifference])` and a block-separable proximal
operator `prox.Stack([L1Reg(...,lamb_l1), L1Reg(...,lamb_tv)])` -- the same
structure sigpy.mri.app.L1WaveletRecon / TotalVariationRecon each use
individually (see their source in sigpy), just combined here rather than
applied one at a time. Solved via PrimalDualHybridGradient, the standard
solver for f(x) + g(Gx) with f smooth and g nonsmooth-but-prox-friendly on
a transformed domain.

Verified against a synthetic SENSE forward model (not a real reconstruction
regression, since no local ScanArchive dataset exists to run this against
end to end): with fully-sampled synthetic k-space and lamb_l1=lamb_tv
shrinking to 0, the reconstruction converges to the true image (relative
error 6.5e-3 -> 7e-4 -> ~0 as lamda goes 1e-2 -> 1e-3 -> 1e-5), confirming
the Vstack/Stack/PDHG composition is solving the intended problem rather
than something subtly mis-wired. BART's `-S` rescaling flag has no direct
replacement; sigpy's own power-iteration step-size calibration
(max_power_iter) serves an analogous auto-scaling purpose, but the exact
numerics will differ from BART's -- an already-accepted tradeoff of
dropping BART.
"""

import numpy as np
import sigpy as sp
import sigpy.mri as mr


def wavelet_tv_recon(
    ksp: np.ndarray,
    smaps: np.ndarray,
    lamb_l1: float,
    lamb_tv: float,
    num_iter: int,
    wave_name: str = 'db4',
    max_power_iter: int = 30,
) -> np.ndarray:
    """[Nx, Ny, Nz] complex image from one frame's zero-filled Cartesian
    k-space and coil sensitivity maps.

    ksp, smaps: [Nx, Ny, Nz, Ncoils] -- coils LAST (this port's usual
    convention); transposed internally since sigpy expects coils first.
    The sampling mask is inferred from where `ksp` is exactly zero (matches
    sigpy.mri.app._estimate_weights' own Cartesian convention, and this
    port's zero-filled-volume convention throughout preprocess.py).
    """
    ksp_cf = np.moveaxis(ksp, -1, 0)
    mps_cf = np.moveaxis(smaps, -1, 0)
    img_shape = mps_cf.shape[1:]

    weights = (sp.rss(ksp_cf, axes=(0,)) > 0).astype(ksp_cf.dtype)
    y = ksp_cf * weights**0.5
    A = mr.linop.Sense(mps_cf, weights=weights)

    W = sp.linop.Wavelet(img_shape, wave_name=wave_name)
    Grad = sp.linop.FiniteDifference(img_shape)
    G = sp.linop.Vstack([W, Grad])
    proxg = sp.prox.Stack([sp.prox.L1Reg(W.oshape, lamb_l1), sp.prox.L1Reg(Grad.oshape, lamb_tv)])

    app = sp.app.LinearLeastSquares(
        A, y, proxg=proxg, G=G,
        solver='PrimalDualHybridGradient',
        max_iter=num_iter, max_power_iter=max_power_iter,
        show_pbar=False,
    )
    return app.run()
