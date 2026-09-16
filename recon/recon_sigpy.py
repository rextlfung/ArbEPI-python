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

Verified against a synthetic SENSE forward model: with fully-sampled
synthetic k-space and lamb_l1=lamb_tv shrinking to 0, the reconstruction
converges to the true image (relative error 6.5e-3 -> 7e-4 -> ~0 as lamda
goes 1e-2 -> 1e-3 -> 1e-5), confirming the Vstack/Stack/PDHG composition is
solving the intended problem rather than something subtly mis-wired.

`y` is rescaled to O(1) before solving (and the result rescaled back) --
this port's replacement for BART's `-S` flag, added after real project
data (wb_2.4mm ball phantom) surfaced the consequence of not having one:
lamb_l1/lamb_tv are tuned for O(1)-scaled data, so without this step
they're negligible against raw scanner-unit k-space (|y| ~ 1e4-1e5),
silently degrading "L1+TV" to an unregularized least-squares SENSE solve.
At this acquisition's undersampling (R=6) that's ill-posed, and a lambda
sweep at the true (unscaled) magnitude confirmed it manifests specifically
as spurious signal loss in a uniform phantom's center -- center/shell
signal ratio was flat and wrong (~0.64, vs RSS's own 0.76) from lamb=0 up
through lamb=80, only correcting once lamb reached ~1000, i.e. roughly
the scale this normalization now reaches automatically at lamb=0.005.
sigpy's power-iteration step-size calibration (max_power_iter) still
serves its own separate auto-scaling purpose (for A/G's operator norms,
not the data/regularizer scale) and remains in place alongside this.
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

    # lamb_l1/lamb_tv are calibrated for O(1)-scaled data -- BART's `pics -S`
    # rescales internally before applying `-R` regularizers (see module
    # docstring); this is this port's replacement for that step. Without it,
    # lamb_l1/lamb_tv are negligible against raw scanner-unit k-space
    # (|y| ~ 1e4-1e5), making the regularizers inert and the "L1+TV" recon
    # silently degrade to an unregularized (and, at real undersampling
    # factors, ill-posed) least-squares SENSE solve -- confirmed on real
    # project data to cause spurious central signal loss in a uniform
    # phantom, fixed by restoring lamb_l1/lamb_tv to their intended
    # relative strength via this normalization.
    scale = 1.0 / np.percentile(np.abs(y[y != 0]), 99)
    y = y * scale

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
    return app.run() / scale
