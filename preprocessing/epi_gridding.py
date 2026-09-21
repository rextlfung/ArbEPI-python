"""1D NUFFT-based ramp-sample-to-Cartesian regridding for EPI.

Ports hmriutils' rampsampepi2cart.m / rampsamp2cart.m / reconecho.m, using
sigpy's NUFFT instead of MIRT's Gmri (see CLAUDE.md for why BART/MIRT were
dropped in favor of sigpy + plain numpy). Same algorithm -- density-
compensated adjoint NUFFT, then forward FFT to grid.

sigpy's nufft coord convention is "cycles/FOV" -- Nyquist at +-N/2 for an
N-point axis -- which is exactly kx (cycles/cm) * fov (cm), the same
scaling MIRT's Gmri([fov*kx(:)], ...) used.

sigpy's nufft_adjoint batches with the non-uniform-sample axis *last* and
oshape's leading dims as the batch (verified empirically -- opposite of this
module's [nr, ...] convention), hence the transposes in rampsamp2cart below.

**Absolute-scale fix (2026-09-21)**: this module's docstring previously
claimed sigpy's differing normalization vs MIRT's Gmri was a harmless,
already-flagged tradeoff ("relative image structure is what matters...any
global scale factor washes out downstream") -- true for RSS combine and
unregularized SENSE (both scale-invariant), but false once real,
noise-calibrated regularization (recon/mslr.py's normalize_noise,
_reg_weights' unit-variance-noise assumption) needs the *absolute* scale
correct. Real background k-space/image noise measured ~70-800x (method-
dependent) away from the unit-variance whitening (preprocessing/coils.py)
guarantees, traced to exactly this gap.

reconecho.m (hmriutils' reference this module ports) is explicit about its
own normalization: `x = A'*(y.*dcf)/nx` -- an EXPLICIT /nx after MIRT's
Gmri adjoint (`A'`), followed by a *plain, unnormalized* `fft` back to
k-space (rampsamp2cart.m's nufft branch: `dc = fftshift(fft(fftshift(x)))`
-- no norm='ortho' or equivalent). sigpy.nufft/nufft_adjoint, unlike raw
MIRT Gmri adjoint, already carry their own internal 1/sqrt(N) prescale
(confirmed directly in sigpy/fourier.py's nufft: `output /=
prod(shape)**0.5`) -- adjoint-consistency means sigpy.nufft_adjoint applies
that same 1/sqrt(N), not MIRT's implicit unnormalized-then-explicit-/nx
convention. The gap between "already has 1/sqrt(nx) built in" and "needs
1/nx total, per reconecho.m" is exactly one more factor of 1/sqrt(nx) --
applied below, right after nufft_adjoint, so the plain/unnormalized final
fft stays byte-for-byte faithful to rampsamp2cart.m's own forward step
(deliberately NOT changed to norm='ortho' -- that would be a different,
not-MATLAB-matching convention, even though it happens to produce the same
net scale here). Confirmed via two independent routes: (1) a controlled
synthetic test with uniform (uniformly-spaced) kx -- where density
compensation degenerates to all-ones, isolating this factor -- measures
this exact sqrt(nx) gap to 4 significant figures across nx in
{32,64,128,256} (see tests/test_preprocessing_epi_gridding.py); (2) this
analytical derivation from reconecho.m's own explicit normalization line.
"""

import numpy as np
import sigpy


def _density_compensation(kx: np.ndarray) -> np.ndarray:
    """Ports reconecho.m: dcf = |diff(kx)| / max(|diff(kx)|), with a
    trailing zero appended so length matches kx (and therefore the data)."""
    dcf = np.abs(np.diff(kx))
    dcf = np.append(dcf, 0.0)
    return dcf / dcf.max()


def rampsamp2cart(dr: np.ndarray, kx: np.ndarray, nx: int, fov_cm: float) -> np.ndarray:
    """Interpolate ramp-sampled data onto a Cartesian grid along axis 0.

    dr: [nr, ...] ramp-sampled raw data along the readout (1st) axis;
        trailing axes (echo, coil, ...) are treated as independent batch
        dims and regridded together.
    Returns [nx, ...] gridded k-space data. Ports rampsamp2cart.m's
    'nufft' branch (the only branch this port implements -- 'spline' is
    not used anywhere in preprocess.m/calibrate_delay.m).
    """
    dr_shape = dr.shape
    nr = dr_shape[0]
    dcf = _density_compensation(kx)
    coord = (kx * fov_cm)[:, None]

    dr2 = dr.reshape(nr, -1).T  # [M, nr], batch-first for sigpy
    ximg = sigpy.nufft_adjoint(dr2 * dcf[None, :], coord, oshape=dr2.shape[:1] + (nx,))
    ximg = ximg.T  # [nx, M]
    # See module docstring's "Absolute-scale fix" -- matches reconecho.m's
    # explicit `/nx` after MIRT's (unnormalized) Gmri adjoint, given
    # sigpy.nufft_adjoint already includes its own 1/sqrt(nx).
    ximg = ximg / np.sqrt(nx)

    dc = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ximg, axes=0), axis=0), axes=0)
    return dc.reshape((nx,) + dr_shape[1:])


def rampsampepi2cart(
    dr: np.ndarray, kxo: np.ndarray, kxe: np.ndarray, nx: int, fov_cm: float
) -> np.ndarray:
    """Interpolate ramp-sampled EPI data (odd/even echoes on different
    trajectories) onto a Cartesian grid. Ports rampsampepi2cart.m.

    dr: [nr, etl, ...] ramp-sampled raw data, EPI echo train along axis 1.
    kxo, kxe: [nr] k-space sample locations (cycles/cm) for odd/even echoes.
    Returns [nx, etl, ...] gridded k-space data.
    """
    dr_shape = dr.shape
    nr, etl = dr_shape[0], dr_shape[1]
    dr2 = dr.reshape(nr, etl, -1)

    dco = rampsamp2cart(dr2[:, 0::2, :], kxo, nx, fov_cm)  # odd echoes (MATLAB 1-based odd)
    dce = rampsamp2cart(dr2[:, 1::2, :], kxe, nx, fov_cm)  # even echoes

    dc = np.empty((nx, etl) + dr2.shape[2:], dtype=np.result_type(dco, dce))
    dc[:, 0::2] = dco
    dc[:, 1::2] = dce
    return dc.reshape((nx,) + dr_shape[1:])
