"""1D NUFFT-based ramp-sample-to-Cartesian regridding for EPI.

Ports hmriutils' rampsampepi2cart.m / rampsamp2cart.m / reconecho.m, using
sigpy's NUFFT instead of MIRT's Gmri (see CLAUDE.md for why BART/MIRT were
dropped in favor of sigpy + plain numpy). Same algorithm -- density-
compensated adjoint NUFFT, then forward FFT to grid -- but sigpy's
nufft/nufft_adjoint use their own normalization rather than MIRT's, so
absolute scale differs from the MATLAB pipeline. This is an accepted,
already-flagged tradeoff: relative image structure is what matters here (see
test_preprocessing_epi_gridding.py's round-trip check), and any global scale
factor washes out downstream (RSS-normalized smaps, regularized recon).

sigpy's nufft coord convention is "cycles/FOV" -- Nyquist at +-N/2 for an
N-point axis -- which is exactly kx (cycles/cm) * fov (cm), the same
scaling MIRT's Gmri([fov*kx(:)], ...) used.

sigpy's nufft_adjoint batches with the non-uniform-sample axis *last* and
oshape's leading dims as the batch (verified empirically -- opposite of this
module's [nr, ...] convention), hence the transposes in rampsamp2cart below.
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
