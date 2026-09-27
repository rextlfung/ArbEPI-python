"""Ramp-sampled EPI readouts -> Cartesian kx, by 1D NUFFT.

Ports hmriutils' rampsampepi2cart.m / rampsamp2cart.m / reconecho.m with
sigpy's NUFFT in place of MIRT's Gmri: a density-compensated adjoint NUFFT
(dcf = |diff(kx)|) to image space, then an FFT back to a Cartesian grid. Odd
and even echoes use their own trajectories (kxo, kxe).

- sigpy's coordinates are in cycles/FOV (Nyquist at +-N/2), i.e. kx (cycles/cm)
  x fov (cm), as with Gmri([fov*kx(:)], ...).
- sigpy batches over leading axes with the sample axis last, hence the
  transposes in rampsamp2cart.
- Absolute scale: reconecho.m divides Gmri's (unnormalized) adjoint by nx, then
  applies a plain fft. sigpy's nufft_adjoint already includes 1/sqrt(nx), so one
  more 1/sqrt(nx) is applied here. This matters because recon's regularization
  weights assume unit-variance noise; without it the noise level was off by
  ~sqrt(nx). tests/test_preprocess_epi_gridding.py checks the scale against an
  orthonormal FFT with uniform kx.
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
    ximg = ximg / np.sqrt(nx)  # reconecho.m's /nx; sigpy already applied 1/sqrt(nx)

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
