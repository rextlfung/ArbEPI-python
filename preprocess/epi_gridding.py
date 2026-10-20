"""Ramp-sampled EPI readouts -> Cartesian kx, by 1D NUFFT or spline.

Ports hmriutils' rampsampepi2cart.m / rampsamp2cart.m / reconecho.m with
sigpy's NUFFT in place of MIRT's Gmri: a density-compensated adjoint NUFFT
(dcf = |diff(kx)|) to image space, then an FFT back to a Cartesian grid. Odd
and even echoes use their own trajectories (kxo, kxe).

`method='spline'` is hmriutils' faster path, as of its commit ca7e9f6 (which
fixed two bugs in it): cubic-spline interpolation onto a 2x oversampled k grid
whose DC sample sits at the DFT's N/2 + 1 index (so the grid is asymmetric
about kx = 0 by dk/4), zero outside the sampled kx range (no extrapolation),
then crop the FOV in image space and deapodize by the image-space response of
an impulse interpolated the same way. Scaled to match the NUFFT path.

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

import functools

import numpy as np
import sigpy
from scipy.interpolate import CubicSpline

METHODS = ('nufft', 'spline')
SPLINE_OSF = 2  # k-space oversampling factor of the spline path's intermediate grid


def _density_compensation(kx: np.ndarray) -> np.ndarray:
    """Ports reconecho.m: dcf = |diff(kx)| / max(|diff(kx)|), with a
    trailing zero appended so length matches kx (and therefore the data)."""
    dcf = np.abs(np.diff(kx))
    dcf = np.append(dcf, 0.0)
    return dcf / dcf.max()


def _spline_resample(d: np.ndarray, kx: np.ndarray, kxc: np.ndarray) -> np.ndarray:
    """MATLAB interp1(kx, d, kxc, 'spline', 0) along axis 0: not-a-knot cubic
    spline, 0 outside [min(kx), max(kx)]. kx may be decreasing."""
    order = np.argsort(kx)
    out = CubicSpline(kx[order], d[order], axis=0, bc_type='not-a-knot')(kxc)
    out[(kxc < kx.min()) | (kxc > kx.max())] = 0
    return out


def _spline_matrix(kx: np.ndarray, nx: int, fov_cm: float) -> np.ndarray:
    """The spline path as one [nx, nr] matrix. Every step (interpolation, ifft,
    crop, deapodization, fft) is linear in the data, so applying the steps to
    the identity gives a matrix that depends only on the trajectory; a
    reconstruction grids ~1e5 readouts with the same two trajectories."""
    if nx % 2:
        raise ValueError(f"rampsamp2cart: method='spline' needs even nx, got {nx}")
    nr = len(kx)
    osf = SPLINE_OSF
    kxc = np.arange(-nx * osf // 2, nx * osf // 2) / osf / fov_cm  # [osf nx], cycles/cm

    def to_image(d):  # [nr, M] -> cropped image-space [nx, M]
        d1 = _spline_resample(d, kx, kxc)
        x1 = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(d1, axes=0), axis=0), axes=0)
        return x1[nx // 2 : nx + nx // 2]  # crop the FOV

    impulse = np.zeros((nr, 1))
    impulse[int(np.floor(nr / 2 + 0.5)), 0] = 1  # MATLAB round(end/2) + 1, 0-based
    ap = np.abs(to_image(impulse))[:, 0]
    ap = ap / ap.max()

    x1 = to_image(np.eye(nr)) / ap[:, None]
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(x1, axes=0), axis=0), axes=0)


@functools.lru_cache(maxsize=8)
def _spline_matrix_cached(kx_bytes: bytes, nx: int, fov_cm: float) -> np.ndarray:
    return _spline_matrix(np.frombuffer(kx_bytes, dtype=np.float64), nx, fov_cm)


def _rampsamp2cart_spline(dr: np.ndarray, kx: np.ndarray, nx: int, fov_cm: float) -> np.ndarray:
    M = _spline_matrix_cached(np.ascontiguousarray(kx, dtype=np.float64).tobytes(), nx, fov_cm)
    M = M.astype(np.result_type(dr.dtype, np.complex64), copy=False)
    return (M @ dr.reshape(dr.shape[0], -1)).reshape((nx,) + dr.shape[1:])


def rampsamp2cart(
    dr: np.ndarray, kx: np.ndarray, nx: int, fov_cm: float, method: str = 'nufft'
) -> np.ndarray:
    """Interpolate ramp-sampled data onto a Cartesian grid along axis 0.

    dr: [nr, ...] ramp-sampled raw data along the readout (1st) axis;
        trailing axes (echo, coil, ...) are treated as independent batch
        dims and regridded together.
    method: 'nufft' (density-compensated adjoint NUFFT) or 'spline' (see the
        module docstring; faster, needs even nx).
    Returns [nx, ...] gridded k-space data. Ports rampsamp2cart.m.
    """
    if method == 'spline':
        return _rampsamp2cart_spline(dr, kx, nx, fov_cm)
    if method != 'nufft':
        raise ValueError(f'method must be one of {METHODS}, got {method!r}')
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
    dr: np.ndarray, kxo: np.ndarray, kxe: np.ndarray, nx: int, fov_cm: float,
    method: str = 'nufft',
) -> np.ndarray:
    """Interpolate ramp-sampled EPI data (odd/even echoes on different
    trajectories) onto a Cartesian grid. Ports rampsampepi2cart.m.

    dr: [nr, etl, ...] ramp-sampled raw data, EPI echo train along axis 1.
    kxo, kxe: [nr] k-space sample locations (cycles/cm) for odd/even echoes.
    method: 'nufft' or 'spline', see rampsamp2cart.
    Returns [nx, etl, ...] gridded k-space data.
    """
    dr_shape = dr.shape
    nr, etl = dr_shape[0], dr_shape[1]
    dr2 = dr.reshape(nr, etl, -1)

    dco = rampsamp2cart(dr2[:, 0::2, :], kxo, nx, fov_cm, method)  # odd echoes (MATLAB 1-based odd)
    dce = rampsamp2cart(dr2[:, 1::2, :], kxe, nx, fov_cm, method)  # even echoes

    dc = np.empty((nx, etl) + dr2.shape[2:], dtype=np.result_type(dco, dce))
    dc[:, 0::2] = dco
    dc[:, 1::2] = dce
    return dc.reshape((nx,) + dr_shape[1:])
