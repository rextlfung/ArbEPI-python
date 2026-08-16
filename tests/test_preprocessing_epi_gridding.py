import numpy as np
import sigpy

from preprocessing.epi_gridding import rampsamp2cart, rampsampepi2cart


def _ramp_trajectory(nx, fov_cm):
    """A smooth nonuniform trajectory standing in for real ramp-sampled
    k-space (denser near center, sparser near the edges) -- enough to
    meaningfully exercise the density-compensation logic, without needing a
    full gradient-waveform simulation."""
    res = fov_cm / nx
    kmax = 1 / (2 * res)
    u = np.linspace(-1, 1, nx)
    return kmax * np.sign(u) * np.abs(u) ** 1.5


def _object(nx):
    x = np.zeros(nx, dtype=complex)
    x[nx // 3 - 3: nx // 3 + 3] = 1.0
    x[2 * nx // 3: 2 * nx // 3 + 3] = 0.6
    return x


def test_rampsamp2cart_recovers_object_location_and_shape():
    nx, fov_cm = 64, 20.0
    kx = _ramp_trajectory(nx, fov_cm)
    coord = (kx * fov_cm)[:, None]

    x_true = _object(nx)
    y = sigpy.nufft(x_true, coord)  # synthesize ramp-sampled k-space

    dc = rampsamp2cart(y[:, None], kx, nx, fov_cm)  # [nx, 1]
    assert dc.shape == (nx, 1)

    img = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(dc[:, 0])))
    img_n = np.abs(img) / np.abs(img).max()
    true_n = np.abs(x_true) / np.abs(x_true).max()

    assert np.argmax(img_n) in range(nx // 3 - 3, nx // 3 + 3)
    np.testing.assert_allclose(img_n, true_n, atol=0.25)


def test_rampsampepi2cart_routes_odd_even_to_correct_trajectory():
    nx, fov_cm = 48, 20.0
    kxo = _ramp_trajectory(nx, fov_cm)
    kxe = kxo * 1.05  # a distinguishably different trajectory

    x_true = _object(nx)
    yo = sigpy.nufft(x_true, (kxo * fov_cm)[:, None])
    ye = sigpy.nufft(x_true, (kxe * fov_cm)[:, None])

    etl = 4  # 2 odd + 2 even echoes (MATLAB 1-based odd/even -> 0-based even/odd index)
    dr = np.zeros((nx, etl), dtype=complex)
    dr[:, 0::2] = yo[:, None]
    dr[:, 1::2] = ye[:, None]

    dc = rampsampepi2cart(dr, kxo, kxe, nx, fov_cm)
    assert dc.shape == (nx, etl)

    # Each echo, gridded with its own trajectory, should reproduce the
    # single-trajectory rampsamp2cart result for that same data/trajectory.
    dco_ref = rampsamp2cart(yo[:, None], kxo, nx, fov_cm)[:, 0]
    dce_ref = rampsamp2cart(ye[:, None], kxe, nx, fov_cm)[:, 0]
    for e in range(etl):
        ref = dco_ref if e % 2 == 0 else dce_ref
        np.testing.assert_allclose(dc[:, e], ref)


def test_rampsampepi2cart_matches_rampsamp2cart_when_trajectories_equal():
    nx, fov_cm = 48, 20.0
    kx = _ramp_trajectory(nx, fov_cm)
    x_true = _object(nx)
    y = sigpy.nufft(x_true, (kx * fov_cm)[:, None])

    etl = 6
    dr = np.tile(y[:, None], (1, etl))
    dc_epi = rampsampepi2cart(dr, kx, kx, nx, fov_cm)
    dc_ref = rampsamp2cart(dr, kx, nx, fov_cm)

    np.testing.assert_allclose(dc_epi, dc_ref)
