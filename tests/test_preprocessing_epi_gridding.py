from dataclasses import replace

import numpy as np
import pytest

sigpy = pytest.importorskip("sigpy")

import hdf5storage  # noqa: E402

from params import load_params  # noqa: E402
from preprocessing.epi_gridding import rampsamp2cart, rampsampepi2cart  # noqa: E402
from sampling.gen_sampling_masks import gen_sampling_masks  # noqa: E402
from sequences.ArbEPI import generate_arbepi  # noqa: E402


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


def test_rampsampepi2cart_recovers_object_under_real_pope_readout_trajectory(tmp_path):
    """The tests above exercise the gridding algorithm against a smooth,
    symmetric power-law stand-in trajectory. This closes the gap to the
    trajectory this repo actually ships: the *real* kxo/kxe measured off
    the assembled asymmetric-POPE-ramp readout (lib/make_readout_grads.py,
    default params.ro_slew_rise/ro_slew_fall = 100/120 T/m/s), the same
    values sequences/ArbEPI.py writes to scan_info.mat and
    preprocessing/preprocess.py feeds straight into rampsampepi2cart.
    kx correctness/coverage/Nyquist under POPE is independently verified by
    test_arbepi_kx_coverage_and_nyquist -- this checks the downstream
    regridded *image* is still recovered accurately, not just that the
    trajectory itself is well-formed.

    Measured relative L2 image error at these (small) params is ~0.04-0.07
    (both parities, a sharp single-pixel feature included to stress
    high-frequency fidelity); the 0.15 threshold below leaves headroom
    rather than pinning the exact value."""
    p = replace(
        load_params(),
        Ny=16, Nz=12, ETL=8, Nshots=3, Nframes=1,
        sampling_method='caipi', R=8, output_dir=str(tmp_path),
    )
    omegas = gen_sampling_masks(p.R, p)
    generate_arbepi(omegas, p, seqname='xcheck')
    scan_info = hdf5storage.loadmat(str(tmp_path / 'scan_info.mat'))
    kxo = scan_info['kxo'].ravel() / 100  # cycles/m -> cycles/cm
    kxe = scan_info['kxe'].ravel() / 100

    nx = p.Nx
    fov_cm = p.fov[0] * 100

    x_true = _object(nx)
    x_true[nx // 2] = 1.5  # a sharp feature to stress high-frequency fidelity
    yo = sigpy.nufft(x_true, (kxo * fov_cm)[:, None])
    ye = sigpy.nufft(x_true, (kxe * fov_cm)[:, None])

    etl = 6  # 3 odd + 3 even echoes
    dr = np.zeros((len(kxo), etl), dtype=complex)
    dr[:, 0::2] = yo[:, None]
    dr[:, 1::2] = ye[:, None]

    dc = rampsampepi2cart(dr, kxo, kxe, nx, fov_cm)
    assert dc.shape == (nx, etl)

    true_n = np.abs(x_true) / np.abs(x_true).max()
    for e in range(etl):
        img = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(dc[:, e])))
        img_n = np.abs(img) / np.abs(img).max()
        rel_err = np.linalg.norm(img_n - true_n) / np.linalg.norm(true_n)
        assert rel_err < 0.15, f'echo {e}: relative L2 image error {rel_err:.4f} too high'
