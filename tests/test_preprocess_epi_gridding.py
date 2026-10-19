from dataclasses import replace

import numpy as np
import pytest

sigpy = pytest.importorskip("sigpy")

import hdf5storage  # noqa: E402

from params import load_params  # noqa: E402
from preprocess.epi_gridding import rampsamp2cart, rampsampepi2cart  # noqa: E402
from sample.gen_sampling_masks import gen_sampling_masks  # noqa: E402
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
    preprocess/preprocess.py feeds straight into rampsampepi2cart.
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


def test_rampsamp2cart_absolute_scale_matches_ortho_fft_for_uniform_kx():
    """Regression test for the 2026-09-21 absolute-scale fix (see module
    docstring's "Absolute-scale fix" for the derivation from reconecho.m).
    A uniform (uniformly-spaced) kx degenerates density compensation to a
    constant (dcf ~= 1 everywhere), isolating the NUFFT/FFT normalization
    gap this fix addresses from any trajectory-dependent density-
    compensation effects -- this is deliberately the *cleanest* case, not
    a claim that DC effects don't also matter for a real ramp trajectory.

    Uses a smooth, non-adversarial test object (this file's own _object
    helper) rather than white noise -- a fully random signal is close to
    worst-case for NUFFT interpolation accuracy and produces large,
    signal-dependent per-sample errors unrelated to the overall scale
    factor being tested here (confirmed during this fix's investigation:
    a white-noise probe gave wildly inconsistent per-sample ratios on a
    real ramp trajectory, while smooth objects and uniform-kx probes both
    give a clean, tight, nx-independent scale factor)."""
    for nx in [32, 64, 128]:
        fov_cm = 20.0
        res = fov_cm / nx
        kmax = 1 / (2 * res)
        kx = np.linspace(-kmax, kmax, nx, endpoint=False)  # uniform -> dcf ~= 1
        coord = (kx * fov_cm)[:, None]

        x_true = _object(nx).astype(complex)
        dr = sigpy.nufft(x_true, coord)
        dc = rampsamp2cart(dr[:, None], kx, nx, fov_cm)[:, 0]

        k_ref = np.fft.fftshift(np.fft.fft(np.fft.fftshift(x_true), norm='ortho'))
        ratio = np.abs(dc) / np.maximum(np.abs(k_ref), 1e-12)
        # only compare where k_ref has meaningful energy (avoid dividing by
        # near-zero reference values at frequencies _object has no content)
        significant = np.abs(k_ref) > 0.05 * np.abs(k_ref).max()
        assert significant.sum() > nx // 4
        median_ratio = np.median(ratio[significant])
        assert abs(median_ratio - 1.0) < 0.1, (
            f'nx={nx}: median scale ratio {median_ratio:.4f}, expected ~1.0'
        )


# --- method='spline' (hmriutils ca7e9f6) ---

def _spline_case(nx=64, fov_cm=20.0, descending=False):
    """A ramp-sampled readout: 1.5x oversampled, denser at the center, a hair
    past Nyquist at the ends (the power-law stand-in above is sparser than a
    Cartesian grid at its ends, which no interpolation recovers)."""
    u = np.linspace(-1, 1, int(1.5 * nx))
    kx = 1.02 * (nx / 2 / fov_cm) * (0.8 * u + 0.2 * np.sign(u) * u**2)
    if descending:
        kx = kx[::-1].copy()
    x_true = _object(nx)
    return nx, fov_cm, kx, x_true, sigpy.nufft(x_true, (kx * fov_cm)[:, None])


@pytest.mark.parametrize('descending', [False, True])
def test_rampsamp2cart_spline_recovers_the_cartesian_kspace(descending):
    nx, fov_cm, kx, x_true, y = _spline_case(descending=descending)
    dc = rampsamp2cart(y[:, None], kx, nx, fov_cm, 'spline')[:, 0]
    # sigpy.nufft is unnormalized-by-nothing: its value at an on-grid k is the orthonormal DFT
    n = np.arange(nx) - nx // 2
    ref = np.exp(-2j * np.pi * np.outer(np.arange(-nx // 2, nx // 2), n) / nx) @ x_true / np.sqrt(nx)
    assert np.linalg.norm(dc - ref) / np.linalg.norm(ref) < 0.02
    dc_n = rampsamp2cart(y[:, None], kx, nx, fov_cm, 'nufft')[:, 0]
    assert np.linalg.norm(dc - ref) < np.linalg.norm(dc_n - ref)  # no worse than the NUFFT


def test_rampsamp2cart_spline_uniform_kx_is_the_orthonormal_dft():
    nx, fov_cm = 64, 20.0
    n = np.arange(nx) - nx // 2
    kx = np.arange(-nx // 2, nx // 2 + 1) / fov_cm  # the Cartesian grid, plus the +N/2 end sample
    x_true = _object(nx)
    dft = np.exp(-2j * np.pi * np.outer(np.arange(-nx // 2, nx // 2 + 1), n) / nx) / np.sqrt(nx)
    dc = rampsamp2cart((dft @ x_true)[:, None], kx, nx, fov_cm, 'spline')[:, 0]
    np.testing.assert_allclose(dc, (dft @ x_true)[:-1], atol=0.01 * np.abs(dc).max())


def test_rampsamp2cart_spline_is_linear_batched_and_rejects_odd_nx():
    nx, fov_cm, kx, _, y = _spline_case()
    batch = np.stack([y, 2j * y, y[::-1]], axis=1).reshape(len(kx), 3, 1)
    dc = rampsamp2cart(batch, kx, nx, fov_cm, 'spline')
    assert dc.shape == (nx, 3, 1)
    np.testing.assert_allclose(dc[:, 1, 0], 2j * dc[:, 0, 0], atol=1e-10)
    np.testing.assert_allclose(dc[:, 0, 0], rampsamp2cart(y[:, None], kx, nx, fov_cm, 'spline')[:, 0])
    with pytest.raises(ValueError, match='even nx'):
        rampsamp2cart(y[:63, None], kx[:63], 63, fov_cm, 'spline')
    with pytest.raises(ValueError, match='method'):
        rampsamp2cart(y[:, None], kx, nx, fov_cm, 'bogus')


def test_rampsampepi2cart_spline_recovers_object_under_real_pope_readout_trajectory(tmp_path):
    p = replace(
        load_params(), Ny=16, Nz=12, ETL=8, Nshots=3, Nframes=1,
        sampling_method='caipi', R=8, output_dir=str(tmp_path),
    )
    generate_arbepi(gen_sampling_masks(p.R, p), p, seqname='xcheck')
    scan_info = hdf5storage.loadmat(str(tmp_path / 'scan_info.mat'))
    kxo = scan_info['kxo'].ravel() / 100
    kxe = scan_info['kxe'].ravel() / 100
    nx, fov_cm = p.Nx, p.fov[0] * 100

    x_true = _object(nx)
    x_true[nx // 2] = 1.5
    dr = np.zeros((len(kxo), 4), dtype=complex)
    dr[:, 0::2] = sigpy.nufft(x_true, (kxo * fov_cm)[:, None])[:, None]
    dr[:, 1::2] = sigpy.nufft(x_true, (kxe * fov_cm)[:, None])[:, None]

    dc = rampsampepi2cart(dr, kxo, kxe, nx, fov_cm, 'spline')
    true_n = np.abs(x_true) / np.abs(x_true).max()
    for e in range(4):
        img = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(dc[:, e])))
        img_n = np.abs(img) / np.abs(img).max()
        assert np.linalg.norm(img_n - true_n) / np.linalg.norm(true_n) < 0.05
