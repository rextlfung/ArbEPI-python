"""simulate_fmri/forward.py against a direct evaluation of the signal equation.

The fast model computes everything spatial once per echo index, with FFTs over
(y, z), a first-order treatment of the perturbation modes and a Taylor
expansion of the evolution during a readout. The reference here sums
exp(...) over every spin for every sample, with none of those shortcuts.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('snake')

from simulate_fmri import forward  # noqa: E402

GAMMA_CYCLES = 1.0  # fields are given in Hz directly


def make_case(shape=(6, 5, 4), g=2, n_coils=3, etl=4, n_frames=2, n_shots=3, seed=0, b0_hz=60.0):
    rng = np.random.default_rng(seed)
    fov = tuple(0.008 * n for n in shape)
    grid = forward.Grid(shape, fov, g)
    fine = grid.fine_shape
    mag = rng.uniform(0.2, 1.0, size=(2, *fine)).astype(np.float32)
    mag[:, :1] = 0  # some empty space
    spins = forward.Spins(
        mag=torch.from_numpy(mag),
        r2s=torch.tensor([18.0, 4.0]),
        b0=torch.from_numpy(rng.uniform(-b0_hz, b0_hz, size=fine).astype(np.float32)),
        smaps=torch.from_numpy(
            (rng.normal(size=(n_coils, *fine)) + 1j * rng.normal(size=(n_coils, *fine)))
            .astype(np.complex64)
        ),
    )
    ny, nz = shape[1:]
    schedules = np.stack(
        [rng.integers(0, ny, size=(n_frames, n_shots, etl)),
         rng.integers(0, nz, size=(n_frames, n_shots, etl))], axis=-1,
    )
    echo_times = 0.030 + (np.arange(etl) - (etl / 2 - 0.5)) * 0.6e-3
    return grid, spins, schedules, echo_times, rng


def ramp_trajectories(nx, fov_x, nfid, rng):
    """Nonuniform, asymmetric odd/even trajectories spanning the acquired kx."""
    kmax = nx / 2 / fov_x
    u = np.linspace(-1, 1, nfid)
    warp = 0.93 * u + 0.07 * u**3 + 0.02 * (1 - u**2)  # denser at the ends, off-center
    return kmax * warp, -kmax * warp[::-1] * 0.98


def brute_force(grid, spins, readout, schedules, modes, frequency):
    """s = scale * sum_r S_c m(r, t) exp(-i 2 pi k.r), with the exact exponentials."""
    x, y, z = (grid.coords(a) for a in range(3))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    mag = spins.mag.numpy().astype(np.float64)
    r2s = spins.r2s.numpy().astype(np.float64)
    b0 = spins.b0.numpy().astype(np.float64)
    smaps = spins.smaps.numpy().astype(np.complex128)
    n_frames, n_shots, etl, _ = schedules.shape
    scale = 1 / (grid.g**3 * math.sqrt(math.prod(grid.shape)))
    out = np.zeros((n_frames * n_shots, etl, smaps.shape[0], readout.nfid), dtype=np.complex128)
    for s in range(n_frames * n_shots):
        f_idx, s_idx = divmod(s, n_shots)
        amp = np.ones_like(b0)
        rate = np.zeros_like(b0)
        freq = b0.copy() + (0.0 if frequency is None else frequency[s])
        group_amp = [np.ones_like(b0) for _ in r2s]
        group_rate = [np.zeros_like(b0) for _ in r2s]
        group_freq = [np.zeros_like(b0) for _ in r2s]
        for m in modes:
            w = m.weight.numpy().astype(np.float64) * m.course[s]
            sel = np.ones(len(r2s)) if m.groups is None else m.groups.numpy()
            for gi in range(len(r2s)):
                if m.kind == 'amp':
                    group_amp[gi] = group_amp[gi] + sel[gi] * w
                elif m.kind == 'r2s':
                    group_rate[gi] = group_rate[gi] + sel[gi] * w
                else:
                    group_freq[gi] = group_freq[gi] + sel[gi] * w
        del amp, rate
        for e in range(etl):
            ky = (schedules[f_idx, s_idx, e, 0] - grid.shape[1] // 2) / grid.fov[1]
            kz = (schedules[f_idx, s_idx, e, 1] - grid.shape[2] // 2) / grid.fov[2]
            for j in range(readout.nfid):
                t = readout.echo_times[e] + readout.tau[e, j]
                m_t = sum(
                    mag[gi] * group_amp[gi]
                    * np.exp(-(r2s[gi] + group_rate[gi]) * t)
                    * np.exp(2j * np.pi * (freq + group_freq[gi]) * t)
                    for gi in range(len(r2s))
                )
                enc = np.exp(-2j * np.pi * (readout.kx[e, j] * X + ky * Y + kz * Z))
                out[s, e, :, j] = (
                    scale * (smaps * (m_t * enc)[None]).sum(axis=(1, 2, 3))
                    * np.exp(1j * readout.phase[e])
                )
    return out


def run_fast(grid, spins, readout, schedules, modes=None, frequency=None, order=2):
    n_exc = schedules.shape[0] * schedules.shape[1]
    out = np.zeros((n_exc, readout.etl, spins.smaps.shape[0], readout.nfid), dtype=np.complex64)

    def sink(e, data):
        out[:, e] = data.numpy()

    cal = forward.epi_readouts(spins, grid, readout, schedules, sink, modes=modes,
                               frequency=frequency, order=order, coil_batch=2)
    return out, cal.numpy()


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.parametrize('shape, g', [((6, 5, 4), 2), ((5, 4, 3), 1), ((5, 4, 3), 3), ((4, 5, 4), 2)])
def test_ramp_sampled_readouts_match_the_signal_equation(shape, g):
    """Ramp sampling, readout delay, odd/even phase and shift, static field and
    T2* decay at every sample's own time, on odd and even matrices and grid
    factors (the fine grid's half-voxel offsets differ between them)."""
    grid, spins, schedules, echo_times, rng = make_case(shape=shape, g=g)
    kxo, kxe = ramp_trajectories(shape[0], grid.fov[0], 9, rng)
    readout = forward.epi_readout(
        kxo, kxe, echo_times, dwell=40e-6, fov_x=grid.fov[0], delay=-0.37,
        a0=[-0.3, -0.25], a1=[0.2, 0.15],
    )
    fast, _ = run_fast(grid, spins, readout, schedules)
    ref = brute_force(grid, spins, readout, schedules, [], None)
    # the only approximation left is the Taylor expansion during the readout:
    # |2 pi 60 Hz x 0.16 ms|^3 / 6 ~ 4e-5
    assert rel_err(fast, ref) < 2e-4


def test_taylor_order_controls_the_within_readout_error():
    grid, spins, schedules, echo_times, rng = make_case(b0_hz=300.0)
    kxo, kxe = ramp_trajectories(6, grid.fov[0], 9, rng)
    readout = forward.epi_readout(kxo, kxe, echo_times, dwell=60e-6, fov_x=grid.fov[0])
    ref = brute_force(grid, spins, readout, schedules, [], None)
    errs = [rel_err(run_fast(grid, spins, readout, schedules, order=n)[0], ref) for n in range(4)]
    assert errs[0] > 0.1  # 300 Hz over +-0.24 ms is not negligible
    assert errs[1] < errs[0] / 3 and errs[2] < errs[1] / 3 and errs[3] < errs[2] / 3
    assert errs[2] < 0.03


def test_modes_and_shot_frequency_match_to_first_order():
    grid, spins, schedules, echo_times, rng = make_case(b0_hz=40.0)
    kxo, kxe = ramp_trajectories(6, grid.fov[0], 9, rng)
    readout = forward.epi_readout(kxo, kxe, echo_times, dwell=20e-6, fov_x=grid.fov[0])
    n_exc = schedules.shape[0] * schedules.shape[1]
    fine = grid.fine_shape

    def weight():
        return torch.from_numpy(rng.uniform(0, 1, size=fine).astype(np.float32))

    modes = [
        forward.Mode('r2s', weight(), rng.uniform(-1.0, 0.0, n_exc), torch.tensor([1.0, 0.0])),
        forward.Mode('amp', weight(), rng.uniform(-0.02, 0.02, n_exc)),
        forward.Mode('freq', weight(), rng.uniform(-0.5, 0.5, n_exc)),
    ]
    frequency = rng.uniform(-2, 2, n_exc)
    fast, _ = run_fast(grid, spins, readout, schedules, modes, frequency)
    ref = brute_force(grid, spins, readout, schedules, modes, frequency)
    unperturbed = brute_force(grid, spins, readout, schedules, [], None)
    change = rel_err(ref, unperturbed)
    assert change > 0.02  # the perturbations are visible ...
    assert rel_err(fast, ref) < 0.05 * change  # ... and reproduced to second order
    # the common frequency offset alone is exact
    fast_f, _ = run_fast(grid, spins, readout, schedules, [], frequency)
    assert rel_err(fast_f, brute_force(grid, spins, readout, schedules, [], frequency)) < 2e-4


def test_calibration_train_is_the_unperturbed_train_at_the_kspace_center():
    grid, spins, schedules, echo_times, rng = make_case()
    kxo, kxe = ramp_trajectories(6, grid.fov[0], 9, rng)
    readout = forward.epi_readout(kxo, kxe, echo_times, dwell=40e-6, fov_x=grid.fov[0],
                                  delay=0.4, a0=-0.3)
    mode = forward.Mode('amp', torch.ones(grid.fine_shape), np.full(6, 0.3))
    _, cal = run_fast(grid, spins, readout, schedules, [mode])
    center = np.zeros((1, 1, readout.etl, 2), dtype=int)
    center[..., 0], center[..., 1] = grid.shape[1] // 2, grid.shape[2] // 2
    ref = brute_force(grid, spins, readout, center, [], None)
    assert rel_err(cal, ref[0]) < 2e-4


def test_cartesian_readout_on_the_acquisition_grid_is_the_centered_fft():
    """g = 1, no field, an ideal readout: the readouts are the orthonormal
    centered FFT of the coil images (recon/operators.py's SENSE) at the
    scheduled locations."""
    grid, spins, schedules, echo_times, _ = make_case(shape=(5, 4, 3), g=1, b0_hz=0.0)
    spins.r2s[:] = 0
    readout = forward.cartesian_readout(5, grid.fov[0], echo_times)
    fast, _ = run_fast(grid, spins, readout, schedules, order=0)
    image = spins.mag.sum(0).numpy()[None] * spins.smaps.numpy()
    axes = (1, 2, 3)
    ksp = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(image, axes=axes), axes=axes, norm='ortho'), axes=axes
    )
    sched = schedules.reshape(-1, readout.etl, 2)
    for s in range(sched.shape[0]):
        for e in range(readout.etl):
            np.testing.assert_allclose(
                fast[s, e], ksp[:, :, sched[s, e, 0], sched[s, e, 1]], atol=1e-5 * np.abs(ksp).max()
            )


@pytest.mark.parametrize('shape, g', [((6, 5, 4), 2), ((5, 4, 3), 3)])
def test_gre_kspace_matches_the_signal_equation(shape, g):
    grid, spins, _, _, _ = make_case(shape=shape, g=g)
    tes = np.array([2.2e-3, 4.5e-3])
    ksp = forward.gre_kspace(spins, grid, tes, coil_batch=2).numpy()
    x, y, z = (grid.coords(a) for a in range(3))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    scale = 1 / (g**3 * math.sqrt(math.prod(shape)))
    rng = np.random.default_rng(1)
    for _ in range(12):
        i = [int(rng.integers(n)) for n in shape]
        k = [(i[a] - shape[a] // 2) / grid.fov[a] for a in range(3)]
        enc = np.exp(-2j * np.pi * (k[0] * X + k[1] * Y + k[2] * Z))
        for n, te in enumerate(tes):
            m_t = sum(
                spins.mag[gi].numpy() * np.exp(-float(spins.r2s[gi]) * te) for gi in range(2)
            ) * np.exp(2j * np.pi * spins.b0.numpy() * te)
            ref = scale * (spins.smaps.numpy() * (m_t * enc)[None]).sum(axis=(1, 2, 3))
            np.testing.assert_allclose(ksp[i[0], i[1], i[2], n], ref, rtol=2e-4, atol=1e-6)


def test_band_limited_image_is_the_block_average_for_a_smooth_object():
    """The truth image: a fine-grid image cut to the acquired k-space. For an
    object that is constant within each acquisition voxel's neighborhood it is
    that constant; and g = 1 returns the image itself."""
    grid = forward.Grid((8, 6, 4), (0.08, 0.06, 0.04), 1)
    image = torch.rand(grid.fine_shape)
    np.testing.assert_allclose(forward.band_limited(image, grid).real.numpy(), image.numpy(),
                               atol=1e-5)
    grid2 = forward.Grid((8, 6, 4), (0.08, 0.06, 0.04), 2)
    flat = torch.full(grid2.fine_shape, 0.7)
    np.testing.assert_allclose(forward.band_limited(flat, grid2).real.numpy(), 0.7, atol=1e-5)


def test_readout_delay_follows_preprocess_convention():
    """epi_readout's `delay` is what preprocess.apply_delay undoes: the same
    trajectories for the same number."""
    pytest.importorskip('sigpy')
    from preprocess.preprocess import apply_delay

    rng = np.random.default_rng(0)
    kxo, kxe = ramp_trajectories(8, 0.2, 16, rng)
    echo_times = 0.03 + np.arange(4) * 5e-4
    for delay in (-0.3, 0.0, 1.25):
        readout = forward.epi_readout(kxo, kxe, echo_times, 4e-6, 0.2, delay=delay)
        ko, ke = apply_delay(kxo, kxe, 16, delay)
        np.testing.assert_allclose(readout.kx[0], ko, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(readout.kx[1], ke, rtol=1e-12, atol=1e-9)
        # tau = 0 where the played trajectory crosses kx = 0
        for e in (0, 1):
            order = np.argsort(readout.kx[e])
            assert abs(np.interp(0.0, readout.kx[e][order], readout.tau[e][order])) < 1e-12
