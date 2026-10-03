"""simulate-fmri/: SNAKE-fMRI simulations of ArbEPI acquisitions.

Everything here runs on the analytic phantom (no BrainWeb download) at small
matrix sizes. The tests that matter most are the ones that tie the simulated
k-space to an independent statement of what it should be: the centered FFT of
the true image at the scheduled locations (and recon/'s own SENSE operator),
the decay each echo should carry, the noise level the file reports, and
recon/testbed.py's scorer recovering the truth from a perfect reconstruction.
"""

import importlib
from dataclasses import replace

import h5py
import numpy as np
import pytest

pytest.importorskip('snake')

pytestmark = [
    # xsdata, on every MRD header read: SNAKE's waveform ids are outside the schema's enum
    pytest.mark.filterwarnings('ignore:Failed to convert value'),
    pytest.mark.filterwarnings('ignore:.*use of fork.*:DeprecationWarning'),
]

simulate = importlib.import_module('simulate-fmri.simulate')
phantoms = importlib.import_module('simulate-fmri.phantom')
handlers = importlib.import_module('simulate-fmri.handlers')
sampler_mod = importlib.import_module('simulate-fmri.sampler')

RES_MM = 8.0
ESP_MS = 0.6
TE_MS = 30.0
TR_SHOT_S = 0.05


def make_protocol(shape, etl, n_frames, n_shots=None, seed=0):
    """A random (ky, kz) schedule: every frame visits n_shots * etl distinct
    locations (all of them when n_shots is None)."""
    nx, ny, nz = shape
    if n_shots is None:
        assert ny * nz % etl == 0
        n_shots = ny * nz // etl
    rng = np.random.default_rng(seed)
    schedules = np.empty((n_frames, n_shots, etl, 2), dtype=np.int64)
    for frame in range(n_frames):
        picks = rng.permutation(ny * nz)[: n_shots * etl].reshape(n_shots, etl)
        schedules[frame, ..., 0], schedules[frame, ..., 1] = np.unravel_index(picks, (ny, nz))
    return simulate.Protocol(
        shape=shape,
        fov_mm=tuple(n * RES_MM for n in shape),
        schedules=schedules,
        echo_times_ms=TE_MS + (np.arange(etl) - (etl / 2 - 0.5)) * ESP_MS,
        volume_tr_s=n_shots * TR_SHOT_S,
        fa_deg=15.0,
    )


def phantom_for(shape):
    """The analytic phantom on a finer grid covering the same field of view,
    and its activation ROI."""
    fine = tuple(2 * n for n in shape)
    return (
        phantoms.ellipsoid_phantom(fine, RES_MM / 2),
        phantoms.ellipsoid_phantom_roi(fine, RES_MM / 2),
    )


def run(tmp_path, protocol, name='sim', **kwargs):
    phantom, roi = phantom_for(protocol.shape)
    kwargs = {'n_coils': 4, 'coils_per_ring': 2, 'snr': np.inf, 'n_workers': 1, **kwargs}
    return simulate.simulate(protocol, str(tmp_path), name, phantom=phantom, roi=roi, **kwargs)


def read(path, *keys):
    with h5py.File(path, 'r') as f:
        return [f[k][()] for k in keys]


def fft3c(x):
    """recon/operators.py's SENSE convention, over the first three axes."""
    axes = (0, 1, 2)
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm='ortho'), axes=axes
    )


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------


def test_sampler_frame_follows_the_schedule_with_alternating_readouts():
    protocol = make_protocol((6, 5, 4), etl=4, n_frames=3, n_shots=2)
    sampler = sampler_mod.ArbEPISampler(
        schedules=protocol.schedules, echo_times_ms=protocol.echo_times_ms
    )
    sim_conf = simulate.make_sim_conf(protocol, n_coils=1)
    for frame in range(3):
        traj = sampler.frame(sim_conf, frame)
        assert traj.shape == (2, 4, 6, 3)
        assert traj.dtype == np.uint32  # the engine reinterprets the stored bytes as uint32
        np.testing.assert_array_equal(traj[..., 0, 1], protocol.schedules[frame, ..., 0])
        np.testing.assert_array_equal(traj[..., 0, 2], protocol.schedules[frame, ..., 1])
        assert (traj[..., 1] == traj[..., :1, 1]).all() and (traj[..., 2] == traj[..., :1, 2]).all()
        np.testing.assert_array_equal(traj[:, 0::2, :, 0], np.broadcast_to(np.arange(6), (2, 2, 6)))
        np.testing.assert_array_equal(
            traj[:, 1::2, :, 0], np.broadcast_to(np.arange(6)[::-1], (2, 2, 6))
        )
    # get_next_frame cycles through the frames
    for frame in (0, 1, 2, 0):
        np.testing.assert_array_equal(
            sampler.get_next_frame(sim_conf), sampler.frame(sim_conf, frame)
        )
    assert sampler.TR_vol_ms(sim_conf) == pytest.approx(protocol.volume_tr_s * 1e3)


def test_sampler_rejects_a_schedule_outside_the_matrix(tmp_path):
    protocol = make_protocol((6, 5, 4), etl=4, n_frames=1, n_shots=2)
    protocol.schedules[0, 0, 0] = (5, 0)  # ky = 5 on a 5-line axis
    with pytest.raises(ValueError, match='outside the simulated'):
        run(tmp_path, protocol, handlers=[])


# ---------------------------------------------------------------------------
# forward model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('shape, etl', [((12, 10, 8), 8), ((11, 9, 7), 7)])
def test_noiseless_kspace_is_the_centered_fft_of_the_true_image(tmp_path, shape, etl):
    """With no decay, noise or activation, every frame's k-space must be the
    recon's forward model of the true image, at the scheduled locations and
    exactly zero elsewhere -- for odd sizes too, where SNAKE's own FFT helper
    is one sample off center. Checks the sampler -> MRD -> engine -> export
    chain: axis order, k-space centering, coil layout, 0-based indices."""
    protocol = make_protocol(shape, etl, n_frames=2, n_shots=6)
    out = run(tmp_path, protocol, model='simple', handlers=[])
    ksp, omegas, smaps, x0 = read(out['preprocessed'], 'ksp_epi_zf', 'omegas', 'smaps', 'truth/x0')

    assert ksp.shape == (*shape, 4, 2)
    assert omegas.sum(axis=(0, 1)).tolist() == [6 * etl] * 2
    expected = fft3c(x0[..., None] * smaps)
    assert np.abs(expected).max() > 0.01
    for frame in range(2):
        want = expected * omegas[None, :, :, frame, None]
        np.testing.assert_allclose(ksp[..., frame], want, atol=2e-6 * np.abs(expected).max())


def test_kspace_matches_recon_sense_operator(tmp_path):
    """The same statement against recon/operators.py itself, read the way
    recon/sense.py reads a preprocessed file."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('mirtorch')
    from recon.operators import build_sense
    from recon.utils import load_and_gather_ksp, load_normalized_smaps, load_omega

    protocol = make_protocol((11, 10, 8), etl=8, n_frames=2, n_shots=5)
    fn = run(tmp_path, protocol, model='simple', handlers=[])['preprocessed']
    device = torch.device('cpu')
    _, smaps_chw = load_normalized_smaps(fn, device)
    A = build_sense(smaps_chw, load_omega(fn, 11, 10, 8, 2, device))
    y = load_and_gather_ksp(fn, A, device)
    (x0,) = read(fn, 'truth/x0')
    x = torch.from_numpy(np.repeat(x0[..., None], 2, axis=-1)).to(torch.complex64)
    np.testing.assert_allclose(A.apply(x).numpy(), y.numpy(), atol=2e-6 * float(y.abs().max()))


def test_t2s_model_decays_each_sample_from_its_scheduled_echo_time(tmp_path):
    """One tissue, one coil: T2s-model k-space / simple-model k-space is
    exp(-(t - TE) / T2*), t being the echo time of the sample's (ky, kz) in
    the schedule plus its place in the (alternating) readout."""
    shape, etl = (12, 10, 8), 8
    protocol = make_protocol(shape, etl, n_frames=1, n_shots=10)
    phantom, _ = phantom_for(shape)
    gm = list(phantom.labels).index('gm')
    single = replace(
        phantom, masks=phantom.masks.sum(axis=0, keepdims=True), labels=np.array(['gm']),
        props=phantom.props[gm : gm + 1],
    )
    t2s_ms = float(phantoms.TISSUE_PROPS_3T['gm'][2])
    ksp = {}
    for model in ('simple', 'T2s'):
        fn = simulate.simulate(
            protocol, str(tmp_path), model, phantom=single, n_coils=1, snr=np.inf, n_workers=1,
            model=model, handlers=[],
        )['preprocessed']
        (ksp[model],) = read(fn, 'ksp_epi_zf')

    nx = shape[0]
    dwell_ms = ESP_MS / nx
    expected = np.zeros(shape)
    for shot in range(protocol.n_shots):
        for echo in range(etl):
            ky, kz = protocol.schedules[0, shot, echo]
            order = np.arange(nx) if echo % 2 == 0 else np.arange(nx)[::-1]  # order[kx] = sample
            t = protocol.echo_times_ms[echo] + (order - (nx - 1) / 2) * dwell_ms
            expected[:, ky, kz] = np.exp(-(t - protocol.te_ms) / t2s_ms)
    assert expected.min() > 0  # fully sampled
    assert expected.max() / expected.min() > 1.05  # the echo train spans a visible decay
    simple = ksp['simple'][..., 0, 0]
    strong = np.abs(simple) > 1e-3 * np.abs(simple).max()
    np.testing.assert_allclose(
        (ksp['T2s'][..., 0, 0] / simple)[strong], expected[strong], rtol=2e-4
    )


def test_noise_var_attr_is_the_variance_of_the_added_noise(tmp_path):
    protocol = make_protocol((12, 10, 8), etl=8, n_frames=4)
    clean = run(tmp_path, protocol, 'clean', handlers=[])['preprocessed']
    noisy = run(tmp_path, protocol, 'noisy', handlers=[], snr=20.0)['preprocessed']
    (k0,), (k1,) = read(clean, 'ksp_epi_zf'), read(noisy, 'ksp_epi_zf')
    with h5py.File(clean, 'r') as f:
        assert 'noise_var' not in f.attrs  # recon/sense.py then leaves the data unscaled
    with h5py.File(noisy, 'r') as f:
        noise_var = f.attrs['noise_var']
        assert f.attrs['whitened']
    noise = k1 - k0  # fully sampled: 12*10*8 locations x 4 coils x 4 frames
    assert np.mean(np.abs(noise) ** 2) == pytest.approx(noise_var, rel=0.03)
    assert np.mean(noise.real**2) == pytest.approx(noise_var / 2, rel=0.05)
    # white across coils
    flat = noise.reshape(-1, 4, 4).transpose(1, 0, 2).reshape(4, -1)
    cov = flat @ flat.conj().T / flat.shape[1]
    assert np.abs(cov - np.diag(np.diag(cov))).max() < 0.05 * noise_var


def test_worker_processes_give_the_same_kspace(tmp_path):
    """The engine's forkserver workers unpickle the sampler/engine classes by
    module name ('simulate-fmri.engine'), which only works because nothing
    refers to the hyphenated package through an import statement."""
    protocol = make_protocol((12, 10, 8), etl=8, n_frames=2)
    one = run(tmp_path, protocol, 'one', block_on=0.3, block_off=0.3)['preprocessed']
    two = run(tmp_path, protocol, 'two', block_on=0.3, block_off=0.3, n_workers=2)['preprocessed']
    np.testing.assert_array_equal(read(one, 'ksp_epi_zf')[0], read(two, 'ksp_epi_zf')[0])


# ---------------------------------------------------------------------------
# activation and ground truth
# ---------------------------------------------------------------------------


def test_activation_roi_is_the_same_anatomy_on_any_grid():
    """The ROI is placed in mm through the affine, so a phantom cropped and
    resampled to another field of view gets it in the same place (SNAKE's
    stock ellipsoid scales with the array shape instead)."""
    centers = []
    for shape, res in (((48, 48, 36), 4.0), ((40, 36, 24), 5.0)):
        phantom = phantoms.ellipsoid_phantom((64, 64, 48), 3.0)
        roi = phantoms.ellipsoid_phantom_roi((64, 64, 48), 3.0)
        protocol = replace(
            make_protocol(shape, etl=shape[1], n_frames=1, n_shots=1),
            fov_mm=tuple(n * res for n in shape),
        )
        sim_conf = simulate.make_sim_conf(
            protocol, 1, phantoms.place_fov(phantom, shape, (res,) * 3)
        )
        on_grid = phantoms.to_acquisition_grid(phantom, sim_conf)
        handler = handlers.EllipsoidActivationHandler(block_on=1, block_off=1, duration=4, **roi)
        with_roi = handler.get_static(on_grid, sim_conf)
        mask = with_roi.masks[with_roi.labels_idx['ROI']]
        gm = with_roi.masks[with_roi.labels_idx['gm']]
        assert mask.sum() > 20
        assert (mask <= gm + 1e-6).all()  # inside gray matter
        idx = np.argwhere(mask > 0.5)
        centers.append((idx @ on_grid.affine[:3, :3].T + on_grid.affine[:3, 3]).mean(axis=0))
        np.testing.assert_allclose(centers[-1], roi['center_mm'], atol=res)
    np.testing.assert_allclose(centers[0], centers[1], atol=5.0)


def test_truth_group_reproduces_the_frames_and_scores_as_a_perfect_recon(tmp_path):
    """Single-shot frames (one phantom state per frame), fully sampled, no
    noise or decay: the inverse FFT of each frame is the true image, which the
    truth group must reproduce as x0 * (1 + amp_map * w(t)). Handing those
    images to recon/testbed.py's scorer as a reconstruction must then score as
    exact -- the check that the group follows the layout the scorer reads."""
    testbed = pytest.importorskip('recon.testbed')
    shape = (24, 24, 16)
    # 1 s per frame: the scorer's low-band (< 0.15 Hz) GLM needs a few bins there
    protocol = replace(make_protocol(shape, etl=24 * 16, n_frames=24), volume_tr_s=1.0)
    fn = run(tmp_path, protocol, model='simple', block_on=5, block_off=5)['preprocessed']
    ksp, smaps, x0, amp_map, waves, rois = read(
        fn, 'ksp_epi_zf', 'smaps', 'truth/x0', 'truth/amp_map', 'truth/waveforms', 'truth/roi_masks'
    )
    with h5py.File(fn, 'r') as f:
        amp = f['truth'].attrs['amp']
        assert list(f['truth'].attrs['roi_names']) == ['block_occipital']
        assert f.attrs['volume_tr'] == pytest.approx(protocol.volume_tr_s)

    axes = (0, 1, 2)
    coil_images = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(ksp, axes=axes), axes=axes, norm='ortho'), axes=axes
    )
    recon = (coil_images * smaps.conj()[..., None]).sum(axis=3)  # unit root-sum-of-squares maps
    truth = x0[..., None] * (1 + amp_map[..., None] * waves[0])
    np.testing.assert_allclose(np.abs(recon), truth, atol=1e-5 * x0.max())

    w = waves[0]
    assert abs(w.mean()) < 1e-9 and np.abs(w).max() == pytest.approx(1)
    assert rois.shape == (1, *shape) and rois.sum() > 10
    assert 0.01 < amp < 0.03  # TE / delta_r2s = 3% at full gray matter, less where mixed
    assert np.median(amp_map[rois[0]]) == pytest.approx(amp)

    fn_recon = str(tmp_path / 'perfect.h5')
    with h5py.File(fn_recon, 'w') as f:
        f['X_recon'] = recon.astype(np.complex64)
    s = testbed.score(fn, fn_recon)
    assert s['block_occipital_amp_ratio'] == pytest.approx(1, abs=0.02)
    assert s['block_occipital_corr'] > 0.999
    assert s['nrmse_frame_pct'] < 0.1
    assert s['fluct_pct'] < 1e-3


def test_resting_run_writes_an_empty_activation_truth(tmp_path):
    protocol = make_protocol((12, 10, 8), etl=8, n_frames=2)
    fn = run(tmp_path, protocol, handlers=[])['preprocessed']
    x0, image_rest, rois, waves = read(
        fn, 'truth/x0', 'truth/image_rest', 'truth/roi_masks', 'truth/waveforms'
    )
    np.testing.assert_array_equal(x0, image_rest)
    assert rois.shape == (0, 12, 10, 8) and waves.shape == (0, 2)


# ---------------------------------------------------------------------------
# from a real scan_info.mat
# ---------------------------------------------------------------------------


def test_simulate_from_generated_scan_info(built_seq_dir, tmp_path):
    """The default protocol's scan_info.mat (as generate_arbepi wrote it):
    the acquisition parameters survive the round trip, and the simulated
    frame's sampling mask is the mask the sequence was built from."""
    from params import load_params
    from preprocess.utils import nominal_te_s
    from sample.gen_sampling_masks import resolve_omegas

    params = replace(load_params(), Nframes=1, seed=0)
    scan_info = str(built_seq_dir / 'scan_info.mat')
    protocol = simulate.load_protocol(scan_info)
    assert protocol.shape == (params.Nx, params.Ny, params.Nz)
    assert (protocol.n_frames, protocol.n_shots, protocol.etl) == (1, params.Nshots, params.ETL)
    np.testing.assert_allclose(protocol.fov_mm, params.fov * 1e3)
    assert protocol.fa_deg == pytest.approx(params.fa)
    assert protocol.tr_shot_ms == pytest.approx(params.TR * 1e3)
    assert protocol.te_ms == pytest.approx(params.TE * 1e3, abs=1e-3)
    assert protocol.t_ref_s == pytest.approx(nominal_te_s(scan_info, params.ETL))
    assert protocol.acceleration == pytest.approx(params.R, rel=0.01)

    out = simulate.simulate(
        scan_info, str(tmp_path), phantom='ellipsoid', n_coils=1, model='simple', snr=np.inf,
        n_workers=1, handlers=[],
    )
    omegas, echo_times, ksp = read(out['preprocessed'], 'omegas', 'echo_times', 'ksp_epi_zf')
    np.testing.assert_array_equal(omegas, resolve_omegas(params))
    sampled_times = echo_times[omegas]
    assert sampled_times.min() == pytest.approx(protocol.echo_times_ms.min() / 1e3)
    assert sampled_times.max() == pytest.approx(protocol.echo_times_ms.max() / 1e3)
    # the echo at the k-space center is the nominal-TE echo
    assert echo_times[params.Ny // 2, params.Nz // 2, 0] == pytest.approx(protocol.t_ref_s)
    assert (np.abs(ksp[..., 0, 0]).sum(axis=0) > 0).tolist() == omegas[..., 0].tolist()


def test_load_protocol_without_a_saved_flip_angle_uses_the_ernst_angle(built_seq_dir, tmp_path):
    """scan_info.mat files written before 'fa' was recorded."""
    import shutil

    old = str(tmp_path / 'scan_info.mat')
    shutil.copy(built_seq_dir / 'scan_info.mat', old)
    with h5py.File(old, 'r+') as f:
        saved = f['fa'][()].item()
        del f['fa']
    assert simulate.load_protocol(old).fa_deg == pytest.approx(saved, rel=1e-6)  # T1 = 1.3 s
    assert simulate.load_protocol(old, fa_deg=20.0).fa_deg == 20.0
    assert simulate.load_protocol(old, frames=1).n_frames == 1
