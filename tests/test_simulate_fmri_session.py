"""simulate_fmri/session.py: a simulated scan session through the real preprocess/.

The sessions here are small (a real ArbEPI sequence at 90 x 16 x 12, a 36 x 36
x 25 deGRE, 8 coils, the analytic phantom, CPU). What they check is that
preprocess() recovers what the simulation put in: the image, the readout
delay, the odd/even phase, the noise level, the sensitivity maps, the field
map.
"""

import shutil
import warnings
from dataclasses import replace

import h5py
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('snake')
pytest.importorskip('sigpy')

from params import load_params  # noqa: E402
from preprocess import utils as pre_utils  # noqa: E402
from preprocess.coils import apply_whitening  # noqa: E402
from preprocess.preprocess import PreprocessConfig, preprocess  # noqa: E402
from sample.gen_sampling_masks import resolve_omegas  # noqa: E402
from sequences.ArbEPI import generate_arbepi  # noqa: E402
from simulate_fmri import phantom as phantoms  # noqa: E402
from simulate_fmri import session  # noqa: E402
from simulate_fmri.physio import PhysioConfig  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings('ignore:Failed to convert value'),
    pytest.mark.filterwarnings('ignore:.*use of fork.*:DeprecationWarning'),
]

N_COILS = 8


def make_scan_info(out_dir, n_frames, n_shots, r):
    """A real ArbEPI sequence's scan_info.mat at 90 x 16 x 12 over the default
    field of view, with a 36 x 36 x 25 deGRE."""
    p = replace(
        load_params(output_dir=str(out_dir)), Ny=16, Nz=12, ETL=8, Nshots=n_shots,
        Nframes=n_frames, sampling_method='caipi', R=r, Nx_degre=36, Ny_degre=36, Nz_degre=25,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        generate_arbepi(resolve_omegas(p), p)
    return str(out_dir / 'scan_info.mat'), p


def anatomy(long_t2s=False):
    a = phantoms.ellipsoid_anatomy((48, 48, 36), 4.0)
    if long_t2s:
        a.phantom.props[:, 2] = 1e9  # no T2* decay
    return a


def small_cfg(**kw):
    base = dict(grid_factor=2, degre_grid_factor=2, n_coils=N_COILS, coils_per_ring=4, seed=3)
    return session.SessionConfig(**{**base, **kw})


def read(path, *keys):
    with h5py.File(path, 'r') as f:
        return [f[k][()] for k in keys]


def ifft3c(k):
    ax = (0, 1, 2)
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=ax), axes=ax, norm='ortho'), axes=ax)


# ---------------------------------------------------------------------------
# the archives
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def clean(tmp_path_factory):
    """Fully sampled, one frame, nothing but the object: no field, decay,
    noise, physiology or activation."""
    seq = tmp_path_factory.mktemp('clean_seq')
    out = tmp_path_factory.mktemp('clean')
    scan_info, p = make_scan_info(seq, n_frames=1, n_shots=24, r=1)
    cfg = small_cfg(b0_scale=0.0, noise=0.0, physio=None, activation=False, delay=0.45,
                    oe_phase=(0.2, 0.35))
    paths = session.simulate_session(scan_info, str(out), anatomy=anatomy(long_t2s=True), cfg=cfg,
                                     device='cpu')
    return paths, p, cfg


def test_archives_have_the_shapes_and_order_preprocess_reads(clean):
    paths, p, cfg = clean
    nfid = len(pre_utils.load_kxoe(paths['scan_info'])[0])
    epi = pre_utils.read_archive(paths['epi'])
    assert epi.shape == (nfid, N_COILS, 1 * 24 * 8) and epi.dtype == np.complex64
    assert pre_utils.read_archive(paths['cal']).shape == (nfid, N_COILS, 24 * 8)
    n_noise = int(np.ceil(20 * N_COILS**2 / nfid))
    assert pre_utils.read_archive(paths['noise']).shape == (nfid, N_COILS, n_noise)
    gre = pre_utils.read_archive(paths['gre'])
    assert gre.shape == (36, N_COILS, 36 * 2 + 25 * 36 * 2)  # gain-cal block, then the volume
    # the gain-cal block repeats the ky = kz = 0 readout of each echo, which in
    # the volume (echo fastest, then iY, then iZ) is at iY = 18, iZ = 12
    np.testing.assert_array_equal(gre[..., 0], gre[..., 2])
    center = 36 * 2 + (12 * 36 + 18) * 2
    np.testing.assert_array_equal(gre[..., :2], gre[..., center : center + 2])
    # streaming gives the same readouts as the bulk read
    reader = pre_utils.ArchiveReader(paths['epi'])
    assert reader.metadata()['scan'] == 'ArbEPI'
    np.testing.assert_array_equal(np.stack([next(reader) for _ in range(5)], axis=-1), epi[..., :5])
    assert sum(1 for _ in reader) == epi.shape[-1] - 5
    # the calibration train: every shot identical (noise-free), blip-free
    cal = pre_utils.read_archive(paths['cal'])
    np.testing.assert_array_equal(cal[..., :8], cal[..., 8:16])


def test_preprocess_reconstructs_the_object_from_the_raw_readouts(clean):
    """With nothing but the object in the data, preprocess()'s k-space must be
    the object's: inverse FFT and combine with the true (whitened) coil maps,
    and the truth image comes back, in the right place and at the right scale.
    This is the end-to-end check of the archive order, the ramp-sample model
    against preprocess's gridding, the readout-delay and odd/even conventions,
    and the k-space centering."""
    paths, p, cfg = clean
    out = preprocess(PreprocessConfig(
        datdir=paths['datdir'], seqnames=['sim'], compress=False, estimate_smaps=False,
        estimate_b0=False, estimate_r2star=False), 'sim')
    ksp, W = read(out, 'ksp_epi_zf', 'W')
    with h5py.File(out, 'r') as f:
        assert f.attrs['delay'] == pytest.approx(cfg.delay, abs=0.051)
        a = np.asarray(f.attrs['oephase_a'])
    np.testing.assert_allclose(a[:, 0], np.linspace(*cfg.oe_phase, len(a)), atol=0.01)
    assert np.abs(a[:, 1]).max() < 0.05  # the delay leaves no linear term behind

    smaps, x0, tissue = read(paths['truth'], 'truth/smaps', 'truth/x0', 'truth/tissues')
    sw = apply_whitening(smaps, W)
    image = (ifft3c(ksp[..., 0]) * sw.conj()).sum(-1) / (np.abs(sw) ** 2).sum(-1)
    inside = tissue.sum(0) > 0.5
    assert inside.sum() > 500
    err = np.abs(np.abs(image) - x0)
    assert err[inside].max() < 0.03 * x0.max()
    assert np.linalg.norm(err) < 0.01 * np.linalg.norm(x0)


# ---------------------------------------------------------------------------
# a realistic session
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def realistic(tmp_path_factory):
    """Undersampled, four frames, everything on."""
    seq = tmp_path_factory.mktemp('real_seq')
    out = tmp_path_factory.mktemp('real')
    scan_info, p = make_scan_info(seq, n_frames=4, n_shots=3, r=8)
    cfg = small_cfg(delay=-0.3, block_on=0.05, block_off=0.05)
    paths = session.simulate_session(scan_info, str(out), anatomy=anatomy(), cfg=cfg, device='cpu')
    julia = shutil.which('julia') is not None
    pre = preprocess(PreprocessConfig(datdir=paths['datdir'], seqnames=['sim'],
                                      estimate_b0=julia), 'sim')
    return paths, pre, cfg, julia


def test_preprocess_recovers_delay_odd_even_phase_and_noise_level(realistic):
    paths, pre, cfg, _ = realistic
    with h5py.File(pre, 'r') as f:
        assert f.attrs['delay'] == pytest.approx(cfg.delay, abs=0.051)
        a = np.asarray(f.attrs['oephase_a'])
        assert f.attrs['whitened']
        # the noise scan whitens the data to unit variance
        assert f.attrs['noise_var'] == pytest.approx(1.0, abs=0.15)
        assert f['ksp_epi_zf'].shape[:3] == (90, 16, 12) and f['ksp_epi_zf'].shape[-1] == 4
        W = f['W'][()]
    np.testing.assert_allclose(a[:, 0], np.linspace(*cfg.oe_phase, len(a)), atol=0.03)
    # W whitens the covariance the simulation used: conj(W) Psi conj(W)^H ~ c I
    (psi,) = read(paths['truth'], 'truth/noise_covariance')
    m = W.conj() @ psi @ W.T
    m = m / np.real(np.trace(m)) * len(m)
    assert np.abs(m - np.eye(len(m))).max() < 0.25


def test_estimated_sensitivity_maps_match_the_true_ones(realistic):
    paths, pre, cfg, _ = realistic
    smaps, W, gcc = read(pre, 'smaps', 'W', 'GCC')
    truth, tissue = read(paths['truth'], 'truth/smaps', 'truth/tissues')
    from preprocess.coils import apply_gcc_image

    st = apply_gcc_image(apply_whitening(truth, W), gcc)
    num = np.abs((smaps.conj() * st).sum(-1))
    den = np.sqrt((np.abs(smaps) ** 2).sum(-1) * (np.abs(st) ** 2).sum(-1))
    inside = (tissue.sum(0) > 0.9) & (den > 0)
    assert inside.sum() > 300
    assert np.median(num[inside] / den[inside]) > 0.99


def test_estimated_field_map_matches_the_true_one(realistic):
    """On the deGRE's own grid, where the map is estimated. (On the EPI grid
    the comparison is dominated by preprocess/grid_resize.py's placement of
    the maps, docs/review-findings.md item 263: 6 mm in z at this test's 12 mm
    slices.)"""
    paths, pre, cfg, julia = realistic
    if not julia:
        pytest.skip('the B0 map needs julia')
    b0, mask = read(pre, 'degre/b0_map', 'degre/mask')
    (truth,) = read(paths['truth'], 'truth/degre/b0_map')
    inside = mask > 0
    assert inside.sum() > 300 and truth[inside].std() > 5  # a field worth mapping
    err = (b0 - truth)[inside]
    print(f'deGRE-grid field map: corr {np.corrcoef(b0[inside], truth[inside])[0, 1]:.4f}, '
          f'median |error| {np.median(np.abs(err)):.2f} Hz, truth std {truth[inside].std():.2f} Hz')
    assert np.corrcoef(b0[inside], truth[inside])[0, 1] > 0.9
    assert np.median(np.abs(err)) < 0.2 * truth[inside].std()
    # and it reaches the EPI grid, if not exactly in place
    b0_epi, mask_epi = read(pre, 'b0_map', 'b0_mask')
    (truth_epi,) = read(paths['truth'], 'truth/b0_map')
    m = mask_epi > 0
    assert np.corrcoef(b0_epi[m], truth_epi[m])[0, 1] > 0.7


def test_truth_file_scores_with_the_testbed_scorer(realistic, tmp_path):
    """The truth file follows recon/testbed.py's layout: handing the scorer the
    true series as a reconstruction gives an exact score."""

    paths, _, _, _ = realistic
    x0, amp_map, waves, rois = read(paths['truth'], 'truth/x0', 'truth/amp_map',
                                    'truth/waveforms', 'truth/roi_masks')
    with h5py.File(paths['truth'], 'r') as f:
        assert f.attrs['volume_tr'] > 0
        assert list(f['truth'].attrs['roi_names']) == ['block_occipital']
        amp = f['truth'].attrs['amp']
    assert rois.shape == (1, 90, 16, 12) and waves.shape == (1, 4)
    assert rois.sum() > 5 and amp > 0  # activation raises the signal (R2* falls)
    assert np.median(amp_map[rois[0]]) == pytest.approx(amp, rel=0.02)


def test_effects_can_be_switched_off(tmp_path):
    """b0_scale = 0 leaves a uniform field, noise = 0 a noise-free run with a
    unit-variance noise scan to whiten by."""
    scan_info, _ = make_scan_info(tmp_path / 'seq', n_frames=1, n_shots=3, r=8)
    cfg = small_cfg(b0_scale=0.0, noise=0.0, physio=PhysioConfig(scale=0.0), activation=False)
    paths = session.simulate_session(scan_info, str(tmp_path / 'out'), anatomy=anatomy(), cfg=cfg,
                                     device='cpu')
    b0, rois = read(paths['truth'], 'truth/b0_map', 'truth/roi_masks')
    assert not b0.any() and rois.shape[0] == 0
    noise = pre_utils.read_archive(paths['noise'])
    assert np.mean(np.abs(noise) ** 2) == pytest.approx(1.0, rel=0.1)
    cal = pre_utils.read_archive(paths['cal'])
    np.testing.assert_array_equal(cal[..., :8], cal[..., 8:16])


def test_frames_truncates_the_copied_scan_info(tmp_path):
    scan_info, _ = make_scan_info(tmp_path / 'seq', n_frames=3, n_shots=3, r=8)
    cfg = small_cfg(noise=0.0, physio=None, activation=False, b0_scale=0.0)
    paths = session.simulate_session(scan_info, str(tmp_path / 'out'), anatomy=anatomy(), cfg=cfg,
                                     frames=2, device='cpu')
    schedules, _ = pre_utils.load_schedules(paths['scan_info'])
    full, _ = pre_utils.load_schedules(scan_info)
    assert schedules.shape[0] == 2 and full.shape[0] == 3
    np.testing.assert_array_equal(schedules, full[:2])
    assert pre_utils.read_archive(paths['epi']).shape[-1] == 2 * 3 * 8
