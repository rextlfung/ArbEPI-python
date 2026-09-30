"""End-to-end preprocess() on synthetic data, with GE's ScanArchive reader
replaced by in-memory fake archives (no GERecon needed)."""

import os
import shutil

import h5py
import numpy as np
import pytest

pytest.importorskip('sigpy')
pytest.importorskip('nibabel')

from preprocess import utils  # noqa: E402
from preprocess.coils import apply_gcc_kspace  # noqa: E402
from preprocess.preprocess import (  # noqa: E402
    PreprocessConfig,
    apply_delay,
    grid_noise,
    preprocess,
    process_epi_frame,
    set_seq_paths,
)

SEQ = 'synth'
NX, NY, NZ, ETL, NSHOTS, NFRAMES, NFID, NC = 12, 8, 8, 4, 6, 3, 16, 6
NXD, NYD, NZD = 16, 16, 12  # deGRE grid
FOV, FOV_DEGRE = (0.12, 0.12, 0.06), (0.12, 0.12, 0.08)
TE_DEGRE = (0.003, 0.0052)
A_FIXED = np.array([0.2, 1.5])
D_TRUE = 0.35  # readout delay (samples) built into the synthetic calibration scan


def _crandn(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _fft3c(x):
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)


def _cal_scan(rng, kxo):
    """Blip-free calibration echo trains [NFID, NC, ETL * 5] of a smooth 1D
    object, read out at the kx positions of a D_TRUE readout delay, with a
    constant (delay-free) odd/even phase on the even echoes."""
    import sigpy

    kxo_t, kxe_t = apply_delay(kxo / 100, kxo[::-1] / 100, NFID, D_TRUE)
    xp = np.arange(NX) - NX / 2
    img = np.exp(-((xp / (NX / 3)) ** 2))[:, None] * _crandn(rng, NC)[None, :]  # [NX, NC]

    def readout(kx, phase):  # -> [NFID, NC]
        return sigpy.nufft(img.T * phase, (kx * FOV[0] * 100)[:, None]).T

    trains = [readout(kxo_t, 1) if e % 2 == 0 else readout(kxe_t, np.exp(0.3j))
              for _ in range(5) for e in range(ETL)]
    cal = np.stack(trains, axis=-1)
    return cal + 1e-3 * _crandn(rng, *cal.shape)


def _schedules(rng):
    """[Nframes, Nshots, ETL, 3]: 1-based (ky, kz) + echo time. Every frame
    samples the central 4x4 block plus 8 random other locations."""
    calib = [(y, z) for y in range(2, 6) for z in range(2, 6)]
    others = [(y, z) for y in range(NY) for z in range(NZ) if (y, z) not in calib]
    out = np.zeros((NFRAMES, NSHOTS, ETL, 3))
    for t in range(NFRAMES):
        pick = rng.choice(len(others), NSHOTS * ETL - len(calib), replace=False)
        locs = np.array(calib + [others[i] for i in pick])[rng.permutation(NSHOTS * ETL)]
        out[t, :, :, :2] = locs.reshape(NSHOTS, ETL, 2) + 1
        out[t, :, :, 2] = 0.01 + 0.001 * np.arange(ETL)
    return out


@pytest.fixture
def dataset(tmp_path, monkeypatch, request):
    """A datdir with scan_info.mat, placeholder archive files, and the fake
    archive contents patched into preprocess.utils. Parametrize indirectly
    with a deGRE FOV to override FOV_DEGRE."""
    fov_degre = getattr(request, 'param', FOV_DEGRE)
    rng = np.random.default_rng(0)
    seqdir = tmp_path / 'seqs' / SEQ
    seqdir.mkdir(parents=True)
    kmax = NX / (2 * FOV[0])  # cycles/m
    kxo = np.linspace(-kmax, kmax, NFID)
    fields = {
        'Nx': NX, 'Ny': NY, 'Nz': NZ, 'ETL': ETL, 'R': 2.0, 'fov': np.array(FOV),
        'volume_tr': 1.0, 'discard_duration': 0.0,
        'Nx_degre': NXD, 'Ny_degre': NYD, 'Nz_degre': NZD, 'fov_degre': np.array(fov_degre),
        'n_echoes_degre': 2, 'TE_degre': np.array(TE_DEGRE),
        'kxo': kxo, 'kxe': kxo[::-1].copy(), 'schedules': _schedules(rng),
    }
    with h5py.File(seqdir / 'scan_info.mat', 'w') as f:
        for k, v in fields.items():
            f.create_dataset(k, data=np.asarray(v).transpose())

    mix = _crandn(rng, NC, NC) * 0.3 + np.eye(NC)  # correlated, unequal coil noise
    noise = (_crandn(rng, NFID, 200, NC) @ mix.T).transpose(0, 2, 1)
    cal = _cal_scan(rng, kxo)
    epi_shots = [_crandn(rng, NFID, NC) for _ in range(NFRAMES * NSHOTS * ETL)]

    # deGRE: a sphere seen by smooth coils, with a small field map.
    xx, yy, zz = np.meshgrid(*(np.linspace(-1, 1, n) for n in (NXD, NYD, NZD)), indexing='ij')
    obj = (xx**2 + yy**2 + zz**2 < 0.6).astype(float)
    centers = rng.uniform(-1, 1, size=(NC, 3))
    coils = np.stack([np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2))
                      for c in centers], axis=-1)
    ksp = np.stack([_fft3c(obj[..., None] * coils * np.exp(2j * np.pi * 20 * xx[..., None] * te))
                    for te in TE_DEGRE], axis=3)  # [x, y, z, echo, coil]
    ksp = ksp + 1e-3 * _crandn(rng, *ksp.shape)
    gre = ksp.transpose(0, 4, 3, 1, 2).reshape(NXD, NC, -1, order='F')
    gre = np.concatenate([np.zeros((NXD, NC, NYD * 2)), gre], axis=2)  # iZ=0 cal block first

    scanarchives = tmp_path / 'scanarchives'
    scanarchives.mkdir()
    archives = {
        str(scanarchives / f'{SEQ}_noise.h5'): noise,
        str(scanarchives / f'{SEQ}_cal.h5'): cal,
        str(scanarchives / 'gre.h5'): gre,
    }
    for path in list(archives) + [str(scanarchives / f'{SEQ}_epi.h5')]:
        open(path, 'w').close()

    state = {'epi_reads': 0, 'fail_after': None}

    class FakeReader:
        def __init__(self, filename):
            self.i = 0

        def next_frame(self):
            if state['fail_after'] is not None and state['epi_reads'] >= state['fail_after']:
                raise RuntimeError('simulated crash')
            if self.i >= len(epi_shots):
                raise StopIteration
            self.i += 1
            state['epi_reads'] += 1
            return epi_shots[self.i - 1]

    monkeypatch.setattr(utils, 'read_archive', lambda fn: archives[str(fn)])
    monkeypatch.setattr(utils, 'ArchiveReader', FakeReader)
    return {
        'datdir': str(tmp_path), 'noise': noise, 'cal': cal, 'epi_shots': epi_shots,
        'schedules': fields['schedules'], 'kxo': kxo, 'state': state,
    }


def _cfg(ds, **kw):
    base = dict(datdir=ds['datdir'], seqnames=[SEQ], estimate_b0=False)
    return PreprocessConfig(**{**base, **kw})


def _trajectories(ds, delay):
    return apply_delay(ds['kxo'] / 100, ds['kxo'][::-1] / 100, NFID, delay)


def _reference(ds, W, GCC, delay):
    """Each frame whitened, gridded and scattered on its own, then compressed
    with GCC (if given): (k-space, noise variance)."""
    kxo, kxe = _trajectories(ds, delay)
    sched = ds['schedules'][..., :2].astype(int) - 1
    spf = NSHOTS * ETL
    frames = [
        process_epi_frame(
            np.stack(ds['epi_shots'][t * spf:(t + 1) * spf], axis=-1), W, kxo, kxe, A_FIXED,
            sched[t], NX, NY, NZ, ETL, NSHOTS, FOV[0] * 100,
        )
        for t in range(NFRAMES)
    ]
    noise = grid_noise(ds['noise'], W, kxo, kxe, A_FIXED, NX, ETL, FOV[0] * 100)
    if GCC is not None:
        frames = [apply_gcc_kspace(k, GCC) for k in frames]
        noise = apply_gcc_kspace(noise, GCC)
    return np.stack(frames, axis=-1), float(np.mean(np.abs(noise) ** 2))


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_gcc_output_is_gcc_applied_to_the_whitened_gridded_data(dataset):
    out = preprocess(_cfg(dataset, estimate_smaps=False), SEQ, a=A_FIXED)
    with h5py.File(out, 'r') as f:
        ksp, GCC, W = f['ksp_epi_zf'][()], f['GCC'][()], f['W'][()]
        nv = f.attrs['Nvcoils']
        assert f.attrs['whitened'] and f.attrs['coil_compressed']
        assert f.attrs['Nvcoils_source'] == 'energy' and f.attrs['cc_energy_kept'] >= 0.99
        np.testing.assert_allclose(f.attrs['oephase_a'], A_FIXED)
        noise_var, delay = f.attrs['noise_var'], f.attrs['delay']
    assert GCC.shape == (NX, nv, NC) and ksp.shape == (NX, NY, NZ, nv, NFRAMES)
    ref_ksp, ref_noise_var = _reference(dataset, W, GCC, delay)
    assert _rel(ksp, ref_ksp) < 1e-5
    assert noise_var == pytest.approx(ref_noise_var, rel=1e-5)


def test_nvcoils_sets_the_exact_virtual_coil_count(dataset):
    out = preprocess(_cfg(dataset, Nvcoils=3), SEQ, a=A_FIXED)
    with h5py.File(out, 'r') as f:
        assert f['ksp_epi_zf'].shape[3] == 3
        assert f['smaps'].shape == (NX, NY, NZ, 3)
        assert f['GCC'].shape == (NX, 3, NC)
        assert f.attrs['Nvcoils'] == 3 and f.attrs['Nvcoils_source'] == 'user'
        # compressed maps are unit-RSS inside their support
        rss = np.sqrt(np.sum(np.abs(f['smaps'][()]) ** 2, axis=-1))
        assert np.allclose(rss[rss > 0], 1, atol=1e-4)
        assert f['r2star_map'].shape == (NX, NY, NZ)
        assert 'degre/img_echoes' in f and 'ksp_calib' in f


@pytest.mark.parametrize('nv', [0, NC + 1])
def test_invalid_nvcoils_raises(dataset, nv):
    with pytest.raises(ValueError, match='Nvcoils'):
        preprocess(_cfg(dataset, Nvcoils=nv, estimate_smaps=False), SEQ)


def test_no_compression_promotes_the_cache_and_keeps_all_coils(dataset):
    cfg = _cfg(dataset, compress=False, estimate_smaps=False)
    out = preprocess(cfg, SEQ, a=A_FIXED)
    paths = set_seq_paths(cfg, SEQ)
    assert not os.path.exists(paths.cache)
    with h5py.File(out, 'r') as f:
        W, delay = f['W'][()], f.attrs['delay']
        ksp = f['ksp_epi_zf'][()]
        assert not f.attrs['coil_compressed'] and 'GCC' not in f
        assert 'complete' not in f.attrs and 'noise_gridded' not in f
    ref, _ = _reference(dataset, W, None, delay)
    assert ksp.shape[3] == NC and _rel(ksp, ref) < 1e-5


def test_missing_noise_scan_is_recorded_as_unwhitened(dataset):
    os.remove(os.path.join(dataset['datdir'], 'scanarchives', f'{SEQ}_noise.h5'))
    with pytest.warns(UserWarning, match='NOT whitened'):
        out = preprocess(_cfg(dataset, estimate_smaps=False), SEQ, a=A_FIXED)
    with h5py.File(out, 'r') as f:
        assert not f.attrs['whitened']
        assert 'noise_var' not in f.attrs
        np.testing.assert_array_equal(f['W'][()], np.eye(NC))


def test_calibration_region_is_extracted(dataset):
    out = preprocess(_cfg(dataset, estimate_smaps=False), SEQ, a=A_FIXED)
    with h5py.File(out, 'r') as f:
        calib = f['ksp_calib']
        assert tuple(calib.attrs['calib_y_range']) == (2, 6)
        assert tuple(calib.attrs['calib_z_range']) == (2, 6)
        np.testing.assert_array_equal(calib[()], f['ksp_epi_zf'][:, 2:6, 2:6])
        assert np.all(calib[()] != 0)


def test_rerun_with_a_kept_cache_skips_stage_a(dataset):
    cfg = _cfg(dataset, keep_cache=True, estimate_smaps=False)
    preprocess(cfg, SEQ, a=A_FIXED)
    reads = dataset['state']['epi_reads']
    out = preprocess(_cfg(dataset, keep_cache=True, Nvcoils=2, estimate_smaps=False), SEQ)
    assert dataset['state']['epi_reads'] == reads  # the EPI archive was not read again
    with h5py.File(out, 'r') as f:
        assert f['ksp_epi_zf'].shape[3] == 2
        np.testing.assert_allclose(f.attrs['oephase_a'], A_FIXED)  # the cache's a


def test_resume_after_a_crash_matches_an_uninterrupted_run(dataset):
    cfg = _cfg(dataset, estimate_smaps=False)
    state = dataset['state']
    state['fail_after'] = NSHOTS * ETL + 3  # dies partway through frame 2
    with pytest.raises(RuntimeError, match='simulated crash'):
        preprocess(cfg, SEQ, a=A_FIXED)
    with h5py.File(set_seq_paths(cfg, SEQ).cache, 'r') as f:
        assert f.attrs['last_completed_frame'] == 0 and not f.attrs['complete']
    state['fail_after'] = None
    out = preprocess(cfg, SEQ, a=A_FIXED)
    with h5py.File(out, 'r') as f:
        ksp, GCC, W = f['ksp_epi_zf'][()], f['GCC'][()], f['W'][()]
        delay = f.attrs['delay']
    ref, _ = _reference(dataset, W, GCC, delay)
    assert _rel(ksp, ref) < 1e-5


def test_readout_delay_is_calibrated_from_the_cal_scan(dataset):
    """No delay is configured: Stage A sweeps it on the calibration scan, finds
    the one the synthetic readouts were made with, and records the sweep."""
    out = preprocess(_cfg(dataset, estimate_smaps=False), SEQ)
    with h5py.File(out, 'r') as f:
        assert f.attrs['delay'] == pytest.approx(D_TRUE, abs=0.051)
        sweep = {k: f['delay_sweep'][k][()] for k in ('delay', 'a1', 'a2', 'wrap_count')}
        a = f.attrs['oephase_a']
    assert len({len(v) for v in sweep.values()}) == 1
    best = np.argmin(np.abs(sweep['delay'] - D_TRUE))
    assert sweep['wrap_count'][best] == 0 and sweep['wrap_count'].max() > 0
    # at the calibrated delay only the constant odd/even phase is left
    assert a[0] == pytest.approx(0.3, abs=0.05) and abs(a[1]) < 0.3


def test_cache_left_by_a_failed_start_is_replaced(dataset):
    """A run that fails during calibration leaves a cache file with no
    attributes; the next run must start over instead of failing on it."""
    cfg = _cfg(dataset, estimate_smaps=False)
    cache = set_seq_paths(cfg, SEQ).cache
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with h5py.File(cache, 'w'):
        pass
    out = preprocess(cfg, SEQ, a=A_FIXED)
    with h5py.File(out, 'r') as f:
        ksp, GCC, W = f['ksp_epi_zf'][()], f['GCC'][()], f['W'][()]
        delay = f.attrs['delay']
    ref, _ = _reference(dataset, W, GCC, delay)
    assert _rel(ksp, ref) < 1e-5


@pytest.mark.skipif(shutil.which('julia') is None, reason='julia not on PATH')
def test_full_pipeline_with_b0(dataset):
    out = preprocess(_cfg(dataset, estimate_b0=True), SEQ, a=A_FIXED)
    with h5py.File(out, 'r') as f:
        assert f['b0_map'].shape == (NX, NY, NZ)
        assert f['b0_mask'].dtype == bool
        assert f['degre/b0_map'].shape == (NXD, NYD, NZD)
        assert {'smaps', 'r2star_map', 'ksp_calib', 'omegas', 'echo_times'} <= set(f.keys())
    paths = utils.plot_gre_b0_diagnostics(out)
    assert all(os.path.exists(p) for p in paths)


@pytest.mark.parametrize('dataset', [(0.15, 0.135, 0.08)], indirect=True)
def test_degre_with_a_larger_xy_fov_than_the_epi(dataset):
    """The deGRE's x/y FOV may exceed the EPI's (params.py rounds it up to
    whole deGRE voxels): the maps and the GCC matrices land on the EPI grid,
    and the output k-space is still the GCC-compressed gridded data."""
    out = preprocess(_cfg(dataset), SEQ, a=A_FIXED)
    with h5py.File(out, 'r') as f:
        ksp, GCC, W, smaps = f['ksp_epi_zf'][()], f['GCC'][()], f['W'][()], f['smaps'][()]
        delay = f.attrs['delay']
    assert GCC.shape[0] == NX
    assert smaps.shape[:3] == (NX, NY, NZ) and np.all(np.isfinite(smaps))
    ref, _ = _reference(dataset, W, GCC, delay)
    assert _rel(ksp, ref) < 1e-5
