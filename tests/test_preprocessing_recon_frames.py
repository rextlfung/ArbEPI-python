import os

import h5py
import numpy as np

from preprocessing.config import SeqParams, load_config, set_seq_paths
from preprocessing.recon_frames import recon_frames


def _sum_coils(data, smaps):
    # A trivial, picklable, module-level recon_fn: just sum over coils --
    # lets the test check that the right frames/shapes flow through
    # recon_frames without needing a real reconstruction algorithm.
    assert smaps.shape == data.shape
    return np.sum(data, axis=-1)


def _make_fixture(tmp_path, seqname, Nx, Ny, Nz, Nvcoils, Nframes_avail):
    recon_dir = tmp_path / 'recon'
    recon_dir.mkdir()

    ksp = (
        np.arange(Nx * Ny * Nz * Nvcoils * Nframes_avail, dtype=np.float64)
        .reshape(Nx, Ny, Nz, Nvcoils, Nframes_avail)
        .astype(np.complex64)
    )
    with h5py.File(recon_dir / f'{seqname}_epi_zf.h5', 'w') as f:
        f.create_dataset('ksp_epi_zf', data=ksp)

    smaps = np.ones((Nx, Ny, Nz, Nvcoils), dtype=np.complex64)
    with h5py.File(recon_dir / f'smaps_{seqname}_sigpy.h5', 'w') as f:
        f.create_dataset('smaps_raw', data=smaps)
        f.create_dataset('emap', data=np.ones((Nx, Ny, Nz)))
        f.create_dataset('smaps', data=smaps)
        f.attrs['Nvcoils'] = Nvcoils

    return ksp


def test_recon_frames_uses_cached_smaps_and_reconstructs_all_frames(tmp_path):
    seqname = 'testseq'
    Nx, Ny, Nz, Nvcoils, Nframes = 4, 3, 2, 5, 6
    ksp = _make_fixture(tmp_path, seqname, Nx, Ny, Nz, Nvcoils, Nframes)

    cfg = load_config(datdir=str(tmp_path), seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    seq_params = SeqParams(
        Nx=Nx, Ny=Ny, Nz=Nz, ETL=1, R=1, fov=(0.1, 0.1, 0.1),
        volume_tr=1.0, discard_duration=0.0,
        Nx_degre=Nx, Ny_degre=Ny, Nz_degre=Nz, fov_degre=(0.1, 0.1, 0.1),
    )

    # No GRE cache file exists -- if recon_frames tried the "estimate from
    # scratch" fallback instead of the cache it should find, this would
    # raise FileNotFoundError, so a clean run also confirms the cache path
    # (not the fallback) was taken.
    img, sp, runtime_s = recon_frames(cfg, paths, seq_params, _sum_coils)

    assert img.shape == (Nx, Ny, Nz, Nframes)
    assert sp['Nframes'] == Nframes
    assert sp['Nvcoils'] == Nvcoils
    assert runtime_s >= 0
    for frame in range(Nframes):
        np.testing.assert_allclose(img[:, :, :, frame], np.sum(ksp[:, :, :, :, frame], axis=-1))


def test_recon_frames_caps_at_cfg_nframes(tmp_path):
    seqname = 'testseq'
    Nx, Ny, Nz, Nvcoils, Nframes_avail = 4, 3, 2, 5, 6
    _make_fixture(tmp_path, seqname, Nx, Ny, Nz, Nvcoils, Nframes_avail)

    cfg = load_config(datdir=str(tmp_path), seqnames=[seqname], Nframes=3)
    paths = set_seq_paths(cfg, seqname)
    seq_params = SeqParams(
        Nx=Nx, Ny=Ny, Nz=Nz, ETL=1, R=1, fov=(0.1, 0.1, 0.1),
        volume_tr=1.0, discard_duration=0.0,
        Nx_degre=Nx, Ny_degre=Ny, Nz_degre=Nz, fov_degre=(0.1, 0.1, 0.1),
    )

    img, sp, _ = recon_frames(cfg, paths, seq_params, _sum_coils)
    assert img.shape == (Nx, Ny, Nz, 3)
    assert sp['Nframes'] == 3


def test_recon_frames_estimates_smaps_when_no_cache(tmp_path, monkeypatch):
    seqname = 'testseq'
    Nx, Ny, Nz, Nvcoils, Nframes = 4, 4, 4, 3, 2

    recon_dir = tmp_path / 'recon'
    recon_dir.mkdir()
    ksp = np.ones((Nx, Ny, Nz, Nvcoils, Nframes), dtype=np.complex64)
    with h5py.File(recon_dir / f'{seqname}_epi_zf.h5', 'w') as f:
        f.create_dataset('ksp_epi_zf', data=ksp)
    with h5py.File(recon_dir / f'{seqname}_gre.h5', 'w') as f:
        f.create_dataset('ksp_gre', data=np.ones((Nx, Ny, Nz, Nvcoils), dtype=np.complex64))

    calls = {}

    def fake_estimate_smaps(ksp_gre, **kwargs):
        calls['estimate_smaps'] = True
        return np.ones((Nx, Ny, Nz, Nvcoils), dtype=np.complex64), np.ones((Nx, Ny, Nz))

    def fake_process_smaps(smaps_raw, emap, fov_gre, fov, n_target, threshold_mask):
        calls['process_smaps'] = True
        return smaps_raw

    monkeypatch.setattr('preprocessing.recon_frames.estimate_smaps', fake_estimate_smaps)
    monkeypatch.setattr('preprocessing.recon_frames.process_smaps', fake_process_smaps)

    cfg = load_config(datdir=str(tmp_path), seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    seq_params = SeqParams(
        Nx=Nx, Ny=Ny, Nz=Nz, ETL=1, R=1, fov=(0.1, 0.1, 0.1),
        volume_tr=1.0, discard_duration=0.0,
        Nx_degre=Nx, Ny_degre=Ny, Nz_degre=Nz, fov_degre=(0.1, 0.1, 0.1),
    )

    recon_frames(cfg, paths, seq_params, _sum_coils)

    assert calls.get('estimate_smaps')
    assert calls.get('process_smaps')
    assert os.path.exists(recon_dir / f'smaps_{seqname}_sigpy.h5')
