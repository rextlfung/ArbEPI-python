import h5py
import numpy as np
import pytest

from preprocessing.config import (
    PreprocessingConfig,
    SeqPaths,
    load_config,
    load_seq_params,
    seq_delay,
    set_seq_paths,
)


def test_load_config_defaults_and_overrides():
    cfg = load_config(datdir='/data/', seqnames=['a', 'b'], cc_energy_thresh=0.8)
    assert cfg.datdir == '/data/'
    assert cfg.seqnames == ['a', 'b']
    assert cfg.fn_gre == '/data/scanarchives/gre.h5'
    assert cfg.cc_energy_thresh == 0.8
    assert cfg.do_sense is True  # untouched default


def test_load_config_rejects_unknown_override():
    with pytest.raises(TypeError):
        load_config(datdir='/data/', seqnames=['a'], not_a_real_field=1)


def test_seq_delay_scalar_and_dict():
    cfg_scalar = load_config(datdir='/data/', seqnames=['a'], delay=-1.5)
    assert seq_delay(cfg_scalar, 'a') == -1.5

    cfg_map = load_config(datdir='/data/', seqnames=['a', 'b'], delay={'a': -1.0, 'b': 2.0})
    assert seq_delay(cfg_map, 'a') == -1.0
    assert seq_delay(cfg_map, 'b') == 2.0


def test_set_seq_paths_builds_expected_layout():
    cfg = PreprocessingConfig(
        datdir='/data/', seqnames=['caipi'], fn_gre='/data/scanarchives/gre.h5'
    )
    paths = set_seq_paths(cfg, 'caipi')
    assert paths.seqdir == '/data/seqs/caipi'
    assert paths.scan_info == '/data/seqs/caipi/scan_info.mat'
    assert paths.cal == '/data/scanarchives/caipi_cal.h5'
    assert paths.noise == '/data/scanarchives/caipi_noise.h5'
    assert paths.epi == '/data/scanarchives/caipi_epi.h5'
    assert paths.recon == '/data/recon/caipi_epi_zf.h5'  # .h5, not .mat -- see SeqPaths docstring


def test_load_seq_params_round_trips_a_scan_info_fixture(tmp_path):
    path = tmp_path / 'scan_info.mat'
    fields = {
        'Nx': 240, 'Ny': 240, 'Nz': 45, 'ETL': 60, 'R': 9,
        'fov': np.array([0.216, 0.216, 0.0405]),
        'volume_tr': 2.0, 'discard_duration': 0.0,
        'Nx_degre': 108, 'Ny_degre': 108, 'Nz_degre': 108,
        'fov_degre': np.array([0.216, 0.216, 0.216]),
        'n_echoes_degre': 2,
        'TE_degre': np.array([0.005, 0.01]),
    }
    with h5py.File(path, 'w') as f:
        for k, v in fields.items():
            f.create_dataset(k, data=np.asarray(v).transpose())

    paths = SeqPaths(
        seqname='x', seqdir=str(tmp_path), scan_info=str(path),
        cal='', noise='', epi='', recon='',
    )
    sp = load_seq_params(paths)
    assert sp.Nx == 240 and sp.Ny == 240 and sp.Nz == 45
    assert sp.ETL == 60 and sp.R == 9
    assert sp.fov == pytest.approx((0.216, 0.216, 0.0405))
    assert sp.volume_tr == pytest.approx(2.0)
    assert sp.Nx_degre == 108
    assert sp.fov_degre == pytest.approx((0.216, 0.216, 0.216))
    assert sp.n_echoes_degre == 2
    assert sp.TE_degre == pytest.approx((0.005, 0.01))


def test_load_seq_params_defaults_degre_echo_fields_for_pre_dual_echo_snapshot(tmp_path):
    """Scan-scalar snapshots written before the dual-echo deGRE upgrade have
    neither n_echoes_degre nor TE_degre -- those acquisitions were
    genuinely single-echo, so load_seq_params must default rather than
    raising KeyError on a durable, non-regeneratable per-acquisition data
    record."""
    path = tmp_path / 'scan_info.mat'
    fields = {
        'Nx': 240, 'Ny': 240, 'Nz': 45, 'ETL': 60, 'R': 9,
        'fov': np.array([0.216, 0.216, 0.0405]),
        'volume_tr': 2.0, 'discard_duration': 0.0,
        'Nx_degre': 108, 'Ny_degre': 108, 'Nz_degre': 108,
        'fov_degre': np.array([0.216, 0.216, 0.216]),
    }
    with h5py.File(path, 'w') as f:
        for k, v in fields.items():
            f.create_dataset(k, data=np.asarray(v).transpose())

    paths = SeqPaths(
        seqname='x', seqdir=str(tmp_path), scan_info=str(path),
        cal='', noise='', epi='', recon='',
    )
    sp = load_seq_params(paths)
    assert sp.n_echoes_degre == 1
    assert sp.TE_degre is None
