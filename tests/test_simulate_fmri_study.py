"""simulate_fmri/study.py: repetitions of an experiment through preprocess/ and
recon/, each step skipped when its output is there."""

import os
import warnings
from dataclasses import replace

import h5py
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('snake')
pytest.importorskip('sigpy')
pytest.importorskip('mirtorch')

from params import load_params  # noqa: E402
from sample.gen_sampling_masks import resolve_omegas  # noqa: E402
from sequences.ArbEPI import generate_arbepi  # noqa: E402
from simulate_fmri import analysis, session, study  # noqa: E402
from simulate_fmri import phantom as phantoms  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings('ignore:Failed to convert value'),
    pytest.mark.filterwarnings('ignore:.*use of fork.*:DeprecationWarning'),
    pytest.mark.filterwarnings('ignore:write_output'),
]


def test_paths_keep_runs_and_reconstructions_apart(tmp_path):
    out = str(tmp_path)
    a, b = study.run_paths(out, 0), study.run_paths(out, 1)
    assert a['dir'].endswith('run1') and b['dir'].endswith('run2')
    assert a['preprocessed'].endswith(os.path.join('run1', 'recon', 'task_preprocessed.h5'))
    assert a['preprocessed_ok'] == a['preprocessed'] + '.ok'
    plain, b0 = (study.recon_path(out, 0, kw) for kw in study.RECONS.values())
    assert plain.endswith(os.path.join('recon', 'sense_wavelet-tv_hp3', 'task_recon.h5'))
    assert b0.endswith(os.path.join('recon', 'sense_wavelet-tv_b0_hp3', 'task_recon.h5'))


def test_repetitions_differ_in_noise_only_and_are_not_redone(tmp_path, monkeypatch):
    """Two repetitions of a small experiment: the same truth image and task,
    different noise; reconstructions land where recon_path says; a second call
    does no work; the analysis runs on the result."""
    seq = tmp_path / 'seq'
    p = replace(
        load_params(output_dir=str(seq)), Ny=16, Nz=12, ETL=8, Nshots=3, Nframes=6,
        sampling_method='caipi', R=8, Nx_degre=36, Ny_degre=36, Nz_degre=25,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        generate_arbepi(resolve_omegas(p), p)
    scan_info, out = str(seq / 'scan_info.mat'), str(tmp_path / 'study')
    cfg = session.SessionConfig(
        grid_factor=2, degre_grid_factor=2, n_coils=8, coils_per_ring=4, seed=3, b0_scale=0.0,
        activations=(session.Activation('occipital', 4, 4, onset=-5),),
    )
    recons = {'plain': dict(sigma1A=1.0, niters=3)}
    kw = dict(cfg=cfg, recons=recons, anatomy=phantoms.ellipsoid_anatomy((48, 48, 36), 4.0),
              keep_raw=1, device='cpu')
    # no field map to estimate here (b0_scale=0), and no julia needed for the test
    from preprocess import preprocess as pre

    real = pre.PreprocessConfig
    monkeypatch.setattr(pre, 'PreprocessConfig', lambda **k: real(estimate_b0=False, **k))
    runs = study.run_repetitions(scan_info, out, 2, **kw)

    assert [os.path.basename(r['dir']) for r in runs] == ['run1', 'run2']
    for k, run in enumerate(runs):
        assert os.path.exists(run['preprocessed']) and os.path.exists(run['preprocessed'] + '.ok')
        assert run['recons'] == {'plain': study.recon_path(out, k, recons['plain'])}
        with h5py.File(run['recons']['plain'], 'r') as f:
            assert f['X_recon'].shape == (90, 16, 12, 6)
    # the first run keeps its raw archives, the second does not
    assert os.path.isdir(study.run_paths(out, 0)['raw'])
    assert not os.path.exists(study.run_paths(out, 1)['raw'])

    t1, t2 = (analysis.load_truth(r['truth']) for r in runs)
    np.testing.assert_array_equal(t1.x0, t2.x0)  # the same brain and task ...
    np.testing.assert_array_equal(t1.response, t2.response)
    np.testing.assert_array_equal(t1.roi_masks, t2.roi_masks)
    cardiac = t1.mode_names.index('cardiac')
    assert not np.allclose(t1.mode_course[cardiac], t2.mode_course[cardiac])  # ... new physiology
    x1, x2 = (analysis.load_series(r['recons']['plain'], t1.brain) for r in runs)
    assert x1.shape == (int(t1.brain.sum()), 6) and not np.allclose(x1, x2)  # ... and new noise

    # nothing is redone: neither the simulation nor preprocess nor recon is called again
    def fail(*a, **k):
        raise AssertionError('redone')

    monkeypatch.setattr(study, 'simulate_session', fail)
    monkeypatch.setattr(pre, 'preprocess', fail)
    import recon.sense

    monkeypatch.setattr(recon.sense, 'main', fail)
    again = study.run_repetitions(scan_info, out, 2, **kw)
    assert again == runs
