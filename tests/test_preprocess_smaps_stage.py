"""preprocess()'s sensitivity-map stage with estimate_smaps=True
(docs/review-findings.md item 174): smaps_smooth_sigma_mm / zero_pad_z reach
process_smaps for the EPI-grid maps (and only the smoothing for the deGRE QA
maps), and the stored maps are unit-RSS, finite, on the right grids. Reuses
test_preprocess_pipeline.py's fake-archive dataset."""

import h5py
import numpy as np
import pytest

pytest.importorskip('sigpy')
pytest.importorskip('nibabel')

from preprocess import smaps as smaps_mod  # noqa: E402
from preprocess.preprocess import preprocess  # noqa: E402
from tests.test_preprocess_pipeline import (  # noqa: E402
    A_FIXED,
    NX,
    NXD,
    NY,
    NYD,
    NZ,
    NZD,
    SEQ,
    _cfg,
)
from tests.test_preprocess_pipeline import dataset as _dataset  # noqa: E402

dataset = _dataset  # re-export the fixture under its name


@pytest.fixture
def spy(monkeypatch):
    calls = []
    real = smaps_mod.process_smaps

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(smaps_mod, 'process_smaps', wrapper)
    return calls


def test_smoothing_and_zero_pad_are_threaded_to_process_smaps(dataset, spy):
    preprocess(_cfg(dataset, estimate_smaps=True, smaps_smooth_sigma_mm=4.5, zero_pad_z=True),
               SEQ, a=A_FIXED)
    assert len(spy) == 2
    (a_epi, k_epi), (a_deg, k_deg) = spy
    # EPI-grid maps: target grid is the EPI's, with both options passed on.
    assert tuple(a_epi[4]) == (NX, NY, NZ)
    assert k_epi['smooth_sigma_mm'] == 4.5
    assert k_epi['zero_pad_z'] is True
    # deGRE QA maps: stay on the deGRE grid, smoothing passed, no z padding.
    assert tuple(a_deg[4]) == (NXD, NYD, NZD)
    assert tuple(a_deg[2]) == tuple(a_deg[3])  # fov_gre == fov
    assert k_deg['smooth_sigma_mm'] == 4.5
    assert not k_deg.get('zero_pad_z', False)


def test_defaults_reach_process_smaps(dataset, spy):
    preprocess(_cfg(dataset, estimate_smaps=True), SEQ, a=A_FIXED)
    assert all(k['smooth_sigma_mm'] == 6.0 for _, k in spy)
    assert spy[0][1]['zero_pad_z'] is False


def test_estimate_smaps_false_skips_the_stage(dataset, spy):
    out = preprocess(_cfg(dataset, estimate_smaps=False), SEQ, a=A_FIXED)
    assert spy == []
    with h5py.File(out, 'r') as f:
        assert 'smaps' not in f and 'degre/smaps' not in f


def test_stored_smaps_are_unit_rss_finite_and_smoothing_changes_them(dataset):
    outs = {}
    for sigma in (0.0, 8.0):
        out = preprocess(
            _cfg(dataset, estimate_smaps=True, smaps_smooth_sigma_mm=sigma, keep_cache=True),
            SEQ, a=A_FIXED,
        )
        with h5py.File(out, 'r') as f:
            nvc = f['GCC'].shape[-1] if 'GCC' in f else None
            outs[sigma] = (f['smaps'][()], f['degre/smaps'][()], f['degre/emap'][()], nvc)

    for sigma, (sm, sm_deg, emap, _) in outs.items():
        assert sm.dtype == np.complex64 and sm_deg.dtype == np.complex64
        assert sm.shape[:3] == (NX, NY, NZ)
        assert sm_deg.shape[:3] == (NXD, NYD, NZD)
        assert emap.shape == (NXD, NYD, NZD)
        assert np.all(np.isfinite(sm)) and np.all(np.isfinite(sm_deg))
        rss = np.sqrt(np.sum(np.abs(sm) ** 2, axis=-1))
        support = rss > 0.5
        assert support.any()
        np.testing.assert_allclose(rss[support], 1.0, atol=1e-4)

    sm0, sm8 = outs[0.0][0], outs[8.0][0]
    assert not np.allclose(sm0, sm8)  # sigma actually has an effect
    # Smoothing makes the maps smoother (smaller voxel-to-voxel jumps).
    def roughness(s):
        return sum(np.abs(np.diff(s, axis=ax)).mean() for ax in range(3))

    assert roughness(sm8) < roughness(sm0)
