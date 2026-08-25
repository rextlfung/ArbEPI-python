import numpy as np
import pytest
import scipy.io as sio

from sampling.external_mask import load_external_mask


def _write_mat(tmp_path, array, key='samp', filename='mask.mat'):
    path = tmp_path / filename
    sio.savemat(str(path), {key: array})
    return str(path)


def test_load_external_mask_2d(tmp_path):
    mask = np.zeros((12, 8), dtype=np.uint8)
    mask[3, 4] = 1
    mask[5, 1] = 1
    path = _write_mat(tmp_path, mask)

    loaded = load_external_mask(path, 12, 8)
    assert loaded.shape == (12, 8)
    assert loaded.dtype == bool
    assert np.array_equal(loaded, mask.astype(bool))


def test_load_external_mask_3d_time_resolved(tmp_path):
    mask = np.zeros((12, 8, 5), dtype=np.uint8)
    mask[0, 0, 2] = 1
    path = _write_mat(tmp_path, mask)

    loaded = load_external_mask(path, 12, 8)
    assert loaded.shape == (12, 8, 5)
    assert loaded.dtype == bool
    assert np.array_equal(loaded, mask.astype(bool))


def test_load_external_mask_wrong_key_raises(tmp_path):
    path = _write_mat(tmp_path, np.zeros((12, 8), dtype=np.uint8), key='mask')

    with pytest.raises(KeyError):
        load_external_mask(path, 12, 8, key='samp')


def test_load_external_mask_wrong_shape_raises(tmp_path):
    path = _write_mat(tmp_path, np.zeros((10, 8), dtype=np.uint8))

    with pytest.raises(ValueError):
        load_external_mask(path, 12, 8)
