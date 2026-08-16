import h5py
import numpy as np

from preprocessing.matio import read_mat, read_mat_array


def _write_hdf5storage_style(path, arrays: dict):
    """Write datasets axis-reversed, mimicking hdf5storage's on-disk
    convention (see matio.py's docstring) -- without depending on
    hdf5storage itself, which isn't in the preprocessing venv."""
    with h5py.File(path, 'w') as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=np.asarray(arr).transpose())


def test_read_mat_array_recovers_logical_shape_and_values(tmp_path):
    path = tmp_path / 'fixture.mat'
    schedules = np.arange(30 * 20 * 60 * 2).reshape(30, 20, 60, 2)
    _write_hdf5storage_style(path, {'schedules': schedules})

    with h5py.File(path, 'r') as f:
        recovered = read_mat_array(f, 'schedules')

    assert recovered.shape == schedules.shape
    np.testing.assert_array_equal(recovered, schedules)


def test_read_mat_array_is_a_noop_on_vector_values(tmp_path):
    path = tmp_path / 'fixture.mat'
    fov = np.array([0.216, 0.216, 0.0405])
    _write_hdf5storage_style(path, {'fov': fov})

    with h5py.File(path, 'r') as f:
        recovered = read_mat_array(f, 'fov')

    np.testing.assert_array_equal(recovered.ravel(), fov)


def test_read_mat_reads_requested_and_all_datasets(tmp_path):
    path = tmp_path / 'fixture.mat'
    a = np.arange(6).reshape(2, 3)
    b = np.arange(12).reshape(3, 4)
    _write_hdf5storage_style(path, {'a': a, 'b': b})

    only_a = read_mat(str(path), ['a'])
    assert set(only_a.keys()) == {'a'}
    np.testing.assert_array_equal(only_a['a'], a)

    everything = read_mat(str(path))
    assert set(everything.keys()) == {'a', 'b'}
    np.testing.assert_array_equal(everything['b'], b)
