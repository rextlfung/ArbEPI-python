import json

import h5py
import numpy as np
import pytest

from preprocess import utils
from preprocess.utils import (
    ift3c,
    load_seq_params,
    matlab_round,
    read_mat,
    read_mat_array,
    rss_echo_images,
    save_recon_nifti,
)


def _write_hdf5storage_style(path, arrays: dict):
    """Write datasets axis-reversed, as hdf5storage does, without depending on
    hdf5storage (not in the preprocessing venv)."""
    with h5py.File(path, 'w') as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=np.asarray(arr).transpose())


# --- hdf5storage .mat ---------------------------------------------------------


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


# --- scan_info.mat ------------------------------------------------------------

_SCAN_FIELDS = {
    'Nx': 240, 'Ny': 240, 'Nz': 45, 'ETL': 60, 'R': 9,
    'fov': np.array([0.216, 0.216, 0.0405]),
    'volume_tr': 2.0, 'discard_duration': 0.0,
    'Nx_degre': 108, 'Ny_degre': 108, 'Nz_degre': 108,
    'fov_degre': np.array([0.216, 0.216, 0.216]),
}


def test_load_seq_params_round_trips_a_scan_info_fixture(tmp_path):
    path = tmp_path / 'scan_info.mat'
    _write_hdf5storage_style(
        path, {**_SCAN_FIELDS, 'n_echoes_degre': 2, 'TE_degre': np.array([0.005, 0.01])}
    )
    sp = load_seq_params(str(path))
    assert sp.Nx == 240 and sp.Ny == 240 and sp.Nz == 45
    assert sp.ETL == 60 and sp.R == 9
    assert sp.fov == pytest.approx((0.216, 0.216, 0.0405))
    assert sp.volume_tr == pytest.approx(2.0)
    assert sp.Nx_degre == 108
    assert sp.fov_degre == pytest.approx((0.216, 0.216, 0.216))
    assert sp.n_echoes_degre == 2
    assert sp.TE_degre == pytest.approx((0.005, 0.01))


def test_load_seq_params_defaults_degre_echo_fields_for_pre_dual_echo_snapshot(tmp_path):
    """Snapshots from before the dual-echo deGRE have no n_echoes_degre/TE_degre;
    those acquisitions were single-echo, so default rather than raise."""
    path = tmp_path / 'scan_info.mat'
    _write_hdf5storage_style(path, _SCAN_FIELDS)
    sp = load_seq_params(str(path))
    assert sp.n_echoes_degre == 1
    assert sp.TE_degre is None


# --- NIfTI --------------------------------------------------------------------


def test_save_recon_nifti_writes_magnitude_with_fov_derived_spacing(tmp_path):
    nib = pytest.importorskip('nibabel')
    fn_base = str(tmp_path / 'seq_recon_rss')
    img = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(np.complex64) * (1 + 1j)
    fov = (0.2, 0.3, 0.4)  # m
    save_recon_nifti(fn_base, img, fov=fov, seqname='seq', runtime_s=1.5)
    nii = nib.load(f'{fn_base}.nii.gz')
    np.testing.assert_allclose(nii.get_fdata(), np.abs(img), atol=1e-3)
    expected_voxel_mm = [1000.0 * fov[axis] / img.shape[axis] for axis in range(3)]
    np.testing.assert_allclose(np.diag(nii.affine)[:3], expected_voxel_mm)


def test_save_recon_nifti_writes_json_sidecar_with_attrs(tmp_path):
    pytest.importorskip('nibabel')
    fn_base = str(tmp_path / 'seq_recon_rss')
    save_recon_nifti(
        fn_base, np.zeros((2, 2, 2, 1)), fov=np.array([0.1, 0.1, 0.1]), seqname='seq',
        num_iter=np.int64(50),
    )
    with open(f'{fn_base}.json') as f:
        attrs = json.load(f)
    assert attrs['seqname'] == 'seq'
    assert attrs['num_iter'] == 50
    assert attrs['fov'] == [0.1, 0.1, 0.1]


# --- numerics -----------------------------------------------------------------


def test_matlab_round_rounds_half_away_from_zero():
    # Nx/4 with Nx = 90 (a real value here) is a .5 tie: round() gives 22, MATLAB 23.
    assert matlab_round(90 / 4) == 23
    assert matlab_round(3 * 90 / 4) == 68
    assert matlab_round(2.4) == 2
    assert matlab_round(2.5) == 3
    assert matlab_round(-2.5) == -3


def test_ift3c_inverts_centered_forward_fft():
    # Odd sizes on purpose: the fftshift-on-both-sides spelling this replaced
    # only agrees with the standard centered pairing on even axes.
    rng = np.random.default_rng(0)
    shape = (7, 8, 5, 3)
    img_true = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    axes = (0, 1, 2)
    ksp = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img_true, axes=axes), axes=axes), axes=axes)
    np.testing.assert_allclose(ift3c(ksp), img_true, atol=1e-10)


def test_rss_echo_images_is_per_echo_root_sum_of_squares():
    rng = np.random.default_rng(1)
    img = rng.standard_normal((4, 5, 6, 2, 3)) + 1j * rng.standard_normal((4, 5, 6, 2, 3))
    axes = (0, 1, 2)
    ksp = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img, axes=axes), axes=axes), axes=axes)
    np.testing.assert_allclose(
        rss_echo_images(ksp), np.sqrt(np.sum(np.abs(img) ** 2, axis=-1)), atol=1e-10
    )


# ---------------------------------------------------------------------------
# Simulated archives
# ---------------------------------------------------------------------------


def test_simulated_archive_round_trips_through_the_archive_reader(tmp_path):
    """simulate_fmri/ writes readouts [Nacq, Ncoils, Nfid]; ArchiveReader hands
    them back one [Nfid, Ncoils] shot at a time, in order, without GERecon."""
    rng = np.random.default_rng(0)
    readouts = (rng.normal(size=(11, 3, 5)) + 1j * rng.normal(size=(11, 3, 5))).astype(np.complex64)
    fn = str(tmp_path / 'sim.h5')
    utils.write_simulated_archive(fn, readouts, scan='test')
    assert utils.is_simulated_archive(fn)

    reader = utils.ArchiveReader(fn)
    assert reader.metadata()['scan'] == 'test'
    shots = list(reader)
    assert len(shots) == 11 and shots[0].shape == (5, 3) and shots[0].dtype == np.complex64
    for i, shot in enumerate(shots):
        np.testing.assert_array_equal(shot, readouts[i].T)
    with pytest.raises(StopIteration):
        reader.next_frame()
    np.testing.assert_array_equal(utils.read_archive(fn), readouts.transpose(2, 1, 0))


def test_simulated_archive_streams_across_block_boundaries(tmp_path):
    readouts = np.arange(10 * 2 * 3, dtype=np.float32).reshape(10, 2, 3).astype(np.complex64)
    fn = str(tmp_path / 'sim.h5')
    with utils.create_simulated_archive(fn, 10, 2, 3) as f:
        f[utils.SIM_DATASET][3::2] = readouts[3::2]  # filled out of order, as the simulator does
        f[utils.SIM_DATASET][:3] = readouts[:3]
        f[utils.SIM_DATASET][4::2] = readouts[4::2]
    archive = utils._SimulatedArchive(fn, block=4)
    got = np.stack([archive.NextFrame() for _ in range(10)])
    np.testing.assert_array_equal(got, readouts.transpose(0, 2, 1))
    with pytest.raises(RuntimeError, match='No next frame available'):
        archive.NextFrame()


def test_other_hdf5_files_are_not_taken_for_simulated_archives(tmp_path):
    """A real ScanArchive is HDF5 too: only the marker makes a file simulated."""
    plain = str(tmp_path / 'plain.h5')
    with h5py.File(plain, 'w') as f:
        f['readouts'] = np.zeros((2, 2, 2))
    assert not utils.is_simulated_archive(plain)
    assert not utils.is_simulated_archive(str(tmp_path / 'missing.h5'))
    text = tmp_path / 'not_hdf5.h5'
    text.write_text('hello')
    assert not utils.is_simulated_archive(str(text))
