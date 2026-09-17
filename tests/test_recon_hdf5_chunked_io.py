"""recon/hdf5_chunked_io.py's read_frames_cropped: both branches (chunked
one frame per chunk vs. not chunked along the last axis) x both modes
(spatial_slices=None vs. given) against a location-encoding synthetic
dataset, in the same style as test_preprocessing_preprocess.py's
scatter_frame test -- each element is given a value encoding its own
(x, y, z, c, t) index, so a wrong crop/reshape is caught directly rather
than only by a shape check. No torch/mirtorch needed (this module is
pure h5py/numpy, see its own docstring for why), so this test collects
in the base test env.
"""

import h5py
import numpy as np

from recon.hdf5_chunked_io import read_frames_cropped

NX, NY, NZ, NC, NT = 6, 5, 4, 3, 7


def _encode(x, y, z, c, t):
    return ((((x * NY + y) * NZ + z) * NC + c) * NT + t).astype(np.complex64)


def _make_dataset(fn, key, chunked: bool):
    x, y, z, c, t = np.meshgrid(
        np.arange(NX), np.arange(NY), np.arange(NZ), np.arange(NC), np.arange(NT), indexing="ij"
    )
    full = _encode(x, y, z, c, t)
    with h5py.File(fn, "w") as f:
        chunks = (NX, NY, NZ, NC, 1) if chunked else None
        f.create_dataset(key, data=full, chunks=chunks)
    return full


def test_read_frames_cropped_no_crop_matches_full_array_when_chunked(tmp_path):
    fn = str(tmp_path / "ksp.h5")
    full = _make_dataset(fn, "ksp_epi_zf", chunked=True)
    out = read_frames_cropped(fn, "ksp_epi_zf")
    np.testing.assert_array_equal(out, full)


def test_read_frames_cropped_no_crop_matches_full_array_when_not_chunked(tmp_path):
    fn = str(tmp_path / "ksp.h5")
    full = _make_dataset(fn, "ksp_epi_zf", chunked=False)
    out = read_frames_cropped(fn, "ksp_epi_zf")
    np.testing.assert_array_equal(out, full)


def test_read_frames_cropped_crops_correctly_when_chunked(tmp_path):
    fn = str(tmp_path / "ksp.h5")
    full = _make_dataset(fn, "ksp_epi_zf", chunked=True)
    xs, ys, zs = slice(1, 4), slice(0, 3), slice(2, 4)
    out = read_frames_cropped(fn, "ksp_epi_zf", spatial_slices=(xs, ys, zs))
    expected = full[xs, ys, zs]
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out, expected)


def test_read_frames_cropped_crops_correctly_when_not_chunked(tmp_path):
    fn = str(tmp_path / "ksp.h5")
    full = _make_dataset(fn, "ksp_epi_zf", chunked=False)
    xs, ys, zs = slice(1, 4), slice(0, 3), slice(2, 4)
    out = read_frames_cropped(fn, "ksp_epi_zf", spatial_slices=(xs, ys, zs))
    expected = full[xs, ys, zs]
    np.testing.assert_array_equal(out, expected)


def test_read_frames_cropped_matches_slow_reference_gather(tmp_path):
    """Same crop, computed the slow/obviously-correct way (load the whole
    array via a bare h5py read, then slice) -- an independent check that
    doesn't reuse any of read_frames_cropped's own chunk-iteration logic,
    unlike the _encode-based tests above, which could in principle share a
    transposition bug with the implementation under test."""
    fn = str(tmp_path / "ksp.h5")
    _make_dataset(fn, "ksp_epi_zf", chunked=True)
    xs, ys, zs = slice(2, 5), slice(1, 5), slice(0, 3)
    with h5py.File(fn, "r") as f:
        reference = f["ksp_epi_zf"][()][xs, ys, zs]
    out = read_frames_cropped(fn, "ksp_epi_zf", spatial_slices=(xs, ys, zs))
    np.testing.assert_array_equal(out, reference)
