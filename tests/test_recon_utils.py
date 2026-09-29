"""recon/utils.py's read_frames_cropped: both branches (chunked
one frame per chunk vs. not chunked along the last axis) x both modes
(spatial_slices=None vs. given) against a location-encoding synthetic
dataset, in the same style as test_preprocess_preprocess.py's
scatter_frame test -- each element is given a value encoding its own
(x, y, z, c, t) index, so a wrong crop/reshape is caught directly rather
than only by a shape check. Also: load_and_gather_ksp's frame selection,
rss.py, and utils.tsnr_report.
"""

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from recon.operators import build_sense  # noqa: E402
from recon.rss import rss, run_rss  # noqa: E402
from recon.utils import (  # noqa: E402
    load_and_gather_ksp,
    object_mask,
    read_frames_cropped,
    tsnr_report,
)

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


def test_load_and_gather_ksp_frames_maps_blocks_to_file_frames(tmp_path):
    fn = tmp_path / "k.h5"
    full = _make_dataset(fn, "ksp_epi_zf", chunked=True)
    smaps = torch.ones(NC, NX, NY, NZ, dtype=torch.complex64)
    omega = torch.ones(NX, NY, NZ, 2, dtype=torch.bool)
    A = build_sense(smaps, omega)
    ksp = load_and_gather_ksp(str(fn), A, torch.device("cpu"), frames=[5, 2])
    np.testing.assert_array_equal(ksp[:, :, 0].numpy(), full[..., 5].reshape(-1, NC))
    np.testing.assert_array_equal(ksp[:, :, 1].numpy(), full[..., 2].reshape(-1, NC))


def test_rss_matches_numpy_reference(tmp_path):
    fn = tmp_path / "k.h5"
    full = _make_dataset(fn, "ksp_epi_zf", chunked=True)
    axes = (0, 1, 2)
    ref = np.fft.ifftn(np.fft.ifftshift(full, axes=axes), axes=axes, norm="ortho")
    ref = np.fft.fftshift(ref, axes=axes)
    ref = np.sqrt((np.abs(ref) ** 2).sum(axis=3))
    img = run_rss(str(fn), device="cpu", max_batch_bytes=1)  # 1 frame per batch
    np.testing.assert_allclose(img, ref, rtol=1e-4, atol=1e-2)
    assert rss(torch.from_numpy(full)).shape == (NX, NY, NZ, NT)


def test_tsnr_report_on_a_stable_phantom(tmp_path):
    nib = pytest.importorskip("nibabel")
    rng = np.random.default_rng(0)
    img = np.zeros((8, 8, 4, 20), dtype=np.float32)
    img[2:6, 2:6] = 100 + rng.standard_normal((4, 4, 4, 20))  # tSNR ~ 100
    fn = str(tmp_path / "recon.nii.gz")
    nib.save(nib.Nifti1Image(img, np.eye(4)), fn)
    stats = tsnr_report([fn], tr_s=1.0)["recon"]
    assert 50 < stats["median_voxel_tsnr"] < 200
    assert (tmp_path / "recon_tsnr.png").exists()


def test_tsnr_report_uses_one_mask_for_all_inputs(tmp_path):
    """Item 256: a mask thresholded at a fraction of each input's own max
    changed size with signal level (5603/4725/13100 voxels for three runs of
    one static phantom), so the medians weren't over the same voxels. Two
    runs of one object, one with a hot spot that shrinks its own mask, must be
    measured over the same voxels."""
    nib = pytest.importorskip("nibabel")
    rng = np.random.default_rng(0)
    base = np.zeros((8, 8, 4, 20), dtype=np.float32)
    base[1:7, 1:7] = 100 + rng.standard_normal((6, 6, 4, 20))
    hot = base.copy()
    hot[3:5, 3:5] *= 8  # its own 0.2-of-max mask now drops the 100-level voxels
    fns = []
    for name, img in (("a", base), ("b", hot)):
        fns.append(str(tmp_path / f"{name}.nii.gz"))
        nib.save(nib.Nifti1Image(img, np.eye(4)), fns[-1])
    stats = tsnr_report(fns, tr_s=1.0)
    n_a = np.isfinite(stats["a"]["tsnr_map"]).sum()
    n_b = np.isfinite(stats["b"]["tsnr_map"]).sum()
    assert n_a == n_b == object_mask(hot).sum() < object_mask(base).sum()  # the intersection

    given = np.zeros((8, 8, 4), dtype=bool)
    given[1:7, 1:7] = True
    stats = tsnr_report(fns, tr_s=1.0, mask=given)
    for label in ("a", "b"):
        assert np.isfinite(stats[label]["tsnr_map"]).sum() == given.sum()
    with pytest.raises(ValueError):
        tsnr_report(fns, tr_s=1.0, mask=given[:, :, :2])
