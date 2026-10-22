"""postprocess.brainmask: mask from a preprocessed file's deGRE echo 1, applied to a series."""

import h5py
import numpy as np
import pytest
from scipy import ndimage

from postprocess import brainmask

SHAPE = (20, 20, 12)


@pytest.fixture
def preprocessed(tmp_path):
    x, y, z = np.meshgrid(*[np.arange(n) - (n - 1) / 2 for n in SHAPE], indexing="ij")
    head = ((x / 7) ** 2 + (y / 7) ** 2 + (z / 4) ** 2 < 1).astype(np.float32)
    fn = str(tmp_path / "pre.h5")
    with h5py.File(fn, "w") as f:
        f["degre/img_echoes"] = np.stack([100 * head + 1, 50 * head + 1], axis=-1)
        f.attrs["fov"] = np.array([0.048, 0.048, 0.036])  # 2.4 mm voxels
    return fn, head > 0


def test_mask_follows_echo_one(preprocessed):
    fn, head = preprocessed
    mask, vox = brainmask.gre_brain_mask(fn, method="threshold")
    assert vox == pytest.approx((2.4, 2.4, 3.0))
    assert (mask == head).all()


def test_apply_zeroes_outside_every_frame_and_scale(preprocessed):
    _, head = preprocessed
    X = np.ones(SHAPE + (5, 2), np.complex64)
    out = brainmask.apply_brain_mask(X, head)
    assert (out[head] == 1).all() and (out[~head] == 0).all()
    assert brainmask.apply_brain_mask(X[..., 0], head).shape == X[..., 0].shape
    with pytest.raises(ValueError):
        brainmask.apply_brain_mask(X, head[:-1])


@pytest.mark.skipif(brainmask.find_flirt() is None, reason="FSL flirt not found")
def test_register_mask_follows_head_motion():
    """The EPI head sits 3 voxels (7.2 mm) from where the deGRE saw it: the registered mask lands on it."""
    shape, vox = (48, 48, 32), (2.4, 2.4, 2.4)
    x, y, z = np.meshgrid(*[np.arange(n) - (n - 1) / 2 for n in shape], indexing="ij")
    rng = np.random.default_rng(0)
    tex = ndimage.gaussian_filter(rng.standard_normal(shape), 2.0)

    def head(dx):
        inside = (((x - dx) / 15) ** 2 + (y / 17) ** 2 + (z / 11) ** 2) < 1
        return inside, 100 * inside * (1 + 0.3 * tex / tex.std()) + 1

    mask0, gre = head(0)
    mask3, epi = head(3)
    reg, xfm = brainmask.register_mask(gre, mask0, epi, vox)
    dice = lambda a, b: 2 * (a & b).sum() / (a.sum() + b.sum())  # noqa: E731
    assert dice(mask0, mask3) < 0.85 < dice(reg, mask3)
    assert xfm.shape == (4, 4) and brainmask.mask_shift_mm(xfm, mask0, vox) == pytest.approx(7.2, abs=1.0)
