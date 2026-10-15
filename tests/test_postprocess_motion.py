"""postprocess.motion on a synthetic head with known rigid motion."""

import numpy as np
import pytest
from scipy import ndimage

from postprocess import motion

VOX = (3.0, 3.0, 3.0)
SHAPE = (40, 40, 30)
TRUTH = np.array([
    [0, 0, 0, 0, 0, 0],
    [0.00, 0.00, 0.03, 1.5, 0.0, 0.0],
    [0.02, -0.02, 0.00, -1.0, 2.0, 1.5],
    [-0.03, 0.01, 0.02, 0.5, -1.5, -2.0],
])


def _phantom(rng):
    x, y, z = np.meshgrid(*[(np.arange(n) - (n - 1) / 2) for n in SHAPE], indexing="ij")
    head = ((x / 14) ** 2 + (y / 16) ** 2 + (z / 11) ** 2 < 1).astype(float)
    # off-center blobs break every symmetry, smooth texture gives the cost a basin
    tex = ndimage.gaussian_filter(rng.standard_normal(SHAPE), 2.0)
    blobs = sum(np.exp(-(((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / 8.0))
                for cx, cy, cz in [(5, 2, 1), (-6, -4, 2), (1, 7, -3)])
    return 100 * head * (1 + 0.09 * tex / tex.std() + blobs)


@pytest.fixture(scope="module")
def moved():
    vol = _phantom(np.random.default_rng(0))
    c = motion._center_mm(SHAPE, VOX)
    # frame t = the reference seen through the *inverse* of TRUTH[t]: aligning it
    # back needs exactly T = TRUTH[t] (moving -> reference)
    frames = [motion._warp(vol, np.linalg.inv(motion.params_to_matrix(p, c)), VOX, order=3)
              for p in TRUTH]
    return vol, np.stack(frames, axis=3).astype(np.float32)


def _check(res, tol_mm=0.3, tol_deg=0.6):
    # frame 0 is the reference only up to rounding
    err = res.params - TRUTH
    assert np.abs(err[:, 3:]).max() < tol_mm
    assert np.degrees(np.abs(err[:, :3]).max()) < tol_deg


def test_params_matrix_roundtrip():
    c = motion._center_mm(SHAPE, VOX)
    p = np.array([0.1, -0.05, 0.2, 3.0, -2.0, 1.0])
    assert np.allclose(motion.matrix_to_params(motion.params_to_matrix(p, c), c), p)


def test_python_backend_recovers_known_motion(moved):
    _, series = moved
    res = motion.correct_motion(series, VOX, ref=0, method="python")
    _check(res)
    # the corrected frames agree with the reference far better than the raw ones
    raw = np.linalg.norm(series[..., 1:] - series[..., :1])
    cor = np.linalg.norm(res.series[..., 1:] - res.series[..., :1])
    assert cor < 0.35 * raw


def test_complex_series_keeps_phase(moved):
    _, series = moved
    cplx = (series * np.exp(0.7j)).astype(np.complex64)
    res = motion.correct_motion(cplx, VOX, ref=0, method="python")
    inside = np.abs(res.series[..., 0]) > 50
    assert np.allclose(np.angle(res.series[..., 1][inside]), 0.7, atol=0.05)


def test_zero_motion_is_identity():
    vol = _phantom(np.random.default_rng(1)).astype(np.float32)
    series = np.stack([vol] * 3, axis=3)
    res = motion.correct_motion(series, VOX, ref="middle", method="python")
    assert np.abs(res.params[:, 3:]).max() < 0.05
    assert np.allclose(res.series, series, atol=1.0)


@pytest.mark.skipif(motion.find_mcflirt() is None, reason="FSL mcflirt not found")
def test_fsl_backend_matches_python_backend(moved):
    _, series = moved
    fsl = motion.correct_motion(series, VOX, ref=0, method="fsl")
    py = motion.correct_motion(series, VOX, ref=0, method="python")
    _check(fsl, tol_mm=0.5, tol_deg=0.75)
    assert np.abs(fsl.params - py.params)[:, 3:].max() < 0.5


def test_cli_writes_corrected_recon(moved, tmp_path):
    import h5py

    from postprocess.__main__ import main

    _, series = moved
    src, out = str(tmp_path / "r_recon.h5"), str(tmp_path / "r_mc_recon.h5")
    with h5py.File(src, "w") as f:
        f["X_recon"] = series.astype(np.complex64)
        f["X"] = np.stack([series, 0 * series], axis=4).astype(np.complex64)
    main(["--recon", src, "--out", out, "--method", "python", "--ref", "0",
          "--voxel-size", "3", "3", "3"])
    with h5py.File(out, "r") as f:
        assert f["X_recon"].shape == series.shape and f["X"].shape == series.shape + (2,)
        assert f["motion/params"].shape == (4, 6)
        got = np.abs(f["X_recon"][()])
    assert np.linalg.norm(got[..., 1:] - got[..., :1]) < 0.35 * np.linalg.norm(
        series[..., 1:] - series[..., :1])
    assert (tmp_path / "r_mc_recon.par").exists()
