"""analyze/score.py: scoring reconstructions against a known truth. The truth
file is written directly (the layout simulate_fmri/from_real.py, session.py and
ideal.py produce), so nothing here needs torch or SNAKE."""

import h5py
import numpy as np
import pytest

from analyze.design import canonical_hrf
from analyze.score import panel, score

N = (32, 32, 24)
NT, TR, AMP = 24, 0.5, 0.02
NAMES = ["block_r3", "block_r1.5", "sin0.10", "sin0.25"]
KINDS = ["block", "block", "sin0.10", "sin0.25"]
RADII = [3.0, 1.5, 3.0, 3.0]
CENTERS = [(11, 16, 12), (21, 16, 12), (16, 10, 12), (16, 22, 12)]


def _wave(kind, nt, tr):
    """Zero-mean, peak |w| = 1: a 10 s on / 10 s off boxcar through the HRF, or a sinusoid."""
    t = np.arange(nt) * tr
    if kind == "block":
        h = canonical_hrf(np.arange(0, 32.0, tr))
        w = np.convolve(((t // 10.0) % 2 == 1).astype(float), h / h.sum())[:nt]
    else:
        w = np.sin(2 * np.pi * float(kind[3:]) * t)
    w = w - w.mean()
    return w / np.abs(w).max()


@pytest.fixture(scope="module")
def truth(tmp_path_factory):
    """A ball with four disjoint spherical ROIs inside it, in the `truth` layout."""
    tmp = tmp_path_factory.mktemp("truth")
    g = np.indices(N) - (np.array(N) / 2)[:, None, None, None]
    x0 = ((np.linalg.norm(g, axis=0) < 11) * (1 + 0.02 * g[0])).astype(np.complex64)
    x0 = x0 * np.exp(1j * 0.02 * g[0])  # only |x0| matters
    grid = np.indices(N)
    rois = np.stack([np.linalg.norm(grid - np.array(c)[:, None, None, None], axis=0) <= r
                     for c, r in zip(CENTERS, RADII)])
    fn = str(tmp / "truth.h5")
    with h5py.File(fn, "w") as f:
        f.attrs["volume_tr"] = TR
        t_ = f.create_group("truth")
        t_["x0"] = x0
        t_["roi_masks"] = rois
        t_["waveforms"] = np.stack([_wave(k, NT, TR) for k in KINDS])
        t_.attrs["roi_names"] = NAMES
        t_.attrs["amp"] = AMP
    return tmp, fn


def test_the_rois_are_disjoint_and_inside_the_object(truth):
    _, fn = truth
    with h5py.File(fn, "r") as f:
        rois, x0 = f["truth/roi_masks"][()], np.abs(f["truth/x0"][()])
    assert rois.sum(0).max() == 1 and all(m.any() for m in rois)
    assert all((x0[m] > 0).all() for m in rois)


def _write_recon(path, X):
    with h5py.File(path, "w") as f:
        f["X_recon"] = X.astype(np.complex64)


def _truth_series(fn):
    with h5py.File(fn, "r") as f:
        x0 = f["truth/x0"][()]
        rois, waves = f["truth/roi_masks"][()], f["truth/waveforms"][()]
        amp = f["truth"].attrs["amp"]
    X = np.repeat(x0[..., None], NT, -1)
    for m, w in zip(rois, waves):
        X[m] *= 1 + amp * w[None, :]
    return X


def test_scoring_the_truth_gives_perfect_scores_even_rescaled(truth):
    tmp, fn = truth
    rec = str(tmp / "truth_recon.h5")
    _write_recon(rec, 3.7 * _truth_series(fn))  # recon scale is arbitrary
    s = score(fn, rec)
    assert s["nrmse_mean_pct"] < 1e-4 and s["nrmse_frame_pct"] < 1e-4
    assert s["fluct_pct"] < 1e-4
    for name in NAMES:
        assert np.isclose(s[f"{name}_amp_ratio"], 1, atol=1e-4)
        assert abs(s[f"{name}_leak_ratio"]) < 1e-4
        assert s[f"{name}_corr"] > 0.9999
    assert np.isclose(s["edge_sharpness_ratio"], 1, atol=1e-4)


def test_scoring_uses_per_roi_amplitudes_when_the_truth_has_them(truth, tmp_path):
    """A truth file may give each ROI its own peak change ('amps'; simulated
    sessions do, their regions differ in tissue mix). The truth series then
    scores exactly for every ROI, which it would not against the shared amp."""
    import shutil

    fn = truth[1]
    fn2 = str(tmp_path / "amps.h5")
    shutil.copy(fn, fn2)
    with h5py.File(fn2, "r+") as f:
        g = f["truth"]
        x0, rois, waves = g["x0"][()], g["roi_masks"][()], g["waveforms"][()]
        amps = g.attrs["amp"] * np.linspace(0.5, 1.5, len(rois))
        g.attrs["amps"] = amps
    X = np.repeat(x0[..., None], NT, -1)
    for m, w, a in zip(rois, waves, amps):
        X[m] *= 1 + a * w[None, :]
    rec = str(tmp_path / "recon.h5")
    _write_recon(rec, X)
    s = score(fn2, rec)
    assert s["nrmse_frame_pct"] < 1e-4
    for name in NAMES:
        assert np.isclose(s[f"{name}_amp_ratio"], 1, atol=1e-4)
    # and against the file without them, the same series is off by those factors
    s = score(fn, rec)
    ratios = [s[f"{name}_amp_ratio"] for name in NAMES]
    np.testing.assert_allclose(ratios, np.linspace(0.5, 1.5, len(rois)), atol=1e-3)


def test_panel_can_show_any_roi(truth, tmp_path):
    tmp, fn = truth
    rec = str(tmp_path / "recon.h5")
    _write_recon(rec, _truth_series(fn))
    for roi in (0, 1):
        png = tmp_path / f"panel{roi}.png"
        panel(fn, rec, str(png), roi=roi)
        assert png.stat().st_size > 10_000


def test_scoring_detects_noise_attenuation_and_blur(truth):
    tmp, fn = truth
    T = _truth_series(fn)
    rng = np.random.default_rng(2)
    noisy = T * (1 + 0.03 * rng.normal(size=T.shape))
    _write_recon(str(tmp / "noisy.h5"), noisy)
    s = score(fn, str(tmp / "noisy.h5"))
    assert 2.5 < s["fluct_pct"] < 3.5
    # white noise splits between bands in proportion to their widths
    keep = int((np.arange(NT) / (2 * NT * TR) <= 0.15).sum())
    assert np.isclose(s["fluct_inband_pct"] ** 2 / s["fluct_pct"] ** 2, (keep - 1) / NT, atol=0.1)
    # a temporal mean throws the activation away
    _write_recon(str(tmp / "static.h5"), np.repeat(T.mean(-1, keepdims=True), NT, -1))
    s = score(fn, str(tmp / "static.h5"))
    assert abs(s["block_r3_amp_ratio"]) < 0.05 and s["fluct_pct"] < 1e-3
    # spatial blur lowers edge sharpness
    from scipy import ndimage

    blur = ndimage.gaussian_filter(T.real, (1.5, 1.5, 1.5, 0)) + 1j * ndimage.gaussian_filter(
        T.imag, (1.5, 1.5, 1.5, 0))
    _write_recon(str(tmp / "blur.h5"), blur)
    assert score(fn, str(tmp / "blur.h5"))["edge_sharpness_ratio"] < 0.8
