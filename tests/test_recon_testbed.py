"""recon/testbed.py: building a known-truth testbed from a fully sampled
acquisition, and scoring reconstructions against it."""

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from recon import testbed  # noqa: E402
from recon.sense import run_sense  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = (32, 32, 24)
NC, NF, NT, TR = 2, 3, 24, 0.5


def _ball_full(tmp_path):
    """A fully sampled 'acquisition' of a ball (3 frames), in preprocess/'s layout."""
    rng = np.random.default_rng(0)
    g = np.indices(N) - (np.array(N) / 2)[:, None, None, None]
    ball = (np.linalg.norm(g, axis=0) < 11).astype(np.complex64) * np.exp(1j * 0.02 * g[0])
    smaps = rng.normal(size=N + (NC,)) + 1j * rng.normal(size=N + (NC,))
    smaps = ndimage_smooth(smaps)
    smaps /= np.sqrt((np.abs(smaps) ** 2).sum(-1, keepdims=True))
    img = ball[..., None] * smaps
    ksp = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img, axes=(0, 1, 2)), axes=(0, 1, 2),
                                      norm="ortho"), axes=(0, 1, 2))
    fn = tmp_path / "full_preprocessed.h5"
    with h5py.File(fn, "w") as f:
        f["ksp_epi_zf"] = np.repeat(ksp[..., None], NF, -1).astype(np.complex64)
        f["smaps"] = smaps.astype(np.complex64)
        f["b0_map"] = (10.0 * g[1] / N[1]).astype(np.float32)
        et = 0.01 + 0.03 * np.arange(N[1])[:, None] / N[1] * np.ones(N[2])
        f["echo_times"] = np.repeat(et[..., None], NF, -1)
        f["omegas"] = np.ones(N[1:] + (NF,), bool)
        f.attrs.update(fov=(0.2, 0.2, 0.15), noise_var=1.0)
    return str(fn)


def ndimage_smooth(x):
    from scipy import ndimage

    return ndimage.gaussian_filter(x.real, (6, 6, 6, 0)) + 1j * ndimage.gaussian_filter(
        x.imag, (6, 6, 6, 0))


def _masks(tmp_path, frac=0.4):
    rng = np.random.default_rng(1)
    k = int(frac * N[1] * N[2])
    om = np.zeros((N[1] * N[2], NT + 1), bool)
    for t in range(NT + 1):
        om[rng.choice(N[1] * N[2], k, replace=False), t] = True
    fn = tmp_path / "masks_preprocessed.h5"
    with h5py.File(fn, "w") as f:
        f["omegas"] = om.reshape(N[1], N[2], NT + 1)
    return str(fn)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("testbed")
    out = str(tmp / "recon" / "tb_preprocessed.h5")
    testbed.build_testbed(_ball_full(tmp), _masks(tmp), out, TR, nt=NT, device=DEVICE, L_b0=4,
                          nbins_b0=32, cg_iters=50)
    return tmp, out


def test_waveforms_are_zero_mean_unit_peak_and_in_or_out_of_band():
    for kind in ("block", "sin0.10", "sin0.25"):
        w = testbed.waveform(kind, 120, 0.5)
        assert abs(w.mean()) < 1e-12 and np.isclose(np.abs(w).max(), 1)
    C = testbed._dct(120)
    keep = int((np.arange(120) / (2 * 120 * 0.5) <= 0.15).sum())
    for kind, inband in (("block", True), ("sin0.10", True), ("sin0.25", False)):
        c = C @ testbed.waveform(kind, 120, 0.5)
        frac_in = (c[:keep] ** 2).sum() / (c**2).sum()
        assert (frac_in > 0.9) if inband else (frac_in < 0.1), (kind, frac_in)


def test_build_writes_a_file_run_sense_reads_with_rois_inside_the_object(built):
    _, fn = built
    with h5py.File(fn, "r") as f:
        assert f["ksp_epi_zf"].shape == N + (NC, NT)
        assert f["omegas"].shape == N[1:] + (NT,)
        rois, interior = f["truth/roi_masks"][()], f["truth/interior"][()]
        assert np.isclose(f.attrs["volume_tr"], TR)
        assert len(set(f["truth"].attrs["roi_names"])) == len(testbed.ROIS)
    assert all(m.any() and interior[m].all() for m in rois)
    assert not (rois.sum(0) > 1).any()  # disjoint
    r = run_sense(fn_ksp=fn, fn_smaps=fn, reg="none", niters=5, device=DEVICE, fn_b0map=fn,
                  L_b0=4, nbins_b0=32)
    assert torch.isfinite(r.X_recon).all()


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


def test_scoring_the_truth_gives_perfect_scores_even_rescaled(built):
    tmp, fn = built
    rec = str(tmp / "truth_recon.h5")
    _write_recon(rec, 3.7 * _truth_series(fn))  # recon scale is arbitrary
    s = testbed.score(fn, rec)
    assert s["nrmse_mean_pct"] < 1e-4 and s["nrmse_frame_pct"] < 1e-4
    assert s["fluct_pct"] < 1e-4
    for name, _, _ in testbed.ROIS:
        assert np.isclose(s[f"{name}_amp_ratio"], 1, atol=1e-4)
        assert abs(s[f"{name}_leak_ratio"]) < 1e-4
        assert s[f"{name}_corr"] > 0.9999
    assert np.isclose(s["edge_sharpness_ratio"], 1, atol=1e-4)


def test_scoring_detects_noise_attenuation_and_blur(built):
    tmp, fn = built
    T = _truth_series(fn)
    rng = np.random.default_rng(2)
    noisy = T * (1 + 0.03 * rng.normal(size=T.shape))
    _write_recon(str(tmp / "noisy.h5"), noisy)
    s = testbed.score(fn, str(tmp / "noisy.h5"))
    assert 2.5 < s["fluct_pct"] < 3.5
    # white noise splits between bands in proportion to their widths
    keep = int((np.arange(NT) / (2 * NT * TR) <= 0.15).sum())
    assert np.isclose(s["fluct_inband_pct"] ** 2 / s["fluct_pct"] ** 2, (keep - 1) / NT, atol=0.1)
    # a temporal mean throws the activation away
    _write_recon(str(tmp / "static.h5"), np.repeat(T.mean(-1, keepdims=True), NT, -1))
    s = testbed.score(fn, str(tmp / "static.h5"))
    assert abs(s["block_r3_amp_ratio"]) < 0.05 and s["fluct_pct"] < 1e-3
    # spatial blur lowers edge sharpness
    from scipy import ndimage

    blur = ndimage.gaussian_filter(T.real, (1.5, 1.5, 1.5, 0)) + 1j * ndimage.gaussian_filter(
        T.imag, (1.5, 1.5, 1.5, 0))
    _write_recon(str(tmp / "blur.h5"), blur)
    assert testbed.score(fn, str(tmp / "blur.h5"))["edge_sharpness_ratio"] < 0.8


def test_build_can_take_the_masks_runs_own_echo_times(tmp_path):
    full = _ball_full(tmp_path)
    fn_m = _masks(tmp_path)
    et = np.random.default_rng(3).uniform(0.01, 0.05, (N[1], N[2], NT + 1))
    with h5py.File(fn_m, "a") as f:
        f["echo_times"] = et
    out = str(tmp_path / "recon" / "tbm_preprocessed.h5")
    testbed.build_testbed(full, fn_m, out, TR, nt=NT, device=DEVICE, L_b0=4, nbins_b0=32,
                          cg_iters=20, timing="masks")
    with h5py.File(out, "r") as f:
        np.testing.assert_allclose(f["echo_times"][()], et[..., 1 : NT + 1], rtol=1e-6)
        assert f.attrs["testbed_timing"] == "masks"


def test_lowband_glm_is_calibrated_for_band_limited_residuals():
    """A temporally smooth (low-pass) residual inflates the plain t; the low-band
    GLM, with its own degrees of freedom, keeps the false-positive rate near
    nominal (p < 0.001)."""
    rng = np.random.default_rng(4)
    nt, keep, V = 80, 12, 20000
    C = testbed._dct(nt)
    w = testbed.waveform("block", nt, 0.4851)
    noise = rng.normal(size=(V, keep)) @ C[:keep]  # band-limited null data
    _, t_plain = testbed._glm(1 + 0.01 * noise, w)
    _, t_lb = testbed._glm(1 + 0.01 * noise, w, C[:keep])
    from scipy import stats

    assert np.mean(np.abs(t_plain) > 3.29) > 0.05  # badly inflated
    assert np.mean(np.abs(t_lb) > stats.t.ppf(1 - 0.0005, keep - 3)) < 0.005
