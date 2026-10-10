"""analyze.io / analyze.mask / analyze.run / CLI on a synthetic reconstruction."""

import json

import h5py
import nibabel as nib
import numpy as np
import pytest

from analyze import mask
from analyze.__main__ import main
from analyze.design import ExperimentParams, block_response
from analyze.io import read_recon
from analyze.run import analyze_recon

TR = 1.0
NT = 120
SHAPE = (12, 12, 8)
ONSETS = [10.0, 50.0, 90.0]
DUR = 20.0
ACTIVE = (slice(2, 5), slice(2, 5), slice(2, 5))


@pytest.fixture(scope="module")
def recon_file(tmp_path_factory):
    """Two-scale 'mslr' run: scale 0 a static background with a phase, scale 1 a
    zero-mean local component carrying the activation (+3%) in one corner."""
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("recon")
    fn = str(d / "run_recon.h5")
    t = (np.arange(NT) + 0.5) * TR
    w = block_response(t, ONSETS, [DUR] * 3)
    w = w - w.mean()
    yy, xx = np.meshgrid(np.arange(12), np.arange(12), indexing="ij")
    obj = ((xx - 5.5) ** 2 + (yy - 5.5) ** 2 < 25).astype(float)[:, :, None] * np.ones(SHAPE)
    phase = np.exp(1j * 0.7)
    X0 = np.repeat((100 * obj * phase)[..., None], NT, -1)
    X1 = np.zeros(SHAPE + (NT,), complex)
    X1[ACTIVE] = 100 * 0.03 * w * phase * 1.0
    noise = 0.5 * (rng.standard_normal(SHAPE + (NT,)) + 1j * rng.standard_normal(SHAPE + (NT,)))
    X1 = X1 + noise * obj[..., None]
    X = np.stack([X0, X1], axis=-1).astype(np.complex64)
    with h5py.File(fn, "w") as f:
        f["X_recon"] = X.sum(-1)
        f["X"] = X
    with open(str(d / "run_recon.json"), "w") as f:
        json.dump({"fov": [0.012 * 2.0, 0.012 * 2.0, 0.008 * 3.0]}, f)  # 2, 2, 3 mm
    return fn


def test_read_recon_scales_and_voxel_size(recon_file):
    rec = read_recon(recon_file, scales=("all",), n_discard=4)
    assert set(rec.series) == {"sum", "scale0", "scale1"}
    assert rec.series["sum"].shape == SHAPE + (NT - 4,)
    assert rec.n_scales == 2
    assert np.allclose(rec.voxel_size_mm, (2.0, 2.0, 3.0))
    # component projections are real, and add up to the magnitude image where the phase is stable
    s = rec.series["scale0"] + rec.series["scale1"]
    assert np.allclose(s[5, 5, 4], rec.series["sum"][5, 5, 4], rtol=0.02)
    with pytest.raises(ValueError):
        read_recon(recon_file, scales=(5,))


def test_analyze_recon_finds_the_activation_in_the_dynamic_scale_only(recon_file, tmp_path):
    p = ExperimentParams.block(TR, ONSETS, DUR)
    brain = np.zeros(SHAPE, bool)
    brain[:, :, :] = True
    out = analyze_recon(recon_file, p, scales=("all",), brain=brain, out_dir=str(tmp_path))
    act = np.zeros(SHAPE, bool)
    act[ACTIVE] = True
    sum_t = out["sum"].t
    assert sum_t[act].mean() > 8
    assert abs(sum_t[~act & (out["sum"].t != 0)]).max() < 6
    # background/foreground separation: the global scale has no task response,
    # the local one has all of it
    assert np.abs(out["scale0"].t[act]).mean() < 3
    assert out["scale1"].t[act].mean() > 8
    # effect size in percent of the voxel's mean signal: 3% peak-to-mean
    assert 1.0 < out["scale1"].psc[act].mean() < 6.0
    assert out["sum"].n_fdr >= act.sum() * 0.8
    for label in ("sum", "scale0", "scale1"):
        for kind in ("t", "z", "psc"):
            img = nib.load(str(tmp_path / f"run_recon_{label}_{kind}.nii.gz"))
            assert img.shape == SHAPE
            assert np.allclose(img.header.get_zooms(), (2.0, 2.0, 3.0))
    t_nii = np.asanyarray(nib.load(str(tmp_path / "run_recon_scale1_t.nii.gz")).dataobj)
    assert t_nii.min() < 0 < t_nii.max()  # signed
    summ = json.load(open(tmp_path / "run_recon_summary.json"))
    assert summ["design"][0] == "task" and summ["design"][-1] == "constant"
    assert summ["scales"]["sum"]["dof"] == NT - len(summ["design"])


def test_threshold_mask_and_mask_file(recon_file, tmp_path):
    rec = read_recon(recon_file)
    m = mask.brain_mask(rec.mean_signal, method="threshold")
    assert m.shape == SHAPE and m.dtype == bool and 0.3 < m.mean() < 0.9
    fn = str(tmp_path / "m.nii.gz")
    nib.save(nib.Nifti1Image(m.astype(np.uint8), np.eye(4)), fn)
    assert (mask.load_mask(fn) == m).all()
    out = analyze_recon(recon_file, ExperimentParams.block(TR, ONSETS, DUR), brain=fn)
    assert out["sum"].n_voxels <= int(m.sum())
    with pytest.raises(ValueError):
        mask.brain_mask(rec.mean_signal, method="nope")


def test_python_bet_returns_a_brain_sized_mask():
    pytest.importorskip("brainextractor")
    n = 40
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"))
    r2 = ((g - n / 2) ** 2 / np.array([12, 14, 11]).reshape(3, 1, 1, 1) ** 2).sum(0)
    vol = np.where(r2 < 1, 100.0, 0.0) + np.random.default_rng(0).uniform(0, 3, (n,) * 3)
    m = mask.bet_python(vol, (3.0, 3.0, 3.0))
    truth = r2 < 1
    dice = 2 * (m & truth).sum() / (m.sum() + truth.sum())
    assert m.shape == vol.shape and dice > 0.8


def test_cli_writes_maps(recon_file, tmp_path, capsys):
    brain = np.ones(SHAPE, np.uint8)
    fn = str(tmp_path / "all.nii.gz")
    nib.save(nib.Nifti1Image(brain, np.eye(4)), fn)
    main(["--recon", recon_file, "--tr", str(TR), "--onsets", *map(str, ONSETS),
          "--duration", str(DUR), "--scales", "sum", "1", "--mask", fn,
          "--out", str(tmp_path / "o"), "--tag", "x"])
    assert (tmp_path / "o" / "x_scale1_z.nii.gz").exists()
    assert (tmp_path / "o" / "x_sum_t.nii.gz").exists()
    assert "scale1: dof" in capsys.readouterr().out
