"""analyze.compare: shared-mask comparison, figures and table, on synthetic recons."""

import matplotlib

matplotlib.use("Agg")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from analyze import compare  # noqa: E402
from analyze.design import ExperimentParams  # noqa: E402
from analyze.run import analyze_recon  # noqa: E402
from tests.test_analyze_run import DUR, NT, ONSETS, SHAPE, TR, recon_file  # noqa: E402,F401


@pytest.fixture(scope="module")
def noisy_file(recon_file, tmp_path_factory):  # noqa: F811
    """The same recon with extra noise: a worse 'reconstruction'."""
    fn = str(tmp_path_factory.mktemp("noisy") / "noisy_recon.h5")
    rng = np.random.default_rng(7)
    with h5py.File(recon_file) as f, h5py.File(fn, "w") as g:
        X = f["X"][()]
        n = 2.0 * (rng.standard_normal(X.shape[:-1]) + 1j * rng.standard_normal(X.shape[:-1]))
        X[..., 1] += n.astype(np.complex64)
        g["X"] = X
        g["X_recon"] = X.sum(-1)
    return fn


def test_compare_recons_shares_mask_and_ranks_the_cleaner_recon_higher(
        recon_file, noisy_file, tmp_path, capsys):  # noqa: F811
    p = ExperimentParams.block(TR, ONSETS, DUR)
    brain = np.ones(SHAPE, bool)
    out = compare.compare_recons({"clean": recon_file, "noisy": noisy_file}, p,
                                 scales=("sum", 1), brain=brain, out_dir=str(tmp_path))
    assert list(out) == ["clean", "clean/scale1", "noisy", "noisy/scale1"]
    assert (out["clean"].brain == out["noisy"].brain).all()
    assert out["clean"].t.max() > out["noisy"].t.max()
    assert out["clean"].series_psc.shape == (out["clean"].n_voxels, NT)
    for f in ("maps.png", "timecourses.png"):
        assert (tmp_path / f).stat().st_size > 1000
    assert "clean/scale1" in (tmp_path / "summary.txt").read_text()
    txt = capsys.readouterr().out
    assert "clean/scale1" in txt and "FDR n" in txt


def test_plot_timecourses_needs_series(recon_file):  # noqa: F811
    p = ExperimentParams.block(TR, ONSETS, DUR)
    m = analyze_recon(recon_file, p, brain=np.ones(SHAPE, bool))
    with pytest.raises(ValueError):
        compare.plot_timecourses(m, p)


def test_top_percent_selects_the_largest_statistic(recon_file):  # noqa: F811
    p = ExperimentParams.block(TR, ONSETS, DUR)
    m = analyze_recon(recon_file, p, brain=np.ones(SHAPE, bool))["sum"]
    sel = compare._select(m, 5.0)
    v = m.t[m.brain]
    assert sel.sum() == round(0.05 * v.size)
    assert v[sel].min() >= np.sort(v)[-sel.sum()]


def test_cli(recon_file, noisy_file, tmp_path):  # noqa: F811
    compare.main(["--recon", f"a={recon_file}", f"b={noisy_file}", "--tr", str(TR),
                  "--onsets", *map(str, ONSETS), "--duration", str(DUR),
                  "--mask", "threshold", "--out", str(tmp_path)])
    assert (tmp_path / "maps.png").exists()
