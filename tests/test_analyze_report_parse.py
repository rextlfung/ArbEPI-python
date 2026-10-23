"""analyze.report.parse: analysis folder name -> (base, run, tag, lowband), for every run naming used so far."""

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("nibabel")

import numpy as np  # noqa: E402

from analyze.report import find_recon, parse, top_peaks, write_index  # noqa: E402


@pytest.mark.parametrize("name, run, tag", [
    ("2.4mm_radial_wtv0.00125", "2.4mm_radial", "wtv0.00125"),
    ("1.6mm_radial_llr_p9s9_cs_l56.48_it100", "1.6mm_radial", "llr_p9s9_cs_l56.48_it100"),
    ("2.4mm_laminar_wtv0.00125_hp30", "2.4mm_laminar", "wtv0.00125_hp30"),
    ("1.6mm_pd_wtv0.005", "1.6mm_pd", "wtv0.005"),
    ("2.4mm_caipi_wtv0.005", "2.4mm_caipi", "wtv0.005"),
    ("2.4mm_caipi_rw_wtv0.005", "2.4mm_caipi_rw", "wtv0.005"),
    ("2.4mm_caipi_rw_llr_p6s6_cs_sqrtR", "2.4mm_caipi_rw", "llr_p6s6_cs_sqrtR"),
    ("2.4mm_pd_mslr_g204_l8.5_p6s6", "2.4mm_pd", "mslr_g204_l8.5_p6s6"),
])
def test_parse(name, run, tag):
    base, r, t, lowband = parse(f"/x/analysis/{name}")
    assert (base, r, t, lowband) == (name, run, tag, False)


def test_parse_lowband():
    assert parse("analysis/2.4mm_pd_wtv0.005_lowband")[1:] == ("2.4mm_pd", "wtv0.005", True)


def test_find_recon_new_layout_then_old(tmp_path):
    root = str(tmp_path)
    old = tmp_path / "recon" / "sense_wavelet-tv_b0_wtv0.005" / "joined" / "2.4mm_radial_recon.h5"
    old.parent.mkdir(parents=True)
    old.touch()
    assert find_recon(root, "2.4mm_radial", "wtv0.005") == str(old)
    new = tmp_path / "recon" / "2.4mm_radial_wtv0.005" / "2.4mm_radial_recon_final.h5"
    new.parent.mkdir(parents=True)
    new.touch()
    assert find_recon(root, "2.4mm_radial", "wtv0.005") == str(new)
    with pytest.raises(FileNotFoundError):
        find_recon(root, "2.4mm_pd", "wtv0.005")


def test_write_index_reads_reports_beside_the_maps(tmp_path):
    import json

    for name, n in (("2.4mm_pd_wtv0.005", 100), ("2.4mm_caipi_wtv0.005", 200)):
        d = tmp_path / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({
            "name": name, "n_active": n, "n_counter_active": 5, "z_fdr": 3.0,
            "split_half_odd_even": {"z_pearson": 0.4, "dice": 0.5}, "test_retest": {"auc": 0.8}}))
    write_index(str(tmp_path))
    text = (tmp_path / "stats.md").read_text()
    assert "[2.4mm_pd_wtv0.005](2.4mm_pd_wtv0.005/report.md)" in text and text.index("caipi") < text.index("_pd_")
    assert not (tmp_path / "reports").exists()


def test_top_peaks_counter_active_takes_the_most_negative_distinct_voxels():
    z = np.zeros((20, 20, 10))
    z[2, 2, 2], z[3, 2, 2] = -9.0, -8.5  # a cluster: the neighbour is skipped
    z[15, 15, 5], z[10, 4, 7] = -7.0, -6.0
    z[1, 1, 1] = 12.0  # an active voxel is never a counter-active peak
    brain = np.ones(z.shape, bool)
    vox = (2.4, 2.4, 2.4)
    assert top_peaks(z, brain, vox, n=3, sign=-1) == [(2, 2, 2), (15, 15, 5), (10, 4, 7)]
    assert top_peaks(z, brain, vox, n=1) == [(1, 1, 1)]
