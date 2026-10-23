"""analyze.report.parse: analysis folder name -> (base, run, tag, lowband), for every run naming used so far."""

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("nibabel")

from analyze.report import parse  # noqa: E402


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
