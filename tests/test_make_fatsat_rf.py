from dataclasses import replace

import numpy as np
import pypulseq as pp
import pytest

from lib.make_fatsat_rf import flip_profile, make_fatsat_rf
from lib.slr import dzrf_ex
from params import load_params


@pytest.fixture(scope='module')
def default_rfsat():
    p = load_params()
    return p, make_fatsat_rf(p.fatsat, p.sys, p.fat_offres_freq)


def test_default_fatsat_leaves_water_alone(default_rfsat):
    """Item 255: the old Gaussian fat-sat tipped on-resonance water by 26 deg
    every shot, which the random spoilers turned into ~3% frame-to-frame
    fluctuation. The default pulse's stop band must cover measured in-object
    B0 (99% of voxels in -144..+79 Hz in vivo, 20260922xiaokai)."""
    p, rf = default_rfsat
    assert flip_profile(rf, np.arange(-150, 151, 5)).max() < 2.0
    assert flip_profile(rf, np.arange(-200, -150, 5)).max() < 5.0


def test_default_fatsat_saturates_fat_without_overshoot(default_rfsat):
    """90 deg at the fat frequency, a pass band that tolerates in-object B0,
    and no overshoot past 90 deg (a min-phase TBW 4 / 12 ms candidate reached
    118 deg on the band edges)."""
    p, rf = default_rfsat
    fat = -p.fat_offres_freq
    assert flip_profile(rf, fat)[0] == pytest.approx(p.fatsat.flip, abs=0.5)
    assert flip_profile(rf, fat + np.arange(-50, 51, 5)).min() > 78.0
    assert flip_profile(rf, fat + np.arange(-100, 101, 5)).min() > 55.0
    assert flip_profile(rf, fat + np.arange(-300, 301, 5)).max() < p.fatsat.flip + 1.0


def test_fatsat_is_on_rf_raster_with_prescribed_duration(default_rfsat):
    p, rf = default_rfsat
    assert len(rf.signal) * p.sys.rf_raster_time == pytest.approx(p.fatsat.dur)
    assert np.allclose(np.diff(rf.t), p.sys.rf_raster_time)
    assert rf.freq_offset == pytest.approx(-p.fat_offres_freq)
    # On resonance with the pulse, a real waveform's flip is exactly its area.
    area_deg = 360 * np.sum(rf.signal) * p.sys.rf_raster_time
    assert area_deg == pytest.approx(p.fatsat.flip, rel=1e-6)
    assert pp.calc_duration(rf) >= p.fatsat.dur


def test_old_gaussian_design_fails_the_water_check():
    """The water check actually discriminates: the design item 255 replaced
    (pypulseq Gaussian, 90 deg, TBW 3, 4 ms) fails it by a wide margin."""
    p = load_params()
    g = pp.make_gauss_pulse(
        np.pi / 2, duration=4e-3, time_bw_product=3, system=p.sys, use='saturation'
    )
    g.freq_offset = -p.fat_offres_freq
    assert flip_profile(g, 0.0)[0] == pytest.approx(26, abs=1.5)


def test_flip_profile_matches_hard_pulse_rotation():
    """On resonance, a constant pulse of area theta tips by exactly theta; off
    resonance it follows the closed-form Rabi result."""
    p = load_params()
    # 1 ms at the 2 us RF raster
    rf = pp.make_arbitrary_rf(np.ones(500), np.pi / 2, system=p.sys, use='saturation')
    b1 = 0.25 / 1e-3  # Hz for 90 deg in 1 ms
    df = np.array([0.0, 300.0, -700.0])
    weff = np.sqrt(b1**2 + df**2)
    expected = np.degrees(np.arccos(1 - 2 * (b1 / weff) ** 2 * np.sin(np.pi * weff * 1e-3) ** 2))
    assert flip_profile(rf, df) == pytest.approx(expected, abs=1e-3)
    with pytest.raises(ValueError):
        flip_profile(pp.make_block_pulse(np.pi / 2, duration=1e-3, system=p.sys), 0.0)


@pytest.mark.parametrize('ftype', ['min', 'ls'])
@pytest.mark.parametrize('tbw', [1.5, 2, 3])
def test_dzrf_ex_matches_sigpy(ftype, tbw):
    """lib/slr.py is a port of sigpy.mri.rf.dzrf's 'ex' path. sigpy casts real
    FFT input to complex64 inside its min-phase factorization, hence the
    looser tolerance for 'min'. Measured: ~1e-5 ('min'), ~2e-7 ('ls')."""
    srf = pytest.importorskip('sigpy.mri.rf')
    ours = dzrf_ex(200, tbw, ftype)
    ref = np.real(srf.dzrf(200, tbw, 'ex', ftype))
    assert np.linalg.norm(ours - ref) / np.linalg.norm(ref) < (1e-4 if ftype == 'min' else 1e-6)


def test_dzrf_ex_ls_is_symmetric_and_ftype_is_checked():
    """'ls' is a linear-phase design, so its waveform is time-symmetric; 'min'
    is not."""
    mp = dzrf_ex(200, 2, 'min')
    ls = dzrf_ex(200, 2, 'ls')
    assert np.allclose(ls, ls[::-1], atol=1e-6 * np.abs(ls).max())
    assert not np.allclose(mp, mp[::-1], atol=1e-2 * np.abs(mp).max())
    with pytest.raises(ValueError):
        dzrf_ex(200, 2, 'pm')


def test_fatsat_duration_and_tbw_are_honored():
    """A different (longer) design still lands on the raster at its own
    duration -- the params knobs aren't silently ignored."""
    p = load_params()
    fs = replace(p.fatsat, dur=8e-3, tbw=3, ftype='ls')
    rf = make_fatsat_rf(fs, p.sys, p.fat_offres_freq)
    assert len(rf.signal) * p.sys.rf_raster_time == pytest.approx(8e-3)
    assert flip_profile(rf, 0.0)[0] < 2.0
