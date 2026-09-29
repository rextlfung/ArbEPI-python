"""Slab-selective binomial water excitation (lib/make_water_excitation.py),
checked by Bloch simulation over slab position and frequency (lib/bloch.py).
The simulator itself is first checked on the existing single-sinc excitation."""

from dataclasses import replace

import numpy as np
import pypulseq as pp
import pytest

from lib.bloch import simulate_excitation
from lib.make_excitation_pulse import make_excitation_from_params, make_excitation_pulse
from lib.make_water_excitation import make_water_excitation
from lib.readout_from_params import derated_sys
from params import load_params


@pytest.fixture(scope='module')
def setup():
    p = load_params()
    return p, derated_sys(p), 0.9 * p.fov[2]


def _profile(rf, gz, gzr, L, f):
    z = np.linspace(-(L / 2 + 0.02), L / 2 + 0.02, 181)
    mxy, flip = simulate_excitation(rf, gz, gzr, f, z)
    return z, mxy, flip


def _fwhm(z, m):
    half = z[m > m.max() / 2]
    return half[-1] - half[0]


def _phase_ptp_deg(z, mxy, L):
    mid = len(z) // 2
    ph = np.unwrap(np.angle(mxy * np.conj(mxy[mid])))
    return np.degrees(np.ptp(ph[np.abs(z) < 0.4 * L]))


def test_simulator_reproduces_the_sinc_excitation(setup):
    """Known-good reference: the fat-sat mode's sinc gives the prescribed
    flip mid-slab, the prescribed slab width, and flat phase after its
    rephaser. A sign or area error in lib/bloch.py's rephaser shows up here as
    a phase twist of hundreds of degrees."""
    p, sys_, L = setup
    rf, gz, gzr = make_excitation_pulse(p.fa, p.rf_dur, p.rf_tb, p.fov, sys_, p.crt)
    z, mxy, flip = _profile(rf, gz, gzr, L, [0.0])
    assert flip[len(z) // 2, 0] == pytest.approx(p.fa, abs=0.05)
    assert _fwhm(z, np.abs(mxy[:, 0])) == pytest.approx(L, rel=0.03)
    assert _phase_ptp_deg(z, mxy[:, 0], L) < 3.0


def test_water_excitation_profile(setup):
    """Water: full flip mid-slab, same slab width, flat phase after the
    rephaser, <= 1% profile overshoot (Hanning subpulses; unapodized TBW 8
    overshoots 16%)."""
    p, sys_, L = setup
    we = p.water_exc
    rf, gz, gzr = make_water_excitation(
        p.fa, we.tbw, p.fov, sys_, p.crt, p.fat_offres_freq,
        binomial=we.binomial, apodization=we.apodization,
    )
    z, mxy, flip = _profile(rf, gz, gzr, L, [0.0, 80.0, -80.0, -150.0])
    mid = len(z) // 2
    assert flip[mid, 0] == pytest.approx(p.fa, abs=0.05)
    assert flip[:, 0].max() < 1.01 * p.fa
    assert _fwhm(z, np.abs(mxy[:, 0])) == pytest.approx(L, rel=0.03)
    assert _phase_ptp_deg(z, mxy[:, 0], L) < 2.0
    # 1-3-3-1 off-resonance cost: cos^3(pi f tau)
    assert flip[mid, 1] / p.fa == pytest.approx(0.886, abs=0.02)
    assert flip[mid, 2] / p.fa == pytest.approx(0.886, abs=0.02)
    assert flip[mid, 3] / p.fa == pytest.approx(0.643, abs=0.03)


def test_water_excitation_rejects_fat_across_the_whole_slab(setup):
    """Monopolar lobes: fat cancels everywhere in the slab, including the
    edges, where bipolar lobes left ~4 deg (~24% of the water flip) from
    alternating chemical-shift displacement."""
    p, sys_, L = setup
    we = p.water_exc
    rf, gz, gzr = make_water_excitation(
        p.fa, we.tbw, p.fov, sys_, p.crt, p.fat_offres_freq,
        binomial=we.binomial, apodization=we.apodization,
    )
    fat = -p.fat_offres_freq + np.arange(-100, 101, 25)
    _, _, flip = _profile(rf, gz, gzr, L, fat)
    assert flip.max() < 1.0  # vs ~18 deg water
    assert flip[:, len(fat) // 2].max() < 0.05  # exactly at the fat peak


def test_water_excitation_timing(setup):
    """Subpulses spaced half a fat period on the gradient raster, the RF
    center at the train's weighted center, on-raster gradient corners, and
    zero gradient moment from rf.center to the end of the rephaser (so
    calculate_kspace's reset at rf.center matches the spins)."""
    p, sys_, _ = setup
    rf, gz, gzr = make_water_excitation(p.fa, 8, p.fov, sys_, p.crt, p.fat_offres_freq)
    tau = round(1 / (2 * p.fat_offres_freq) / p.crt) * p.crt
    assert tau == pytest.approx(1120e-6)
    assert pp.calc_rf_center(rf)[0] == pytest.approx(rf.shape_dur / 2, abs=rf.t[1] - rf.t[0])
    assert np.allclose(np.round(gz.tt / p.crt) * p.crt, gz.tt)
    assert rf.freq_offset == 0
    t_c = rf.delay + pp.calc_rf_center(rf)[0] - gz.delay
    t = np.linspace(t_c, gz.tt[-1], 200001)
    after = np.trapezoid(np.interp(t, gz.tt, gz.waveform), t)
    assert after + gzr.area == pytest.approx(0, abs=1e-3 * abs(gzr.area))


def test_excitation_factory_modes():
    p = load_params()
    assert p.excitation == 'fatsat'  # default stays fat-sat
    sys_ = derated_sys(p)
    *_, rfsat = make_excitation_from_params(p, sys_)
    assert rfsat is not None
    *_, rfsat = make_excitation_from_params(replace(p, excitation='water'), sys_)
    assert rfsat is None
