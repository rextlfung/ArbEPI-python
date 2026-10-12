"""mGRE (sequences/mGRE.py): what the assembled sequence encodes, read back
out of the Pulseq object rather than from the design numbers."""

import warnings

import numpy as np
import pytest

from params import load_params
from sequences.mGRE import design_mgre, generate_mgre, iter_encodes

# 7.2 mm voxels over the real 216 x 216 x 144 mm FOV: 30 x 30 x 20, small
# enough to build and read back in a few seconds.
KW = dict(res=(7.2e-3,) * 3, fov=(0.216, 0.216, 0.144), n_echoes=6, n_dummy_zloops=1)


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    p = load_params(output_dir=str(tmp_path_factory.mktemp('mgre')))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        seq, d = generate_mgre(p, verbose=False, **KW)
    return p, seq, d


def test_timing_and_scan_duration_match_design(built):
    _, seq, d = built
    ok, report = seq.check_timing()
    assert ok, report
    assert seq.duration()[0] == pytest.approx(d.scan_time, abs=1e-9)


def test_echo_times_are_fat_water_periods(built):
    p, seq, d = built
    _, _, t_exc, _, t_adc = seq.calculate_kspace()
    nx, ny, _ = d.N
    per_shot = d.n_echoes * nx
    # First image shot: after the dummies (no ADC) and the iZ = 0 gain-cal
    # block, ny shots with ADC. t_exc has one entry per excitation.
    adc_shot = ny
    t_rf = t_exc[d.blocks['n_dummy_zloops'] * ny + ny]
    t = t_adc[adc_shot * per_shot:(adc_shot + 1) * per_shot].reshape(d.n_echoes, nx)
    te = (t[:, nx // 2 - 1] + t[:, nx // 2]) / 2 - t_rf  # kx = 0 is between those samples
    assert te == pytest.approx(d.TE, abs=1e-9)
    assert np.diff(te) == pytest.approx(1 / p.fat_offres_freq, abs=3e-6)
    assert te[0] == pytest.approx(1 / p.fat_offres_freq, abs=8e-6)


def test_every_echo_sees_the_same_kx_and_the_shots_phase_encode(built):
    _, seq, d = built
    k_traj_adc, *_ = seq.calculate_kspace()
    nx, ny, nz = d.N
    n_adc_shots = (1 + nz) * ny
    k = k_traj_adc.reshape(3, n_adc_shots, d.n_echoes, nx)
    dk = [1 / f for f in d.fov]
    # monopolar: every lobe's kx axis is identical and centred
    for e in range(1, d.n_echoes):
        np.testing.assert_allclose(k[0, :, e], k[0, :, 0], atol=1e-3 * dk[0])
    kx = k[0, 0, 0]
    np.testing.assert_allclose(np.diff(kx), dk[0], rtol=1e-6)
    assert kx.mean() == pytest.approx(0, abs=1e-6 * dk[0])  # centred on the echo
    # ky/kz constant along the train, and equal to the encode index
    assert np.ptp(k[1], axis=(1, 2)).max() < 1e-3 * dk[1]
    assert np.ptp(k[2], axis=(1, 2)).max() < 1e-3 * dk[2]
    enc = [e for e in iter_encodes(d.N, d.blocks['n_dummy_zloops']) if e[0] >= 0]
    for s, (iZ, iY) in enumerate(enc):
        if iZ == 0:
            assert abs(k[1, s, 0, 0]) < 1e-3 * dk[1] and abs(k[2, s, 0, 0]) < 1e-3 * dk[2]
        else:
            assert k[1, s, 0, 0] == pytest.approx((iY - ny / 2) * dk[1], abs=1e-3 * dk[1])
            assert k[2, s, 0, 0] == pytest.approx((iZ - 1 - nz / 2) * dk[2], abs=1e-3 * dk[2])


def test_default_protocol_fits_five_minutes_at_twenty_echoes():
    p = load_params()
    d = design_mgre(p, n_echoes=20)
    assert d.N == (90, 90, 60)
    assert d.scan_time < 300
    assert d.TE[0] == pytest.approx(1 / p.fat_offres_freq, abs=8e-6)
    assert np.diff(d.TE) == pytest.approx(1 / p.fat_offres_freq, abs=3e-6)
