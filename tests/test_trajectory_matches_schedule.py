"""Independent correctness check: read k-space back out of the assembled
Pulseq sequence and confirm it matches the `schedules` array used to build
it. `seq.check_timing()` passing only proves the sequence is well-formed —
it says nothing about whether gradient amplitudes/signs/scales are correct
(a sign-flipped blip or an off-by-one prephaser scale still produces a
valid, well-timed .seq that samples the wrong k-space locations). This
closes that gap for both the encoded (ArbEPI) and zeroed (EPIcal) cases.
"""

import copy
from dataclasses import replace

import numpy as np
import pytest

from lib.make_readout_grads import make_readout_grads
from params import load_params
from sampling.gen_sampling_masks import gen_sampling_masks
from sequences.ArbEPI import _compute_schedules, generate_arbepi
from sequences.EPIcal import generate_epical


def _derated_sys(p):
    """Matches the PNS derate ArbEPI.py/EPIcal.py apply to their own sys copy."""
    sys = copy.deepcopy(p.sys)
    sys.max_slew = 100 * sys.gamma
    return sys


def _small_params(tmp_path):
    p = load_params()
    # Ny*Nz/R = 16*12/8 = 24 = Nshots*ETL
    return replace(
        p,
        Ny=16,
        Nz=12,
        ETL=8,
        Nshots=3,
        Nframes=1,
        sampling_method='caipi',
        R=8,
        output_dir=str(tmp_path),
    )


def test_arbepi_trajectory_matches_schedule(tmp_path):
    p = _small_params(tmp_path)
    omegas = gen_sampling_masks(p.R, p)
    schedules, _ = _compute_schedules(
        omegas, p.ETL, p.Nshots, p.epi_trajectory, deltak=(1 / p.fov[1], 1 / p.fov[2]),
    )  # 0-based, pre-savemat

    seq = generate_arbepi(omegas, p, seqname='xcheck')

    max_ky_step = np.max(np.abs(np.diff(schedules[..., 0], axis=2)))
    max_kz_step = np.max(np.abs(np.diff(schedules[..., 1], axis=2)))
    rg = make_readout_grads(max_ky_step, max_kz_step, p.Nx, p.fov, p.dwell, _derated_sys(p), p.crt)

    k_traj_adc, *_ = seq.calculate_kspace()
    Ny, Nz, Nfid = p.Ny, p.Nz, rg.Nfid

    for s in range(p.Nshots):
        for e in range(p.ETL):
            # Blips are delayed to play after the ADC window closes, so
            # ky/kz are flat across each echo's ADC window; sample the
            # midpoint.
            idx = (s * p.ETL + e) * Nfid + Nfid // 2
            iy_measured = round(k_traj_adc[1, idx] / rg.deltak[1] + Ny / 2)
            iz_measured = round(k_traj_adc[2, idx] / rg.deltak[2] + Nz / 2)
            assert iy_measured == schedules[0, s, e, 0], f'shot={s} echo={e} ky mismatch'
            assert iz_measured == schedules[0, s, e, 1], f'shot={s} echo={e} kz mismatch'


def test_arbepi_schedule_echo_times(tmp_path):
    """scan_info.mat's 'schedules' carries a 3rd channel, echo time (s
    since RF excitation) per acquisition, alongside (ky, kz) -- see
    sequences/ArbEPI.py's echo_times computation and
    preprocessing/preprocess.py's load_schedules(), which splits this back
    out for (ky, kz)-only consumers."""
    import hdf5storage

    p = _small_params(tmp_path)
    omegas = gen_sampling_masks(p.R, p)
    generate_arbepi(omegas, p, seqname='xcheck')

    schedules = hdf5storage.loadmat(str(tmp_path / 'scan_info.mat'))['schedules']
    assert schedules.shape == (p.Nframes, p.Nshots, p.ETL, 3)

    echo_times = schedules[..., 2]
    # Identical across every shot/frame -- readout timing doesn't vary.
    np.testing.assert_allclose(
        echo_times, np.broadcast_to(echo_times[0, 0], echo_times.shape)
    )
    et = echo_times[0, 0]
    # Strictly increasing, one uniform gro-duration step per echo.
    steps = np.diff(et)
    assert np.all(steps > 0)
    np.testing.assert_allclose(steps, steps[0], rtol=1e-9)
    # The nominal (TE-defining) echo sits at the continuous, generally
    # non-integer index ETL/2 - 0.5 (see sequences/ArbEPI.py's
    # echo_times comment) -- for even ETL that's exactly the midpoint
    # between echoes ETL/2 - 1 and ETL/2, whose average should equal the
    # achieved TE at that nominal echo, i.e. p.TE (within one raster step,
    # from calc_te_tr_delays.py's te_delay flooring).
    mid = 0.5 * (et[p.ETL // 2 - 1] + et[p.ETL // 2])
    assert mid == pytest.approx(p.TE, abs=1e-4)


def test_arbepi_kxoe_matches_epical(tmp_path):
    """kxo/kxe are now written into scan_info.mat by generate_arbepi()
    (sequences/ArbEPI.py) instead of a separate kxoe<Nx>.mat written by
    generate_epical() (sequences/EPIcal.py) -- confirm the rationale for
    that move (EPIcal is just ArbEPI's readout with the y/z blips zeroed,
    which doesn't affect gx timing/area) actually holds by independently
    recomputing the same odd/even kx trajectory from EPIcal's own sequence
    and checking it matches what ArbEPI wrote."""
    import hdf5storage

    p = _small_params(tmp_path)
    omegas = gen_sampling_masks(p.R, p)
    generate_arbepi(omegas, p, seqname='xcheck')
    epical_seq = generate_epical(p, seqname='xcheck_cal')

    scan_info = hdf5storage.loadmat(str(tmp_path / 'scan_info.mat'))

    # Recompute rg exactly as generate_epical() does internally (readout
    # gradients sized off the real schedule's blip steps, even though
    # EPIcal's own blips are scaled to zero at assembly time -- see its
    # module docstring) so Nfid/Tread match, not a zeroed-blip rg (whose
    # gro trapezoid, and hence Nfid, would be a different size).
    schedules = scan_info['schedules']
    max_ky_step = np.max(np.abs(np.diff(schedules[..., 0], axis=2)))
    max_kz_step = np.max(np.abs(np.diff(schedules[..., 1], axis=2)))
    rg = make_readout_grads(max_ky_step, max_kz_step, p.Nx, p.fov, p.dwell, _derated_sys(p), p.crt)

    k_traj_adc, *_ = epical_seq.calculate_kspace()
    kxo_epical = k_traj_adc[0, : rg.Nfid]
    kxe_epical = k_traj_adc[0, rg.Nfid : rg.Nfid * 2]

    np.testing.assert_allclose(scan_info['kxo'].ravel(), kxo_epical, atol=1e-6)
    np.testing.assert_allclose(scan_info['kxe'].ravel(), kxe_epical, atol=1e-6)


def test_epical_trajectory_is_centered(tmp_path):
    """EPIcal zeroes all ky/kz encoding — every echo should read back k~0."""
    p = _small_params(tmp_path)
    omegas = gen_sampling_masks(p.R, p)
    generate_arbepi(omegas, p, seqname='xcheck')  # writes scan_info.mat that EPIcal loads

    seq = generate_epical(p, seqname='xcheck_cal')

    import hdf5storage

    schedules = hdf5storage.loadmat(str(tmp_path / 'scan_info.mat'))['schedules'].astype(float)
    max_ky_step = np.max(np.abs(np.diff(schedules[..., 0], axis=2)))
    max_kz_step = np.max(np.abs(np.diff(schedules[..., 1], axis=2)))
    rg = make_readout_grads(max_ky_step, max_kz_step, p.Nx, p.fov, p.dwell, _derated_sys(p), p.crt)

    k_traj_adc, *_ = seq.calculate_kspace()
    Nfid = rg.Nfid

    # Only real (non-dummy) shots carry an ADC event in EPIcal.
    for s in range(p.Nshots):
        for e in range(p.ETL):
            idx = (s * p.ETL + e) * Nfid + Nfid // 2
            ky = k_traj_adc[1, idx]
            kz = k_traj_adc[2, idx]
            assert abs(ky) < 1e-6, f'shot={s} echo={e} ky={ky} (expected ~0)'
            assert abs(kz) < 1e-6, f'shot={s} echo={e} kz={kz} (expected ~0)'
