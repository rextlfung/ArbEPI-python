"""Independent correctness check: read k-space back out of the assembled
Pulseq sequence and confirm it matches the `schedules` array used to build
it. `seq.check_timing()` passing only proves the sequence is well-formed —
it says nothing about whether gradient amplitudes/signs/scales are correct
(a sign-flipped blip or an off-by-one prephaser scale still produces a
valid, well-timed .seq that samples the wrong k-space locations). This
closes that gap for both the encoded (ArbEPI) and zeroed (EPIcal) cases.
"""

from dataclasses import replace

import numpy as np
import pytest

from lib.readout_from_params import make_readout_grads_from_params
from params import load_params
from sampling.gen_sampling_masks import gen_sampling_masks
from sequences.ArbEPI import _compute_schedules, generate_arbepi
from sequences.deGRE import generate_degre
from sequences.EPIcal import generate_epical
from sequences.noise import generate_noise


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
    rg = make_readout_grads_from_params(max_ky_step, max_kz_step, p)

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
    rg = make_readout_grads_from_params(max_ky_step, max_kz_step, p)

    k_traj_adc, *_ = epical_seq.calculate_kspace()
    kxo_epical = k_traj_adc[0, : rg.Nfid]
    kxe_epical = k_traj_adc[0, rg.Nfid : rg.Nfid * 2]

    np.testing.assert_allclose(scan_info['kxo'].ravel(), kxo_epical, atol=1e-6)
    np.testing.assert_allclose(scan_info['kxe'].ravel(), kxe_epical, atol=1e-6)


def test_noise_nfid_matches_arbepi(tmp_path):
    """generate_noise() rebuilds its readout gradients from scan_info.mat
    to match the imaging readout's Nfid/gro duration -- it must go through
    make_readout_grads_from_params() (the derated POPE sys + blip_sys), the
    same as generate_arbepi()/generate_epical(), not params.sys directly.
    Regression test for a real bug where noise.py bypassed that helper and
    got a shorter Nfid, which preprocessing's calibration guard rejects
    outright (see preprocessing/preprocess.py's Nfid mismatch check)."""
    import hdf5storage

    p = _small_params(tmp_path)
    omegas = gen_sampling_masks(p.R, p)
    generate_arbepi(omegas, p, seqname='xcheck')

    schedules = hdf5storage.loadmat(str(tmp_path / 'scan_info.mat'))['schedules']
    max_ky_step = np.max(np.abs(np.diff(schedules[..., 0], axis=2)))
    max_kz_step = np.max(np.abs(np.diff(schedules[..., 1], axis=2)))
    rg = make_readout_grads_from_params(max_ky_step, max_kz_step, p)

    noise_seq = generate_noise(p, seqname='xcheck_noise')
    assert noise_seq.get_block(1).adc.num_samples == rg.Nfid


def test_arbepi_kx_coverage_and_nyquist(tmp_path):
    """Both readout parities must actually cover the full +-kmax range
    (the recentering fix: gx pre-winds to -S/2, not -kmax, so the a1 lost
    to the circular shift no longer shifts the sampled window off-center),
    and no two consecutive kx samples may be farther apart than deltak
    (Nyquist), including on the asymmetric POPE ramps."""
    p = _small_params(tmp_path)
    omegas = gen_sampling_masks(p.R, p)
    seq = generate_arbepi(omegas, p, seqname='xcheck')

    schedules, _ = _compute_schedules(
        omegas, p.ETL, p.Nshots, p.epi_trajectory, deltak=(1 / p.fov[1], 1 / p.fov[2]),
    )
    max_ky_step = np.max(np.abs(np.diff(schedules[..., 0], axis=2)))
    max_kz_step = np.max(np.abs(np.diff(schedules[..., 1], axis=2)))
    rg = make_readout_grads_from_params(max_ky_step, max_kz_step, p)

    k_traj_adc, *_ = seq.calculate_kspace()
    deltak_x = rg.deltak[0]
    k_need = (p.Nx / 2 - 0.5) * deltak_x
    for parity in (0, 1):  # odd/even echoes of the first shot
        kx = k_traj_adc[0, parity * rg.Nfid : (parity + 1) * rg.Nfid]
        assert kx.min() <= -k_need, f'parity {parity}: -kmax not covered ({kx.min():.1f})'
        assert kx.max() >= k_need, f'parity {parity}: +kmax not covered ({kx.max():.1f})'
        assert np.max(np.abs(np.diff(kx))) <= deltak_x * (1 + 1e-6), (
            f'parity {parity}: kx sample spacing exceeds Nyquist'
        )
    # With asymmetric ramps each parity's own window is NOT symmetric about
    # kx = 0 (odd spans [-S/2 + a1, S/2 - a_d]); it's the two parities that
    # mirror each other sample-for-sample (even-echo kx(t) = -odd-echo
    # kx(t) at the same in-block time) -- the property the -S/2 pre-wind
    # guarantees, and what makes the kx = 0 crossing time parity-independent.
    kx_odd = k_traj_adc[0, : rg.Nfid]
    kx_even = k_traj_adc[0, rg.Nfid : 2 * rg.Nfid]
    np.testing.assert_allclose(kx_even, -kx_odd, atol=1e-6)


def test_arbepi_schedule_echo_times_match_measured_kx_zero_crossings(tmp_path):
    """Ground-truth check of schedules[..., 2] (per-echo acquisition time,
    used by a future B0-correction consumer): the saved echo times must
    match the kx(t) = 0 crossing times measured from the assembled
    sequence itself, for every echo of a shot -- both parities. Before the
    echo_offset fix these were early by the gro1 lead-in block plus the
    in-block crossing time (~0.5-0.6 ms at default params)."""
    import hdf5storage

    p = _small_params(tmp_path)
    omegas = gen_sampling_masks(p.R, p)
    seq = generate_arbepi(omegas, p, seqname='xcheck')

    scan_info = hdf5storage.loadmat(str(tmp_path / 'scan_info.mat'))
    et_saved = scan_info['schedules'][0, 0, :, 2]

    schedules = scan_info['schedules']
    max_ky_step = np.max(np.abs(np.diff(schedules[..., 0], axis=2)))
    max_kz_step = np.max(np.abs(np.diff(schedules[..., 1], axis=2)))
    rg = make_readout_grads_from_params(max_ky_step, max_kz_step, p)

    k_traj_adc, _, t_excitation, _, t_adc = seq.calculate_kspace()
    # First shot's echoes: interpolate the kx = 0 crossing within each
    # echo's ADC window from the measured trajectory.
    t_rf = t_excitation[0]
    for e in range(p.ETL):
        kx = k_traj_adc[0, e * rg.Nfid : (e + 1) * rg.Nfid]
        t = t_adc[e * rg.Nfid : (e + 1) * rg.Nfid]
        sign_change = np.nonzero(np.diff(np.sign(kx)))[0]
        assert len(sign_change) == 1, f'echo {e}: expected exactly one kx=0 crossing'
        i = sign_change[0]
        t_cross = t[i] + (0 - kx[i]) * (t[i + 1] - t[i]) / (kx[i + 1] - kx[i])
        # ~5 gradient-raster steps of tolerance: the RF-center time
        # convention (dead time, ringdown) isn't exactly the saved
        # anchor's `0.5 * calc_duration(rf)`.
        assert t_cross - t_rf == pytest.approx(et_saved[e], abs=5 * p.sys.grad_raster_time), (
            f'echo {e}: measured kx=0 crossing differs from saved echo time'
        )


def test_readout_ramps_are_asymmetric():
    """The composite readout waveform's realized edge slews must reflect
    ro_slew_rise/ro_slew_fall, and the gro1 lead-in must span exactly half
    the blip window (the circular-shift split point)."""
    p = load_params()
    assert p.ro_slew_fall > p.ro_slew_rise  # POPE: fall faster than rise
    rg = make_readout_grads_from_params(4, 4, p)

    gamma = p.sys.gamma
    w, tt = rg.gro.waveform, rg.gro.tt
    slews = np.diff(w) / np.diff(tt) / gamma  # T/m/s
    # Steepest downward edge ~ ro_slew_fall; steepest upward edge (the
    # remainder of the throttled rise) ~ ro_slew_rise. Ramp times are
    # ceil'd to the raster, so realized slew is <= nominal, within one
    # raster step's worth.
    assert np.max(-slews) == pytest.approx(p.ro_slew_fall, rel=0.05)
    assert np.max(-slews) <= p.ro_slew_fall * (1 + 1e-6)
    assert np.max(slews) == pytest.approx(p.ro_slew_rise, rel=0.05)
    assert np.max(slews) <= p.ro_slew_rise * (1 + 1e-6)

    assert rg.gro1.shape_dur == pytest.approx(rg.blip_duration / 2)


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
    rg = make_readout_grads_from_params(max_ky_step, max_kz_step, p)

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


def test_degre_excitation_is_centered(tmp_path):
    """deGRE inlines its slab-selective excitation rather than calling
    make_excitation_pulse, but must apply the same
    gz_ss.delay = rf.delay - gz_ss.rise_time resync after trap4ge --
    trap4ge always rebuilds the trapezoid from scratch (lib/trap4ge.py),
    which resets delay to 0 regardless of whether its rise/flat/fall
    rounding changes anything, so without the resync the RF pulse is
    decentered within gz_ss's flat top right now, not just if crt changes.
    The iZ=0 rows are pure receive-gain-calibration lines (y_step =
    z_step = 0, see generate_degre's loop) -- ky is unaffected by this bug
    (it never touches gz_ss/gz_ssr), but a decentered excitation leaves a
    nonzero residual kz at readout via gz_ssr's now-mistimed rewind."""
    p = replace(
        load_params(), Ny_degre=2, Nz_degre=2, Ndummy_zloops=0, output_dir=str(tmp_path),
    )
    seq = generate_degre(p, seqname='xcheck_degre')

    k_traj_adc, *_ = seq.calculate_kspace()
    # First ADC-bearing acquisition overall is iZ=0, iY=0, echo c=0.
    ky, kz = k_traj_adc[1, 0], k_traj_adc[2, 0]
    assert abs(ky) < 1e-6, f'ky={ky} (expected ~0)'
    assert abs(kz) < 1e-6, f'kz={kz} (expected ~0)'


def test_degre_raises_actionable_error_below_minimum_tr(tmp_path):
    """generate_degre raises a clear, actionable error for a below-minimum
    TE_degre (delay_te < 0); TR_degre must be guarded the same way, or a
    too-short TR_degre fails deep inside pp.make_delay with a generic
    error that names neither TR_degre nor the achievable minimum."""
    p = replace(load_params(), TR_degre=1e-6, output_dir=str(tmp_path))
    with pytest.raises(ValueError, match='TR_degre'):
        generate_degre(p, seqname='xcheck_degre')
