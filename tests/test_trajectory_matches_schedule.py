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

from lib.make_readout_grads import make_readout_grads
from params import load_params
from sampling.gen_sampling_masks import gen_sampling_masks
from sequences.arbepi import _compute_schedules, generate_arbepi
from sequences.epical import generate_epical


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
    rg = make_readout_grads(max_ky_step, max_kz_step, p.Nx, p.fov, p.dwell, p.sys, p.crt)

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


def test_epical_trajectory_is_centered(tmp_path):
    """EPIcal zeroes all ky/kz encoding — every echo should read back k~0."""
    p = _small_params(tmp_path)
    omegas = gen_sampling_masks(p.R, p)
    generate_arbepi(omegas, p, seqname='xcheck')  # writes samp_locs.mat that EPIcal loads

    seq = generate_epical(p, seqname='xcheck_cal')

    import hdf5storage

    schedules = hdf5storage.loadmat(str(tmp_path / 'samp_locs.mat'))['schedules'].astype(float)
    max_ky_step = np.max(np.abs(np.diff(schedules[..., 0], axis=2)))
    max_kz_step = np.max(np.abs(np.diff(schedules[..., 1], axis=2)))
    rg = make_readout_grads(max_ky_step, max_kz_step, p.Nx, p.fov, p.dwell, p.sys, p.crt)

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
