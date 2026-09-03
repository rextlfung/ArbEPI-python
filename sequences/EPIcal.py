"""Ported from ../ArbEPI/src/EPIcal.m — EPI ghost-correction calibration sequence.

Structurally mirrors ArbEPI but with no ky/kz blips, producing unencoded
readout lines at k-space center. Requires scan_info.mat from a prior
generate_arbepi() run (used to match readout gradient design so the
calibration trajectory matches the imaging trajectory).

Outputs: <output_dir>/EPIcal.seq. kxoe<Nx>.mat (odd/even echo k-space
trajectories for ghost correction, now scan_info.mat's kxo/kxe) used to be
written here too, but is now written by sequences/ArbEPI.py instead --
EPIcal's own kx trajectory is identical to ArbEPI's (this sequence is just
ArbEPI's readout with the y/z blips zeroed, which doesn't affect gx
timing/area at all), so there's no need for a second, redundant copy.
"""

import copy
import os

import hdf5storage
import numpy as np
import pypulseq as pp

from lib.calc_te_tr_delays import calc_te_tr_delays
from lib.make_excitation_pulse import make_excitation_pulse
from lib.make_fatsat_rf import make_fatsat_rf
from lib.make_prephasers import make_prephasers
from lib.make_spoilers import make_spoilers
from lib.mask2epi import max_blip_steps
from lib.readout_from_params import derated_sys, make_readout_grads_from_params
from params import Params


def generate_epical(params: Params, seqname: str = 'EPIcal') -> pp.Sequence:
    os.makedirs(params.output_dir, exist_ok=True)
    # Derated system for the non-readout gradients, matching ArbEPI's
    # (same gradient design, see sequences/ArbEPI.py's sys comment).
    sys = derated_sys(params)

    # Load EPI schedule to match readout gradient design. Index base
    # doesn't matter here: only consecutive-echo differences are used.
    # scan_info.mat is written as MATLAB v7.3 (HDF5-based, see
    # sequences/ArbEPI.py) — scipy.io.loadmat cannot read v7.3 at all, so
    # this must use hdf5storage too.
    schedules = hdf5storage.loadmat(os.path.join(params.output_dir, 'scan_info.mat'))['schedules']

    # Excitation pulse (identical to ArbEPI)
    rf, gz_ss, gz_ssr = make_excitation_pulse(params.fa, params.rf_dur, params.rf_tb, params.fov, sys, params.crt)

    # Fat-sat pulse (identical to ArbEPI)
    rfsat = make_fatsat_rf(params.fatsat, sys, params.fat_offres_freq)

    # Readout gradients — sized to match ArbEPI so calibration trajectory applies
    max_ky_step, max_kz_step = max_blip_steps(schedules)
    rg = make_readout_grads_from_params(max_ky_step, max_kz_step, params)

    # Prephasers and spoilers (identical to ArbEPI)
    gx_pre, gy_pre, gz_pre = make_prephasers(params.Nx, params.Ny, params.Nz, params.fov, sys, params.crt)
    gx_spoil, gy_spoil, gz_spoil = make_spoilers(
        params.Nx, params.Ny, params.Nz, params.fov, params.n_cycles_spoil, sys, params.crt
    )

    # TE and TR delays (identical to ArbEPI)
    te_delay, tr_delay, min_te, min_tr = calc_te_tr_delays(
        rf, rfsat, gz_ss, gz_ssr, gx_pre, gy_pre, gz_pre, rg.gro, gx_spoil, gy_spoil, gz_spoil,
        params.ETL, params.TE, params.TR, sys, echo_offset=rg.echo_offset,
    )

    # Assemble sequence (full-hardware system, matching ArbEPI -- the POPE
    # fall ramp deliberately exceeds the derate; see ArbEPI.py's comment)
    sys_seq = copy.deepcopy(params.sys)
    sys_seq.adc_dead_time = 0  # suppress warnings; no back-to-back ADC blocks
    seq = pp.Sequence(system=sys_seq)

    rf_count = 1

    for shot in range(-params.Ndummyshots, params.Nshots):
        is_dummy = shot < 0
        TRID = 1 if is_dummy else 2  # TRID 1 = dummy, TRID 2 = real (see Pulseq on GE manual)

        # Fat-sat
        seq.add_block(rfsat, pp.make_label('TRID', 'SET', TRID))
        seq.add_block(gx_spoil, gz_spoil)

        # RF spoiling (quadratic phase cycling)
        rf_phase = (0.5 * params.rf_phase_0 * rf_count**2) % 360.0
        rf.phase_offset = rf_phase / 180 * np.pi
        rg.adc.phase_offset = rf_phase / 180 * np.pi
        rf_count += 1

        # Slab-selective excitation + slice-select rephaser
        seq.add_block(rf, gz_ss)
        seq.add_block(gz_ssr)

        # TE padding delay
        if params.TE > min_te:
            seq.add_block(pp.make_delay(te_delay))

        # Prephase to k-space center (no ky/kz encoding — blips will be
        # zero). gx pre-winds to -S/2, identical to ArbEPI (see
        # ReadoutGrads.gx_pre_scale) -- required for the calibration kx
        # trajectory to match the imaging one.
        seq.add_block(
            pp.scale_grad(gx_pre, rg.gx_pre_scale),
            pp.scale_grad(gy_pre, 0),
            pp.scale_grad(gz_pre, 0),
        )

        # EPI readout — same waveform as ArbEPI, blips scaled to zero
        seq.add_block(rg.gro1)
        for echo in range(params.ETL - 1):
            gro_line = pp.scale_grad(rg.gro, (-1) ** echo)
            gy_blip0 = pp.scale_grad(rg.gy_blip, 0)
            gz_blip0 = pp.scale_grad(rg.gz_blip, 0)
            if not is_dummy:
                seq.add_block(rg.adc, gro_line, gy_blip0, gz_blip0)
            else:
                seq.add_block(gro_line, gy_blip0, gz_blip0)
        # Last echo line
        gro2_line = pp.scale_grad(rg.gro2, (-1) ** (params.ETL - 1))
        if not is_dummy:
            seq.add_block(rg.adc, gro2_line)
        else:
            seq.add_block(gro2_line)

        # Spoilers: x dephase, y/z no net phase (no encoding was applied)
        seq.add_block(gx_spoil, pp.scale_grad(gy_spoil, 0), pp.scale_grad(gz_spoil, 1))

        # TR padding delay
        if params.TR > min_tr:
            seq.add_block(pp.make_delay(tr_delay))

    # Check sequence timing
    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed! Error listing follows:')
        print(error_report)

    # Write Pulseq .seq file
    seq.set_definition('FOV', params.fov)
    seq.set_definition('Name', seqname)
    seq.write(os.path.join(params.output_dir, f'{seqname}.seq'))

    return seq
