"""Ported from ../ArbEPI/src/noise.m — noise prescan for receiver noise
covariance estimation.

Acquires ADC data without RF or gradients (aside from one trailing dummy RF
block, needed so the scanner recognizes a second block type). Requires
scan_info.mat from a prior generate_arbepi() run to match readout geometry.

Outputs: <output_dir>/noise.seq (Pulseq format).
"""

import copy
import math
import os

import hdf5storage
import pypulseq as pp

from lib.make_excitation_pulse import make_excitation_pulse
from lib.mask2epi import max_blip_steps
from lib.readout_from_params import derated_sys, make_readout_grads_from_params
from params import Params


def generate_noise(params: Params, seqname: str = 'noise') -> pp.Sequence:
    os.makedirs(params.output_dir, exist_ok=True)
    sys = derated_sys(params)

    # Rebuild readout gradient objects matching the EPI sequence.
    # scan_info.mat is written as MATLAB v7.3 (HDF5-based, see
    # sequences/ArbEPI.py) — scipy.io.loadmat cannot read v7.3 at all.
    schedules = hdf5storage.loadmat(os.path.join(params.output_dir, 'scan_info.mat'))['schedules']
    max_ky_step, max_kz_step = max_blip_steps(schedules)
    rg = make_readout_grads_from_params(max_ky_step, max_kz_step, params)
    rf, _, _ = make_excitation_pulse(params.fa, params.rf_dur, params.rf_tb, params.fov, sys, params.crt)

    # Number of ADC repetitions needed
    Nsamples_noise = 20 * params.Ncoils**2
    Nreps = math.ceil(Nsamples_noise / rg.Nfid)

    # Delay to pad each block to EPI readout duration
    adc_dead_time = sys.adc_dead_time
    adc_total_dur = adc_dead_time + pp.calc_duration(rg.adc)
    pad_duration = pp.calc_duration(rg.gro) - adc_total_dur

    delay_block = pp.make_delay(pad_duration) if pad_duration > 1e-9 else None

    # Assemble sequence. The Sequence system is the full-hardware params.sys,
    # not the derated `sys` above -- matching ArbEPI.py/EPIcal.py (see their
    # sys_seq comment for why: the POPE fall ramp deliberately runs above the
    # derate). This sequence has no gradients, so the choice has no
    # observable effect here, but keeping it consistent avoids a gratuitous
    # divergence for readers comparing the four sequence files.
    sys_seq = copy.deepcopy(params.sys)
    sys_seq.adc_dead_time = 0  # suppress warnings; no back-to-back ADC blocks
    seq = pp.Sequence(system=sys_seq)
    for _ in range(Nreps):
        seq.add_block(rg.adc, pp.make_label('TRID', 'SET', 1))
        if delay_block is not None:
            seq.add_block(delay_block)

    # Dummy RF pulse at end so the scanner recognises a second block type
    seq.add_block(rf, pp.make_label('TRID', 'SET', 2))

    # Check sequence timing
    print('Validating sequence...')
    ok, error_report = seq.check_timing()
    if ok:
        print('  Timing check PASSED.\n')
    else:
        print('  Timing check WARNINGS/ERRORS:')
        print(error_report)

    # Write Pulseq .seq file
    seq.set_definition('FOV', params.fov)
    seq.set_definition('Name', seqname)
    fn_seq = os.path.join(params.output_dir, f'{seqname}.seq')
    seq.write(fn_seq)
    print(f'Sequence written to: {fn_seq}')

    return seq
