"""Ported from ../ArbEPI/src/noise.m — noise prescan for receiver noise
covariance estimation.

Acquires ADC data without RF or gradients (aside from one trailing dummy RF
block, needed so the scanner recognizes a second block type). Requires
samp_locs.mat from a prior generate_arbepi() run to match readout geometry.

Outputs: <output_dir>/noise.seq (Pulseq format).
"""

import copy
import math
import os

import hdf5storage
import pypulseq as pp

from lib.make_excitation_pulse import make_excitation_pulse
from lib.make_readout_grads import make_readout_grads
from lib.mask2epi import max_blip_steps
from params import Params


def generate_noise(params: Params, seqname: str = 'noise') -> pp.Sequence:
    os.makedirs(params.output_dir, exist_ok=True)
    sys = params.sys

    # Rebuild readout gradient objects matching the EPI sequence.
    # samp_locs.mat is written as MATLAB v7.3 (HDF5-based, see
    # sequences/ArbEPI.py) — scipy.io.loadmat cannot read v7.3 at all.
    schedules = hdf5storage.loadmat(os.path.join(params.output_dir, 'samp_locs.mat'))['schedules']
    max_ky_step, max_kz_step = max_blip_steps(schedules)
    rg = make_readout_grads(max_ky_step, max_kz_step, params.Nx, params.fov, params.dwell, sys, params.crt)
    rf, _, _ = make_excitation_pulse(params.fa, params.rf_dur, params.rf_tb, params.fov, sys, params.crt)

    # Number of ADC repetitions needed
    Nsamples_noise = 20 * params.Ncoils**2
    Nreps = math.ceil(Nsamples_noise / rg.Nfid)

    # Delay to pad each block to EPI readout duration
    sys_seq = copy.deepcopy(sys)
    sys_seq.adc_dead_time = 0  # suppress warnings; no back-to-back ADC blocks
    adc_total_dur = sys_seq.adc_dead_time + pp.calc_duration(rg.adc)
    pad_duration = pp.calc_duration(rg.gro) - adc_total_dur

    delay_block = pp.make_delay(pad_duration) if pad_duration > 1e-9 else None

    # Assemble sequence
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
    fn_seq = os.path.join(params.output_dir, f'{seqname}.seq')
    seq.write(fn_seq)
    print(f'Sequence written to: {fn_seq}')

    return seq
