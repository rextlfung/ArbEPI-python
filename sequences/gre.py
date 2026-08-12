"""Ported from ../ArbEPI/src/GRE.m — T1-weighted gradient echo sequence.

Used for coil sensitivity map estimation. Acquires a slightly larger FOV
than EPI, cropped during reconstruction.

Outputs: <output_dir>/GRE.seq (Pulseq format).

Note: this file inlines the slab-selective excitation pulse construction
(trap4ge only, no `gzSS.delay = rf.delay - gzSS.riseTime` resync) rather
than calling make_excitation_pulse, exactly mirroring GRE.m — which does
the same inline construction, omitting that resync line that
make_excitation_pulse.m has. That asymmetry is preserved here rather than
silently fixed, since it's unclear whether it's intentional.
"""

import math
import os

import numpy as np
import pypulseq as pp

from lib.make_fatsat_rf import make_fatsat_rf
from lib.trap4ge import trap4ge
from params import FatsatParams, Params


def generate_gre(params: Params, seqname: str = 'GRE') -> pp.Sequence:
    os.makedirs(params.output_dir, exist_ok=True)
    sys = params.sys
    crt = params.crt

    seq = pp.Sequence(system=sys)

    # Fat-sat (longer pulse than EPI for better fat suppression). Local
    # override, not touching the shared params.fatsat (mirrors GRE.m
    # overwriting its own local `fatsat` workspace variable).
    fatsat_gre = FatsatParams(flip=90, sl_thick=1e5, tbw=3, dur=8e-3)
    rfsat = make_fatsat_rf(fatsat_gre, sys, params.fat_offres_freq)

    # Slab-selective excitation (same pulse as EPI)
    rf, gz_ss, gz_ssr = pp.make_sinc_pulse(
        params.fa / 180 * math.pi,
        duration=params.rf_dur,
        slice_thickness=0.9 * params.fov[2],
        time_bw_product=params.rf_tb,
        system=sys,
        use='excitation',
        return_gz=True,
    )
    gz_ss = trap4ge(gz_ss, crt, sys)
    gz_ssr = trap4ge(gz_ssr, crt, sys)

    # Readout and phase-encode gradients
    deltak = [1 / f for f in params.fov_gre]
    Tread = params.Nx_gre * params.dwell

    gy_pre = trap4ge(
        pp.make_trapezoid('y', system=sys, area=params.Ny_gre * deltak[1] / 2, duration=params.Tpre), crt, sys
    )
    gz_pre = trap4ge(
        pp.make_trapezoid('z', system=sys, area=params.Nz_gre * deltak[2] / 2, duration=params.Tpre), crt, sys
    )

    gxtmp = pp.make_trapezoid('x', system=sys, amplitude=params.Nx_gre * deltak[0] / Tread, flat_time=Tread)
    gx_pre = trap4ge(
        pp.make_trapezoid('x', system=sys, area=-gxtmp.area / 2, duration=params.Tpre), crt, sys
    )

    adc = pp.make_adc(params.Nx_gre, system=sys, duration=Tread, delay=gxtmp.rise_time)

    # Extend flat time to split at end of ADC dead time
    gx = trap4ge(
        pp.make_trapezoid(
            'x', system=sys, amplitude=params.Nx_gre * deltak[0] / Tread, flat_time=Tread + adc.dead_time
        ),
        crt,
        sys,
    )

    # Ported literally from GRE.m, which uses deltak[0] (kx spacing) for
    # both the z and x spoilers rather than deltak[2] for gz_spoil.
    gz_spoil = pp.make_trapezoid('z', system=sys, area=params.Nx_gre * deltak[0] * params.n_cycles_spoil_gre)
    gx_spoil = pp.make_trapezoid('x', system=sys, area=params.Nx_gre * deltak[0] * params.n_cycles_spoil_gre)

    # Phase-encode step vectors
    pe1_steps = (np.arange(params.Ny_gre) - params.Ny_gre / 2) / params.Ny_gre * 2
    pe2_steps = (np.arange(params.Nz_gre) - params.Nz_gre / 2) / params.Nz_gre * 2

    # TE and TR delays
    te_min = (
        max(pp.calc_duration(rf), pp.calc_duration(gz_ss)) / 2
        + pp.calc_duration(gz_ssr)
        + pp.calc_duration(gx_pre)
        + adc.delay
        + params.Nx_gre / 2 * params.dwell
    )
    delay_te = math.ceil((params.TE_gre - te_min) / sys.grad_raster_time) * sys.grad_raster_time
    tr_min = (
        max(pp.calc_duration(rf), pp.calc_duration(gz_ss))
        + pp.calc_duration(gz_ssr)
        + delay_te
        + pp.calc_duration(gx_pre)
        + pp.calc_duration(gx)
        + pp.calc_duration(gx_spoil)
    )
    delay_tr = math.ceil((params.TR_gre - tr_min) / sys.grad_raster_time) * sys.grad_raster_time

    # Assemble sequence.
    # iZ < 0: dummy shots to reach steady state
    # iZ = 0: ADC on for receive gain calibration (GE auto-prescan)
    # iZ > 0: image acquisition
    rf_count = 1

    for iZ in range(-params.Ndummy_zloops, params.Nz_gre + 1):
        is_dummy_tr = iZ < 0
        print(f'z encode {iZ} of {params.Nz_gre}')

        # Fat-sat
        TRID = 1 if is_dummy_tr else 2
        seq.add_block(rfsat, pp.make_label('TRID', 'SET', TRID))
        seq.add_block(gz_spoil)

        for iY in range(params.Ny_gre):
            # Phase-encode steps off during dummy and gain-cal shots
            y_step = pe1_steps[iY] if iZ > 0 else 0.0
            z_step = pe2_steps[max(0, iZ - 1)] if iZ > 0 else 0.0

            # RF spoiling
            rf_phase = (0.5 * params.rf_phase_0 * rf_count**2) % 360.0
            rf.phase_offset = rf_phase / 180 * math.pi
            adc.phase_offset = rf_phase / 180 * math.pi
            rf_count += 1

            seq.add_block(rf, gz_ss)
            seq.add_block(gz_ssr)
            seq.add_block(pp.make_delay(delay_te))
            seq.add_block(gx_pre, pp.scale_grad(gy_pre, y_step), pp.scale_grad(gz_pre, z_step))
            if is_dummy_tr:
                seq.add_block(gx)
            else:
                seq.add_block(gx, adc)
            seq.add_block(gx_spoil, pp.scale_grad(gy_pre, -y_step), pp.scale_grad(gz_pre, -z_step))
            seq.add_block(pp.make_delay(delay_tr))

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
