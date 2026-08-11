"""Ported from ../ArbEPI/src/ArbEPI.m — the main 3D-EPI sequence.

Outputs: <output_dir>/ArbEPI.seq (Pulseq format), <output_dir>/samp_locs.mat
(schedules and partition map for reconstruction).

GE `.pge` export (write_to_ge.m) and the trailing MATLAB plotting figures
are intentionally not ported here — see ge_export.py (stage 8 of the
port plan) and plotting/plotting.py respectively.
"""

import copy
import os
from typing import Tuple

import hdf5storage
import numpy as np
import pypulseq as pp

from lib.calc_te_tr_delays import calc_te_tr_delays
from lib.make_excitation_pulse import make_excitation_pulse
from lib.make_fatsat_rf import make_fatsat_rf
from lib.make_prephasers import make_prephasers
from lib.make_readout_grads import make_readout_grads
from lib.make_spoilers import make_spoilers
from lib.mask2epi import mask2epi
from params import Params


def _compute_schedules(omegas: np.ndarray, ETL: int, Nshots: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run mask2epi per frame. Returns 0-based schedules/parts (see mask2epi)."""
    Ny, Nz, Nframes = omegas.shape
    schedules = np.zeros((Nframes, Nshots, ETL, 2), dtype=int)
    parts = np.zeros((Ny, Nz, Nframes), dtype=int)
    for frame in range(Nframes):
        schedules[frame], parts[:, :, frame] = mask2epi(omegas[:, :, frame], ETL, Nshots)
    return schedules, parts


def generate_arbepi(omegas: np.ndarray, params: Params, seqname: str = 'ArbEPI') -> pp.Sequence:
    """
    Parameters
    ----------
    omegas : (Ny, Nz, Nframes) boolean sampling mask (see gen_sampling_masks).
    params : loaded Params (see params.load_params).
    seqname : sequence name, used for the output filename and Pulseq 'Name'
        definition.

    Returns
    -------
    seq : the assembled pypulseq Sequence.
    """
    os.makedirs(params.output_dir, exist_ok=True)
    sys = params.sys

    # Excitation pulse
    rf, gz_ss, gz_ssr = make_excitation_pulse(params.fa, params.rf_dur, params.rf_tb, params.fov, sys, params.crt)

    # Fat-sat pulse
    rfsat = make_fatsat_rf(params.fatsat, sys, params.fat_offres_freq)

    # Generate EPI sampling schedule from mask
    schedules, parts = _compute_schedules(omegas, params.ETL, params.Nshots)

    # Infer maximum ky and kz blip steps across all frames and shots
    max_ky_step = np.max(np.abs(np.diff(schedules[..., 0], axis=2)))
    max_kz_step = np.max(np.abs(np.diff(schedules[..., 1], axis=2)))

    # Readout gradients and ADC event
    rg = make_readout_grads(max_ky_step, max_kz_step, params.Nx, params.fov, params.dwell, sys, params.crt)

    # Prephasers and spoilers
    gx_pre, gy_pre, gz_pre = make_prephasers(params.Nx, params.Ny, params.Nz, params.fov, sys, params.crt)
    gx_spoil, gy_spoil, gz_spoil = make_spoilers(
        params.Nx, params.Ny, params.Nz, params.fov, params.n_cycles_spoil, sys, params.crt
    )

    # Delays to achieve desired TE and TR
    te_delay, tr_delay, min_te, min_tr = calc_te_tr_delays(
        rf, rfsat, gz_ss, gz_ssr, gx_pre, gy_pre, gz_pre, rg.gro, gx_spoil, gy_spoil, gz_spoil,
        params.ETL, params.TE, params.TR, sys,
    )

    # Assemble sequence. Local copy: MATLAB's params.m re-executes fresh for
    # every sequence function, so `sys.adcDeadTime = 0` there never leaks
    # across ArbEPI/EPIcal/GRE/noise. Here `params.sys` may be one shared
    # object across all four calls, so mutate a copy instead.
    sys_seq = copy.deepcopy(sys)
    sys_seq.adc_dead_time = 0  # suppress warnings; no back-to-back ADC blocks
    seq = pp.Sequence(system=sys_seq)

    rf_count = 1
    Ny, Nz = params.Ny, params.Nz

    for frame in range(params.Nframes):
        print(f'Writing frame {frame + 1}')

        for shot in range(params.Nshots):
            # Fat-sat (label first block in each unique section with TRID for GE)
            seq.add_block(rfsat, pp.make_label('TRID', 'SET', 1))
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

            # Load this shot's k-space locations (0-based)
            y_locs = schedules[frame, shot, :, 0]
            z_locs = schedules[frame, shot, :, 1]

            # Move to first k-space location
            gy_pre_tmp = pp.scale_grad(gy_pre, (y_locs[0] - Ny / 2) / (-Ny / 2))
            gz_pre_tmp = pp.scale_grad(gz_pre, (z_locs[0] - Nz / 2) / (-Nz / 2))
            seq.add_block(gx_pre, gy_pre_tmp, gz_pre_tmp)

            # EPI readout — zip through k-space with alternating readout polarity
            seq.add_block(rg.gro1)
            for echo in range(len(y_locs) - 1):
                seq.add_block(
                    rg.adc,
                    pp.scale_grad(rg.gro, (-1) ** echo),
                    pp.scale_grad(rg.gy_blip, y_locs[echo + 1] - y_locs[echo]),
                    pp.scale_grad(rg.gz_blip, z_locs[echo + 1] - z_locs[echo]),
                )
            # Last echo line (no blip needed). Sign is (-1)**(ETL-1): this
            # equals the MATLAB original's final `echo` loop-variable value
            # in both the normal case and the ETL=1 empty-loop case.
            seq.add_block(rg.adc, pp.scale_grad(rg.gro2, (-1) ** (params.ETL - 1)))

            # Spoilers: x/z dephase, y rewind to center. Ported literally
            # from ArbEPI.m's 1-based formula (substituting 0-based
            # y_locs[-1]+1 for y_locs(end)) rather than re-derived, since
            # it's unclear whether the missing "-1" relative to the
            # prephaser's k-offset formula is intentional (this is a
            # spoiler, so a 1-index residual is likely inconsequential).
            seq.add_block(
                gx_spoil,
                pp.scale_grad(gy_spoil, -((y_locs[-1] + 1 - Ny / 2) * rg.deltak[1]) / gy_spoil.area),
                pp.scale_grad(
                    gz_spoil, (gz_spoil.area - (z_locs[-1] + 1 - Nz / 2) * rg.deltak[2]) / gz_spoil.area
                ),
            )

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

    # Save sampling locations for reconstruction. schedules/parts were
    # computed 0-based (see mask2epi.py); convert schedules to 1-based here
    # so samp_locs.mat matches what MATLAB-side reconstruction code expects
    # (parts is already a 1-based shot label with 0 = unsampled, no
    # conversion needed). Written as MATLAB v7.3 (HDF5-based) via
    # hdf5storage, matching the original MATLAB code's `save(..., '-v7.3')`
    # — scipy.io.savemat can only write v5/v4, never v7.3.
    hdf5storage.savemat(
        os.path.join(params.output_dir, 'samp_locs.mat'),
        {'schedules': schedules + 1, 'parts': parts},
        fmt='7.3',
    )

    # Write Pulseq .seq file
    seq.set_definition('FOV', params.fov)
    seq.set_definition('Name', seqname)
    seq.write(os.path.join(params.output_dir, f'{seqname}.seq'))

    return seq
