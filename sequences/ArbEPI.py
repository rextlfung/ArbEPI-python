"""Ported from ../ArbEPI/src/ArbEPI.m — the main 3D-EPI sequence.

Outputs: <output_dir>/ArbEPI.seq (Pulseq format) and <output_dir>/scan_info.mat
-- kxo/kxe (odd/even echo k-space trajectories for ghost correction, shared
with EPIcal.py, see this module's kxoe comment below), schedules ((ky, kz,
echo time) per acquisition) and parts (the partition map), and a snapshot of
scan scalars preprocessing/ needs -- everything preprocessing/ reads for
this acquisition, in one file.

GE `.pge` export (write_to_ge.m) and the trailing MATLAB plotting figures
are intentionally not ported here — see ge/ge_export.py (stage 8 of the
port plan) and plotting/plotting.py respectively.
"""

import copy
import os
from typing import Tuple

import hdf5storage
import numpy as np
import pypulseq as pp
from tqdm import tqdm

from lib.calc_te_tr_delays import calc_te_tr_delays
from lib.make_excitation_pulse import make_excitation_pulse
from lib.make_fatsat_rf import make_fatsat_rf
from lib.make_prephasers import make_prephasers
from lib.make_spoilers import make_spoilers
from lib.mask2epi import mask2epi_laminar, mask2epi_radial, max_blip_steps
from lib.readout_from_params import derated_sys, make_readout_grads_from_params
from params import Params

_MASK2EPI = {
    'laminar': mask2epi_laminar,
    'radial': mask2epi_radial,
}


def _compute_schedules(
    omegas: np.ndarray, ETL: int, Nshots: int, trajectory: str, deltak: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Run mask2epi_{trajectory} per frame (see params.Params.epi_trajectory).
    `deltak` = (deltak_y, deltak_z), i.e. (1/fov_y, 1/fov_z) -- weights
    mask2epi's pass-2 bottleneck-blip ordering so a ky index step and a kz
    index step are compared in physical k-space units, matching how
    make_readout_grads.py sizes the two blip axes independently (see
    lib/mask2epi.py's module docstring).
    Returns 0-based schedules/parts (see lib/mask2epi.py)."""
    if trajectory not in _MASK2EPI:
        raise ValueError(
            f'Unknown epi_trajectory {trajectory!r}; expected one of {sorted(_MASK2EPI)}'
        )
    mask2epi = _MASK2EPI[trajectory]

    Ny, Nz, Nframes = omegas.shape
    schedules = np.zeros((Nframes, Nshots, ETL, 2), dtype=int)
    parts = np.zeros((Ny, Nz, Nframes), dtype=int)
    for frame in range(Nframes):
        schedules[frame], parts[:, :, frame] = mask2epi(omegas[:, :, frame], ETL, Nshots, deltak)
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
    # Derated system for all non-readout gradients (excitation, fat-sat,
    # prephasers, spoilers), from the params.slew_derate knob: this
    # sequence is slew-, not amplitude-, PNS-limited (measured on a full
    # build: capping max_grad alone barely moved peak PNS, 125.9% ->
    # 124.6%, while a symmetric max_slew=100 T/m/s brought it to 84.3%;
    # the asymmetric POPE readout ramps -- see lib/make_readout_grads.py's
    # module docstring and params.py's ro_slew_rise/ro_slew_fall -- take it
    # under the 80% normal-mode limit). See CLAUDE.md's PNS section.
    sys = derated_sys(params)

    # Excitation pulse
    rf, gz_ss, gz_ssr = make_excitation_pulse(params.fa, params.rf_dur, params.rf_tb, params.fov, sys, params.crt)

    # Fat-sat pulse
    rfsat = make_fatsat_rf(params.fatsat, sys, params.fat_offres_freq)

    # Generate EPI sampling schedule from mask
    schedules, parts = _compute_schedules(
        omegas, params.ETL, params.Nshots, params.epi_trajectory,
        deltak=(1 / params.fov[1], 1 / params.fov[2]),
    )

    # Infer maximum ky and kz blip steps across all frames and shots
    max_ky_step, max_kz_step = max_blip_steps(schedules)

    # Readout gradients and ADC event (asymmetric POPE ramps + blip slew
    # from params -- see lib/readout_from_params.py)
    rg = make_readout_grads_from_params(max_ky_step, max_kz_step, params)

    # Prephasers and spoilers
    gx_pre, gy_pre, gz_pre = make_prephasers(params.Nx, params.Ny, params.Nz, params.fov, sys, params.crt)
    gx_spoil, gy_spoil, gz_spoil = make_spoilers(
        params.Nx, params.Ny, params.Nz, params.fov, params.n_cycles_spoil, sys, params.crt
    )

    # Delays to achieve desired TE and TR. echo_offset anchors the
    # prescribed TE at the true kx = 0 crossing (see calc_te_tr_delays).
    te_delay, tr_delay, min_te, min_tr = calc_te_tr_delays(
        rf, rfsat, gz_ss, gz_ssr, gx_pre, gy_pre, gz_pre, rg.gro, gx_spoil, gy_spoil, gz_spoil,
        params.ETL, params.TE, params.TR, sys, echo_offset=rg.echo_offset,
    )

    # Per-echo acquisition time (s since RF excitation), for a future
    # off-resonance-correction consumer (recon/) that needs per-sampled-
    # k-space-location acquisition time, not just its (ky, kz) index --
    # exported as scan_info.mat's 3rd 'schedules' channel below. Every echo
    # occupies one gro-duration block (calc_te_tr_delays.py's own min_tr
    # term is `ETL * calc_duration(gro)`, i.e. this same uniform-spacing
    # assumption), and min_te anchors the nominal (TE-defining) echo's true
    # kx = 0 crossing (via rg.echo_offset -- identical for both readout
    # parities, see lib/make_readout_grads.py) at continuous echo-train
    # index ETL/2 - 0.5 -- so per-echo time is that same anchor offset by
    # (echo - (ETL/2 - 0.5)) gro-durations. `min_te + te_delay` is the TE
    # actually achieved at that nominal echo (equals params.TE when
    # achievable; falls back to min_te, matching calc_te_tr_delays' own
    # "warns, never raises" behavior, when it isn't -- see CLAUDE.md).
    # Identical for every shot/frame, since readout timing doesn't vary
    # across them.
    gro_dur = pp.calc_duration(rg.gro)
    te_nominal_echo = min_te + te_delay
    echo_times = te_nominal_echo + (np.arange(params.ETL) - (params.ETL / 2 - 0.5)) * gro_dur

    # Assemble sequence. Local copy: MATLAB's params.m re-executes fresh for
    # every sequence function, so `sys.adcDeadTime = 0` there never leaks
    # across ArbEPI/EPIcal/GRE/noise. Here `params.sys` may be one shared
    # object across all four calls, so mutate a copy instead. The Sequence
    # system is the full-hardware params.sys, NOT the derated `sys` above:
    # the readout's POPE fall ramp deliberately runs above the derate
    # (params.ro_slew_fall, still within hardware limits, which
    # ge/check.py verifies against params.spec).
    sys_seq = copy.deepcopy(params.sys)
    sys_seq.adc_dead_time = 0  # suppress warnings; no back-to-back ADC blocks
    seq = pp.Sequence(system=sys_seq)

    rf_count = 1
    Ny, Nz = params.Ny, params.Nz

    for frame in tqdm(range(params.Nframes), desc='Writing frames'):

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

            # Move to first k-space location. gx pre-winds to -S/2 (not its
            # built-in -kmax) so both readout parities sample symmetrically
            # about kx = 0 and reach +-kmax -- see ReadoutGrads.gx_pre_scale.
            gy_pre_tmp = pp.scale_grad(gy_pre, (y_locs[0] - Ny / 2) / (-Ny / 2))
            gz_pre_tmp = pp.scale_grad(gz_pre, (z_locs[0] - Nz / 2) / (-Nz / 2))
            seq.add_block(pp.scale_grad(gx_pre, rg.gx_pre_scale), gy_pre_tmp, gz_pre_tmp)

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

    # Odd/even echo k-space trajectories for ghost correction. This used to
    # be computed by EPIcal.py from its own separately-built sequence, but
    # EPIcal is just this sequence's readout with ky/kz blips zeroed -- the
    # y/z blips ride on separate gradient channels and don't affect gx
    # timing/area, so the first two echoes' kx trajectory (the alternating
    # odd/even readout polarity) is identical whether read from ArbEPI's or
    # EPIcal's sequence. Computing it here means EPIcal.py no longer needs
    # to run at all just to produce this, and there's only one
    # calculate_kspace() call instead of two. Physical k-space values
    # (cycles/m), not indices -- no 1-based conversion applies.
    k_traj_adc, *_ = seq.calculate_kspace()
    kxo = k_traj_adc[0, : rg.Nfid]
    kxe = k_traj_adc[0, rg.Nfid : rg.Nfid * 2]

    # Sampling schedule for reconstruction. schedules/parts were computed
    # 0-based (see mask2epi.py); convert schedules to 1-based here so
    # scan_info.mat matches what MATLAB-side reconstruction code expects
    # (parts is already a 1-based shot label with 0 = unsampled, no
    # conversion needed). schedules gains a 3rd channel here, echo_times
    # (broadcast to every frame/shot -- see its computation above), so the
    # array holds [iy (1-based), iz (1-based), echo time (s)] per
    # acquisition -- float64 throughout (iy/iz stay exactly representable),
    # since MATLAB has no separate-dtype-per-column array type anyway.
    # preprocessing/preprocess.py's load_schedules() splits the 3rd channel
    # back out for consumers that still want a plain (ky, kz) index array.
    schedules_te = np.broadcast_to(echo_times, (params.Nframes, params.Nshots, params.ETL))
    schedules_out = np.concatenate(
        [(schedules + 1).astype(np.float64), schedules_te[..., None]], axis=-1,
    )

    # Save everything preprocessing/ needs for this acquisition in one file,
    # scan_info.mat: the kxo/kxe trajectories and schedules/parts above, plus
    # a snapshot of the scan scalars preprocessing/config.py's
    # load_seq_params reads (MATLAB's preprocess.m gets these by run()-ing a
    # per-acquisition params.m into its workspace, which has no Python
    # equivalent -- copying the whole params.py module would drag
    # pypulseq/scanners.py into the separate, GE-SDK-constrained
    # preprocessing venv just to read a handful of scalars). These three
    # used to be three separate files (kxoe<Nx>.mat, samp_locs.mat,
    # params.mat); consolidated now that all of them are written from this
    # one place, at this one point in the pipeline (kxoe<Nx>.mat's
    # Nx-dependent name was the original reason it needed its own file,
    # resolved in a second step once Nx was known -- moot once it's just
    # another key in the same dict as everything else here). Written as
    # MATLAB v7.3 (HDF5-based) via hdf5storage, matching the original
    # MATLAB code's `save(..., '-v7.3')` -- scipy.io.savemat can only write
    # v5/v4, never v7.3.
    hdf5storage.savemat(
        os.path.join(params.output_dir, 'scan_info.mat'),
        {
            'kxo': kxo, 'kxe': kxe,
            'schedules': schedules_out, 'parts': parts,
            'Nx': params.Nx, 'Ny': params.Ny, 'Nz': params.Nz,
            'ETL': params.ETL, 'R': params.R,
            'fov': params.fov,
            'volume_tr': params.volume_tr,
            'discard_duration': params.discard_duration,
            'Nx_degre': params.Nx_degre, 'Ny_degre': params.Ny_degre, 'Nz_degre': params.Nz_degre,
            'fov_degre': params.fov_degre,
            # generate_degre's `c` loop excites every phase encode once per
            # echo (see sequences/deGRE.py) -- preprocess.py's raw deGRE
            # unflatten needs this to recover the acquisition-order shape.
            'n_echoes_degre': len(params.TE_degre),
            # ΔTE for external phase-difference B0 mapping from preprocess.py's
            # ksp_gre_echoes cache -- see that module's STEP 2.
            'TE_degre': params.TE_degre,
        },
        fmt='7.3',
    )

    # Write Pulseq .seq file
    seq.set_definition('FOV', params.fov)
    seq.set_definition('Name', seqname)
    seq.write(os.path.join(params.output_dir, f'{seqname}.seq'))

    return seq
