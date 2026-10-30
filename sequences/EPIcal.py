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

import os

import pypulseq as pp

from lib.epi_shot import build_epi_shot_parts, new_spoil_rng
from lib.readout_from_params import sequence_sys
from lib.seq_io import finalize_sequence, load_blip_steps
from lib.shot_start import add_fatsat_and_excitation, draw_spoil_scales
from params import Params


def generate_epical(params: Params, seqname: str = 'EPIcal', n_frames: int = 1) -> pp.Sequence:
    os.makedirs(params.output_dir, exist_ok=True)
    # Readout gradients are sized to match ArbEPI's, from the schedule it saved, so the
    # calibration trajectory applies; everything else is built exactly as ArbEPI builds it
    # (lib/epi_shot.py).
    max_ky_step, max_kz_step = load_blip_steps(params)
    (sys, rf, gz_ss, gz_ssr, rfsat, rg, gx_pre, gy_pre, gz_pre, gx_spoil, gy_spoil, gz_spoil,
     gx_residual, te_delay, tr_delay, min_te, min_tr) = build_epi_shot_parts(
        params, max_ky_step, max_kz_step)

    # Assemble sequence (full-hardware system, matching ArbEPI -- the POPE
    # fall ramp deliberately exceeds the derate; see ArbEPI.py's comment)
    seq = pp.Sequence(system=sequence_sys(params))

    rf_count = 1
    spoil_rng = new_spoil_rng()

    # n_frames > 1 repeats the real (ADC-on) shots for that many volumes,
    # e.g. as a phase-encode-free time series for readout/hardware
    # stability testing; the default 1 is the usual single calibration pass.
    # Dummy (ADC-off) shots first, for params.discard_duration to the nearest
    # shot: the same steady-state warm-up ArbEPI plays as its discarded
    # frames, so the calibration is taken in the state the time series is.
    # Loop-invariant scaled gradients (scale_grad deep-copies): every shot
    # plays the same unencoded waveform, blips scaled to zero.
    gx_pre_scaled = pp.scale_grad(gx_pre, rg.gx_pre_scale)
    gy_pre0, gz_pre0 = pp.scale_grad(gy_pre, 0), pp.scale_grad(gz_pre, 0)
    gy_blip0, gz_blip0 = pp.scale_grad(rg.gy_blip, 0), pp.scale_grad(rg.gz_blip, 0)
    gro_signed = {s: pp.scale_grad(rg.gro, s) for s in (1, -1)}
    gro2_line = pp.scale_grad(rg.gro2, (-1) ** (params.ETL - 1))

    for shot in range(-params.n_dummy_shots, params.Nshots * n_frames):
        is_dummy = shot < 0
        TRID = 1 if is_dummy else 2  # TRID 1 = dummy, TRID 2 = real (see Pulseq on GE manual)

        # Per-shot random spoiler cycles/voxel (scale factors of the built
        # maximum), reused for both this shot's fat-sat crusher and its
        # post-readout spoiler (after the readout).
        x_scale, y_scale, z_scale = draw_spoil_scales(params, spoil_rng)
        rf_count = add_fatsat_and_excitation(
            seq, params, rf, rg.adc, gz_ss, gz_ssr, rfsat, gx_spoil, gy_spoil, gz_spoil,
            (x_scale, y_scale, z_scale), TRID, rf_count,
        )

        # TE padding delay
        if params.TE > min_te:
            seq.add_block(pp.make_delay(te_delay))

        # Prephase to k-space center (no ky/kz encoding — blips will be
        # zero). gx pre-winds to -S/2, identical to ArbEPI (see
        # ReadoutGrads.gx_pre_scale) -- required for the calibration kx
        # trajectory to match the imaging one.
        seq.add_block(gx_pre_scaled, gy_pre0, gz_pre0)

        # EPI readout — same waveform as ArbEPI, blips scaled to zero
        adc = [] if is_dummy else [rg.adc]
        seq.add_block(rg.gro1)
        for echo in range(params.ETL - 1):
            seq.add_block(*adc, gro_signed[(-1) ** echo], gy_blip0, gz_blip0)
        # Last echo line
        seq.add_block(*adc, gro2_line)

        # Spoilers: this shot's randomly-drawn x/y/z cycles/voxel. x still
        # rewinds gx_residual (the readout's own fixed net kx moment, same
        # as ArbEPI -- EPIcal plays the identical kx trajectory); no rewind
        # term is needed on y/z here since, unlike ArbEPI, no ky/kz
        # encoding was ever applied.
        seq.add_block(
            pp.scale_grad(gx_spoil, (x_scale * gx_spoil.area - gx_residual) / gx_spoil.area),
            pp.scale_grad(gy_spoil, y_scale),
            pp.scale_grad(gz_spoil, z_scale),
        )

        # TR padding delay
        if params.TR > min_tr:
            seq.add_block(pp.make_delay(tr_delay))

    finalize_sequence(seq, params, seqname, params.fov)

    return seq
