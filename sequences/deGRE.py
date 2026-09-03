"""deGRE: 3D spoiled dual-echo GRE B0 field-mapping sequence.

Ported from ../ArbEPI/src/GRE.m (this repo's former single-echo GRE)
upgraded to match Jayden's MATLAB/Pulseq B0 mapping sequence:
https://github.com/HarmonizedMRI/B0shimming/blob/main/sequence/Pulseq/writeB0.m
A B0 map is just a GRE acquired at two echo times: field offset is
recovered from the phase difference between the two echoes -- hence
"deGRE" (dual-echo GRE), not plain "GRE", once this stopped being a
single-echo sequence.

Two deliberate deviations from writeB0.m, both carried over from this
repo's original single-echo GRE rather than re-derived from writeB0.m:
  - Excitation stays the slab-selective sinc pulse the single-echo GRE
    already used (writeB0.m instead excites non-selectively with a short
    block pulse) -- this repo's GRE-family sequences image the same slab
    as the main EPI sequence, so a non-selective pulse would change what
    volume gets mapped, not just how.
  - No fat saturation (writeB0.m has none either): the two echo times are
    themselves the fat-suppression mechanism -- see TE_degre's definition
    in params.py.

Outputs: <output_dir>/deGRE.seq (Pulseq format). Still usable for
coil-sensitivity-map estimation (smaps.py, preprocess.py, recon_frames.py)
via either echo, in addition to now producing B0 map data -- those
preprocessing modules key off their own `_gre` raw-data naming convention
for acquired ScanArchives, independent of this file's name.

Note: this file inlines the slab-selective excitation pulse construction
rather than calling make_excitation_pulse, but does apply the same
`gz_ss.delay = rf.delay - gz_ss.rise_time` resync after trap4ge that
lib/make_excitation_pulse.py does. GRE.m (and an earlier version of this
file) omitted that resync, exactly mirroring GRE.m's own inline
construction -- but trap4ge always rebuilds the trapezoid from scratch via
pp.make_trapezoid (see lib/trap4ge.py), which resets delay to 0
regardless of whether the rise/flat/fall rounding itself changes anything;
without the resync this decenters the RF pulse within gz_ss's flat top
right now, not just if crt ever changes, leaving a nonzero residual kz at
readout (confirmed via test_degre_excitation_is_centered, the deGRE
analogue of test_epical_trajectory_is_centered).
"""

import copy
import math
import os

import numpy as np
import pypulseq as pp
from tqdm import tqdm

from lib.trap4ge import trap4ge
from params import Params


def generate_degre(params: Params, seqname: str = 'deGRE') -> pp.Sequence:
    os.makedirs(params.output_dir, exist_ok=True)
    # Own copy of params.sys (not the shared instance -- see params.py's
    # Params.sys docstring) with its own slew derate: this sequence's
    # single-line spoiled readout has a much lower PNS profile than the
    # EPI train's blip train, so it doesn't need nearly as aggressive a
    # cap. Measured on a full build (peak PNS vs max_slew, IEC 60601-2-33's
    # 80% normal-mode threshold): 200 T/m/s (hardware) -> 83.4%, 180 ->
    # 79.7%, 175 -> 78.7%. 175 T/m/s clears normal mode with a small
    # margin.
    sys = copy.deepcopy(params.sys)
    sys.max_slew = 175 * sys.gamma  # T/m/s -> Hz/m/s (pypulseq's internal unit)
    crt = params.crt

    seq = pp.Sequence(system=sys)

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
    gz_ss.delay = rf.delay - gz_ss.rise_time  # sync RF onset with slice-select gradient
    gz_ssr = trap4ge(gz_ssr, crt, sys)

    # Readout and phase-encode gradients
    deltak = [1 / f for f in params.fov_degre]
    # Exact-Nyquist flat top (Nx_degre*deltak[0]/Tread) can exceed
    # sys.max_grad once that's derated below hardware for PNS (see
    # params.py) -- stretch dwell just enough to keep the flat-top
    # amplitude within max_grad (same oversampling tradeoff
    # make_readout_grads.py makes for the EPI readout), rounded up to the
    # ADC raster so Tread stays sampleable.
    dwell_degre = max(params.dwell, deltak[0] / sys.max_grad)
    dwell_degre = math.ceil(dwell_degre / sys.adc_raster_time) * sys.adc_raster_time
    Tread = params.Nx_degre * dwell_degre

    gy_pre = trap4ge(
        pp.make_trapezoid(
            'y', system=sys, area=params.Ny_degre * deltak[1] / 2, duration=params.Tpre
        ),
        crt,
        sys,
    )
    gz_pre = trap4ge(
        pp.make_trapezoid(
            'z', system=sys, area=params.Nz_degre * deltak[2] / 2, duration=params.Tpre
        ),
        crt,
        sys,
    )

    gxtmp = pp.make_trapezoid(
        'x', system=sys, amplitude=params.Nx_degre * deltak[0] / Tread, flat_time=Tread
    )
    gx_pre = trap4ge(
        pp.make_trapezoid('x', system=sys, area=-gxtmp.area / 2, duration=params.Tpre), crt, sys
    )

    adc = pp.make_adc(params.Nx_degre, system=sys, duration=Tread, delay=gxtmp.rise_time)

    # Extend flat time to split at end of ADC dead time
    gx = trap4ge(
        pp.make_trapezoid(
            'x', system=sys, amplitude=params.Nx_degre * deltak[0] / Tread,
            flat_time=Tread + adc.dead_time,
        ),
        crt,
        sys,
    )

    # Ported literally from GRE.m, which uses deltak[0] (kx spacing, not
    # deltak[2]) for the spoiler area. Kept as a plain trapezoid
    # (trap4ge'd) rather than writeB0.m's makeExtendedTrapezoidArea (which
    # shapes gxSpoil to start from the readout's trailing amplitude for a
    # shorter spoiler) -- that produces an arbitrary-waveform gradient,
    # which trap4ge (and this repo's GE raster-alignment invariant, see
    # CLAUDE.md) isn't set up to handle; the plain trapezoid this repo
    # already used is a deliberate, documented deviation, not an
    # oversight. No z-spoiler here (unlike this file's former single-echo
    # GRE, which used one as a fat-sat crusher) -- writeB0.m has none
    # either, since this sequence has no fat-sat to crush.
    gx_spoil = trap4ge(
        pp.make_trapezoid(
            'x', system=sys, area=params.Nx_degre * deltak[0] * params.n_cycles_spoil_degre
        ),
        crt,
        sys,
    )

    # Phase-encode step vectors
    pe1_steps = (np.arange(params.Ny_degre) - params.Ny_degre / 2) / params.Ny_degre * 2
    pe2_steps = (np.arange(params.Nz_degre) - params.Nz_degre / 2) / params.Nz_degre * 2

    # TE and TR delays, one pair per echo (TE_degre is a 2-element array --
    # see params.py). te_min doesn't depend on which echo, since both
    # echoes share the same excitation/prephasing timing.
    te_min = (
        max(pp.calc_duration(rf), pp.calc_duration(gz_ss)) - (rf.delay + pp.calc_rf_center(rf)[0])
        + pp.calc_duration(gz_ssr)
        + pp.calc_duration(gx_pre)
        + adc.delay
        + Tread / 2
    )
    raster = sys.grad_raster_time
    delay_te = np.array([math.ceil((te - te_min) / raster) * raster for te in params.TE_degre])
    if np.any(delay_te < 0):
        raise ValueError(
            f'params.TE_degre {params.TE_degre * 1e3} ms is below the minimum achievable TE '
            f'{te_min * 1e3:.3f} ms for at least one echo -- increase TE_degre.'
        )
    tr_min = (
        max(pp.calc_duration(rf), pp.calc_duration(gz_ss))
        + pp.calc_duration(gz_ssr)
        + delay_te
        + max(pp.calc_duration(gx_pre), pp.calc_duration(gy_pre), pp.calc_duration(gz_pre))
        + pp.calc_duration(gx)
        + max(pp.calc_duration(gx_spoil), pp.calc_duration(gy_pre), pp.calc_duration(gz_pre))
    )
    delay_tr = np.array([math.ceil((params.TR_degre - tr) / raster) * raster for tr in tr_min])
    if np.any(delay_tr < 0):
        raise ValueError(
            f'params.TR_degre {params.TR_degre * 1e3:.3f} ms is below the minimum achievable TR '
            f'{tr_min.max() * 1e3:.3f} ms for at least one echo -- increase TR_degre.'
        )

    # Assemble sequence.
    # iZ < 0: dummy shots to reach steady state
    # iZ = 0: ADC on for receive gain calibration (GE auto-prescan)
    # iZ > 0: image acquisition
    # Each (iY, iZ) phase-encode location is excited once per echo (the
    # `c` loop) -- two full, separately-spoiled acquisitions sharing the
    # same phase encodes, not a single-excitation dual-echo readout,
    # exactly matching writeB0.m.
    rf_count = 1

    for iZ in tqdm(
        range(-params.Ndummy_zloops, params.Nz_degre + 1), desc='z encode'
    ):
        is_dummy_tr = iZ < 0
        TRID = 1 if is_dummy_tr else 2

        for iY in range(params.Ny_degre):
            # Phase-encode steps off during dummy and gain-cal shots
            y_step = pe1_steps[iY] if iZ > 0 else 0.0
            z_step = pe2_steps[iZ - 1] if iZ > 0 else 0.0

            for c in range(len(params.TE_degre)):
                # RF spoiling
                rf_phase = (0.5 * params.rf_phase_0 * rf_count**2) % 360.0
                rf.phase_offset = rf_phase / 180 * math.pi
                adc.phase_offset = rf_phase / 180 * math.pi
                rf_count += 1

                # Mark start of segment (block group) with a label, as
                # writeB0.m does on its excitation block.
                seq.add_block(rf, gz_ss, pp.make_label('TRID', 'SET', TRID))
                seq.add_block(gz_ssr)
                seq.add_block(pp.make_delay(delay_te[c]))
                seq.add_block(gx_pre, pp.scale_grad(gy_pre, y_step), pp.scale_grad(gz_pre, z_step))
                if is_dummy_tr:
                    seq.add_block(gx)
                else:
                    seq.add_block(gx, adc)
                seq.add_block(
                    gx_spoil, pp.scale_grad(gy_pre, -y_step), pp.scale_grad(gz_pre, -z_step)
                )
                seq.add_block(pp.make_delay(delay_tr[c]))

    # Check sequence timing
    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed! Error listing follows:')
        print(error_report)

    # Write Pulseq .seq file
    seq.set_definition('FOV', params.fov_degre)
    seq.set_definition('Name', seqname)
    seq.write(os.path.join(params.output_dir, f'{seqname}.seq'))

    return seq
