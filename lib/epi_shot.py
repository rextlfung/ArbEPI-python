"""The gradients, pulses and delays of one EPI shot, shared by sequences/ArbEPI.py
and sequences/EPIcal.py.

EPIcal is ArbEPI's readout with the ky/kz blips zeroed, so it must be built from
exactly the same objects (test_arbepi_kxoe_matches_epical); this is where they
are built, once, from the largest blip steps of the schedule.
"""

from typing import NamedTuple

import numpy as np
import pypulseq as pp

from lib.calc_te_tr_delays import calc_te_tr_delays
from lib.make_excitation_pulse import make_excitation_from_params
from lib.make_prephasers import make_prephasers
from lib.make_readout_grads import ReadoutGrads
from lib.make_spoilers import make_spoilers
from lib.readout_from_params import derated_sys, make_readout_grads_from_params, readout_kx_residual
from params import Params


class EpiShotParts(NamedTuple):
    sys: pp.Opts  # the derated system the non-readout gradients were built with
    rf: object
    gz_ss: object
    gz_ssr: object
    rfsat: object  # None for water excitation (params.excitation)
    rg: ReadoutGrads
    gx_pre: object
    gy_pre: object
    gz_pre: object
    gx_spoil: object  # built at params.spoil_cycles_max; scaled down per shot
    gy_spoil: object
    gz_spoil: object
    gx_residual: float  # net kx moment left by gx_pre + the readout train
    te_delay: float
    tr_delay: float
    min_te: float
    min_tr: float


def build_epi_shot_parts(
    params: Params, max_ky_step: float, max_kz_step: float
) -> EpiShotParts:
    # Derated system for all non-readout gradients (excitation, fat-sat,
    # prephasers, spoilers), from the params.slew_derate knob: this
    # sequence is slew-, not amplitude-, PNS-limited (measured on a full
    # build of the earlier 240x240x45 GE_UHP protocol: capping max_grad
    # alone barely moved peak PNS, 125.9% -> 124.6%, while a symmetric
    # max_slew=100 T/m/s brought it to 84.3%; the asymmetric POPE readout
    # ramps -- see lib/make_readout_grads.py's module docstring and
    # params.py's ro_slew_rise/ro_slew_fall -- took it under the 80%
    # normal-mode limit). See CLAUDE.md's PNS section.
    sys = derated_sys(params)

    # Excitation, and the fat-sat pulse (None for water excitation, see
    # params.excitation)
    rf, gz_ss, gz_ssr, rfsat = make_excitation_from_params(params, sys)

    # Readout gradients and ADC event (asymmetric POPE ramps + blip slew
    # from params -- see lib/readout_from_params.py)
    rg = make_readout_grads_from_params(max_ky_step, max_kz_step, params)
    print(
        f'Selected ADC dwell: {rg.adc.dwell * 1e6:.1f} us '
        f'(fastest feasible for Nx={params.Nx}, fov_x={params.fov[0] * 1e3:.1f} mm -- '
        "see lib/readout_from_params.py's find_min_feasible_dwell)"
    )

    # Prephasers and spoilers
    gx_pre, gy_pre, gz_pre = make_prephasers(
        params.Nx, params.Ny, params.Nz, params.fov, sys, params.crt
    )
    gx_residual = readout_kx_residual(gx_pre, rg, params.ETL)
    # Built at the maximum cycles/voxel value on every axis; the per-shot
    # loop scales each down (via pp.scale_grad) to that shot's randomly-drawn
    # cycles/voxel, so these fixed trapezoids' duration (and therefore
    # calc_te_tr_delays' min_te/min_tr) never changes shot-to-shot -- only
    # the amplitude does.
    gx_spoil, gy_spoil, gz_spoil = make_spoilers(
        params.res, [params.spoil_cycles_max] * 3, sys, params.crt
    )

    # Delays to achieve desired TE and TR. echo_offset anchors the
    # prescribed TE at the true kx = 0 crossing (see calc_te_tr_delays).
    te_delay, tr_delay, min_te, min_tr = calc_te_tr_delays(
        rf, rfsat, gz_ss, gz_ssr, gx_pre, gy_pre, gz_pre, rg.gro, gx_spoil, gy_spoil, gz_spoil,
        params.ETL, params.TE, params.TR, sys, echo_offset=rg.echo_offset,
    )
    return EpiShotParts(
        sys, rf, gz_ss, gz_ssr, rfsat, rg, gx_pre, gy_pre, gz_pre,
        gx_spoil, gy_spoil, gz_spoil, gx_residual, te_delay, tr_delay, min_te, min_tr,
    )


def new_spoil_rng() -> np.random.Generator:
    """The per-shot spoiler draws' generator, deliberately unseeded (fresh
    entropy every run/process, independent of params.seed): the draws only need
    to break a residual coherence pathway within a sequence's own repeated TR
    structure (see params.py's spoil_cycles_min/max comment), not to be
    reproducible run-to-run like the sampling mask (params.seed) is -- unlike
    the mask, no test or downstream consumer depends on a specific spoiler draw
    sequence, and reproducibility here would only make it easier to
    accidentally rely on one (docs/review-findings.md item 176)."""
    return np.random.default_rng()
