"""Single source of truth for the PNS slew derates in params.py.

ArbEPI.py, EPIcal.py, and the trajectory tests all need the exact same
derated system objects and the exact same make_readout_grads call --
EPIcal must stay waveform-identical to ArbEPI (see sequences/EPIcal.py's
module docstring and test_arbepi_kxoe_matches_epical), and the tests
recompute ReadoutGrads independently to cross-check the assembled
sequences. Before this module, the `sys.max_slew = 100 * sys.gamma`
derate was hand-copied in three places.
"""

import copy

import pypulseq as pp

from lib.make_readout_grads import ReadoutGrads, make_readout_grads
from params import Params


def derated_sys(params: Params) -> pp.Opts:
    """params.sys derated to params.slew_derate (T/m/s), as a fresh copy.

    Used for every gradient except the readout trapezoid's own ramps:
    excitation, fat-sat, prephasers, spoilers, and the ky/kz blips.
    Always a deepcopy -- params.sys is one shared mutable object across
    all four sequence-generation calls (see params.py's Params.sys note).
    """
    sys = copy.deepcopy(params.sys)
    sys.max_slew = params.slew_derate * sys.gamma  # T/m/s -> Hz/m/s
    return sys


def blip_sys(params: Params) -> pp.Opts:
    """params.sys with max_slew = params.blip_slew (T/m/s), as a fresh copy.

    Handed to make_readout_grads, whose sys.max_slew governs the blips
    (the readout ramps get their own explicit slew_rise/slew_fall).
    """
    sys = copy.deepcopy(params.sys)
    sys.max_slew = params.blip_slew * sys.gamma  # T/m/s -> Hz/m/s
    return sys


def make_readout_grads_from_params(
    max_ky_step: float, max_kz_step: float, params: Params
) -> ReadoutGrads:
    """The one canonical make_readout_grads call (POPE slews from params)."""
    gamma = params.sys.gamma
    return make_readout_grads(
        max_ky_step,
        max_kz_step,
        params.Nx,
        params.fov,
        params.dwell,
        blip_sys(params),
        params.crt,
        slew_rise=params.ro_slew_rise * gamma,
        slew_fall=params.ro_slew_fall * gamma,
    )
