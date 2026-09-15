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

from ge.acoustics import _ESP_BANDS_US
from lib.make_readout_grads import ReadoutGrads, make_readout_grads
from params import Params

# Extra clearance (us) enforced on top of the exact coil forbidden-echo-
# spacing bands below, since those band edges are themselves an
# approximate hardware table (check_grad_acoustics.m), not a precise
# analytic cutoff -- see find_min_feasible_dwell's docstring.
ACOUSTIC_MARGIN_US = 20.0


def derated_sys(params: Params) -> pp.Opts:
    """params.sys derated to params.slew_derate (T/m/s), as a fresh copy.

    Used for every gradient except the readout trapezoid's own ramps and
    the ky/kz blips (which get their own derate via blip_sys() below):
    excitation, fat-sat, prephasers, spoilers. Always a deepcopy --
    params.sys is one shared mutable object across all four
    sequence-generation calls (see params.py's Params.sys note).
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


def _echo_spacing_in_forbidden_band(echo_us: float, ge_coil: str) -> bool:
    """True if `echo_us` (the readout block's own repeat period -- what
    drives the EPI train's fundamental gradient-switching frequency) falls
    within `ACOUSTIC_MARGIN_US` of any of `ge_coil`'s forbidden
    echo-spacing bands (ge/acoustics.py's `_ESP_BANDS_US`, copied verbatim
    from check_grad_acoustics.m). Landing in one of these bands drives the
    gradient waveform's frequency content into a coil mechanical
    resonance -- ge/check.py's real FFT-based acoustics check only warns
    (matches MATLAB's own non-blocking behavior), but exceeding it is a
    genuine hardware-damage risk on real scanner hardware, not just a
    modeling artifact, so this search actively steers clear of it rather
    than leaving it to be caught (non-fatally) after the fact.

    Checked against the union of all three axes' band lists, matching
    check_grad_acoustics.m's own cross-product check (every waveform axis
    is compared against every axis's band list -- see ge/acoustics.py's
    check_grad_acoustics docstring) -- not just the x-axis list, even
    though `echo_us` here is the gro (x readout) block duration.
    """
    for axis_bands in _ESP_BANDS_US[ge_coil.lower()]:
        for lo_us, hi_us, _ in axis_bands:
            if lo_us - ACOUSTIC_MARGIN_US <= echo_us <= hi_us + ACOUSTIC_MARGIN_US:
                return True
    return False


def find_min_feasible_dwell(
    max_ky_step: float, max_kz_step: float, params: Params, max_multiple: int = 100,
) -> float:
    """Smallest ADC dwell (an integer multiple of sys.adc_raster_time, the
    GE hardware dwell quantum) for which make_readout_grads's readout-lobe
    geometry is feasible -- i.e. doesn't hit the "triangular lobe" crash
    (docs/review-findings.md item 146: at a small readout FOV/dwell
    combination, the flat-top amplitude pins at sys.max_grad and the
    asymmetric POPE ramps alone can already exceed the k-space area a
    Nyquist-sampled readout of width Nx needs, making the required
    flat-top duration negative) -- *and* for which the resulting echo
    spacing doesn't land in one of the scanner coil's forbidden
    acoustic-resonance bands (see _echo_spacing_in_forbidden_band).

    Smaller dwell means denser/faster sampling (shorter Tread, shorter
    echo spacing), and the required flat-top duration is monotonically
    non-decreasing in dwell (larger dwell -> smaller flat-top amplitude
    A = min(deltak[0]/dwell, max_grad) -> smaller-or-equal ramp times ->
    smaller-or-equal ramp area to make up for), so a linear search
    upward from the hardware dwell quantum finds the *fastest* feasible
    dwell, not just *a* feasible one -- no need to solve the flat >= 0
    inequality in closed form (it also depends on max_ky_step/max_kz_step
    via the blip-duration/raster-rounding terms, not just Nx/fov/slews).
    The acoustic-band check breaks that monotonic "first hit wins"
    shortcut (echo spacing keeps growing with dwell past the forbidden
    band, so a dwell just past it is not necessarily feasible on the
    geometry check too), hence still trying every quantum in order rather
    than stopping at the first geometrically-feasible one.
    """
    quantum = params.sys.adc_raster_time
    gamma = params.sys.gamma
    for n in range(1, max_multiple + 1):
        dwell = n * quantum
        try:
            rg = make_readout_grads(
                max_ky_step,
                max_kz_step,
                params.Nx,
                params.fov,
                dwell,
                blip_sys(params),
                params.crt,
                slew_rise=params.ro_slew_rise * gamma,
                slew_fall=params.ro_slew_fall * gamma,
            )
        except AssertionError:
            continue
        echo_us = pp.calc_duration(rg.gro) * 1e6
        if _echo_spacing_in_forbidden_band(echo_us, params.spec.ge_coil):
            continue
        return dwell
    raise RuntimeError(
        f'No feasible ADC dwell found up to {max_multiple} x adc_raster_time '
        f'({max_multiple * quantum * 1e6:.0f} us) for max_ky_step={max_ky_step}, '
        f'max_kz_step={max_kz_step} that also clears {params.spec.ge_coil!r}\'s '
        'forbidden acoustic-resonance echo-spacing bands -- this Nx/fov/slew/mask '
        'combination may be fundamentally infeasible; see docs/review-findings.md '
        'item 146.'
    )


def make_readout_grads_from_params(
    max_ky_step: float, max_kz_step: float, params: Params
) -> ReadoutGrads:
    """The one canonical make_readout_grads call (POPE slews from params).

    dwell is not a fixed params field: it's searched here via
    find_min_feasible_dwell for the fastest (smallest) ADC dwell that
    keeps this mask's own max_ky_step/max_kz_step readout-lobe geometry
    feasible (see that function's docstring).
    """
    gamma = params.sys.gamma
    dwell = find_min_feasible_dwell(max_ky_step, max_kz_step, params)
    return make_readout_grads(
        max_ky_step,
        max_kz_step,
        params.Nx,
        params.fov,
        dwell,
        blip_sys(params),
        params.crt,
        slew_rise=params.ro_slew_rise * gamma,
        slew_fall=params.ro_slew_fall * gamma,
    )
