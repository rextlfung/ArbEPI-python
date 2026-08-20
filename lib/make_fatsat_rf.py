"""Adapted from ../ArbEPI/lib/make_fatsat_rf.m.

The MATLAB original designs the fat-sat pulse via GE's
`toppe.utils.rf.makeslr` (a min-phase SLR pulse, per the Shinnar-Le Roux
algorithm: Pauly J, Le Roux P, Nishimura D, Macovski A. "Parameter
relations for the Shinnar-Le Roux selective excitation pulse design
algorithm." IEEE Trans Med Imaging. 1991;10(1):53-65), which has no Python
equivalent. Per the project's scoping decision (see the port plan), this
uses pypulseq's built-in `make_gauss_pulse` instead — a simpler design with
a less sharp spectral profile than the original SLR pulse, but no new
dependency and no bit-exact-waveform requirement.
"""

import math
from types import SimpleNamespace

import pypulseq as pp

from params import FatsatParams


def make_fatsat_rf(fatsat: FatsatParams, sys: pp.Opts, fat_offres_freq: float) -> SimpleNamespace:
    """Create a fat-saturation RF pulse object.

    Parameters
    ----------
    fatsat : flip (deg), sl_thick (m, dummy value), tbw, dur (s)
    sys : pypulseq system (Opts)
    fat_offres_freq : fat off-resonance frequency (Hz)
    """
    rfsat = pp.make_gauss_pulse(
        fatsat.flip / 180 * math.pi,
        duration=fatsat.dur,
        slice_thickness=fatsat.sl_thick,
        time_bw_product=fatsat.tbw,
        system=sys,
        use='saturation',
    )
    rfsat.freq_offset = -fat_offres_freq  # Hz
    return rfsat
