"""GE scanner hardware profiles.

A single source of truth for the gradient/RF hardware limits that both
`.seq` generation (Siemens-format Pulseq, via `params.py`'s `sys`) and GE
`.pge` export/feasibility checking (`ge_export.py`) must agree on, so a
sequence built for a given scanner cannot silently drift out of sync with
that scanner's actual limits (see the `g_max`/`slew_max` unit-drift bug
this module replaces, described in CLAUDE.md/README.md history).

`max_grad`/`max_slew` here are the values used to build the `.seq` file
itself (via `pp.Opts`); `ge_export.py` derives GE's G/cm-based `g_max`/
`slew_max` from these same two numbers rather than storing them twice.

`ge_coil` is the coil identifier string passed straight through to
MATLAB's `pge2.opts(...)` and `check_grad_acoustics(...)`, which each
carry their own internal, more detailed tables (PNS SAFE-model
chronaxie/rheobase/alpha coefficients, and per-coil acoustic resonance
frequencies respectively) keyed off that same string. Python does not
duplicate those tables -- see ../PulCeq/matlab/+pge2/opts.m and
../ArbEPI/lib/check_grad_acoustics.m for the authoritative values.
"""

from dataclasses import dataclass


@dataclass
class ScannerSpec:
    name: str
    max_grad: float  # mT/m
    max_slew: float  # T/m/s
    b1_max: float  # Gauss
    psd_rf_wait: float  # s, RF-gradient delay
    psd_grd_wait: float  # s, ADC-gradient delay
    pislquant: int  # ADC events at scan start for receive gain calibration
    ge_coil: str  # coil code for pge2.opts / check_grad_acoustics


SCANNERS: dict[str, ScannerSpec] = {
    # Values from ../PulCeq/matlab/+pge2/opts.m's 'xrm' table row and
    # ../ArbEPI/params.m (the original MATLAB repo's only target scanner).
    # psd_rf_wait/psd_grd_wait/b1_max/pislquant confirmed MR750-specific
    # in ../ArbEPI/CLAUDE.md.
    'GE_MR750': ScannerSpec(
        name='GE Discovery MR750 (XRM gradient coil)',
        max_grad=50,
        max_slew=200,
        b1_max=0.25,
        psd_rf_wait=150e-6,
        psd_grd_wait=120e-6,
        pislquant=10,
        ge_coil='xrm',
    ),
    # max_grad/max_slew from ../PulCeq/matlab/+pge2/opts.m's 'hrmbuhp'
    # table row. psd_rf_wait/psd_grd_wait/b1_max/pislquant are UNVERIFIED
    # placeholders copied from GE_MR750 -- confirm against the scanner's
    # GRSubsystemHWO.xml or Scandbdt.cfg (see opts.m's header comment) and
    # correct here.
    'GE_UHP': ScannerSpec(
        name='GE Ultra-High Performance (HRMB gradient coil)',
        max_grad=100,
        max_slew=200,
        b1_max=0.25,
        psd_rf_wait=150e-6,
        psd_grd_wait=120e-6,
        pislquant=10,
        ge_coil='hrmbuhp',
    ),
}
