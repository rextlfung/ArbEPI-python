"""GE scanner hardware profiles.

A single source of truth for the gradient/RF hardware limits that both
`.seq` generation (Siemens-format Pulseq, via `params.py`'s `sys`) and GE
`.pge` export/feasibility checking (`seq2ge/ge_export.py`) must agree on, so a
sequence built for a given scanner cannot silently drift out of sync with
that scanner's actual limits (see the `g_max`/`slew_max` unit-drift bug
this module replaces, described in CLAUDE.md/README.md history).

`max_grad`/`max_slew` here are the values used to build the `.seq` file
itself (via `params.py`'s `pp.Opts` construction, which reads every one of
`pp.Opts`'s kwargs straight off the selected `ScannerSpec` rather than
hard-coding any of them there); `seq2ge/ge_export.py` derives GE's
G/cm-based `g_max`/`slew_max` from these same two numbers rather than
storing them twice. `ScannerSpec` intentionally holds only plain data (no
`pypulseq` import here) -- `params.py` is the one place that converts it
into a `pp.Opts`.

`ge_coil` is the coil identifier string passed through to MATLAB's
`pge2.opts(...)`/`check_grad_acoustics(...)` when using the MATLAB export
path (seq2ge/ge_export.py). `chronaxie`/`rheobase`/`alpha` duplicate the PNS
SAFE-model coefficients from the same `pge2.opts.m` table (used by
seq2ge/pns.py's pure-Python PNS check -- see seq2ge/check.py); `check_grad_acoustics`'s
per-coil acoustic-resonance bands are ported directly into seq2ge/acoustics.py
instead (they're keyed by `ge_coil`, not duplicated per-scanner here).
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
    ge_coil: str  # coil code for pge2.opts / check_grad_acoustics / seq2ge/acoustics.py
    chronaxie: float  # s, PNS SAFE-model nerve impulse response time constant
    rheobase: float  # stimulation threshold for constant slew of infinite duration
    alpha: float  # effective coil length (dimensionless); s_min = rheobase / alpha
    # Remaining fields feed pp.Opts (via params.py) 1:1; kept here rather
    # than hard-coded in params.py so every pp.Opts kwarg traces back to
    # this single source of truth, same as max_grad/max_slew above.
    rf_dead_time: float  # s
    rf_ringdown_time: float  # s (hardware ringdown + safety margin)
    adc_dead_time: float  # s
    adc_raster_time: float  # s
    rf_raster_time: float  # s
    grad_raster_time: float  # s
    block_duration_raster: float  # s
    B0: float  # T


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
        chronaxie=334e-6,
        rheobase=23.4,
        alpha=0.333,
        rf_dead_time=100e-6,
        rf_ringdown_time=260e-6,  # 60us hardware ringdown + 200us safety margin
        adc_dead_time=20e-6,
        adc_raster_time=2e-6,
        rf_raster_time=2e-6,
        grad_raster_time=4e-6,
        block_duration_raster=4e-6,
        B0=3.0,
    ),
    # max_grad/max_slew/chronaxie/rheobase/alpha from
    # ../PulCeq/matlab/+pge2/opts.m's 'hrmbuhp' table row.
    # psd_rf_wait/psd_grd_wait/b1_max/pislquant are UNVERIFIED placeholders
    # copied from GE_MR750 -- confirm against the scanner's
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
        chronaxie=359e-6,
        rheobase=26.5,
        alpha=0.370,
        rf_dead_time=100e-6,
        rf_ringdown_time=260e-6,  # 60us hardware ringdown + 200us safety margin
        adc_dead_time=20e-6,
        adc_raster_time=2e-6,
        rf_raster_time=2e-6,
        grad_raster_time=4e-6,
        block_duration_raster=4e-6,
        B0=3.0,
    ),
}
