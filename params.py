"""Central configuration, mirroring ../ArbEPI/params.m.

params.m is a MATLAB script that injects variables into the caller's
workspace; there is no Python equivalent of that pattern. Instead,
``load_params()`` returns a single ``Params`` dataclass instance that is
passed explicitly to every function that needs it.
"""

import math
from dataclasses import dataclass

import numpy as np
import pypulseq as pp

from scanners import SCANNERS, ScannerSpec


@dataclass
class FatsatParams:
    flip: float  # degrees
    sl_thick: float  # m (dummy value; just needs to be large)
    tbw: float  # time-bandwidth product
    dur: float  # s


@dataclass
class Params:
    sys: pp.Opts
    crt: float  # common raster time (s)
    dwell: float  # s, ADC sample time

    # EPI spatial parameters
    res: np.ndarray  # m, [x, y, z]
    N: np.ndarray  # acquisition tensor size [Nx, Ny, Nz]
    fov: np.ndarray  # m
    Nx: int
    Ny: int
    Nz: int

    # Acceleration
    R: float
    ETL: int
    Nshots: int

    # Decay / timing
    TE: float  # s
    volume_tr: float  # s
    TR: float  # s
    T1: float  # s

    duration: float  # s
    discard_duration: float  # s
    Nframes: int
    Ndummyshots: int

    fa: float  # degrees, Ernst angle
    rf_dur: float  # s
    rf_tb: float
    rf_phase_0: float  # degrees
    n_cycles_spoil: int

    # Fat saturation
    fat_chem_shift: float  # ppm (dimensionless ratio)
    fat_offres_freq: float  # Hz
    fatsat: FatsatParams

    # GRE parameters
    res_gre: np.ndarray
    fov_gre: np.ndarray
    N_gre: np.ndarray
    Nx_gre: int
    Ny_gre: int
    Nz_gre: int
    Ndummy_zloops: int
    TE_gre: float
    TR_gre: float
    T1_gre: float
    alpha_gre: float  # degrees
    rf_dur_gre: float
    n_cycles_spoil_gre: int
    Tpre: float

    # Noise prescan
    Ncoils: int

    # Output
    output_dir: str

    # The selected ScannerSpec itself (see scanners.py) -- seq2ge/ge_export.py's
    # pure-Python feasibility check/export reads max_grad/max_slew/b1_max/
    # chronaxie/rheobase/alpha/ge_coil/pislquant straight from this, so
    # they can't drift out of sync with `sys` above (both come from the
    # same ScannerSpec instance).
    spec: ScannerSpec
    # PNS is a physiological safety limit, not a hardware one -- kept
    # separate from ScannerSpec since it's phantom-vs-human scan context,
    # not a scanner constant.
    PNSwt: np.ndarray

    # Sampling mask parameters
    sampling_method: str
    pd_calib_frac: float
    pd_crop_corner: bool
    pd_decay: float
    rand_gaussian_sigma: np.ndarray | None


DEFAULT_SCANNER = 'GE_UHP'


def load_params(scanner: str = DEFAULT_SCANNER, output_dir: str = 'output') -> Params:
    """
    scanner : one of scanners.SCANNERS' keys (currently 'GE_MR750',
        'GE_UHP') -- selects the gradient/RF hardware limits used both to
        build the .seq file and, if `--ge` is used, to check/export it for
        GE. See scanners.py for where these values come from.
    """
    spec = SCANNERS[scanner]

    sys = pp.Opts(
        max_grad=spec.max_grad,
        grad_unit='mT/m',
        max_slew=spec.max_slew,
        slew_unit='T/m/s',
        rf_dead_time=spec.rf_dead_time,
        rf_ringdown_time=spec.rf_ringdown_time,
        adc_dead_time=spec.adc_dead_time,
        adc_raster_time=spec.adc_raster_time,
        rf_raster_time=spec.rf_raster_time,
        grad_raster_time=spec.grad_raster_time,
        block_duration_raster=spec.block_duration_raster,
        B0=spec.B0,
    )

    crt = 20e-6  # s, common raster time (Siemens 10us, GE 4us)
    dwell = 2e-6  # s

    # Spatial parameters. 0.9mm isotropic resolution; x/y FOV held at the
    # previous 216mm, z (slice-select) FOV reduced to 40.5mm.
    res = np.array([0.9, 0.9, 0.9]) * 1e-3
    N = np.array([240, 240, 45])
    fov = N * res
    Nx, Ny, Nz = int(N[0]), int(N[1]), int(N[2])

    # Acceleration parameters
    R = 9
    ETL = 60
    Nshots = math.ceil(Ny * Nz / R / ETL)

    # Decay parameters
    TE = 30e-3
    volume_tr = 2
    TR = volume_tr / Nshots
    T1 = 1.3

    duration = 60
    discard_duration = 0
    Nframes = round((duration + discard_duration) / volume_tr)

    Ndummyshots = round(discard_duration / TR)

    fa = 180 / math.pi * math.acos(math.exp(-TR / T1))
    rf_dur = 2e-3
    rf_tb = 6
    rf_phase_0 = 117
    n_cycles_spoil = 2

    fat_chem_shift = 3.5 * 1e-6
    fat_offres_freq = sys.gamma * sys.B0 * fat_chem_shift
    fatsat = FatsatParams(flip=90, sl_thick=1e5, tbw=3, dur=4e-3)

    # GRE parameters
    res_gre = np.array([2, 2, 2]) * 1e-3
    fov_gre = np.array([21.6, 21.6, 21.6]) * 1e-2
    N_gre = np.round(fov_gre / res_gre).astype(int)
    Nx_gre, Ny_gre, Nz_gre = int(N_gre[0]), int(N_gre[1]), int(N_gre[2])

    Ndummy_zloops = 4
    TE_gre = 1 / fat_offres_freq + 8e-4
    TR_gre = 6e-3
    T1_gre = 1.3
    alpha_gre = 180 / math.pi * math.acos(math.exp(-TR_gre / T1_gre))

    rf_dur_gre = 0.4e-3
    n_cycles_spoil_gre = 2
    Tpre = 1.0e-3

    Ncoils = 32

    # IEC 60601-2-33:2022-recommended PNS channel weights -- see CLAUDE.md's
    # "Open finding" for why this was zero (disabling the PNS check
    # entirely) until now, and that this default now correctly blocks
    # `main.py --ge` for EPIcal/ArbEPI/GRE until slew/blip timing changes.
    PNSwt = np.array([0.8, 1.0, 0.7]) # human
    # PNSwt = np.array([0.0, 0.0, 0.0]) # phantom

    # Sampling mask parameters
    sampling_method = 'pd'
    # Fully-sampled central calibration region: a centered ellipse,
    # aspect-matched to (Ny, Nz), sized to hold 30% of the R-dependent
    # sample budget (floor(Ny*Nz/R) -- see sampling/pd_sample.py).
    pd_calib_frac = 0.3
    pd_crop_corner = True
    pd_decay = 1.4
    rand_gaussian_sigma = None

    return Params(
        sys=sys,
        crt=crt,
        dwell=dwell,
        res=res,
        N=N,
        fov=fov,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        R=R,
        ETL=ETL,
        Nshots=Nshots,
        TE=TE,
        volume_tr=volume_tr,
        TR=TR,
        T1=T1,
        duration=duration,
        discard_duration=discard_duration,
        Nframes=Nframes,
        Ndummyshots=Ndummyshots,
        fa=fa,
        rf_dur=rf_dur,
        rf_tb=rf_tb,
        rf_phase_0=rf_phase_0,
        n_cycles_spoil=n_cycles_spoil,
        fat_chem_shift=fat_chem_shift,
        fat_offres_freq=fat_offres_freq,
        fatsat=fatsat,
        res_gre=res_gre,
        fov_gre=fov_gre,
        N_gre=N_gre,
        Nx_gre=Nx_gre,
        Ny_gre=Ny_gre,
        Nz_gre=Nz_gre,
        Ndummy_zloops=Ndummy_zloops,
        TE_gre=TE_gre,
        TR_gre=TR_gre,
        T1_gre=T1_gre,
        alpha_gre=alpha_gre,
        rf_dur_gre=rf_dur_gre,
        n_cycles_spoil_gre=n_cycles_spoil_gre,
        Tpre=Tpre,
        Ncoils=Ncoils,
        output_dir=output_dir,
        spec=spec,
        PNSwt=PNSwt,
        sampling_method=sampling_method,
        pd_calib_frac=pd_calib_frac,
        pd_crop_corner=pd_crop_corner,
        pd_decay=pd_decay,
        rand_gaussian_sigma=rand_gaussian_sigma,
    )
