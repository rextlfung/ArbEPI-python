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

    # deGRE (dual-echo GRE) parameters
    res_degre: np.ndarray
    fov_degre: np.ndarray
    N_degre: np.ndarray
    Nx_degre: int
    Ny_degre: int
    Nz_degre: int
    Ndummy_zloops: int
    TE_degre: np.ndarray  # s, [TE1, TE2] -- two echo times for B0 field mapping
    TR_degre: float
    T1_degre: float
    alpha_degre: float  # degrees
    rf_dur_degre: float
    n_cycles_spoil_degre: int
    Tpre: float

    # Noise prescan
    Ncoils: int

    # Output
    output_dir: str

    # The selected ScannerSpec itself (see scanners.py) -- ge/ge_export.py's
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
    seed: int | None  # passed to gen_sampling_masks' rng; None = unseeded (fresh mask each run)

    # Sampling trajectory
    epi_trajectory: str  # 'laminar' (mask2epi_laminar) or 'radial' (mask2epi_radial)

def load_params(output_dir: str = 'output') -> Params:
    scanner = 'GE_MR750' # 'GE_UHP' or 'GE_MR750'. See scanners.py
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

    crt = 4e-6  # s, GE raster time only; would need 20e-6 (lcm of Siemens 10us, GE 4us) for GE AND Siemens compatibility
    dwell = 2e-6  # s, for ADC/RF

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
    # 39ms for now (min achievable is ~38.1ms at this ETL/R/slew derate) --
    # BOLD-contrast feasibility of this TE still needs checking.
    TE = 39e-3
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

    # deGRE (dual-echo GRE) parameters
    res_degre = np.array([2, 2, 2]) * 1e-3
    # N_degre (and hence fov_degre = N_degre*res_degre) tracks the EPI FOV
    # (`fov` above) rather than independently hardcoded numbers, so it
    # stays in sync if the EPI FOV ever changes. N_degre is the smallest
    # integer voxel count at res_degre spacing whose FOV is still >= the
    # EPI FOV per axis (np.ceil, not round) -- e.g. fov[2]=40.5mm at
    # res_degre[2]=2mm needs ceil(40.5/2)=21 voxels, giving
    # fov_degre[2]=42mm exactly, not some in-between value 2mm voxels
    # can't actually represent. z needs fov_degre[2] >= fov[2]
    # specifically (preprocessing/smaps.py's process_smaps raises
    # otherwise); x/y get the same treatment for consistency even though
    # nothing currently enforces it there. This replaces the old
    # independent 216mm x/y, 42mm z values.
    # Epsilon before ceil, same fix as lib/trap4ge.py's _round_up_to_raster
    # -- float64 noise can make an exact ratio like 216mm/2mm evaluate to
    # 108.00000000000001 instead of 108.0, which would otherwise make
    # ceil silently add a spurious extra voxel.
    N_degre = np.ceil(fov / res_degre - 1e-9).astype(int)
    fov_degre = N_degre * res_degre
    Nx_degre, Ny_degre, Nz_degre = int(N_degre[0]), int(N_degre[1]), int(N_degre[2])

    Ndummy_zloops = 4
    # Two echo times for B0 field mapping (see sequences/deGRE.py, ported
    # from HarmonizedMRI/B0shimming's writeB0.m): ΔTE is fixed at exactly
    # 1/fat_offres_freq so fat accumulates one full extra 2*pi of phase
    # between TE1 and TE2, making its contribution to the echo-to-echo
    # phase difference (what the field map is computed from) identical at
    # both echoes and cancel out -- no separate fat-sat pulse needed. The
    # +8e-4 s offset applied to both (preserving that exact delta) is the
    # same margin the old single-echo TE_degre used, needed because this
    # sequence's slab-selective excitation/prephasing (unlike writeB0.m's
    # non-selective block pulse) doesn't clear a bare 1/fat_offres_freq.
    TE_degre = 1 / fat_offres_freq * np.array([1.0, 2.0]) + 8e-4
    # TR_degre must clear tr_min for the *longer* of the two TEs (~7.6ms at
    # the values above, given this sequence's slab-selective excitation
    # timing) -- the old single-echo GRE's 6e-3 no longer fits once TE2 is
    # roughly double TE1. 8e-3 leaves a small margin.
    TR_degre = 8e-3
    T1_degre = 1.3
    alpha_degre = 180 / math.pi * math.acos(math.exp(-TR_degre / T1_degre))

    rf_dur_degre = 0.4e-3
    n_cycles_spoil_degre = 2
    Tpre = 1.0e-3

    Ncoils = 32

    # IEC 60601-2-33:2022-recommended PNS channel weights -- see CLAUDE.md's
    # "Open finding" for why this was zero (disabling the PNS check
    # entirely) until now, and that this default now correctly blocks
    # `main.py --ge` for EPIcal/ArbEPI/deGRE until slew/blip timing changes.
    PNSwt = np.array([0.8, 1.0, 0.7]) # human
    # PNSwt = np.array([0.0, 0.0, 0.0]) # phantom

    # Sampling mask
    sampling_method = 'pd'
    # Fully-sampled central calibration region: a centered ellipse,
    # aspect-matched to (Ny, Nz), sized to hold 30% of the R-dependent
    # sample budget (floor(Ny*Nz/R) -- see sampling/pd_sample.py).
    pd_calib_frac = 0.3
    pd_crop_corner = True
    pd_decay = 1.4
    rand_gaussian_sigma = None
    # None = a fresh unseeded rng each run (current default behavior). Set
    # to an int for a reproducible mask, e.g. to compare epi_trajectory
    # settings against the same sampling mask -- main.py passes this
    # straight to gen_sampling_masks(..., rng=np.random.default_rng(seed)).
    seed = 0

    # Sampling trajectory: which mask2epi_* variant orders each shot's echo
    # train. 'laminar' = mask2epi_laminar (ky non-decreasing rows, ported
    # from MATLAB). 'radial' = mask2epi_radial (each shot sweeps through
    # k-space center as one spoke, this repo's own addition) -- see
    # lib/mask2epi.py's module docstring for the tradeoffs between them.
    epi_trajectory = 'radial'

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
        res_degre=res_degre,
        fov_degre=fov_degre,
        N_degre=N_degre,
        Nx_degre=Nx_degre,
        Ny_degre=Ny_degre,
        Nz_degre=Nz_degre,
        Ndummy_zloops=Ndummy_zloops,
        TE_degre=TE_degre,
        TR_degre=TR_degre,
        T1_degre=T1_degre,
        alpha_degre=alpha_degre,
        rf_dur_degre=rf_dur_degre,
        n_cycles_spoil_degre=n_cycles_spoil_degre,
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
        seed=seed,
        epi_trajectory=epi_trajectory,
    )
