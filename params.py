"""Central configuration, mirroring ../ArbEPI/params.m.

params.m is a MATLAB script that injects variables into the caller's
workspace; there is no Python equivalent of that pattern. Instead,
``load_params()`` returns a single ``Params`` dataclass instance that is
passed explicitly to every function that needs it.

To configure a scan, edit the "USER CONFIGURATION" section near the top of
``load_params()`` below. Everything under "ADVANCED / DERIVED PARAMETERS"
is pre-tuned for this sequence's hardware/PNS/timing constraints (see
CLAUDE.md) and typically doesn't need to change.
"""

import math
from dataclasses import dataclass

import numpy as np
import pypulseq as pp

from sampling.external_mask import resolve_custom_omegas
from scanners import SCANNERS, ScannerSpec


@dataclass
class FatsatParams:
    flip: float  # degrees
    sl_thick: float  # m (dummy value; just needs to be large)
    tbw: float  # time-bandwidth product
    dur: float  # s


@dataclass
class Params:
    # =====================================================================
    # Fields set directly from load_params()'s USER CONFIGURATION section
    # (some, like fov/Nx/Ny/Nz/Nshots/TR, are simple derived values kept
    # next to the user-set field they come from).
    # =====================================================================

    # The selected ScannerSpec itself (see scanners.py) -- ge/ge_export.py's
    # pure-Python feasibility check/export reads max_grad/max_slew/b1_max/
    # chronaxie/rheobase/alpha/ge_coil/pislquant straight from this, so
    # they can't drift out of sync with `sys` below (both come from the
    # same ScannerSpec instance).
    spec: ScannerSpec

    # EPI spatial parameters
    res: np.ndarray  # m, [x, y, z]
    fov: np.ndarray  # m
    Nx: int
    Ny: int
    Nz: int

    # Acceleration
    R: float
    ETL: int
    Nshots: int

    # Sampling mask parameters. Both are ignored (set to None by load_params)
    # when custom_mask_path below is set -- see that field's comment.
    sampling_method: str | None
    seed: int | None  # passed to gen_sampling_masks' rng; None = unseeded (fresh mask each run)

    # Custom ky-kz(-t) sampling mask, loaded via sampling/external_mask.py's
    # load_external_mask when custom_mask_path is not None -- see
    # README.md's "Using custom ky-kz-t sampling masks" section. Replaces
    # gen_sampling_masks (and the R/sampling_method/seed fields above)
    # entirely: R/Nshots are instead derived from the mask's own sample
    # count. custom_omegas holds the already-loaded, already-broadcast
    # (Ny, Nz, Nframes) mask (None on the built-in gen_sampling_masks path);
    # main.py uses it directly in place of calling gen_sampling_masks.
    custom_mask_path: str | None
    custom_omegas: np.ndarray | None

    # Sampling trajectory
    epi_trajectory: str  # 'laminar' (mask2epi_laminar) or 'radial' (mask2epi_radial) -- always
    # required, even with a custom mask: mask2epi still partitions any
    # (ky, kz) mask into Nshots EPI trajectories of length ETL, regardless
    # of where the mask came from.

    # Decay / timing
    TE: float  # s
    volume_tr: float  # s
    TR: float  # s

    discard_duration: float  # s
    Nframes: int
    Ndummyshots: int

    # Noise prescan
    Ncoils: int

    # Output
    output_dir: str

    # =====================================================================
    # Advanced / derived fields -- see load_params()'s ADVANCED / DERIVED
    # PARAMETERS section. Pre-tuned for this sequence's hardware/PNS/timing
    # constraints; usually don't need to change.
    # =====================================================================

    sys: pp.Opts
    crt: float  # common raster time (s)
    dwell: float  # s, ADC sample time

    # PNS-driven slew limits (T/m/s), the single source of truth consumed
    # via lib/readout_from_params.py (ArbEPI/EPIcal used to hardcode a
    # `sys.max_slew = 100 * sys.gamma` derate in three separate places).
    # slew_derate: general derate applied to everything except the readout
    # ramps and the ky/kz blips (excitation, fat-sat, prephasers, spoilers)
    # -- the blips get their own derate, blip_slew below, via blip_sys().
    # ro_slew_rise/ro_slew_fall: the readout trapezoid's asymmetric POPE
    # ramps (see lib/make_readout_grads.py's module docstring: PNS peaks at
    # the end of each ramp-up, so only the rise is throttled while the fall
    # runs at/near hardware slew).
    slew_derate: float
    ro_slew_rise: float
    ro_slew_fall: float
    # Blip slew (T/m/s), separate from both of the above: the blips play
    # centered on the kx turnaround (right after the fast fall), so they
    # are their own lever in the RSS PNS combination -- faster blips
    # shorten the blip window (and with it every readout lobe) but raise
    # the y/z contribution at the turnaround hotspot.
    blip_slew: float

    # PNS is a physiological safety limit, not a hardware one -- kept
    # separate from ScannerSpec since it's phantom-vs-human scan context,
    # not a scanner constant.
    PNSwt: np.ndarray

    fa: float  # degrees, Ernst angle
    rf_dur: float  # s
    rf_tb: float
    rf_phase_0: float  # degrees

    # Gradient spoiler design (lib/make_spoilers.py): area is expressed
    # directly in cycles of phase twist per voxel along each axis. Varied
    # shot-to-shot (uniform-random within this range, independently per
    # axis) rather than held constant, so a residual coherence pathway
    # that survives one RF-spoiling phase-cycle period doesn't also see
    # an identical net spoiler moment -- see CLAUDE.md's RF-spoiling
    # section for why a constant per-shot spoiler moment lets that
    # periodicity through undisturbed.
    spoil_cycles_min: float  # cycles/voxel
    spoil_cycles_max: float  # cycles/voxel

    # Fat saturation
    fat_chem_shift: float  # ppm (dimensionless ratio)
    fat_offres_freq: float  # Hz
    fatsat: FatsatParams

    # deGRE (dual-echo GRE) parameters
    fov_degre: np.ndarray
    Nx_degre: int
    Ny_degre: int
    Nz_degre: int
    Ndummy_zloops: int
    TE_degre: np.ndarray  # s, [TE1, TE2] -- two echo times for B0 field mapping
    TR_degre: float
    alpha_degre: float  # degrees
    rf_dur_degre: float
    n_cycles_spoil_degre: int
    Tpre: float

    pd_calib_frac: float
    pd_crop_corner: bool
    pd_decay: float
    rand_gaussian_sigma: np.ndarray | None


def load_params(output_dir: str = 'output') -> Params:
    # =================================================================
    # USER CONFIGURATION
    #
    # Edit the values in this section to set up a scan. Everything under
    # "ADVANCED / DERIVED PARAMETERS" below is pre-tuned for this
    # sequence's hardware/PNS/timing constraints (see CLAUDE.md) and
    # typically doesn't need to change.
    # =================================================================

    # Scanner hardware profile: 'GE_MR750' or 'GE_UHP'. See scanners.py.
    scanner = 'GE_MR750'

    # Spatial parameters: voxel resolution [x, y, z] (m) and acquisition
    # matrix size [Nx, Ny, Nz]. x/y are the in-plane (readout/phase-encode)
    # axes, z is the slice-select/partition axis.
    res = np.array([0.9, 0.9, 0.9]) * 1e-3
    N = np.array([240, 240, 45])
    fov = N * res
    Nx, Ny, Nz = int(N[0]), int(N[1]), int(N[2])

    # Nominal echo time, s. NOTE: this sits close to the minimum achievable
    # TE for the default ETL/R/scanner/slews below (see CLAUDE.md's "PNS
    # finding history") -- raising ETL, lowering R, or changing resolution
    # can make this value unreachable. calc_te_tr_delays only *warns* and
    # silently falls back to zero padding delay if so, so check its output
    # after changing any of the above.
    TE = 34.9e-3
    # Time to acquire one full 3D volume (all shots), s.
    volume_tr = 2
    # Total scan duration across all frames/timepoints, s.
    duration = 60
    # Tissue T1, s -- used below to compute the Ernst-angle flip angle.
    T1 = 1.3

    # Frames to discard at the start of the scan (steady-state warm-up), s,
    # and the resulting number of acquired frames. Hoisted up from the
    # "ADVANCED / DERIVED PARAMETERS" section below -- unlike Nshots,
    # Nframes depends only on duration/volume_tr/discard_duration, not on
    # R/ETL/the sampling mask -- so a custom mask's own time-frame count
    # (below) can be validated against it immediately.
    discard_duration = 0
    Nframes = round((duration + discard_duration) / volume_tr)

    # Echo train length (number of echoes acquired per shot).
    ETL = 60

    # Custom ky-kz(-t) sampling mask (optional): path to a collaborator-
    # provided .mat file holding an externally-designed 0/1 sampling mask
    # (2D (Ny, Nz), reused every frame, or 3D (Ny, Nz, Nframes),
    # time-resolved), loaded via sampling/external_mask.py's
    # load_external_mask -- see README.md's "Using custom ky-kz-t sampling
    # masks" section. When set, this replaces gen_sampling_masks (and the
    # R/sampling_method/seed fields below) entirely: R and Nshots are
    # instead derived from the mask's own per-frame sample count (R purely
    # for display/scan_info.mat bookkeeping at that point -- no longer a
    # design input), and sampling_method/seed go unused. Leave None to use
    # the built-in sampling_method/seed/R path instead.
    custom_mask_path = None  # e.g. 'my_custom_mask.mat'
    custom_mask_key = 'samp'  # variable name inside that .mat file

    if custom_mask_path is not None:
        custom_omegas, Nshots, R = resolve_custom_omegas(
            custom_mask_path, Ny, Nz, Nframes, ETL, key=custom_mask_key
        )
        sampling_method = None
        seed = None
    else:
        custom_omegas = None

        # Acceleration factor applied to the (ky, kz) sampling pattern.
        R = 9
        Nshots = math.ceil(Ny * Nz / R / ETL)

        # ky-kz(-t) sampling pattern: 'pd' (Poisson-disc, recommended), 'caipi',
        # 'ticaipi', or 'rand'. See sampling/gen_sampling_masks.py.
        sampling_method = 'pd'
        # Sampling-mask RNG seed: an int for a reproducible mask across runs
        # (every PNS/timing number quoted in CLAUDE.md/README uses seed=0), or
        # None for a fresh, unseeded mask every run.
        seed = 0

    # Echo-train ordering within each shot: 'radial' (every shot sweeps
    # through k-space center as one spoke) or 'laminar' (ky non-decreasing
    # rows, ported from the original MATLAB repo). It's still unclear which
    # gives better image quality -- see lib/mask2epi.py's module docstring
    # for the tradeoffs.
    epi_trajectory = 'radial'

    # Number of receive coil channels (used for the noise prescan).
    Ncoils = 32

    # =================================================================
    # ADVANCED / DERIVED PARAMETERS
    #
    # Pre-tuned for this sequence's hardware/PNS/timing constraints (see
    # CLAUDE.md). Usually don't need to change these.
    # =================================================================

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

    # PNS-driven slew limits (T/m/s) -- see the Params field comments.
    # Values below are the outcome of an empirical sweep (2026-08-27, ~600
    # rise/fall/blip candidates, each a full-dims worst-frame ArbEPI build
    # evaluated with ge/pns.py's RSS-combined total; see CLAUDE.md's PNS
    # section). rise/fall = 100/120 is the fastest readout-ramp pair under
    # the 80% normal-mode line (rise 105 tips the full build to 80.06% at
    # identical echo spacing; fall beyond ~120 buys almost no echo spacing
    # because with ramp sampling the flat top regrows to keep +-kmax
    # coverage). blip_slew = 105 is a deliberate ride-the-line choice
    # (explicit user decision, 2026-08-27): the full seed=0 build measures
    # 79.8% peak PNS at min TE 34.86 ms, vs 78.3% at 35.10 ms for
    # blip_slew = 100 -- only ~0.2% margin to the 80% line, thinner than
    # observed mask-to-mask variation, so re-verify (regression test +
    # main.py --ge) after ANY change to seed/mask/R/ETL/resolution and
    # drop back to 100 if a new mask pushes it over. blip 110 sits at
    # ~80.0% (coin flip), 115+ is over; the symmetric-100 design measures
    # 77.4% at min TE 35.80 ms. The POPE gain is
    # deliberately modest here: on this whole-body GE gradient the y-blip
    # plays centered on the kx turnaround, i.e. exactly where the readout
    # fall ramp ends, so an aggressive fall slew RSS-combines with the
    # blip into a 3-channel hotspot (e.g. rise/fall/blip 95/200/170 looks
    # great per-channel but its RSS total is 106%) -- the sweep therefore
    # lands on a mild fall/rise ratio and a moderate blip slew rather than
    # the paper's hardware-limit fall. The prescribed-TE target of 30 ms
    # is unreachable under the 80% normal-mode line: this config is the
    # fastest sub-80% one found (min TE 34.86 ms); reaching ~33 ms costs
    # >85%, and ~30 ms well over 100%.
    slew_derate = 100.0
    ro_slew_rise = 100.0  # POPE-throttled ramp-up
    ro_slew_fall = 120.0  # ramp-down; not PNS-limited per se, but see above
    blip_slew = 105.0  # ride-the-line choice, ~0.2% PNS margin -- see above

    # PNS channel weights: the IEC 60601-2-33:2022-recommended
    # [0.8, 1.0, 0.7] for human scanning, or [0, 0, 0] to disable the PNS
    # check entirely for phantom scanning. See CLAUDE.md's "PNS finding
    # history" before changing this away from the human default.
    PNSwt = np.array([0.8, 1.0, 0.7])  # human
    # PNSwt = np.array([0.0, 0.0, 0.0])  # phantom

    TR = volume_tr / Nshots

    Ndummyshots = round(discard_duration / TR)

    fa = 180 / math.pi * math.acos(math.exp(-TR / T1))
    rf_dur = 2e-3
    rf_tb = 6
    # Quadratic RF-spoiling phase increment. 115.4 degrees (not the more
    # commonly-cited 117) per Leupold, Weigel & Bär, PLOS ONE 2025 ("On
    # the choice of the phase difference increment in RF-spoiled
    # gradient-echo MRI of liquids with consideration of diffusion"),
    # which tested exactly this phantom-imaging scenario (liquid
    # phantoms) and found 115.4 outperforms 117 among commonly-used
    # increments.
    rf_phase_0 = 115.4
    # Gradient spoiler cycles/voxel range -- see Params.spoil_cycles_min's
    # comment. 3-4 cycles/voxel, independently per axis, drawn fresh each
    # shot (see sequences/ArbEPI.py's per-shot loop).
    spoil_cycles_min = 3.0
    spoil_cycles_max = 4.0

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
    alpha_degre = 180 / math.pi * math.acos(math.exp(-TR_degre / T1))  # T1 set above

    rf_dur_degre = 0.4e-3
    n_cycles_spoil_degre = 2
    Tpre = 1.0e-3

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
        slew_derate=slew_derate,
        ro_slew_rise=ro_slew_rise,
        ro_slew_fall=ro_slew_fall,
        blip_slew=blip_slew,
        res=res,
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
        discard_duration=discard_duration,
        Nframes=Nframes,
        Ndummyshots=Ndummyshots,
        fa=fa,
        rf_dur=rf_dur,
        rf_tb=rf_tb,
        rf_phase_0=rf_phase_0,
        spoil_cycles_min=spoil_cycles_min,
        spoil_cycles_max=spoil_cycles_max,
        fat_chem_shift=fat_chem_shift,
        fat_offres_freq=fat_offres_freq,
        fatsat=fatsat,
        fov_degre=fov_degre,
        Nx_degre=Nx_degre,
        Ny_degre=Ny_degre,
        Nz_degre=Nz_degre,
        Ndummy_zloops=Ndummy_zloops,
        TE_degre=TE_degre,
        TR_degre=TR_degre,
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
        custom_mask_path=custom_mask_path,
        custom_omegas=custom_omegas,
        epi_trajectory=epi_trajectory,
    )
