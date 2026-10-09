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

from sample.external_mask import resolve_custom_omegas
from scanners import SCANNERS, ScannerSpec


@dataclass
class FatsatParams:
    flip: float  # degrees
    tbw: float  # time-bandwidth product (band = tbw / dur, centered on fat)
    dur: float  # s
    ftype: str = 'min'  # SLR filter: 'min' (minimum phase) or 'ls' (linear phase), see lib/slr.py
    # False replaces the fat-sat RF pulse with an equal-duration delay in
    # ArbEPI/EPIcal, keeping its crusher and every block timing identical --
    # isolates the pulse's own contribution (e.g. as an unintended refocusing
    # pulse on off-resonant water) from everything else.
    enabled: bool = True


@dataclass
class WaterExcParams:
    """Slab-selective binomial water excitation (lib/make_water_excitation.py),
    used when Params.excitation == 'water'."""
    binomial: tuple = (1, 3, 3, 1)  # subpulse flip weights
    tbw: float = 8  # time-bandwidth product of each sinc subpulse
    apodization: float = 0.5  # Hanning window on each subpulse (0 = plain sinc)


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

    # Custom ky-kz(-t) sampling mask, loaded via sample/external_mask.py's
    # load_external_mask when custom_mask_path is not None -- see
    # README.md's "Using custom ky-kz-t sampling masks" section. Replaces
    # gen_sampling_masks (and the R/sampling_method/seed fields above)
    # entirely: R/Nshots are instead derived from the mask's own sample
    # count. custom_omegas holds the already-loaded, already-broadcast
    # (Ny, Nz, Nframes) mask (None on the built-in gen_sampling_masks path);
    # sample/gen_sampling_masks.py's resolve_omegas returns it in place of
    # calling gen_sampling_masks.
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

    discard_duration: float  # s; n_frames_discard * volume_tr by default
    Nframes: int

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
    # an identical net spoiler moment: a constant per-shot spoiler moment
    # would let that periodicity through undisturbed.
    spoil_cycles_min: float  # cycles/voxel
    spoil_cycles_max: float  # cycles/voxel

    # Fat saturation
    fat_chem_shift: float  # ppm (dimensionless ratio)
    fat_offres_freq: float  # Hz
    fatsat: FatsatParams

    # Fat suppression strategy for ArbEPI/EPIcal: 'fatsat' (fat-sat pulse +
    # crusher, then the slab-selective sinc excitation) or 'water' (binomial
    # water excitation, `water_exc`, with no fat-sat pulse or crusher;
    # `fatsat` is then unused, including its `enabled` flag).
    excitation: str
    water_exc: WaterExcParams

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
    slew_degre: float  # T/m/s, deGRE's own slew limit (sequences/deGRE.py)

    pd_calib_frac: float
    pd_crop_corner: bool
    pd_decay: float
    rand_gaussian_sigma: np.ndarray | None

    @property
    def n_frames_discard(self) -> int:
        """Leading warm-up frames, derived from discard_duration so it cannot
        go stale under dataclasses.replace(discard_duration=...)."""
        return round(self.discard_duration / self.volume_tr)


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
    res = np.array([2.4, 2.4, 2.4]) * 1e-3
    N = np.array([90, 90, 60])
    fov = N * res
    Nx, Ny, Nz = int(N[0]), int(N[1]), int(N[2])

    # Nominal echo time, s. The minimum achievable TE depends on ETL, R,
    # resolution, scanner and the slews below (at the current defaults it
    # is well under 30 ms), and raising ETL, lowering R, or finer resolution
    # can push it past this value. calc_te_tr_delays only *warns* and then
    # silently uses the minimum instead, so check its output after changing
    # any of the above.
    TE = 30e-3
    # Time to acquire one full 3D volume (all shots), s. Must clear min_tr
    # (Nshots * per-shot min TR, printed as a warning by calc_te_tr_delays
    # if not -- it then silently plays the longer TR). Set to the fastest
    # achievable, rounded up to the next ms (explicit user decision,
    # 2026-09-30): measured at the ETL=54/R=10/Nx=90/res=2.4mm/TE=30ms
    # water-excitation defaults (seed=0, slews below), min_tr = 50.524
    # ms/shot * 10 shots = 505.24 ms. Any change to TE, ETL, R, resolution,
    # the slews, the spoilers, the excitation mode, epi_trajectory or the
    # mask seed moves this minimum (e.g. 'laminar' needs 51.71 ms/shot, for
    # its larger blips); re-measure (min_tr from calc_te_tr_delays) and
    # update.
    volume_tr = 0.506
    # Total scan duration across all frames/timepoints, s.
    duration = 60
    # Tissue T1, s -- used below to compute the Ernst-angle flip angle.
    T1 = 1.3

    # Frames to discard at the start of the scan (steady-state warm-up), the
    # same warm-up as a duration in s (derived: n_frames_discard * volume_tr),
    # and the resulting number of acquired frames. EPIcal plays dummy shots
    # for the same duration (sequences/EPIcal.py). Params.n_frames_discard
    # reads it back (preprocess/ records it as an attribute). Hoisted up from the
    # "ADVANCED / DERIVED PARAMETERS" section below -- unlike Nshots,
    # Nframes depends only on duration/volume_tr/discard_duration, not on
    # R/ETL/the sampling mask -- so a custom mask's own time-frame count
    # (below) can be validated against it immediately.
    n_frames_discard = 0
    discard_duration = n_frames_discard * volume_tr
    Nframes = round((duration + discard_duration) / volume_tr)

    # Echo train length (number of echoes acquired per shot).
    ETL = 54

    # Custom ky-kz(-t) sampling mask (optional): path to a collaborator-
    # provided .mat file holding an externally-designed 0/1 sampling mask
    # (2D (Ny, Nz), reused every frame, or 3D (Ny, Nz, Nframes),
    # time-resolved), loaded via sample/external_mask.py's
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
        R = 10
        Nshots = math.ceil(Ny * Nz / R / ETL)

        # ky-kz(-t) sampling pattern: 'pd' (Poisson-disc, recommended), 'caipi',
        # 'ticaipi', or 'rand'. See sample/gen_sampling_masks.py.
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

    # Fat suppression: 'fatsat' (spectrally selective fat-sat pulse + crusher
    # before the slab-selective sinc excitation, every shot) or 'water'
    # (slab-selective binomial 1-3-3-1 water excitation, no fat-sat or crusher:
    # 7.3 ms shorter per-shot min TR and 1.1 ms longer min TE on the default
    # protocol, but water off resonance gets less flip -- see
    # lib/make_water_excitation.py and `water_exc` below). Dropping the crusher
    # also halves the random spoiling between excitations (only the
    # post-readout spoiler remains), which spoil_cycles_min/max below account
    # for. 'water' is the default since 2026-09-30 (explicit user decision).
    excitation = 'water'

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
    # ADC dwell is not set here: lib/readout_from_params.py's
    # find_min_feasible_dwell searches for the fastest (smallest) dwell --
    # an integer multiple of sys.adc_raster_time -- that keeps the EPI
    # readout-lobe geometry feasible for the actual sampling mask's
    # max_ky_step/max_kz_step (see docs/review-findings.md item 146: a
    # fixed dwell can make the POPE readout ramps alone exceed the
    # required k-space area at some Nx/fov/slew/mask combinations).

    # PNS-driven slew limits (T/m/s) -- see the Params field comments.
    # Values below are the fastest found by an empirical sweep (2026-09-30,
    # ~9000 rise/fall/blip/slew_derate candidates on the current default
    # protocol, each a full-dims worst-frame ArbEPI build with every shot's
    # spoiler forced to spoil_cycles_max, scored by ge/pns.py's RSS-combined
    # whole-sequence peak; objective: shortest per-shot min TR at <= 79%
    # PNS, leaving ~1% under GE's 80% normal-mode limit). The sweep's own
    # measured percentages/TEs are NOT reproduced here since they're
    # protocol-dependent and go stale on every resolution/R/ETL/TE change
    # (docs/review-findings.md item 204). For the *current* build's real
    # numbers, read docs/review-findings.md's "Current baseline" table or
    # run `main.py --ge`/`tests/test_ge_check.py`'s
    # `test_arbepi_default_params_peak_pns_under_normal_mode_limit`, which
    # regression-guards peak PNS staying under the 80% normal-mode line on
    # every test run regardless of what these values are set to.
    #
    # What the sweeps have shown (the qualitative shape of the tuning, not a
    # specific number): rise < fall, and blip_slew is tuned independently of
    # both, because the y/z blips play centered on the kx turnaround --
    # exactly where the readout's POPE fall ramp ends -- so an aggressive
    # fall slew RSS-combines with the blip into a 3-channel PNS hotspot
    # (e.g. rise/fall/blip 95/200/170 can look fine per-channel but RSS-total
    # well over 100% on the 2026-08 protocol). slew_derate is limited by the
    # post-readout spoiler, not the readout: past ~120 its PNS takes over
    # (2026-09-30 sweep: 125 -> +0.8%, 150 -> +9%), while it only moves min
    # TR by tens of microseconds per shot. Re-run the sweep (not just
    # eyeball these numbers) after any change to seed/mask/R/ETL/
    # resolution/TE/spoilers -- the right combination is a joint,
    # non-monotonic function of all of those.
    slew_derate = 120.0
    ro_slew_rise = 155.0  # POPE-throttled ramp-up
    ro_slew_fall = 190.0  # ramp-down; see above
    blip_slew = 200.0  # tuned jointly with rise/fall above; re-sweep after any protocol change

    # PNS channel weights: the IEC 60601-2-33:2022-recommended
    # [0.8, 1.0, 0.7] for human scanning, or [0, 0, 0] to disable the PNS
    # check entirely for phantom scanning. See CLAUDE.md's "PNS finding
    # history" before changing this away from the human default.
    PNSwt = np.array([0.8, 1.0, 0.7])  # human
    # PNSwt = np.array([0.0, 0.0, 0.0])  # phantom

    TR = volume_tr / Nshots

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
    # comment. 2-6 cycles/voxel (explicit user decision, 2026-09-30),
    # independently per axis, drawn fresh each shot (see
    # sequences/ArbEPI.py's per-shot loop). The spoiler trapezoids are built
    # at spoil_cycles_max, so it sets their duration (and min TR) and their
    # PNS (the slew sweep above assumes the max on every shot).
    spoil_cycles_min = 2.0
    spoil_cycles_max = 6.0

    fat_chem_shift = 3.5 * 1e-6
    fat_offres_freq = sys.gamma * sys.B0 * fat_chem_shift
    # Min-phase SLR, 333 Hz band (TBW 2 / 6 ms) centered on fat. Bloch-simulated
    # (lib/make_fatsat_rf.py's flip_profile): water <= 1.6 deg over -150..+150 Hz
    # and <= 3.8 deg down to -200 Hz; fat >= 81 deg within +-50 Hz, >= 60 deg
    # within +-100 Hz. Sized against measured in-object B0 (99% of voxels in
    # -144..+79 Hz in vivo, 20260922xiaokai; -57..+35 Hz in the ball phantom)
    # and the per-shot TR budget: +2 ms over the old 4 ms pulse fit the
    # 2026-09 fat-sat default protocol (90x90x60, R=6, ETL=60; 6.2 ms slack)
    # and 20260924ball's 5.4 mm one (2.85 ms). On the current default
    # protocol fat-sat mode needs 58.55 ms/shot (vs 50.52 ms in water mode),
    # so switching `excitation` back needs volume_tr >= 0.586 s.
    # A longer pulse protects water further out but narrows the fat band and
    # costs TR -- re-check with flip_profile (and tests/test_make_fatsat_rf.py)
    # before changing these. The old pypulseq Gaussian (90 deg, TBW 3, 4 ms)
    # tipped on-resonance water 26 deg: docs/review-findings.md item 255.
    fatsat = FatsatParams(flip=90, tbw=2, dur=6e-3, ftype='min')
    # Used when excitation == 'water'. Bloch-simulated at the default protocol
    # (15.9 deg, 129.6 mm slab; lib/bloch.py; 2026-09-30): fat <= 0.7 deg
    # anywhere in the slab within +-100 Hz of the fat peak; water 0.89x the
    # flip at +-80 Hz, 0.64x at -150 Hz; slab FWHM 128.1 mm, <= 0.5% profile
    # overshoot (Hanning; 16% unapodized at TBW 8).
    # tests/test_make_water_excitation.py guards it.
    water_exc = WaterExcParams(binomial=(1, 3, 3, 1), tbw=8, apodization=0.5)
    if excitation not in ('fatsat', 'water'):
        raise ValueError(f"excitation must be 'fatsat' or 'water', got {excitation!r}")

    # deGRE (dual-echo GRE) parameters
    res_degre = np.array([3, 3, 3]) * 1e-3
    # N_degre (and hence fov_degre = N_degre*res_degre) tracks the EPI FOV
    # (`fov` above) rather than independently hardcoded numbers, so it
    # stays in sync if the EPI FOV ever changes. N_degre is the smallest
    # integer voxel count at res_degre spacing whose FOV is still >= the
    # EPI FOV per axis (np.ceil, not round) -- e.g. fov[2]=40.5mm at
    # res_degre[2]=3mm needs ceil(40.5/3)=14 voxels, giving
    # fov_degre[2]=42mm exactly, not some in-between value 3mm voxels
    # can't actually represent. So the deGRE FOV is always >= the EPI FOV.
    # Epsilon before ceil, same fix as lib/trap4ge.py's _round_up_to_raster
    # -- float64 noise can make an exact ratio like 216mm/3mm evaluate to
    # 72.00000000000001 instead of 72.0, which would otherwise make
    # ceil silently add a spurious extra voxel.
    #
    # degre_z_margin (m, per side) extends only the z-FOV past the EPI's, so
    # the deGRE slab covers the EPI slab with room to spare even when a
    # single deGRE is shared across EPI variants whose z-FOVs differ by a
    # rounding step (e.g. 144mm vs 145.8mm at 2.4mm vs 5.4mm res) -- avoids
    # needing preprocess/grid_resize.py's zero_pad_z workaround.
    # Preprocessing handles any deGRE FOV >= the EPI's on every axis:
    # preprocess/grid_resize.py resamples each EPI voxel center from its
    # physical position on the deGRE grid, and preprocess/coils.py's
    # gcc_calibration evaluates the deGRE at the EPI's x positions. So the
    # EPI x/y FOV need not be a multiple of res_degre (216 mm = 72 x 3 mm at
    # the default; a 202.5 mm EPI FOV would get a 68-voxel, 204 mm deGRE).
    degre_z_margin = 4e-3
    N_degre = np.ceil((fov + np.array([0, 0, 2 * degre_z_margin])) / res_degre - 1e-9).astype(int)
    fov_degre = N_degre * res_degre
    Nx_degre, Ny_degre, Nz_degre = int(N_degre[0]), int(N_degre[1]), int(N_degre[2])

    Ndummy_zloops = 4
    # Two echo times for B0 field mapping (see sequences/deGRE.py, ported
    # from HarmonizedMRI/B0shimming's writeB0.m, which uses the same pair):
    # ΔTE is exactly 1/fat_offres_freq, so fat accumulates one full extra
    # 2*pi of phase between TE1 and TE2, making its contribution to the
    # echo-to-echo phase difference (what the field map is computed from)
    # identical at both echoes and cancel out -- no separate fat-sat pulse
    # needed. TE1 = 1/fat_offres_freq also puts fat and water in phase at
    # both echoes. Until 2026-09-30 both TEs carried a +0.8 ms offset,
    # because at 2 mm with fixed 1 ms prephasers the minimum TE didn't
    # clear a bare 1/fat_offres_freq (2.237 ms at 3 T); at 3 mm it is
    # 1.26 ms, and sequences/deGRE.py stretches the prephasers to fill the
    # rest.
    TE_degre = 1 / fat_offres_freq * np.array([1.0, 2.0])
    # TR_degre: the minimum for the longer TE, measured 5.700 ms at the
    # 3 mm/slew_degre=175 defaults (2026-09-30). generate_degre raises, with
    # the achievable minimum, if a parameter change pushes it past this.
    TR_degre = 5.70e-3
    alpha_degre = 180 / math.pi * math.acos(math.exp(-TR_degre / T1))  # T1 set above

    rf_dur_degre = 0.4e-3
    n_cycles_spoil_degre = 2
    # deGRE slew limit (T/m/s). Its PNS peak is the x spoiler ramping up
    # right after the readout. Measured peak PNS on the full 3 mm default
    # build (GE_MR750, PNSwt [0.8, 1.0, 0.7], 2026-09-30): 170 -> 78.3%,
    # 175 -> 79.2%, 190 -> 81.4%, 200 (hardware) -> 83.3%, while TR only
    # drops 5.70 -> 5.65 ms from 175 to 200. 175 is the fastest (in 5 T/m/s
    # steps) under the 80% normal-mode limit.
    slew_degre = 175.0

    # Fully-sampled central calibration region: a centered rectangle sized
    # so its pixel area equals this fraction of the R-dependent sample
    # budget (floor(Ny*Nz/R)) -- a constant *share of the acquisition*
    # across every acceleration factor, not a fixed fraction of k-space
    # (see sample/pd_sample.py's calib_frac/_calib_side_frac docstrings,
    # and docs/review-findings.md item 195: a fixed-kmax-fraction region
    # left almost no samples outside calibration at high R -- e.g. 487 of
    # 520 target samples at the real 0.8mm/R~94 config).
    pd_calib_frac = 0.2
    pd_crop_corner = True
    pd_decay = 1.4
    rand_gaussian_sigma = None

    return Params(
        sys=sys,
        crt=crt,
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
        fa=fa,
        rf_dur=rf_dur,
        rf_tb=rf_tb,
        rf_phase_0=rf_phase_0,
        spoil_cycles_min=spoil_cycles_min,
        spoil_cycles_max=spoil_cycles_max,
        fat_chem_shift=fat_chem_shift,
        fat_offres_freq=fat_offres_freq,
        fatsat=fatsat,
        excitation=excitation,
        water_exc=water_exc,
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
        slew_degre=slew_degre,
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
