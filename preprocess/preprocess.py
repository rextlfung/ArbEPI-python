"""Raw GE ScanArchives -> one preprocessed h5 per sequence. Ports preprocess.m.

preprocess(cfg, seqname) runs, for one sequence:

Stage A -- mandatory, cached in <outdir>/<seq>_gridded.h5 (grid_epi)
    noise scan -> whitening matrix W (identity if there is no noise scan)
    cal scan   -> readout delay (k-space center offset), calibrated by a sweep,
                  then the odd/even (Nyquist ghost) phase model `a`; both on the
                  whitened, uncompressed calibration echo trains
    EPI        -> per frame: whiten, regrid the ramp-sampled readouts onto
                  Cartesian kx, odd/even phase correction, scatter into the
                  zero-filled (kx, ky, kz) grid. Streamed and resumable.
    noise      -> the noise readouts through the same gridding/correction,
                  so noise_var can be measured in the output coil space.

Coil compression -- optional (coil_compression)
    GCC: geometric-decomposition coil compression, per-x matrices along the
    fully sampled readout, fit on the whitened deGRE; the number of virtual
    coils from an energy threshold or set explicitly (cfg.Nvcoils).

Maps from the dual-echo deGRE (estimate_maps)
    ESPIRiT sensitivity maps, B0 field map (MRIFieldmaps.jl), R2* map.

Stage B -- output, <outdir>/<seq>_preprocessed.h5 (write_output)
    the cached k-space with the compression applied, the fully sampled
    calibration region, the maps, noise_var and provenance attributes.

Why compression can run after the cache: gridding, the odd/even correction
(one phase function applied identically to every coil) and the scatter are
linear and act on each coil separately, so a coil-mixing matrix -- or GCC's
per-x matrices, which act in hybrid (x, ky, kz) space where the sampling mask
does not depend on x -- gives the same result applied before or after them.
Estimating `a`, choosing the compression and ESPIRiT do not commute with coil
mixing, which is why they run on fixed, whitened, uncompressed data.

MATLAB permute/reshape calls are translated mechanically: permute(A, p) ->
A.transpose(p - 1), reshape(A, dims) -> A.reshape(dims, order='F').
"""

import os
import shutil
import warnings
from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy.interpolate import interp1d

from preprocess import utils
from preprocess.b0map import estimate_b0map, fit_mask, resize_to_epi
from preprocess.coils import (
    align_gcc,
    apply_gcc_image,
    apply_gcc_kspace,
    apply_whitening,
    compute_whitening_matrix,
    gcc_calibration,
    gcc_compression,
    select_nvcoils,
)
from preprocess.epi_gridding import rampsampepi2cart
from preprocess.grid_resize import resize_to_epi_grid
from preprocess.oephase import epiphasecorrect, getoephase
from preprocess.r2star import R2STAR_METHOD, fit_r2star
from preprocess.utils import load_kxoe, load_schedules, load_seq_params, matlab_round

# ---------------------------------------------------------------------------
# Configuration and paths
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    """Settings for preprocess(). Only datdir is required; see demo.ipynb."""

    datdir: str
    seqnames: list[str] = field(default_factory=list)
    fn_gre: str | None = None  # deGRE ScanArchive; default <datdir>/scanarchives/gre.h5
    outdir: str | None = None  # default <datdir>/recon

    # Coil compression of the output (whitening is always applied when the
    # sequence has a noise scan).
    compress: bool = True
    cc_energy_thresh: float = 0.999  # fraction of eigenvalue energy kept
    Nvcoils: int | None = None  # exact number of virtual coils; overrides the threshold
    cc_calib_size: int = 24  # GCC: central (ky, kz) block of the deGRE per x

    # Maps from the deGRE
    estimate_smaps: bool = True
    estimate_b0: bool = True  # needs julia
    estimate_r2star: bool = True
    gre_echo_idx: int = 0  # echo for ESPIRiT and compression (0 = shorter TE, more SNR)
    crop: float = 0.95  # ESPIRiT eigenvalue threshold, also the smaps/B0 support mask
    smaps_smooth_sigma_mm: float = 6.0  # Gaussian smoothing of smaps on the EPI grid; 0 = off
    b0map_mask_thresh: float = 0.1  # fraction of peak echo-1 magnitude fit by b0map.jl
    b0map_min_component: int = 64  # drop fit-mask islands smaller than this (voxels)
    b0map_precon: str = 'diag'  # MRIFieldmaps NCG preconditioner (see julia/b0map.jl)
    r2star_max: float = 200.0  # 1/s
    # Zero-fill EPI slices outside the deGRE's z coverage instead of raising
    # (for an EPI z-FOV slightly larger than the deGRE slab).
    zero_pad_z: bool = False

    # Output
    extract_calib: bool = True  # write the fully sampled calibration region
    keep_cache: bool = False  # keep <seq>_gridded.h5 (lets a rerun skip Stage A)

    def __post_init__(self):
        self.seqnames = list(self.seqnames)
        if self.fn_gre is None:
            self.fn_gre = os.path.join(self.datdir, 'scanarchives', 'gre.h5')
        if self.outdir is None:
            self.outdir = os.path.join(self.datdir, 'recon')


@dataclass
class SeqPaths:
    seqname: str
    seqdir: str
    scan_info: str
    cal: str
    noise: str
    epi: str
    gre: str
    cache: str
    output: str


def set_seq_paths(cfg: PreprocessConfig, seqname: str) -> SeqPaths:
    seqdir = os.path.join(cfg.datdir, 'seqs', seqname)
    scanarchives = os.path.join(cfg.datdir, 'scanarchives')
    return SeqPaths(
        seqname=seqname,
        seqdir=seqdir,
        scan_info=os.path.join(seqdir, 'scan_info.mat'),
        cal=os.path.join(scanarchives, f'{seqname}_cal.h5'),
        noise=os.path.join(scanarchives, f'{seqname}_noise.h5'),
        epi=os.path.join(scanarchives, f'{seqname}_epi.h5'),
        gre=cfg.fn_gre,
        # .h5, not .mat: plain numpy-order h5py, the opposite axis convention
        # from the hdf5storage-written scan_info.mat (see utils.read_mat_array).
        cache=os.path.join(cfg.outdir, f'{seqname}_gridded.h5'),
        output=os.path.join(cfg.outdir, f'{seqname}_preprocessed.h5'),
    )


# ---------------------------------------------------------------------------
# EPI building blocks
# ---------------------------------------------------------------------------


def apply_delay(
    kxo0: np.ndarray, kxe0: np.ndarray, nfid: int, delay: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shift the kx trajectories by the k-space center offset (samples), by
    linear interpolation that extrapolates at the ends (MATLAB's
    interp1(..., 'linear', 'extrap'); np.interp would clamp instead)."""
    idx = np.arange(1, nfid + 1, dtype=float)
    shifted = idx - 0.5 - delay
    kxo = interp1d(idx, kxo0, kind='linear', fill_value='extrapolate')(shifted)
    kxe = interp1d(idx, kxe0, kind='linear', fill_value='extrapolate')(shifted)
    return kxo, kxe


def compute_oephase(
    ksp_cal: np.ndarray, kxo: np.ndarray, kxe: np.ndarray, nx: int, fov_x_cm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Odd/even phase model from blip-free calibration echo trains.

    ksp_cal: [Nfid, ETL_even, N_cal_shots, Nc] from prepare_cal_data().
    Returns (a, th): getoephase's per-echo-pair model [ETL_even//2, 2] and its
    per-echo-pair phase mismatch.
    Uses the same centered-IFFT pairing (ifftshift in, fftshift out) along x as
    oephase.epiphasecorrect, so `a` is estimated in the pixel frame it is
    applied in (docs/review-findings.md item 251).
    """
    oephase_data = rampsampepi2cart(ksp_cal, kxo, kxe, nx, fov_x_cm)
    oephase_data = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(oephase_data, axes=0), n=nx, axis=0), axes=0
    )
    return getoephase(np.mean(oephase_data, axis=2))  # average over cal shots


def prepare_cal_data(ksp_cal_raw: np.ndarray, W: np.ndarray, ETL: int) -> np.ndarray:
    """Calibration readouts [Nfid, Ncoils, N_cal] -> [Nfid, ETL_even, N_cal_shots, Nc]:
    whitened, grouped into echo trains, truncated to an even ETL (getoephase
    pairs odd/even echoes)."""
    Nfid = ksp_cal_raw.shape[0]
    ksp_cal = apply_whitening(ksp_cal_raw.transpose(0, 2, 1), W)  # [Nfid, N_cal, Ncoils]
    ksp_cal = ksp_cal.reshape(Nfid, ETL, -1, ksp_cal.shape[-1], order='F')
    return ksp_cal[:, :ETL - (ETL % 2)]


DELAY_SWEEP = np.round(np.arange(-6, 6 + 0.025, 0.05), 2)  # candidate delays (samples)


def sweep_delay(
    ksp_cal: np.ndarray,
    kxo0: np.ndarray,
    kxe0: np.ndarray,
    Nx: int,
    fov_x_cm: float,
    delays: np.ndarray = DELAY_SWEEP,
    wrap_thresh: float = np.pi,
) -> dict[str, np.ndarray]:
    """The odd/even phase model at each candidate readout delay. Ports
    calibrate_delay.m.

    A wrong delay (k-space center offset) puts a linear phase ramp between odd
    and even echoes; once the ramp passes +-pi inside the object, getoephase's
    no-wrap linear fit breaks. For each delay this counts adjacent-pixel jumps
    > wrap_thresh in the odd/even phase over the central half of x (later echo
    pairs). ksp_cal: [Nfid, ETL_even, N_cal_shots, Nc] from prepare_cal_data().
    Returns {'delay', 'a1', 'a2', 'wrap_count'}, one entry per delay; a1/a2 are
    the model's mid-train (TE) constant and linear terms.
    """
    Nfid = ksp_cal.shape[0]
    rows = slice(matlab_round(Nx / 4), matlab_round(3 * Nx / 4))
    a_all, wraps = [], []
    for d in delays:
        a, th = compute_oephase(ksp_cal, *apply_delay(kxo0, kxe0, Nfid, d), Nx, fov_x_cm)
        d_th = np.diff(th[rows, th.shape[1] // 2:], axis=0)
        a_all.append(a.mean(axis=0))  # mid-train value: getoephase's fit is centered there
        wraps.append(int(np.sum(np.abs(d_th) > wrap_thresh)))
    a_all = np.array(a_all)
    return {'delay': np.asarray(delays, dtype=float), 'a1': a_all[:, 0], 'a2': a_all[:, 1],
            'wrap_count': np.array(wraps)}


def select_best_delay(report: dict) -> float:
    """Among delays with zero detected phase wraps, pick the one whose fitted
    linear term a2 is closest to zero (a correctly aligned readout leaves only
    a constant odd/even offset); if every candidate wrapped, fall back to the
    delay with the fewest wraps. Ports the selection at the end of
    calibrate_delay.m."""
    wrap_count = np.asarray(report['wrap_count'])
    a2 = np.asarray(report['a2'])
    delay = np.asarray(report['delay'])
    safe = np.flatnonzero(wrap_count == 0)
    if len(safe) == 0:
        idx = int(np.argmin(wrap_count))
    else:
        idx = safe[np.argmin(np.abs(a2[safe]))]
    return float(delay[idx])


def calibrate_odd_even(
    ksp_cal_raw: np.ndarray,
    W: np.ndarray,
    scan_info: str,
    Nx: int,
    ETL: int,
    fov_x_cm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Calibration scan -> (kxo, kxe, a, sweep): the readout delay, calibrated
    by sweep_delay/select_best_delay and applied to the odd/even readout
    trajectories; the odd/even phase model at that delay; and the sweep, with
    the chosen delay under 'best'."""
    Nfid = ksp_cal_raw.shape[0]
    ksp_cal = prepare_cal_data(ksp_cal_raw, W, ETL)
    kxo0, kxe0 = load_kxoe(scan_info)
    sweep = sweep_delay(ksp_cal, kxo0, kxe0, Nx, fov_x_cm)
    delay = select_best_delay(sweep)
    sweep['best'] = delay
    wraps = int(sweep['wrap_count'][np.argmin(np.abs(sweep['delay'] - delay))])
    print(f'  readout delay (k-space center offset): {delay:+.2f} samples, calibrated')
    if wraps:
        warnings.warn(
            f'calibrate_odd_even: every candidate delay leaves odd/even phase wraps (fewest: '
            f'{wraps}, at {delay:+.2f}); the ghost correction may be poor', stacklevel=2
        )
    elif delay in (sweep['delay'][0], sweep['delay'][-1]):
        warnings.warn(
            f'calibrate_odd_even: the calibrated delay {delay:+.2f} is at the edge of the '
            f'sweep ({sweep["delay"][0]:+.2f} to {sweep["delay"][-1]:+.2f})', stacklevel=2
        )
    kxo, kxe = apply_delay(kxo0, kxe0, Nfid, delay)
    a, _ = compute_oephase(ksp_cal, kxo, kxe, Nx, fov_x_cm)
    print(f'  odd/even phase at mid-train: constant {a[:, 0].mean():.4f} rad, linear '
          f'{a[:, 1].mean():.4f} rad/FOV; first to last echo pair: {a[-1, 0] - a[0, 0]:+.4f} rad, '
          f'{a[-1, 1] - a[0, 1]:+.4f} rad/FOV')
    return kxo, kxe, a, sweep


def unflatten_gre_echoes(
    ksp_gre_raw: np.ndarray, Ny_degre: int, Nz_degre: int, n_echoes: int
) -> np.ndarray:
    """deGRE archive [Nx_degre, Ncoils, Nacq] -> [Nx_degre, Ny_degre, Nz_degre,
    n_echoes, Ncoils]. The archive starts with a blip-free iZ=0 receive-gain
    calibration block (Ny_degre * n_echoes readouts), dropped here; the image
    block follows in echo-fastest, then iY, then iZ order."""
    Nx_degre, Ncoils, _ = ksp_gre_raw.shape
    ksp_gre = ksp_gre_raw[:, :, Ny_degre * n_echoes:]
    ksp_gre = ksp_gre.reshape(Nx_degre, Ncoils, n_echoes, Ny_degre, Nz_degre, order='F')
    return ksp_gre.transpose(0, 3, 4, 2, 1)


def scatter_frame(
    ksp_frame_cart: np.ndarray, schedule_frame: np.ndarray, Ny: int, Nz: int
) -> np.ndarray:
    """One frame's gridded k-space [Nx, ETL, Nshots, Nc] -> zero-filled
    [Nx, Ny, Nz, Nc], placed by the frame's 0-based (ky, kz) schedule
    [Nshots, ETL, 2]. Data and schedule are both flattened shot-fastest
    (column-major), as in preprocess.m."""
    Nx, ETL, Nshots, Nc = ksp_frame_cart.shape
    ksp_flat = ksp_frame_cart.transpose(0, 2, 1, 3).reshape(Nx, Nshots * ETL, Nc, order='F')
    iy = schedule_frame[:, :, 0].reshape(-1, order='F')
    iz = schedule_frame[:, :, 1].reshape(-1, order='F')
    lin_idx = np.ravel_multi_index((iy, iz), (Ny, Nz), order='F')
    if len(np.unique(lin_idx)) < len(lin_idx):
        warnings.warn(
            'scatter_frame: frame has duplicate (ky, kz) entries. Check schedule.', stacklevel=2
        )
    ksp_frame_zf = np.zeros((Nx, Ny * Nz, Nc), dtype=ksp_frame_cart.dtype)
    ksp_frame_zf[:, lin_idx, :] = ksp_flat
    return ksp_frame_zf.reshape(Nx, Ny, Nz, Nc, order='F')


def process_epi_frame(
    ksp_frame_raw: np.ndarray,
    W: np.ndarray,
    kxo: np.ndarray,
    kxe: np.ndarray,
    a: np.ndarray,
    schedule_frame: np.ndarray,
    Nx: int,
    Ny: int,
    Nz: int,
    ETL: int,
    Nshots: int,
    fov_x_cm: float,
) -> np.ndarray:
    """One EPI frame of raw shots [Nfid, Ncoils, ETL*Nshots] (acquisition
    order, one shot's echoes contiguous) -> zero-filled [Nx, Ny, Nz, Nc]:
    whiten, regrid, odd/even correct, scatter."""
    Nfid = ksp_frame_raw.shape[0]
    ksp = apply_whitening(ksp_frame_raw.transpose(0, 2, 1), W)  # [Nfid, shots, Ncoils]
    Nc = ksp.shape[-1]
    # MATLAB: reshape(ksp, Nfid, Nc, ETL, Nshots) then permute([1 3 4 2]).
    ksp = ksp.transpose(0, 2, 1).reshape(Nfid, Nc, ETL, Nshots, order='F').transpose(0, 2, 3, 1)
    ksp_cart = epiphasecorrect(rampsampepi2cart(ksp, kxo, kxe, Nx, fov_x_cm), a)
    return scatter_frame(ksp_cart, schedule_frame, Ny, Nz)


def grid_noise(
    ksp_noise: np.ndarray,
    W: np.ndarray,
    kxo: np.ndarray,
    kxe: np.ndarray,
    a: np.ndarray,
    Nx: int,
    ETL: int,
    fov_x_cm: float,
) -> np.ndarray:
    """Noise-scan readouts [Nfid, Ncoils, Nacq] through the same whitening,
    regridding and odd/even correction as the EPI data,
    grouped ETL at a time as if they were echo trains -> [Nx, ETL, n_trains, Nc]
    Cartesian k-space. Its mean |.|^2 is the thermal-noise variance of the
    final k-space; the noise scan has no signal, so this can't be biased by
    image content the way an estimate from the EPI data would be."""
    Nfid, _, Nacq = ksp_noise.shape
    n_trains = Nacq // ETL
    if n_trains == 0:
        raise ValueError(f'grid_noise: noise scan has {Nacq} readouts, fewer than ETL={ETL}')
    x = apply_whitening(ksp_noise[:, :, : n_trains * ETL].transpose(0, 2, 1), W)
    Nc = x.shape[-1]
    x = x.transpose(0, 2, 1).reshape(Nfid, Nc, ETL, n_trains, order='F').transpose(0, 2, 3, 1)
    return epiphasecorrect(rampsampepi2cart(x, kxo, kxe, Nx, fov_x_cm), a)


def resume_start_frame(mf: h5py.File, epi_reader, shots_per_frame: int) -> int:
    """Frame to resume Stage A from, given the cache file open for append.
    The archive reader has no seek, so the shots of every completed frame are
    read and discarded; otherwise the next frame would silently get frame 0's
    data."""
    start_frame = int(mf.attrs.get('last_completed_frame', -1)) + 1
    if start_frame > 0:
        n_skip = start_frame * shots_per_frame
        print(f'  resuming at frame {start_frame}, skipping {n_skip} already-gridded shots')
        for _ in range(n_skip):
            epi_reader.next_frame()
    return start_frame


def _iy_iz(schedules: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Flattened (ky, kz) indices of one frame -- shared by _build_omegas and
    _build_echo_times so the two can't drift apart (review item 58)."""
    return schedules[frame, :, :, 0].ravel(), schedules[frame, :, :, 1].ravel()


def _build_omegas(schedules: np.ndarray, Ny: int, Nz: int) -> np.ndarray:
    """[Ny, Nz, Nframes] bool sampling mask from the 0-based schedule."""
    Nframes = schedules.shape[0]
    omegas = np.zeros((Ny, Nz, Nframes), dtype=bool)
    for frame in range(Nframes):
        iy, iz = _iy_iz(schedules, frame)
        omegas[iy, iz, frame] = True
    return omegas


def _build_echo_times(
    schedules: np.ndarray, echo_times: np.ndarray, Ny: int, Nz: int
) -> np.ndarray:
    """[Ny, Nz, Nframes] acquisition time (s since excitation) of each sampled
    (ky, kz); 0 where unsampled (check 'omegas')."""
    Nframes = schedules.shape[0]
    t = np.zeros((Ny, Nz, Nframes), dtype=np.float64)
    for frame in range(Nframes):
        iy, iz = _iy_iz(schedules, frame)
        t[iy, iz, frame] = echo_times[frame].ravel()
    return t


def find_calib_region(omegas: np.ndarray, min_size: int = 4) -> tuple[slice, slice] | None:
    """The fully sampled central calibration region: the rectangle of (ky, kz)
    locations sampled in *every* frame, grown outward from the k-space center
    (Ny//2, Nz//2) one edge line at a time while the new line is fully sampled.
    For Poisson-disc masks this is the pd calibration rectangle. None if the
    center isn't sampled in every frame or the rectangle is smaller than
    min_size along either axis (e.g. CAIPI or random masks)."""
    common = np.all(omegas, axis=-1)
    Ny, Nz = common.shape
    cy, cz = Ny // 2, Nz // 2
    if not common[cy, cz]:
        return None
    y0, y1, z0, z1 = cy, cy + 1, cz, cz + 1
    grown = True
    while grown:
        grown = False
        if y0 > 0 and common[y0 - 1, z0:z1].all():
            y0 -= 1
            grown = True
        if y1 < Ny and common[y1, z0:z1].all():
            y1 += 1
            grown = True
        if z0 > 0 and common[y0:y1, z0 - 1].all():
            z0 -= 1
            grown = True
        if z1 < Nz and common[y0:y1, z1].all():
            z1 += 1
            grown = True
    if y1 - y0 < min_size or z1 - z0 < min_size:
        return None
    return slice(y0, y1), slice(z0, z1)


# ---------------------------------------------------------------------------
# Stage A: grid the EPI data into the cache
# ---------------------------------------------------------------------------


def grid_epi(cfg: PreprocessConfig, paths: SeqPaths, a: np.ndarray | None = None) -> None:
    """Stage A. Writes <outdir>/<seq>_gridded.h5: whitened, all-coil,
    zero-filled k-space 'ksp_epi_zf' [Nx, Ny, Nz, Ncoils, Nframes] (one frame
    per chunk), plus 'omegas', 'echo_times', 'W', 'kxo'/'kxe', 'noise_gridded'.
    Skipped if the cache is already complete; resumed if it is partial (with
    the W, kxo/kxe and `a` it was started with). `a` overrides the odd/even
    phase estimate (tests, validation)."""
    sp = load_seq_params(paths.scan_info)
    Nx, Ny, Nz, ETL = sp.Nx, sp.Ny, sp.Nz, sp.ETL
    fov_x_cm = sp.fov[0] * 100

    resuming = os.path.exists(paths.cache)
    if resuming:
        with h5py.File(paths.cache, 'r') as f:
            # 'oephase_a' is written once calibration and every setup dataset are in
            # place; without it the file is left over from a run that failed first.
            initialized = 'oephase_a' in f.attrs
            complete = bool(f.attrs.get('complete', False))
        if not initialized:
            print(f'Stage A: {paths.cache} is from a failed run; starting over')
            os.remove(paths.cache)
            resuming = False
        elif complete:
            print(f'Stage A: using complete cache {paths.cache}')
            return

    schedules, echo_times = load_schedules(paths.scan_info)
    Nframes, Nshots, ETL_sched, _ = schedules.shape
    if ETL_sched != ETL:
        raise ValueError(f'grid_epi: schedule ETL ({ETL_sched}) != scan_info ETL ({ETL})')

    os.makedirs(cfg.outdir, exist_ok=True)
    with h5py.File(paths.cache, 'a' if resuming else 'w') as f:
        if resuming:
            W, kxo, kxe = f['W'][()], f['kxo'][()], f['kxe'][()]
            if a is not None and not np.allclose(a, f.attrs['oephase_a']):
                warnings.warn('grid_epi: resuming a cache made with a different `a`; keeping its a')
            a = f.attrs['oephase_a']
            Ncoils = f['ksp_epi_zf'].shape[3]
            Nfid = len(kxo)
        else:
            print('Stage A: calibration')
            ksp_cal_raw = utils.read_archive(paths.cal)  # [Nfid, Ncoils, N_cal]
            Nfid, Ncoils, _ = ksp_cal_raw.shape
            if os.path.exists(paths.noise):
                ksp_noise = utils.read_archive(paths.noise)
                if ksp_noise.shape[0] != Nfid:
                    raise ValueError(
                        f'grid_epi: noise Nfid ({ksp_noise.shape[0]}) != cal Nfid ({Nfid})'
                    )
                W = compute_whitening_matrix(ksp_noise.transpose(0, 2, 1))
                whitened = True
            else:
                warnings.warn(f'grid_epi: no noise scan at {paths.noise}; data are NOT whitened')
                ksp_noise, W, whitened = None, np.eye(Ncoils, dtype=np.complex128), False

            kxo, kxe, a_est, sweep = calibrate_odd_even(
                ksp_cal_raw, W, paths.scan_info, Nx, ETL, fov_x_cm
            )
            if a is None:
                a = a_est
            else:
                a = np.asarray(a, dtype=np.float64)
                print(f'  using the given odd/even phase model {a} instead')

            f.create_dataset(
                'ksp_epi_zf', shape=(Nx, Ny, Nz, Ncoils, Nframes), dtype=np.complex64,
                chunks=(Nx, Ny, Nz, Ncoils, 1), compression='gzip', compression_opts=4,
            )
            f.create_dataset('omegas', data=_build_omegas(schedules, Ny, Nz))
            f.create_dataset('echo_times', data=_build_echo_times(schedules, echo_times, Ny, Nz))
            f.create_dataset('W', data=W)
            f.create_dataset('kxo', data=kxo)
            f.create_dataset('kxe', data=kxe)
            if ksp_noise is not None:
                noise_gridded = grid_noise(ksp_noise, W, kxo, kxe, a, Nx, ETL, fov_x_cm)
                f.create_dataset('noise_gridded', data=noise_gridded.astype(np.complex64))
            g = f.create_group('delay_sweep')
            for k in ('delay', 'a1', 'a2', 'wrap_count'):
                g.create_dataset(k, data=sweep[k])
            f.attrs['delay'] = sweep['best']
            f.attrs['whitened'] = whitened
            f.attrs['n_frames_discard'] = round(sp.discard_duration / sp.volume_tr)
            f.attrs['complete'] = False
            f.attrs['oephase_a'] = a  # last: marks the cache as initialized (see above)

        epi_reader = utils.ArchiveReader(paths.epi)
        shots_per_frame = ETL * Nshots
        start = resume_start_frame(f, epi_reader, shots_per_frame) if resuming else 0
        print(f'Stage A: gridding frames {start + 1}-{Nframes} ({shots_per_frame} shots each)')
        try:
            for frame in range(start, Nframes):
                shots = [epi_reader.next_frame() for _ in range(shots_per_frame)]
                if shots[0].shape[0] != Nfid:
                    raise ValueError(
                        f'grid_epi: EPI readout length ({shots[0].shape[0]}) != Nfid ({Nfid})'
                    )
                ksp_frame_zf = process_epi_frame(
                    np.stack(shots, axis=-1), W, kxo, kxe, a, schedules[frame],
                    Nx, Ny, Nz, ETL, Nshots, fov_x_cm,
                )
                f['ksp_epi_zf'][..., frame] = ksp_frame_zf.astype(np.complex64)
                f.attrs['last_completed_frame'] = frame
                print(f'  frame {frame + 1}/{Nframes}')
        except StopIteration as e:
            raise RuntimeError(
                f'grid_epi: EPI archive ran out before {Nframes} frames x '
                f'{shots_per_frame} shots were read'
            ) from e
        f.attrs['complete'] = True


# ---------------------------------------------------------------------------
# Coil compression and deGRE maps
# ---------------------------------------------------------------------------


def load_gre(cfg: PreprocessConfig, sp: utils.SeqParams, W: np.ndarray) -> np.ndarray:
    """Whitened deGRE k-space [Nx_degre, Ny_degre, Nz_degre, n_echoes, Ncoils]."""
    ksp = unflatten_gre_echoes(
        utils.read_archive(cfg.fn_gre), sp.Ny_degre, sp.Nz_degre, sp.n_echoes_degre
    )
    return apply_whitening(ksp, W).astype(np.complex64)


def coil_compression(
    cfg: PreprocessConfig, sp: utils.SeqParams, ksp_gre: np.ndarray
) -> tuple[np.ndarray | None, dict]:
    """(GCC, info) from one whitened deGRE echo [Nx_degre, Ny, Nz, Ncoils].

    GCC is None (no compression) or the [Nx, Nv, Nc] geometric-decomposition
    coil compression matrices on the EPI readout grid (sp.Nx over the EPI
    x-FOV, which the deGRE's may exceed). Nv is cfg.Nvcoils if
    set, else the smallest count keeping cfg.cc_energy_thresh of the
    eigenvalue energy summed over x."""
    Ncoils = ksp_gre.shape[-1]
    if not cfg.compress:
        return None, {'coil_compressed': False, 'Nvcoils': Ncoils}

    A0, evals = gcc_compression(gcc_calibration(
        ksp_gre, sp.Nx, cfg.cc_calib_size, fov_src=sp.fov_degre[0], fov=sp.fov[0]
    ))

    if cfg.Nvcoils is not None:
        if not 1 <= cfg.Nvcoils <= Ncoils:
            raise ValueError(f'Nvcoils must be between 1 and {Ncoils}, got {cfg.Nvcoils}')
        nv, source = cfg.Nvcoils, 'user'
    else:
        nv, source = select_nvcoils(evals, cfg.cc_energy_thresh), 'energy'

    e = np.clip(evals, 0, None)
    kept = e[:, :nv].sum() / e.sum()
    GCC = align_gcc(A0[:, :nv], energy=e.sum(axis=1))
    kept_x = e[:, :nv].sum(axis=1) / np.maximum(e.sum(axis=1), 1e-30)
    worst = np.argsort(kept_x)[:3]
    print(f'  GCC: lowest per-x energy kept {kept_x[worst].round(4)} at x = {worst}')
    print(f'Coil compression (GCC): {Ncoils} -> {nv} virtual coils '
          f'({source}; {100 * kept:.2f}% of energy kept)')
    info = {
        'coil_compressed': True, 'Nvcoils': nv,
        'Nvcoils_source': source, 'cc_energy_kept': kept, 'cc_evals': e,
    }
    return GCC, info


def compress_kspace(ksp: np.ndarray, GCC: np.ndarray | None) -> np.ndarray:
    """Apply coil_compression's GCC to k-space [Nx(kx), ..., Nc] (coils last)."""
    return ksp if GCC is None else apply_gcc_kspace(ksp, GCC)


def compress_image(img: np.ndarray, GCC: np.ndarray | None) -> np.ndarray:
    """Apply coil_compression's GCC to image-domain data [Nx, ..., Nc] (e.g. maps)."""
    return img if GCC is None else apply_gcc_image(img, GCC)


def estimate_maps(
    cfg: PreprocessConfig, sp: utils.SeqParams, ksp_gre: np.ndarray, GCC: np.ndarray | None
) -> dict[str, np.ndarray]:
    """Sensitivity, B0 and R2* maps from the whitened deGRE [..., n_echoes, Nc].
    Keys are output-file dataset paths: EPI-grid maps at the top level, deGRE-
    grid QA volumes under 'degre/'. smaps are in the output coil space (GCC)."""
    from preprocess.smaps import estimate_smaps, process_smaps

    fov, fov_degre = tuple(sp.fov), tuple(sp.fov_degre)
    n_epi = (sp.Nx, sp.Ny, sp.Nz)
    n_degre = ksp_gre.shape[:3]
    img_echoes = utils.rss_echo_images(ksp_gre)
    maps = {'degre/img_echoes': img_echoes.astype(np.float32)}
    te = None if sp.TE_degre is None else np.asarray(sp.TE_degre, dtype=np.float64)

    smaps_degre = emap_degre = None
    if cfg.estimate_smaps:
        print('Sensitivity maps (ESPIRiT)...')
        s_cal, emap = estimate_smaps(ksp_gre[..., cfg.gre_echo_idx, :], crop=cfg.crop)
        smaps_epi = process_smaps(
            s_cal, emap, fov_degre, fov, n_epi, cfg.crop,
            smooth_sigma_mm=cfg.smaps_smooth_sigma_mm, zero_pad_z=cfg.zero_pad_z,
        )
        smaps_degre = process_smaps(
            s_cal, emap, fov_degre, fov_degre, n_degre, cfg.crop,
            smooth_sigma_mm=cfg.smaps_smooth_sigma_mm,
        )
        emap_degre = resize_to_epi_grid(emap, fov_degre, fov_degre, n_degre, order=3)
        # Same linear combination as the k-space, so maps and data share a coil
        # space (exact under the SENSE model); then unit RSS per voxel again.
        smaps = compress_image(smaps_epi, GCC)
        rss = np.sqrt(np.sum(np.abs(smaps) ** 2, axis=-1, keepdims=True))
        smaps = smaps / np.where(rss > np.finfo(np.float32).eps, rss, 1)
        maps['smaps'] = smaps.astype(np.complex64)
        maps['degre/smaps'] = smaps_degre.astype(np.complex64)
        maps['degre/emap'] = emap_degre.astype(np.float32)

    mask_degre = None
    if cfg.estimate_b0:
        if te is None or te.size < 2:
            raise ValueError('estimate_b0 needs a multi-echo deGRE (TE_degre in scan_info.mat)')
        print('B0 field map (MRIFieldmaps.jl)...')
        r = estimate_b0map(
            ksp_gre, te, smaps_degre, emap_degre, crop=cfg.crop,
            mask_thresh=cfg.b0map_mask_thresh, precon=cfg.b0map_precon,
            min_component=cfg.b0map_min_component,
        )
        mask_degre = r['mask']
        b0, b0_mask = resize_to_epi(r['b0_map'], mask_degre, fov_degre, fov, n_epi,
                                    zero_pad_z=cfg.zero_pad_z)
        maps.update({
            'b0_map': b0, 'b0_mask': b0_mask,
            'degre/b0_map': r['b0_map'].astype(np.float32),
            'degre/finit_hz': r['finit_hz'].astype(np.float32),
            'degre/mask': mask_degre,
        })

    if cfg.estimate_r2star:
        if te is None or te.size < 2:
            raise ValueError('estimate_r2star needs a multi-echo deGRE (TE_degre in scan_info.mat)')
        if mask_degre is None:
            mask_degre = fit_mask(img_echoes[..., 0], cfg.b0map_mask_thresh, emap_degre, cfg.crop,
                                  cfg.b0map_min_component)
        r2_degre = fit_r2star(img_echoes, te, mask_degre, cfg.r2star_max)
        r2 = resize_to_epi_grid(
            r2_degre * mask_degre, fov_degre, fov, n_epi, order=3, zero_pad_z=cfg.zero_pad_z
        )
        maps['r2star_map'] = np.clip(r2, 0, None).astype(np.float32)  # spline overshoot
        maps['degre/r2star_map'] = r2_degre
    return maps


# ---------------------------------------------------------------------------
# Stage B: write the output file
# ---------------------------------------------------------------------------


def write_output(
    cfg: PreprocessConfig,
    paths: SeqPaths,
    sp: utils.SeqParams,
    GCC: np.ndarray | None,
    cc_info: dict,
    maps: dict[str, np.ndarray],
) -> None:
    """Stage B: <outdir>/<seq>_preprocessed.h5 from the Stage A cache.

    Datasets: 'ksp_epi_zf' [Nx, Ny, Nz, Nc_out, Nframes] (compressed with GCC),
    'omegas', 'echo_times', 'ksp_calib' (if a calibration region exists),
    the maps, 'W', 'GCC'. Attrs include 'noise_var' (thermal-noise
    variance per complex sample, ~1 after whitening; recon divides by its
    square root), 'whitened', 'coil_compressed' and 't_ref_s'."""
    with h5py.File(paths.cache, 'r') as c:
        omegas = c['omegas'][()]
        Nx, Ny, Nz, Ncoils, Nframes = c['ksp_epi_zf'].shape
        noise_gridded = c['noise_gridded'][()] if 'noise_gridded' in c else None
        cache_attrs = dict(c.attrs)
    region = find_calib_region(omegas) if cfg.extract_calib else None
    if cfg.extract_calib and region is None:
        warnings.warn('write_output: no fully sampled calibration region found; no ksp_calib')

    if GCC is None:
        # Nothing to compress: the cache already is the output k-space.
        if cfg.keep_cache:
            shutil.copyfile(paths.cache, paths.output)
        else:
            os.replace(paths.cache, paths.output)
        f = h5py.File(paths.output, 'a')
        for k in ('noise_gridded', 'kxo', 'kxe'):
            if k in f:
                del f[k]
        for k in ('complete', 'last_completed_frame'):
            if k in f.attrs:
                del f.attrs[k]
        Nc_out = Ncoils
    else:
        f = h5py.File(paths.output, 'w')
        Nc_out = GCC.shape[-2]
        with h5py.File(paths.cache, 'r') as c:
            for k in ('omegas', 'echo_times', 'W'):
                f.create_dataset(k, data=c[k][()])
            if 'delay_sweep' in c:  # absent in caches from before delay calibration
                c.copy('delay_sweep', f)
        f.create_dataset(
            'ksp_epi_zf', shape=(Nx, Ny, Nz, Nc_out, Nframes), dtype=np.complex64,
            chunks=(Nx, Ny, Nz, Nc_out, 1), compression='gzip', compression_opts=4,
        )
    try:
        if region is not None:
            ys, zs = region
            f.create_dataset(
                'ksp_calib', shape=(Nx, ys.stop - ys.start, zs.stop - zs.start, Nc_out, Nframes),
                dtype=np.complex64,
            )
            f['ksp_calib'].attrs['calib_y_range'] = (ys.start, ys.stop)
            f['ksp_calib'].attrs['calib_z_range'] = (zs.start, zs.stop)
            print(f'Calibration region: ky {ys.start}:{ys.stop}, kz {zs.start}:{zs.stop}')
        print(f'Stage B: writing {paths.output}')
        if GCC is not None or region is not None:
            src = h5py.File(paths.cache, 'r') if GCC is not None else None
            try:
                d = src['ksp_epi_zf'] if src is not None else f['ksp_epi_zf']
                for frame in range(Nframes):
                    ksp = d[..., frame]
                    if GCC is not None:
                        ksp = compress_kspace(ksp, GCC).astype(np.complex64)
                        f['ksp_epi_zf'][..., frame] = ksp
                    if region is not None:
                        f['ksp_calib'][..., frame] = ksp[:, region[0], region[1], :]
            finally:
                if src is not None:
                    src.close()

        if noise_gridded is not None:
            f.attrs['noise_var'] = float(np.mean(np.abs(compress_kspace(noise_gridded, GCC)) ** 2))
            print(f'  thermal-noise variance of the output k-space: {f.attrs["noise_var"]:.4f}')
        f.attrs['whitened'] = bool(cache_attrs['whitened'])
        f.attrs['oephase_a'] = cache_attrs['oephase_a']
        f.attrs['delay'] = cache_attrs['delay']
        f.attrs['n_frames_discard'] = cache_attrs['n_frames_discard']
        f.attrs['Ncoils'] = Ncoils
        f.attrs['Nc_out'] = Nc_out
        f.attrs['coil_compressed'] = cc_info['coil_compressed']
        for k in ('Nvcoils', 'Nvcoils_source', 'cc_energy_kept'):
            if k in cc_info:
                f.attrs[k] = cc_info[k]
        if GCC is not None:
            f.create_dataset('GCC', data=GCC.astype(np.complex64))
            f.create_dataset('cc_evals', data=cc_info['cc_evals'].astype(np.float32))
        f.attrs['seqname'] = paths.seqname
        f.attrs['fov'] = np.asarray(sp.fov)
        f.attrs['fov_degre'] = np.asarray(sp.fov_degre)
        f.attrs['t_ref_s'] = utils.nominal_te_s(paths.scan_info, sp.ETL)
        if sp.TE_degre is not None:
            f.attrs['TE_degre'] = np.asarray(sp.TE_degre)
        if 'r2star_map' in maps:
            f.attrs['r2star_method'] = R2STAR_METHOD
            f.attrs['r2star_n_echoes'] = len(sp.TE_degre)
        for key, val in maps.items():
            f.create_dataset(key, data=val)
    finally:
        f.close()

    base = os.path.join(cfg.outdir, paths.seqname)
    for key in ('smaps', 'b0_map', 'r2star_map'):
        if key in maps:
            utils.save_recon_nifti(f'{base}_{key}', maps[key], fov=sp.fov, seqname=paths.seqname)
    if GCC is not None and not cfg.keep_cache:
        os.remove(paths.cache)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def preprocess(cfg: PreprocessConfig, seqname: str, a: np.ndarray | None = None) -> str:
    """Run the whole pipeline for one sequence; returns the output path.
    `a` overrides the odd/even phase estimate (tests, validation)."""
    paths = set_seq_paths(cfg, seqname)
    sp = load_seq_params(paths.scan_info)
    print(f'=== {seqname} ===')
    grid_epi(cfg, paths, a)

    with h5py.File(paths.cache, 'r') as f:
        W = f['W'][()]
    ksp_gre = load_gre(cfg, sp, W)
    GCC, cc_info = coil_compression(cfg, sp, ksp_gre[..., cfg.gre_echo_idx, :])
    maps = estimate_maps(cfg, sp, ksp_gre, GCC)
    del ksp_gre
    write_output(cfg, paths, sp, GCC, cc_info, maps)
    print(f'Done: {paths.output}')
    return paths.output
