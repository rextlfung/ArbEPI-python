"""Stage 1 -- raw ScanArchive data -> zero-filled k-space volume, ready for
Stage 2 iterative reconstruction. Ports preprocess.m.

No local dataset has every file this needs for one acquisition (noise, deGRE,
cal, EPI, samp_locs.mat, params.mat all for the same sequence -- in
particular no noise.h5 exists anywhere under ~/github/data at the time of
this port), so this module has NOT been run end-to-end against real data.
Every building block it calls (raw_io, coils, epi_gridding, oephase, smaps)
has been independently verified (real-data smoke tests or synthetic
round-trip tests -- see their own test files), and the scatter step that
places gridded k-space into the zero-filled volume has a dedicated
location-encoding test (test_preprocess_scatter_frame_places_data_at_
correct_indices in tests/test_preprocessing_preprocess.py) that exercises
the exact permute/reshape chain this file uses. Running this module against
a real acquisition is the user's own end-to-end verification step.

Every MATLAB permute/reshape in preprocess.m is translated mechanically:
`permute(A,[p...])` -> `A.transpose(p_0based...)`, `reshape(A,dims)` ->
`A.reshape(dims, order='F')` -- MATLAB reshape is always column-major over
the whole array, exactly numpy's order='F'. Two permute/reshape round trips
in the original that cancel out (a permute immediately undone by the next
line's reshape-of-a-permute) are collapsed rather than replicated step for
step; this is only skipping a literal no-op, not re-deriving the algorithm,
and is called out inline where it happens.
"""

import os
import warnings

import h5py
import numpy as np
from scipy.interpolate import interp1d

from preprocessing.coils import (
    apply_coil_compression,
    apply_whitening,
    coil_compression_matrix,
    compute_coil_covariance,
    compute_whitening_matrix,
    select_nvcoils,
)
from preprocessing.config import (
    PreprocessingConfig,
    SeqParams,
    SeqPaths,
    load_seq_params,
    seq_delay,
)
from preprocessing.epi_gridding import rampsampepi2cart
from preprocessing.matio import read_mat
from preprocessing.oephase import epiphasecorrect, getoephase
from preprocessing.smaps import estimate_smaps, process_smaps

# raw_io imports GERecon, GE's proprietary (non-pip) SDK -- deliberately not
# imported at module level, so every other function here (and the tests for
# them) stay importable without it installed. Only preprocess() itself,
# which actually needs to read ScanArchives, imports it, lazily, below.


def load_kxoe(seqdir: str, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """kxo, kxe (cycles/cm) from kxoe<Nx>.mat -- resolved here (once Nx is
    known from params) rather than at set_seq_paths time, mirroring
    preprocess.m's own two-stage resolution (its cfg.fn.kxoe placeholder is
    overwritten the same way once Nx is available)."""
    path = os.path.join(seqdir, f'kxoe{nx}.mat')
    d = read_mat(path, ['kxo', 'kxe'])
    return d['kxo'].ravel() / 100, d['kxe'].ravel() / 100


def load_schedules(samp_log_path: str) -> np.ndarray:
    """schedules [Nframes, Nshots, ETL, 2], converted from the 1-based
    convention samp_locs.mat is written in (see CLAUDE.md's index
    convention section) to 0-based for Python indexing."""
    schedules = read_mat(samp_log_path, ['schedules'])['schedules']
    return schedules.astype(np.int64) - 1


def apply_delay(
    kxo0: np.ndarray, kxe0: np.ndarray, nfid: int, delay: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shift the base kx trajectory by the k-space center offset (samples),
    via linear interpolation with true extrapolation at the boundary
    (np.interp clamps outside its domain instead of extrapolating, unlike
    MATLAB's interp1(...,'linear','extrap') -- scipy's interp1d matches).
    Ports preprocess.m's interp1(1:Nfid, kxo, (1:Nfid)-0.5-delay,...) calls.
    """
    idx = np.arange(1, nfid + 1, dtype=float)
    shifted = idx - 0.5 - delay
    kxo = interp1d(idx, kxo0, kind='linear', fill_value='extrapolate')(shifted)
    kxe = interp1d(idx, kxe0, kind='linear', fill_value='extrapolate')(shifted)
    return kxo, kxe


def compute_oephase(
    ksp_cal: np.ndarray, kxo: np.ndarray, kxe: np.ndarray, nx: int, fov_x_cm: float
) -> np.ndarray:
    """Estimate the odd/even ghost-correction linear-phase model `a` from
    an unencoded (no phase-encoding blips) calibration echo train.

    ksp_cal: [Nfid, ETL, N_cal_shots, Nvcoils] whitened, coil-compressed
        calibration data (ETL already truncated to even, see preprocess()).
    Ports preprocess.m's STEP 4 oephase computation.
    """
    oephase_data = rampsampepi2cart(ksp_cal, kxo, kxe, nx, fov_x_cm)
    # fftshift/ifftshift with no axes argument shift *every* axis, matching
    # MATLAB's no-dim fftshift(oephase_data)/ifftshift(...) here exactly.
    oephase_data = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(oephase_data), n=nx, axis=0))
    oephase_mean = np.mean(oephase_data, axis=2)  # average over cal shots (MATLAB dim 3)
    a, _ = getoephase(oephase_mean)
    return a


def scatter_frame(
    ksp_frame_cart: np.ndarray, schedule_frame: np.ndarray, Ny: int, Nz: int
) -> np.ndarray:
    """Scatter one frame's gridded + phase-corrected k-space into a
    zero-filled [Nx, Ny, Nz, Nvcoils] volume, per that frame's (ky, kz)
    sampling schedule. Ports preprocess.m's STEP 6 scatter.

    ksp_frame_cart: [Nx, ETL, Nshots, Nvcoils].
    schedule_frame: [Nshots, ETL, 2], 0-based (ky, kz) -- schedules[frame]
        from load_schedules().

    Both the k-space data and the schedule are flattened shot-fastest,
    echo-slowest (Fortran/column-major order) to match each other -- this
    mirrors preprocess.m's own comment ("column-major reshape gives
    shot-outer order, matching permute([1 3 2 4])"); see
    test_preprocessing_preprocess.py for a location-encoding test that
    verifies this pairing is correct end to end.
    """
    Nx, ETL, Nshots, Nvcoils = ksp_frame_cart.shape

    ksp_flat = ksp_frame_cart.transpose(0, 2, 1, 3).reshape(Nx, Nshots * ETL, Nvcoils, order='F')

    iy = schedule_frame[:, :, 0].reshape(-1, order='F')
    iz = schedule_frame[:, :, 1].reshape(-1, order='F')
    lin_idx = np.ravel_multi_index((iy, iz), (Ny, Nz), order='F')
    if len(np.unique(lin_idx)) < len(lin_idx):
        warnings.warn(
            'scatter_frame: frame has duplicate (ky, kz) entries. Check schedule.', stacklevel=2
        )

    ksp_frame_zf = np.zeros((Nx, Ny * Nz, Nvcoils), dtype=ksp_frame_cart.dtype)
    ksp_frame_zf[:, lin_idx, :] = ksp_flat
    return ksp_frame_zf.reshape(Nx, Ny, Nz, Nvcoils, order='F')


def process_epi_frame(
    ksp_frame_raw: np.ndarray,
    W: np.ndarray,
    cc_matrix: np.ndarray,
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
    """One EPI frame, raw shots -> zero-filled k-space volume. Ports
    preprocess.m's per-frame body inside its STEP 6 loop.

    ksp_frame_raw: [Nfid, Ncoils, shots_per_frame] (shots_per_frame =
        ETL*Nshots), raw (unwhitened, uncompressed) shots for one frame, in
        acquisition order -- consecutive echoes of one shot are contiguous
        (see the reshape below).
    """
    Nfid, _, shots_per_frame = ksp_frame_raw.shape
    Nvcoils = cc_matrix.shape[0]

    ksp = ksp_frame_raw.transpose(0, 2, 1)  # [Nfid, shots_per_frame, Ncoils]
    ksp = apply_whitening(ksp, W)
    ksp = apply_coil_compression(ksp, cc_matrix)  # [Nfid, shots_per_frame, Nvcoils]
    ksp = ksp.transpose(0, 2, 1)  # [Nfid, Nvcoils, shots_per_frame]

    # MATLAB: reshape(ksp, Nfid,Nvcoils,ETL,Nshots) then permute([1 3 4 2]).
    ksp = ksp.reshape(Nfid, Nvcoils, ETL, Nshots, order='F')
    ksp = ksp.transpose(0, 2, 3, 1)  # [Nfid, ETL, Nshots, Nvcoils]

    ksp_cart = rampsampepi2cart(ksp, kxo, kxe, Nx, fov_x_cm)
    ksp_cart = epiphasecorrect(ksp_cart, a)

    return scatter_frame(ksp_cart, schedule_frame, Ny, Nz)


def preprocess(cfg: PreprocessingConfig, paths: SeqPaths) -> None:
    """Full Stage 1 pipeline for one sequence: noise -> deGRE -> smaps ->
    cal (odd/even phase) -> EPI (streamed frame-by-frame, checkpointed).
    Writes paths.recon.
    """
    from preprocessing.raw_io import ArchiveReader, read_archive

    seq_params: SeqParams = load_seq_params(paths)
    Nx, Ny, Nz, ETL, R = seq_params.Nx, seq_params.Ny, seq_params.Nz, seq_params.ETL, seq_params.R
    fov = seq_params.fov

    NframesDiscard = round(seq_params.discard_duration / seq_params.volume_tr)

    # STEP 1 -- noise scan (small; needed to whiten every subsequent dataset)
    print('Loading noise scan...')
    ksp_noise = read_archive(paths.noise)  # [Nfid, Ncoils, Nacq]
    Nfid, Ncoils, _ = ksp_noise.shape
    W = compute_whitening_matrix(ksp_noise.transpose(0, 2, 1))  # coils last for coils.py

    # STEP 2 -- deGRE data -> cc_matrix (coil compression) + ksp_gre (for smaps)
    print('Loading deGRE data...')
    ksp_gre_raw = read_archive(cfg.fn_gre)
    Nx_degre, Ny_degre, Nz_degre = seq_params.Nx_degre, seq_params.Ny_degre, seq_params.Nz_degre
    # Discard the blipless calibration block the scanner prepends (first
    # Ny_degre acquisitions), then unflatten the remaining Ny_degre*Nz_degre
    # phase-encode loop -- mechanical port of preprocess.m's
    # reshape(...,Nx_degre,Ncoils,Ny_degre,Nz_degre) + permute([1 3 4 2]).
    ksp_gre = ksp_gre_raw[:, :, Ny_degre:]
    ksp_gre = ksp_gre.reshape(Nx_degre, Ncoils, Ny_degre, Nz_degre, order='F')
    ksp_gre = ksp_gre.transpose(0, 2, 3, 1)  # [Nx_degre, Ny_degre, Nz_degre, Ncoils]

    ksp_gre = apply_whitening(ksp_gre, W)

    cov = compute_coil_covariance(ksp_gre)
    Nvcoils, Nvcoils_energy = select_nvcoils(cov, cfg.cc_energy_thresh, floor=int(2 * R))
    print(f'  Energy threshold {cfg.cc_energy_thresh:.4f}: {Nvcoils_energy} components needed')
    print(f'  Lower bound 2R = {int(2 * R)}  ->  selected Nvcoils = {Nvcoils}')

    cc_matrix = coil_compression_matrix(cov, Nvcoils)
    ksp_gre = apply_coil_compression(ksp_gre, cc_matrix)

    gre_cache_path = os.path.join(cfg.datdir, 'recon', f'{paths.seqname}_gre.h5')
    os.makedirs(os.path.dirname(gre_cache_path), exist_ok=True)
    with h5py.File(gre_cache_path, 'w') as f:
        f.create_dataset('ksp_gre', data=ksp_gre)

    # STEP 3 -- sensitivity maps (before EPI is loaded, so ksp_gre can be freed)
    fn_smaps = os.path.join(cfg.datdir, 'recon', f'smaps_{paths.seqname}_sigpy.h5')
    smaps = None
    if cfg.do_sense:
        smaps_valid = False
        if os.path.exists(fn_smaps):
            with h5py.File(fn_smaps, 'r') as f:
                smaps_valid = f.attrs.get('Nvcoils') == Nvcoils
        if smaps_valid:
            print(f'Loading precomputed sensitivity maps from {fn_smaps}')
            with h5py.File(fn_smaps, 'r') as f:
                smaps = f['smaps'][()]
        else:
            print('Estimating sensitivity maps via sigpy ESPIRiT...')
            smaps_raw, emap = estimate_smaps(ksp_gre)
            smaps = process_smaps(
                smaps_raw, emap, tuple(seq_params.fov_degre), tuple(fov),
                (Nx, Ny, Nz), cfg.threshold_mask,
            )
            with h5py.File(fn_smaps, 'w') as f:
                f.create_dataset('smaps_raw', data=smaps_raw)
                f.create_dataset('emap', data=emap)
                f.create_dataset('smaps', data=smaps)
                f.attrs['Nvcoils'] = Nvcoils
    del ksp_gre

    # STEP 4 -- calibration data -> odd/even phase offsets a, trajectory kxo/kxe
    print('Loading calibration data...')
    ksp_cal_raw = read_archive(paths.cal)
    if ksp_cal_raw.shape[0] != Nfid:
        raise ValueError(
            f'preprocess: Calibration Nfid ({ksp_cal_raw.shape[0]}) != '
            f'noise Nfid ({Nfid}) -- wrong noise file?'
        )
    ksp_cal = apply_whitening(ksp_cal_raw.transpose(0, 2, 1), W)  # [Nfid, N_cal, Ncoils]
    ksp_cal = apply_coil_compression(ksp_cal, cc_matrix)  # [Nfid, N_cal, Nvcoils]

    kxo0, kxe0 = load_kxoe(paths.seqdir, Nx)
    delay = seq_delay(cfg, paths.seqname)
    print(f'Applying k-space center offset: {delay:.2f} samples')
    kxo, kxe = apply_delay(kxo0, kxe0, Nfid, delay)

    # MATLAB permutes ksp_cal to [Nfid,Nvcoils,N_cal] then immediately
    # permutes back to [Nfid,N_cal,Nvcoils] for this reshape -- a no-op
    # round trip skipped here, not a change to the algorithm (see module
    # docstring).
    ksp_cal = ksp_cal.reshape(Nfid, ETL, -1, Nvcoils, order='F')
    ETL_even = ETL - (ETL % 2)
    a = compute_oephase(ksp_cal[:, :ETL_even, :, :], kxo, kxe, Nx, fov[0] * 100)
    print(f'  Constant phase offset: {a[0]:.4f} rad')
    print(f'  Linear phase offset:   {a[1]:.4f} rad/fov')
    del ksp_cal

    # STEP 5 -- sampling schedule (tiny; needed before allocating the output)
    schedules = load_schedules(paths.samp_log)
    Nframes, Nshots, ETL_sched, _ = schedules.shape
    if ETL_sched != ETL:
        raise ValueError(
            f'preprocess: Sampling schedule ETL ({ETL_sched}) != params ETL ({ETL}) '
            '-- wrong schedule file?'
        )

    # STEP 6 -- EPI data, frame-by-frame (bounds memory; checkpointed for resume)
    print(f'Opening EPI archive: {paths.epi}')
    epi_reader = ArchiveReader(paths.epi)
    shots_per_frame = ETL * Nshots
    total_shots_expected = shots_per_frame * Nframes

    os.makedirs(os.path.dirname(paths.recon), exist_ok=True)
    if os.path.exists(paths.recon):
        mf = h5py.File(paths.recon, 'a')
        start_frame = mf.attrs.get('last_completed_frame', -1) + 1
        if start_frame > 0:
            print(
                f'Resuming from frame {start_frame} '
                f'(checkpoint found at frame {start_frame - 1}).'
            )
    else:
        print(f'Pre-allocating output file: {paths.recon}')
        mf = h5py.File(paths.recon, 'w')
        mf.create_dataset(
            'ksp_epi_zf',
            shape=(Nx, Ny, Nz, Nvcoils, Nframes),
            dtype=np.complex64,
            chunks=(Nx, Ny, Nz, Nvcoils, 1),
        )
        mf.create_dataset('omegas', data=_build_omegas(schedules, Ny, Nz))
        mf.attrs['n_frames_discard'] = NframesDiscard
        start_frame = 0

    try:
        print(f'Processing frames {start_frame + 1}-{Nframes} ({shots_per_frame} shots/frame)...')
        for frame in range(start_frame, Nframes):
            print(f'  Frame {frame + 1} / {Nframes}')
            shots = [epi_reader.next_frame() for _ in range(shots_per_frame)]
            if frame == start_frame and shots[0].shape[0] != Nfid:
                raise ValueError(
                    f'preprocess: EPI readout length ({shots[0].shape[0]}) != Nfid ({Nfid}) '
                    '-- check EPI and cal files.'
                )
            ksp_frame_raw = np.stack(shots, axis=-1)  # [Nfid, Ncoils, shots_per_frame]

            ksp_frame_zf = process_epi_frame(
                ksp_frame_raw, W, cc_matrix, kxo, kxe, a, schedules[frame],
                Nx, Ny, Nz, ETL, Nshots, fov[0] * 100,
            )
            mf['ksp_epi_zf'][:, :, :, :, frame] = ksp_frame_zf.astype(np.complex64)
            mf.attrs['last_completed_frame'] = frame
    except StopIteration as e:
        raise RuntimeError(
            f'preprocess: EPI archive exhausted before {total_shots_expected} shots '
            f'({Nframes} frames x {shots_per_frame} shots/frame) were read.'
        ) from e
    finally:
        mf.close()

    print(f'EPI processing complete. Output: {paths.recon}')


def _build_omegas(schedules: np.ndarray, Ny: int, Nz: int) -> np.ndarray:
    """Rebuild the binary sampling mask [Ny, Nz, Nframes] from the 0-based
    schedule -- omegas[iy, iz, frame] is True iff that location is
    acquired. Ports preprocess.m's STEP 5 omegas rebuild."""
    Nframes = schedules.shape[0]
    omegas = np.zeros((Ny, Nz, Nframes), dtype=bool)
    for frame in range(Nframes):
        iy = schedules[frame, :, :, 0].ravel()
        iz = schedules[frame, :, :, 1].ravel()
        omegas[iy, iz, frame] = True
    return omegas
