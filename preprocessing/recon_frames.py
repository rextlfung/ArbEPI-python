"""Stage 2 shared logic: load/compute smaps, stream k-space, reconstruct
every frame. Ports recon_frames.m.

Deliberate deviation from recon_frames.m's GRE-cache fallback path: MATLAB's
recon_frames.m looks for a *shared* <datdir>/recon/gre.mat, but preprocess.m
actually saves its whitened+compressed ksp_gre to <datdir>/scanarchives/
gre.mat -- those paths don't match in the original MATLAB, and since the
whitening matrix comes from a per-sequence noise scan (see set_seq_paths.m:
cfg.fn.noise is per-seqname), the compressed ksp_gre genuinely differs by
sequence, so a single shared cache is a latent correctness bug (whichever
sequence's preprocess.m ran last silently wins for every other sequence's
fallback smaps estimation). This port avoids it by construction: both
preprocess.py and this file use the same per-sequence path,
<datdir>/recon/<seqname>_gre.h5. The smaps-cache legacy-format fallback
(recon_frames.m's "cache file has smaps_raw/emaps but no smaps yet" branch)
isn't ported either -- preprocess.py, this port's only writer, always
writes the full format, so that branch can never be reached here.
"""

import concurrent.futures
import functools
import os
import time
from typing import Callable

import h5py
import numpy as np

from preprocessing.config import PreprocessingConfig, SeqParams, SeqPaths
from preprocessing.smaps import estimate_smaps, process_smaps

ReconFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _load_smaps(
    cfg: PreprocessingConfig, paths: SeqPaths, seq_params: SeqParams
) -> tuple[np.ndarray, int]:
    fn_smaps = os.path.join(cfg.datdir, 'recon', f'smaps_{paths.seqname}_sigpy.h5')
    if os.path.exists(fn_smaps):
        print(f'Loading precomputed sensitivity maps from {fn_smaps}')
        with h5py.File(fn_smaps, 'r') as f:
            return f['smaps'][()], int(f.attrs['Nvcoils'])

    fn_gre = os.path.join(cfg.datdir, 'recon', f'{paths.seqname}_gre.h5')
    print('Sensitivity maps not found. Estimating via sigpy ESPIRiT...')
    with h5py.File(fn_gre, 'r') as f:
        ksp_gre = f['ksp_gre'][()]
    nvcoils = ksp_gre.shape[-1]
    smaps_raw, emap = estimate_smaps(ksp_gre)
    smaps = process_smaps(
        smaps_raw, emap, tuple(seq_params.fov_degre), tuple(seq_params.fov),
        (seq_params.Nx, seq_params.Ny, seq_params.Nz), cfg.threshold_mask,
    )
    with h5py.File(fn_smaps, 'w') as f:
        f.create_dataset('smaps_raw', data=smaps_raw)
        f.create_dataset('emap', data=emap)
        f.create_dataset('smaps', data=smaps)
        f.attrs['Nvcoils'] = nvcoils
    return smaps, nvcoils


def _recon_one_frame(data: np.ndarray, recon_fn: ReconFn, smaps: np.ndarray) -> np.ndarray:
    try:
        return recon_fn(data, smaps)
    except Exception as e:  # noqa: BLE001 -- mirrors recon_frames.m's per-frame try/catch
        print(f'recon_frames: reconstruction failed on a frame -- skipping. {e}')
        return None


def recon_frames(
    cfg: PreprocessingConfig, paths: SeqPaths, seq_params: SeqParams, recon_fn: ReconFn
) -> tuple[np.ndarray, dict, float]:
    """[Nx,Ny,Nz,Nframes] image, seq_params dict, wall-clock seconds.

    recon_fn(data, smaps) -> [Nx,Ny,Nz] image for one frame. Must be
    picklable (e.g. a functools.partial of a module-level function, not a
    lambda/closure) if cfg.use_parfor is True, since it's sent to worker
    processes via concurrent.futures.ProcessPoolExecutor.
    """
    Nx, Ny, Nz = seq_params.Nx, seq_params.Ny, seq_params.Nz

    smaps, _nvcoils = _load_smaps(cfg, paths, seq_params)

    with h5py.File(paths.recon, 'r') as f:
        nframes_avail = f['ksp_epi_zf'].shape[4]
        nframes = min(cfg.Nframes, nframes_avail)
        if nframes < nframes_avail:
            print(
                f'Reconstructing {nframes} of {nframes_avail} available frames '
                '(cfg.Nframes cap).'
            )

        print(f'Reconstructing {nframes} frames...')
        t_start = time.time()
        # Read one frame at a time rather than slicing the whole [Nx,Ny,Nz,
        # Nvcoils,Nframes] dataset into memory up front -- for a full-res
        # acquisition that array can exceed physical RAM (e.g. ~10GiB for a
        # 240x240x45x18x30 volume), which OOM-kills the process.
        frame_data = (f['ksp_epi_zf'][:, :, :, :, frame] for frame in range(nframes))
        if cfg.use_parfor:
            worker = functools.partial(_recon_one_frame, recon_fn=recon_fn, smaps=smaps)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(worker, frame_data))
        else:
            results = [_recon_one_frame(data, recon_fn, smaps) for data in frame_data]
    runtime_s = time.time() - t_start
    print(f'Reconstruction done in {runtime_s:.1f} s.')

    img = np.zeros((Nx, Ny, Nz, nframes), dtype=np.complex64)
    for frame, result in enumerate(results):
        if result is not None:
            img[:, :, :, frame] = np.asarray(result).reshape(Nx, Ny, Nz)

    if not np.any(img):
        print(
            'recon_frames: all output frames are zero -- '
            'recon_fn may have failed on every frame.'
        )

    seq_params_out = {
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'fov': seq_params.fov, 'volume_tr': seq_params.volume_tr,
        'Nvcoils': _nvcoils, 'Nframes': nframes,
    }
    return img, seq_params_out, runtime_s
