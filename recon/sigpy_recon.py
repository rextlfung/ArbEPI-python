"""Stage 2 sigpy/numpy reconstructions of preprocessing/'s zero-filled k-space
(.venv-preprocessing; no torch): the shared per-frame loop with smaps loading
(recon_frames), the combined L1-wavelet + TV solver (wavelet_tv_recon), and
the three batch drivers built on them -- RSS, CG-SENSE and L1-wavelet+TV --
selectable as subcommands:

    .venv-preprocessing/bin/python -m recon.sigpy_recon {rss,cg-sense,l1-tv} <datdir> <seqname> [<seqname> ...]

The sections below are the original module docstrings, kept verbatim.

Formerly recon/basic/recon_sigpy.py
-----------------------------------
Combined L1-wavelet + total-variation regularized SENSE reconstruction,
replacing BART's `pics -R W:7:0:lamb_l1 -R T:7:0:lamb_tv -i N -S` (run_bart.m)
with sigpy (see CLAUDE.md for why BART was dropped in favor of sigpy).

Both regularizers are combined via sigpy's standard multi-regularizer
pattern for sigpy.app.LinearLeastSquares: a stacked operator
`G = Vstack([Wavelet, FiniteDifference])` and a block-separable proximal
operator `prox.Stack([L1Reg(...,lamb_l1), L1Reg(...,lamb_tv)])` -- the same
structure sigpy.mri.app.L1WaveletRecon / TotalVariationRecon each use
individually (see their source in sigpy), just combined here rather than
applied one at a time. Solved via PrimalDualHybridGradient, the standard
solver for f(x) + g(Gx) with f smooth and g nonsmooth-but-prox-friendly on
a transformed domain.

Verified against a synthetic SENSE forward model: with fully-sampled
synthetic k-space and lamb_l1=lamb_tv shrinking to 0, the reconstruction
converges to the true image (relative error 6.5e-3 -> 7e-4 -> ~0 as lamda
goes 1e-2 -> 1e-3 -> 1e-5), confirming the Vstack/Stack/PDHG composition is
solving the intended problem rather than something subtly mis-wired.

`y` is rescaled to O(1) before solving (and the result rescaled back) --
this port's replacement for BART's `-S` flag, added after real project
data (wb_2.4mm ball phantom) surfaced the consequence of not having one:
lamb_l1/lamb_tv are tuned for O(1)-scaled data, so without this step
they're negligible against raw scanner-unit k-space (|y| ~ 1e4-1e5),
silently degrading "L1+TV" to an unregularized least-squares SENSE solve.
At this acquisition's undersampling (R=6) that's ill-posed, and a lambda
sweep at the true (unscaled) magnitude confirmed it manifests specifically
as spurious signal loss in a uniform phantom's center -- center/shell
signal ratio was flat and wrong (~0.64, vs RSS's own 0.76) from lamb=0 up
through lamb=80, only correcting once lamb reached ~1000, i.e. roughly
the scale this normalization now reaches automatically at lamb=0.005.
sigpy's power-iteration step-size calibration (max_power_iter) still
serves its own separate auto-scaling purpose (for A/G's operator norms,
not the data/regularizer scale) and remains in place alongside this.


Formerly recon/basic/recon_frames.py
------------------------------------
Stage 2 shared logic: load/compute smaps, stream k-space, reconstruct
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


Formerly recon/basic/run_rss.py
-------------------------------
Stage 2 batch driver: root-sum-of-squares reconstruction (no smaps, no
BART). Ports run_rss.m + the toppe.utils.ift3.m helper it depends on
(dozfft=true branch only -- the only one run_rss.m actually uses; ift3.m's
oversampling-trim/decimation options aren't exercised here either, since
run_rss.m calls it with no extra arguments).


Formerly recon/basic/run_cg_sense.py
------------------------------------
Stage 2 batch driver: CG-SENSE reconstruction. Ports run_cg_sense.m.
No BART dependency -- cg_sense.py is plain numpy.


Formerly recon/basic/run_recon_sigpy.py
---------------------------------------
Stage 2 batch driver: combined L1-wavelet + TV regularized reconstruction
via sigpy. Ports run_bart.m (BART's pics replaced by sigpy_recon.py -- see
that module's docstring for why and how).
"""

import argparse
import concurrent.futures
import functools
import os
import sys
import time
from typing import Callable

import h5py
import numpy as np
import sigpy as sp
import sigpy.mri as mr

from preprocessing.cg_sense import cg_sense
from preprocessing.config import (
    PreprocessingConfig,
    SeqParams,
    SeqPaths,
    load_config,
    load_seq_params,
    set_seq_paths,
)
from preprocessing.nifti_io import save_recon_nifti
from preprocessing.smaps import load_smaps


def wavelet_tv_recon(
    ksp: np.ndarray,
    smaps: np.ndarray,
    lamb_l1: float,
    lamb_tv: float,
    num_iter: int,
    wave_name: str = 'db4',
    max_power_iter: int = 30,
) -> np.ndarray:
    """[Nx, Ny, Nz] complex image from one frame's zero-filled Cartesian
    k-space and coil sensitivity maps.

    ksp, smaps: [Nx, Ny, Nz, Ncoils] -- coils LAST (this port's usual
    convention); transposed internally since sigpy expects coils first.
    The sampling mask is inferred from where `ksp` is exactly zero (matches
    sigpy.mri.app._estimate_weights' own Cartesian convention, and this
    port's zero-filled-volume convention throughout preprocess.py).
    """
    ksp_cf = np.moveaxis(ksp, -1, 0)
    mps_cf = np.moveaxis(smaps, -1, 0)
    img_shape = mps_cf.shape[1:]

    weights = (sp.rss(ksp_cf, axes=(0,)) > 0).astype(ksp_cf.dtype)
    y = ksp_cf * weights**0.5

    # lamb_l1/lamb_tv are calibrated for O(1)-scaled data -- BART's `pics -S`
    # rescales internally before applying `-R` regularizers (see module
    # docstring); this is this port's replacement for that step. Without it,
    # lamb_l1/lamb_tv are negligible against raw scanner-unit k-space
    # (|y| ~ 1e4-1e5), making the regularizers inert and the "L1+TV" recon
    # silently degrade to an unregularized (and, at real undersampling
    # factors, ill-posed) least-squares SENSE solve -- confirmed on real
    # project data to cause spurious central signal loss in a uniform
    # phantom, fixed by restoring lamb_l1/lamb_tv to their intended
    # relative strength via this normalization.
    scale = 1.0 / np.percentile(np.abs(y[y != 0]), 99)
    y = y * scale

    A = mr.linop.Sense(mps_cf, weights=weights)

    W = sp.linop.Wavelet(img_shape, wave_name=wave_name)
    Grad = sp.linop.FiniteDifference(img_shape)
    G = sp.linop.Vstack([W, Grad])
    proxg = sp.prox.Stack([sp.prox.L1Reg(W.oshape, lamb_l1), sp.prox.L1Reg(Grad.oshape, lamb_tv)])

    app = sp.app.LinearLeastSquares(
        A, y, proxg=proxg, G=G,
        solver='PrimalDualHybridGradient',
        max_iter=num_iter, max_power_iter=max_power_iter,
        show_pbar=False,
    )
    return app.run() / scale

ReconFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _recon_one_frame(data: np.ndarray, recon_fn: ReconFn, smaps: np.ndarray) -> np.ndarray:
    try:
        return recon_fn(data, smaps)
    except Exception as e:  # noqa: BLE001 -- mirrors recon_frames.m's per-frame try/catch
        print(f'recon_frames: reconstruction failed on a frame -- skipping. {e}')
        return None


_worker_state: dict = {}


def _init_worker(recon_fn: ReconFn, smaps: np.ndarray) -> None:
    """ProcessPoolExecutor initializer: sets recon_fn/smaps once per worker
    process at pool startup, instead of binding them into the per-task
    callable via functools.partial -- ProcessPoolExecutor.map pickles each
    dispatched task (callable + bound args) independently, so a
    functools.partial(..., smaps=smaps) re-serializes the full smaps array
    (356 MiB at this repo's real 240x240x45x18-coil scale) once per frame,
    not once for the pool's lifetime (see docs/review-findings.md item
    105). Module-level dict rather than a global assigned in this
    function's body, so ruff doesn't need a `global` statement."""
    _worker_state['recon_fn'] = recon_fn
    _worker_state['smaps'] = smaps


def _recon_one_frame_worker(data: np.ndarray) -> np.ndarray:
    return _recon_one_frame(data, _worker_state['recon_fn'], _worker_state['smaps'])


def recon_frames(
    cfg: PreprocessingConfig, paths: SeqPaths, seq_params: SeqParams, recon_fn: ReconFn
) -> tuple[np.ndarray, dict, float]:
    """[Nx,Ny,Nz,Nframes] image, seq_params dict, wall-clock seconds.

    recon_fn(data, smaps) -> [Nx,Ny,Nz] image for one frame. Must be
    picklable (e.g. a functools.partial of a module-level function, not a
    lambda/closure) if cfg.use_parfor is True: it's sent to each worker
    process once at pool startup (via ProcessPoolExecutor's initializer/
    initargs, along with smaps -- not re-pickled per dispatched frame, see
    _init_worker).
    """
    Nx, Ny, Nz = seq_params.Nx, seq_params.Ny, seq_params.Nz

    smaps, _smaps_degre, _emap_degre, nvcoils, _smaps_degre_unc = load_smaps(cfg, paths, seq_params)

    with h5py.File(paths.recon, 'r') as f:
        nframes_avail = f['ksp_epi_zf'].shape[4]
        nframes = int(min(cfg.Nframes, nframes_avail))
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
            with concurrent.futures.ProcessPoolExecutor(
                initializer=_init_worker, initargs=(recon_fn, smaps)
            ) as executor:
                results = list(executor.map(_recon_one_frame_worker, frame_data))
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
        'Nvcoils': nvcoils, 'Nframes': nframes,
    }
    return img, seq_params_out, runtime_s

def _ift3(d: np.ndarray) -> np.ndarray:
    """Centered inverse 3D FFT, batched over the trailing (coil) axis.
    Ports toppe.utils.ift3.m's sub_ift3: fftshift(ifftn(fftshift(D))) per
    coil -- note this is fftshift on *both* sides, not the more common
    ifftshift-before/fftshift-after pairing, replicated literally rather
    than switched to the conventional spelling. These are NOT shift-
    equivalent on an odd-length axis (a one-sample circular shift of the
    k-space input, i.e. a pure linear phase ramp in image space) --
    params.py's N_degre has Nz_degre=21 (odd), so this does bite on that
    grid. Safe here only because every consumer of this function takes a
    magnitude (`_rss_recon`) or a difference of two same-grid transforms
    (preprocessing/julia/b0map.jl's mirror of this convention), both of
    which cancel the ramp -- a future complex-valued consumer would
    inherit it silently."""
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(d, axes=axes), axes=axes), axes=axes)


def _rss_recon(data, _smaps):
    return np.sqrt(np.sum(np.abs(_ift3(data)) ** 2, axis=-1))


def run_rss(cfg: PreprocessingConfig) -> None:
    print(f'Batch: {len(cfg.seqnames)} sequence(s) in {cfg.datdir}')
    for i, seqname in enumerate(cfg.seqnames, start=1):
        print(f'\n[{i}/{len(cfg.seqnames)}] {seqname}')
        paths = set_seq_paths(cfg, seqname)
        seq_params = load_seq_params(paths)

        out_dir = os.path.join(cfg.datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)
        fn_recon = os.path.join(out_dir, f'{seqname}_recon_rss')

        try:
            img, sp, runtime_s = recon_frames(cfg, paths, seq_params, _rss_recon)

            print(f'Saving reconstruction to {fn_recon}.nii.gz')
            save_recon_nifti(fn_recon, img, seqname=seqname, runtime_s=runtime_s, **sp)
        except Exception as e:  # noqa: BLE001 -- mirrors run_rss.m's per-sequence try/catch
            print(f"ERROR [{seqname}]: {e}\nSkipping...")
    print('\nBatch complete.')

def _cg_sense_recon(data, smaps, num_iter):
    return cg_sense(data, smaps, num_iter)[..., 0]


def run_cg_sense(cfg: PreprocessingConfig) -> None:
    print(f'Batch: {len(cfg.seqnames)} sequence(s) in {cfg.datdir}')
    for i, seqname in enumerate(cfg.seqnames, start=1):
        print(f'\n[{i}/{len(cfg.seqnames)}] {seqname}')
        paths = set_seq_paths(cfg, seqname)
        seq_params = load_seq_params(paths)

        out_dir = os.path.join(cfg.datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)
        fn_recon = os.path.join(out_dir, f'{seqname}_recon_cgs_i{cfg.num_iter}')

        try:
            recon_fn = functools.partial(_cg_sense_recon, num_iter=cfg.num_iter)
            img, sp, runtime_s = recon_frames(cfg, paths, seq_params, recon_fn)

            print(f'Saving reconstruction to {fn_recon}.nii.gz')
            save_recon_nifti(
                fn_recon, img,
                num_iter=cfg.num_iter,
                crop=cfg.crop,
                do_sense=cfg.do_sense,
                cc_energy_thresh=cfg.cc_energy_thresh,
                seqname=seqname,
                runtime_s=runtime_s,
                **sp,
            )
        except Exception as e:  # noqa: BLE001 -- mirrors run_cg_sense.m's per-sequence try/catch
            print(f"ERROR [{seqname}]: {e}\nSkipping...")
    print('\nBatch complete.')

def run_recon_sigpy(cfg: PreprocessingConfig) -> None:
    print(f'Batch: {len(cfg.seqnames)} sequence(s) in {cfg.datdir}')
    for i, seqname in enumerate(cfg.seqnames, start=1):
        print(f'\n[{i}/{len(cfg.seqnames)}] {seqname}')
        paths = set_seq_paths(cfg, seqname)
        seq_params = load_seq_params(paths)

        out_dir = os.path.join(cfg.datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)
        fn_recon = os.path.join(
            out_dir, f'{seqname}_recon_sigpy_l1_r{cfg.lamb_l1:.4f}_tv_r{cfg.lamb_tv:.4f}'
        )

        try:
            recon_fn = functools.partial(
                wavelet_tv_recon, lamb_l1=cfg.lamb_l1, lamb_tv=cfg.lamb_tv, num_iter=cfg.num_iter
            )
            img, sp, runtime_s = recon_frames(cfg, paths, seq_params, recon_fn)

            print(f'Saving reconstruction to {fn_recon}.nii.gz')
            save_recon_nifti(
                fn_recon, img,
                lamb_l1=cfg.lamb_l1,
                lamb_tv=cfg.lamb_tv,
                num_iter=cfg.num_iter,
                crop=cfg.crop,
                do_sense=cfg.do_sense,
                cc_energy_thresh=cfg.cc_energy_thresh,
                seqname=seqname,
                runtime_s=runtime_s,
                **sp,
            )
        except Exception as e:  # noqa: BLE001 -- mirrors run_bart.m's per-sequence try/catch
            print(f"ERROR [{seqname}]: {e}\nSkipping...")
    print('\nBatch complete.')


_BATCH_RUNNERS = {'rss': run_rss, 'cg-sense': run_cg_sense, 'l1-tv': run_recon_sigpy}


def _cli_batch(method: str) -> None:
    parser = argparse.ArgumentParser(description=f'Stage 2 batch reconstruction ({method}).')
    parser.add_argument('datdir')
    parser.add_argument('seqnames', nargs='+')
    args = parser.parse_args()
    _BATCH_RUNNERS[method](load_config(datdir=args.datdir, seqnames=args.seqnames))


def _cli_rss() -> None:
    _cli_batch('rss')


def _cli_cg_sense() -> None:
    _cli_batch('cg-sense')


def _cli_l1_tv() -> None:
    _cli_batch('l1-tv')


_COMMANDS = {
    'rss': _cli_rss,
    'cg-sense': _cli_cg_sense,
    'l1-tv': _cli_l1_tv,
}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        sys.exit('usage: python -m recon.sigpy_recon {' + ','.join(_COMMANDS) + '} ...')
    _cmd = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {_cmd}'
    _COMMANDS[_cmd]()
