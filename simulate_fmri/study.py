"""Repeated runs of one simulated experiment, taken through preprocess/ and recon/.

A reliability analysis (analysis.py) needs the same experiment several times
over: the same subject, protocol and task, with fresh noise each time. This
module runs that study and is resumable: every step is skipped when its output
exists, so a notebook can call it to pick up finished work.

    python -m simulate_fmri.study output/simulate_fmri --duration 320 --runs 4

writes

    <outdir>/seq/scan_info.mat                  the sequence (make_scan_info)
    <outdir>/run<k>/                            one session directory per run
        scanarchives/, seqs/task/scan_info.mat, task_truth.h5     (session.py)
        recon/task_preprocessed.h5                                 (preprocess/)
        recon/sense_wavelet-tv[_b0]_hp3/task_recon.h5              (recon/)

Runs differ in the seed only: thermal noise, the coil noise covariance and the
physiological noise (time courses and spatial patterns) are redrawn; the
anatomy, field, coils, sequence and task are the same.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import replace

import h5py

from .session import SessionConfig, simulate_session

NAME = 'task'

# Reconstructions to compare: recon.sense.main keyword arguments by label.
# wavelet-TV with the temporal high-pass penalty, with and without the B0
# model. sigma1A is passed so that no power iteration runs (75 min for the B0
# operator of a 60 s run; it measured 1.488, and the plain operator's is 1).
# L_b0 = 16: on the default protocol (a 31 ms echo train, a field clipped to
# about 300 Hz) the B0 operator with 16 time segments is within 0.001% of the
# one with 64, at half the cost of the default 32; 12 is 0.2% off, 8 is 2%.
RECONS = {
    'no B0': dict(sigma1A=1.0),
    'B0': dict(B0=True, L_b0=16, sigma1A=1.5),
}
RECON_COMMON = dict(reg='wavelet-tv', hp_weight=3.0, niters=100, tag='hp3')


def make_scan_info(outdir: str, duration_s: float | None = None) -> str:
    """Generate the ArbEPI and deGRE sequences of params.py for a run of
    duration_s (None: params.py's own) into <outdir>/seq and return the path
    of their scan_info.mat. Skipped when one of that length is there."""
    from params import load_params
    from sample.gen_sampling_masks import resolve_omegas
    from sequences.ArbEPI import generate_arbepi
    from sequences.deGRE import generate_degre

    seq_dir = os.path.join(outdir, 'seq')
    params = load_params(output_dir=seq_dir)
    if duration_s is not None:
        params = replace(params, Nframes=round(duration_s / params.volume_tr))
    fn = os.path.join(seq_dir, 'scan_info.mat')
    if os.path.exists(fn):
        with h5py.File(fn, 'r') as f:
            if f['schedules'].shape[-1] == params.Nframes and 'dwell_degre' in f:
                return fn
    generate_arbepi(resolve_omegas(params), params)
    generate_degre(params)
    return fn


def run_paths(outdir: str, k: int) -> dict[str, str]:
    """Where run k (0-based) keeps its files."""
    run_dir = os.path.join(outdir, f'run{k + 1}')
    pre = os.path.join(run_dir, 'recon', f'{NAME}_preprocessed.h5')
    return {
        'dir': run_dir,
        'truth': os.path.join(run_dir, f'{NAME}_truth.h5'),
        'raw': os.path.join(run_dir, 'scanarchives'),
        'preprocessed': pre,
        # written once preprocess() has returned: a half-written file is not taken for done
        'preprocessed_ok': pre + '.ok',
    }


def simulate_run(scan_info: str, outdir: str, k: int, cfg: SessionConfig | None = None,
                 anatomy='brainweb', device=None) -> dict[str, str]:
    """Simulate run k's raw session (seed cfg.seed + 10 k), unless it is there
    or already preprocessed."""
    paths = run_paths(outdir, k)
    have_raw = os.path.exists(paths['truth']) and os.path.isdir(paths['raw'])
    if not (have_raw or os.path.exists(paths['preprocessed_ok'])):
        cfg = cfg or SessionConfig()
        simulate_session(scan_info, paths['dir'], NAME, anatomy=anatomy,
                         cfg=replace(cfg, seed=cfg.seed + 10 * k), device=device)
    return paths


def preprocess_run(outdir: str, k: int, keep_raw: bool = True) -> dict[str, str]:
    """preprocess/ on run k's raw session, unless done. keep_raw=False deletes
    the raw archives afterwards (9 GB per 320 s run)."""
    from preprocess.preprocess import PreprocessConfig, preprocess

    paths = run_paths(outdir, k)
    if not os.path.exists(paths['preprocessed_ok']):
        t0 = time.time()
        preprocess(PreprocessConfig(datdir=paths['dir'], seqnames=[NAME]), NAME)
        with open(paths['preprocessed_ok'], 'w') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S\n'))
        print(f'run {k + 1}: preprocessed in {(time.time() - t0) / 60:.0f} min')
    if not keep_raw and os.path.isdir(paths['raw']):
        shutil.rmtree(paths['raw'])
    return paths


def recon_path(outdir: str, k: int, kw: dict) -> str:
    kw = {**RECON_COMMON, **kw}
    sub = 'sense_' + kw['reg'] + ('_b0' if kw.get('B0') else '') + f"_{kw['tag']}"
    return os.path.join(run_paths(outdir, k)['dir'], 'recon', sub, f'{NAME}_recon.h5')


def reconstruct_run(outdir: str, k: int, recons: dict[str, dict] | None = None,
                    device=None) -> dict[str, str]:
    """Reconstruct run k with every setting in recons (default RECONS; each on
    top of RECON_COMMON), unless done. Returns {label: path of the .h5}."""
    from recon.sense import main as recon_sense

    paths = run_paths(outdir, k)
    with h5py.File(paths['truth'], 'r') as f:
        volume_tr = float(f.attrs['volume_tr'])
    done = {}
    for label, kw in (RECONS if recons is None else recons).items():
        out = recon_path(outdir, k, kw)
        if not os.path.exists(out):
            t0 = time.time()
            kw = {**RECON_COMMON, **kw}
            reg = kw.pop('reg')
            made = recon_sense(paths['dir'], NAME, reg, volume_tr_s=volume_tr,
                               **({'device': device} if device is not None else {}), **kw)
            assert os.path.samefile(made + '.h5', out), (made, out)
            print(f'run {k + 1}, {label}: reconstructed in {(time.time() - t0) / 60:.0f} min')
        done[label] = out
    return done


def run_repetitions(
    scan_info: str,
    outdir: str,
    n_runs: int = 4,
    *,
    cfg: SessionConfig | None = None,
    recons: dict[str, dict] | None = None,
    anatomy='brainweb',
    keep_raw: int = 1,
    device=None,
) -> list[dict]:
    """Simulate, preprocess and reconstruct n_runs repetitions of the
    experiment in scan_info, one after the other; whatever is already there is
    kept. Run k uses cfg with seed cfg.seed + 10 k.

    recons: recon.sense.main keyword arguments by label (default RECONS), on
    top of RECON_COMMON. keep_raw: how many runs keep their raw archives.

    Returns, per run: {'dir', 'truth', 'preprocessed', 'recons': {label: path
    of the reconstruction's .h5}}.
    """
    runs = []
    for k in range(n_runs):
        simulate_run(scan_info, outdir, k, cfg, anatomy, device)
        paths = preprocess_run(outdir, k, keep_raw=k < keep_raw)
        runs.append({**{key: paths[key] for key in ('dir', 'truth', 'preprocessed')},
                     'recons': reconstruct_run(outdir, k, recons, device)})
    return runs


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('outdir')
    p.add_argument('--duration', type=float, default=None,
                   help='run length, s (default: the duration in params.py)')
    p.add_argument('--runs', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--phantom', default='brainweb', choices=('brainweb', 'ellipsoid'))
    p.add_argument('--keep-raw', type=int, default=1,
                   help='how many runs keep their raw archives')
    p.add_argument('--device', default=None)
    a = p.parse_args()
    scan_info = make_scan_info(a.outdir, a.duration)
    run_repetitions(scan_info, a.outdir, a.runs, cfg=SessionConfig(seed=a.seed),
                    anatomy=a.phantom, keep_raw=a.keep_raw, device=a.device)


if __name__ == '__main__':
    _cli()
