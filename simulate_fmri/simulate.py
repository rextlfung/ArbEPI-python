"""Simulate an fMRI experiment acquired with an ArbEPI sequence.

    python -m simulate_fmri.simulate output/scan_info.mat <outdir>

reads the acquisition from the scan_info.mat that sequences/ArbEPI.py wrote
next to ArbEPI.seq (matrix size, field of view, the (ky, kz) location and echo
time of every echo of every shot of every frame, volume TR, flip angle), has
SNAKE-fMRI acquire a brain phantom with a block-design BOLD activation along
exactly that schedule, and writes

    <outdir>/<name>.mrd                      SNAKE's raw file
    <outdir>/recon/<name>_preprocessed.h5    input to recon.rss / recon.sense,
                                             with the ground truth in `truth`

Reconstruct with recon/ as for a real scan, then score the result against the
ground truth with `python -m recon.testbed score`. See README.md in this folder
for what the simulation models and what it does not.
"""

from __future__ import annotations

import argparse
import os

import h5py
from snake.core.phantom import Phantom
from snake.core.simulation import FOVConfig, GreConfig, HardwareConfig, SimConfig

from .engine import ArbEPIAcquisitionEngine
from .export import export
from .handlers import EllipsoidActivationHandler
from .phantom import (
    brainweb_phantom,
    ellipsoid_phantom,
    ellipsoid_phantom_roi,
    place_fov,
    to_acquisition_grid,
)
from .protocol import Protocol, load_protocol
from .sampler import ArbEPISampler


def make_sim_conf(
    protocol: Protocol, n_coils: int, fov: FOVConfig | None = None, seed: int = 0
) -> SimConfig:
    """SNAKE's configuration for `protocol`.

    One SNAKE repetition is one ArbEPI shot (one excitation), so seq.TR is the
    per-shot TR: the phantom, and with it the BOLD signal, is updated at every
    excitation, and the steady-state contrast is that of a spoiled gradient
    echo at the per-shot TR and flip angle. hardware.dwell_time_ms is the echo
    spacing divided by Nx, an effective dwell that places a readout's samples
    evenly across its echo spacing.
    """
    n_excitations = protocol.n_frames * protocol.n_shots
    return SimConfig(
        # Half a TR of margin: the activation handler makes one sample per TR
        # in [0, max_sim_time), and a float-rounded exact length can lose one.
        max_sim_time=(n_excitations + 0.5) * protocol.tr_shot_ms / 1e3,
        seq=GreConfig(TR=protocol.tr_shot_ms, TE=protocol.te_ms, FA=protocol.fa_deg),
        hardware=HardwareConfig(
            n_coils=n_coils, dwell_time_ms=protocol.echo_spacing_ms / protocol.shape[0], field=3.0
        ),
        fov=fov if fov is not None else FOVConfig(size=protocol.fov_mm, res_mm=protocol.res_mm),
        rng_seed=seed,
    )


def simulate(
    scan_info: str | Protocol,
    outdir: str,
    name: str = 'sim',
    *,
    phantom: str | Phantom = 'brainweb',
    n_coils: int = 16,
    coils_per_ring: int = 8,
    model: str = 'T2s',
    snr: float = 1000.0,
    block_on: float = 10.0,
    block_off: float = 10.0,
    delta_r2s: float = 1000.0,
    roi: dict | None = None,
    handlers: list | None = None,
    fa_deg: float | None = None,
    frames: int | None = None,
    n_workers: int = 0,
    seed: int = 0,
) -> dict[str, str]:
    """Simulate the acquisition in `scan_info` and export it for recon/.

    scan_info: a scan_info.mat path, or a Protocol.
    phantom: 'brainweb' (downloaded on first use), 'ellipsoid' (analytic, no
        download), or a snake Phantom on any grid.
    n_coils, coils_per_ring: receive array (phantom.birdcage_smaps).
    model: 'T2s' (decay along each echo train) or 'simple' (contrast at TE).
    snr: SNAKE's noise level (engine.ArbEPIAcquisitionEngine); np.inf for none.
    block_on, block_off: block design, s. The first block starts at t = 0.
    delta_r2s: sets the activation's peak fractional signal change to
        TE (ms) / delta_r2s, so 1000 gives 3% at TE = 30 ms.
    roi: the activated ellipsoid, as EllipsoidActivationHandler's center_mm,
        semi_axes_mm and euler_angles (world mm). Default: SNAKE's occipital
        region for BrainWeb (also used for a Phantom passed in), and
        phantom.ellipsoid_phantom_roi for 'ellipsoid'. The activation is this
        ellipsoid's gray matter.
    handlers: SNAKE handlers to use instead of that block activation; [] for
        none.
    fa_deg, frames: see load_protocol (ignored when scan_info is a Protocol).
    n_workers: worker processes; 0 = half the CPU count.
    seed: for the noise.

    Returns the paths written: 'mrd' and 'preprocessed'.
    """
    if isinstance(scan_info, Protocol):
        protocol = scan_info
    else:
        protocol = load_protocol(scan_info, fa_deg=fa_deg, frames=frames)
    if isinstance(phantom, str):
        makers = {'brainweb': brainweb_phantom, 'ellipsoid': ellipsoid_phantom}
        if phantom not in makers:
            raise ValueError(f'phantom={phantom!r}, expected one of {sorted(makers)} or a Phantom')
        if phantom == 'ellipsoid' and roi is None:
            roi = ellipsoid_phantom_roi()
        phantom = makers[phantom]()
    fov = place_fov(phantom, protocol.shape, protocol.res_mm)
    sim_conf = make_sim_conf(protocol, n_coils, fov, seed)
    phantom = to_acquisition_grid(phantom, sim_conf, coils_per_ring)

    duration = protocol.n_frames * protocol.volume_tr_s
    if handlers is None:
        handlers = [
            EllipsoidActivationHandler(
                block_on=block_on, block_off=block_off, duration=duration,
                delta_r2s=delta_r2s, **(roi or {}),
            )
        ]
    print(
        f'{protocol.n_frames} frames x {protocol.n_shots} shots x {protocol.etl} echoes, '
        f'{protocol.shape} at {tuple(round(r, 2) for r in protocol.res_mm)} mm, '
        f'R = {protocol.acceleration:.1f}\n'
        f'volume TR {protocol.volume_tr_s * 1e3:.0f} ms, shot TR {protocol.tr_shot_ms:.2f} ms, '
        f'TE {protocol.te_ms:.2f} ms, echo spacing {protocol.echo_spacing_ms * 1e3:.0f} us, '
        f'flip {protocol.fa_deg:.1f} deg, {duration:.1f} s'
    )

    os.makedirs(outdir, exist_ok=True)
    fn_mrd = os.path.join(outdir, f'{name}.mrd')
    sampler = ArbEPISampler(schedules=protocol.schedules, echo_times_ms=protocol.echo_times_ms)
    engine = ArbEPIAcquisitionEngine(model=model, snr=snr)
    engine(
        fn_mrd,
        sampler,
        phantom,
        sim_conf,
        handlers=handlers,
        worker_chunk_size=protocol.n_shots,
        n_workers=n_workers,
        resample_early=False,  # to_acquisition_grid already put it on the grid
    )
    fn_pre = export(fn_mrd, protocol, outdir, name, snr)
    with h5py.File(fn_pre, 'r') as f:
        if 'sim_snr0_gm' in f.attrs:
            print(
                f'snr = {snr:g}: gray-matter SNR of a fully sampled reconstruction '
                f'{f.attrs["sim_snr0_gm"]:.0f}'
            )
    print(f'Wrote {fn_mrd}\n      {fn_pre}')
    return {'mrd': fn_mrd, 'preprocessed': fn_pre}


def _cli() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('scan_info', help='scan_info.mat written by sequences/ArbEPI.py')
    p.add_argument('outdir', help='output directory (the datdir to give recon/)')
    p.add_argument('--name', default='sim', help='output name (the seqname to give recon/)')
    p.add_argument('--phantom', default='brainweb', choices=('brainweb', 'ellipsoid'))
    p.add_argument('--coils', type=int, default=16, help='receive coils (default 16)')
    p.add_argument('--coils-per-ring', type=int, default=8)
    p.add_argument('--model', default='T2s', choices=('T2s', 'simple'))
    p.add_argument('--snr', type=float, default=1000.0, help="SNAKE's noise level; inf for none")
    p.add_argument('--block-on', type=float, default=10.0, help='stimulus duration, s')
    p.add_argument('--block-off', type=float, default=10.0, help='rest duration, s')
    p.add_argument(
        '--delta-r2s', type=float, default=1000.0,
        help='peak fractional signal change = TE (ms) / this (default 1000)',
    )
    p.add_argument('--no-activation', action='store_true', help='simulate a resting phantom')
    p.add_argument('--fa', type=float, default=None, help='flip angle, degrees')
    p.add_argument('--frames', type=int, default=None, help='only the first N frames')
    p.add_argument('--workers', type=int, default=0, help='processes (default: half the CPUs)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    simulate(
        a.scan_info, a.outdir, a.name,
        phantom=a.phantom, n_coils=a.coils, coils_per_ring=a.coils_per_ring, model=a.model,
        snr=a.snr, block_on=a.block_on, block_off=a.block_off, delta_r2s=a.delta_r2s,
        handlers=[] if a.no_activation else None,
        fa_deg=a.fa, frames=a.frames, n_workers=a.workers, seed=a.seed,
    )


if __name__ == '__main__':
    _cli()
