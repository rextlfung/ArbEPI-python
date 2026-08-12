"""Plot the sampling mask, k-space trajectory, and PSF for the most
recently generated sequence in output/. Run standalone after main.py via
`python -m plotting.plot_last_run` (from the repo root -- not
`python plotting/plot_last_run.py`, which breaks its own package-relative
imports), or via `python main.py --plot` (see main.py).

Reconstructs the sampling masks from samp_locs.mat (the schedule actually
used to build the sequence) rather than regenerating them, since sampling
methods other than 'caipi' are randomized and wouldn't reproduce the exact
masks from the last run.
"""

import hdf5storage
import numpy as np
import pypulseq as pp

from params import Params, load_params
from plotting.plotting import plot_one_tr, plot_psf, plot_sampling_mask, plot_trajectory


def plot_last_run(params: Params, frame_idx: int = 0) -> None:
    # Reconstruct the exact sampling masks (all frames) from the schedule
    # that was actually used (samp_locs.mat's schedules are 1-based).
    schedules = hdf5storage.loadmat(f'{params.output_dir}/samp_locs.mat')['schedules']
    omegas = np.zeros((params.Ny, params.Nz, params.Nframes), dtype=bool)
    for frame in range(params.Nframes):
        iy, iz = schedules[frame, ..., 0].astype(int) - 1, schedules[frame, ..., 1].astype(int) - 1
        omegas[iy, iz, frame] = True

    plot_sampling_mask(omegas, params, params.R, frame_idx=frame_idx).savefig(
        f'{params.output_dir}/mask.png', dpi=300
    )
    plot_psf(omegas, params, frame_idx=frame_idx).savefig(f'{params.output_dir}/psf.png', dpi=300)

    # Reload the sequence you just generated and plot its trajectory plus a
    # single-TR snippet (the frame's first shot).
    seq = pp.Sequence(system=params.sys)
    seq.read(f'{params.output_dir}/ArbEPI.seq')
    plot_trajectory(seq, params, params.R, frame_idx=frame_idx).savefig(
        f'{params.output_dir}/trajectory.png', dpi=300
    )
    plot_one_tr(seq, params, shot_index=frame_idx * params.Nshots).savefig(
        f'{params.output_dir}/one_tr.png', dpi=300
    )

    print(f'Wrote {params.output_dir}/mask.png, psf.png, trajectory.png, one_tr.png (frame {frame_idx})')


if __name__ == '__main__':
    plot_last_run(load_params())
