"""Generate all 4 sequences for an ArbEPI scan session.

Ported from ../ArbEPI/main.m. Edit params.py to configure the experiment,
then run this script. Sequences are written to the output/ directory.

Order matters: generate_arbepi must run first because generate_epical and
generate_noise load samp_locs.mat that it produces.
"""

import argparse
import os

import numpy as np

from params import load_params
from sampling.gen_sampling_masks import gen_sampling_masks
from sequences.ArbEPI import generate_arbepi
from sequences.deGRE import generate_degre
from sequences.EPIcal import generate_epical
from sequences.noise import generate_noise


def main(export_ge: bool = False, plot: bool = False):
    params = load_params()

    # 1. Generate sampling masks and main EPI sequence. params.seed is
    # None by default (np.random.default_rng(None) is unseeded, same as
    # gen_sampling_masks' own fallback) -- set it for a reproducible mask.
    omegas = gen_sampling_masks(params.R, params, rng=np.random.default_rng(params.seed))
    generate_arbepi(omegas, params)

    # 2. Calibration sequence (ghost correction + receiver gain)
    generate_epical(params)

    # 3. deGRE: dual-echo gradient echo reference (sensitivity maps + B0 field map)
    generate_degre(params)

    # 4. Noise prescan (noise covariance)
    generate_noise(params)

    if plot:
        # Diagnostic sampling-mask/trajectory/PSF plots — see plot_last_run.py.
        # Written before GE export (only depends on samp_locs.mat/ArbEPI.seq,
        # both already on disk by this point) so they're available even if
        # export_ge fails below.
        from plotting.plot_last_run import plot_last_run

        plot_last_run(params)

    if export_ge:
        from seq2ge.ge_export import check_ge_feasibility, export_to_ge

        seq_paths = {
            name: os.path.join(params.output_dir, f'{name}.seq')
            for name in ('ArbEPI', 'EPIcal', 'deGRE', 'noise')
        }

        # Check all four sequences for GE hardware/PNS/acoustic-resonance
        # feasibility before writing any .pge file, so an infeasible
        # sequence is caught up front rather than after several full
        # exports have already run.
        for seq_path in seq_paths.values():
            check_ge_feasibility(seq_path, params)

        for name, seq_path in seq_paths.items():
            export_to_ge(seq_path, os.path.join(params.output_dir, name), params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ge', action='store_true', dest='export_ge',
        help='also export each sequence to GE .pge format (see seq2ge/ge_export.py)',
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='also write diagnostic sampling-mask/trajectory/PSF plots (see plot_last_run.py)',
    )
    args = parser.parse_args()
    main(export_ge=args.export_ge, plot=args.plot)
