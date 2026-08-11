"""Generate all 4 sequences for an ArbEPI scan session.

Ported from ../ArbEPI/main.m. Edit params.py to configure the experiment,
then run this script. Sequences are written to the output/ directory.

Order matters: generate_arbepi must run first because generate_epical and
generate_noise load samp_locs.mat that it produces.
"""

import argparse
import os

from params import load_params
from sampling.gen_sampling_masks import gen_sampling_masks
from sequences.arbepi import generate_arbepi
from sequences.epical import generate_epical
from sequences.gre import generate_gre
from sequences.noise import generate_noise


def main(export_ge: bool = False, plot: bool = False):
    params = load_params()

    # 1. Generate sampling masks and main EPI sequence
    omegas = gen_sampling_masks(params.R, params)
    generate_arbepi(omegas, params)

    # 2. Calibration sequence (ghost correction + receiver gain)
    generate_epical(params)

    # 3. Gradient echo reference (sensitivity maps)
    generate_gre(params)

    # 4. Noise prescan (noise covariance)
    generate_noise(params)

    if export_ge:
        # Requires a local MATLAB install with pulseq/toppe/PulCeq/ArbEPI
        # checked out as sibling directories — see ge_export.py / README.
        from ge_export import export_to_ge

        for name in ('ArbEPI', 'EPIcal', 'GRE', 'noise'):
            export_to_ge(
                os.path.join(params.output_dir, f'{name}.seq'),
                os.path.join(params.output_dir, name),
                params,
            )

    if plot:
        # Diagnostic sampling-mask/trajectory/PSF plots — see plot_last_run.py.
        from plot_last_run import plot_last_run

        plot_last_run(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ge', action='store_true', dest='export_ge',
        help='also export each sequence to GE .pge format via a local MATLAB install',
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='also write diagnostic sampling-mask/trajectory/PSF plots (see plot_last_run.py)',
    )
    args = parser.parse_args()
    main(export_ge=args.export_ge, plot=args.plot)
