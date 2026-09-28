"""Preprocess several sequences in one go. Ports run_preprocessing.m.

    .venv-preprocessing/bin/python -m preprocess.batch_preprocess <datdir> <seqname> [...]

Writes <outdir>/<seqname>_preprocessed.h5 per sequence (default outdir:
<datdir>/recon). A failure in one sequence is reported and the rest continue.
See preprocess/demo.ipynb for a walkthrough of every setting.
"""

import argparse
import traceback

from preprocess.preprocess import PreprocessConfig, preprocess


def batch_preprocess(cfg: PreprocessConfig) -> dict[str, str | None]:
    """preprocess() every name in cfg.seqnames; returns {seqname: output path,
    or None if it failed}."""
    print(f'Batch: {len(cfg.seqnames)} sequence(s) in {cfg.datdir}')
    results: dict[str, str | None] = {}
    for i, seqname in enumerate(cfg.seqnames, start=1):
        print(f'\n[{i}/{len(cfg.seqnames)}] {seqname}')
        try:
            results[seqname] = preprocess(cfg, seqname)
        except Exception:  # noqa: BLE001 -- keep going with the other sequences
            traceback.print_exc()
            print(f"ERROR in '{seqname}', continuing")
            results[seqname] = None
    print('\nBatch complete.')
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('datdir')
    p.add_argument('seqnames', nargs='+')
    p.add_argument('--outdir', help='default: <datdir>/recon')
    p.add_argument('--fn-gre', help='deGRE archive; default <datdir>/scanarchives/gre.h5')
    p.add_argument('--delay', type=float, default=-1.0, help='k-space center offset (samples)')
    p.add_argument('--no-compress', action='store_true', help='keep all physical coils')
    p.add_argument('--cc-method', choices=('gcc', 'pca'), default='gcc')
    p.add_argument('--cc-energy', type=float, default=0.999,
                   help='fraction of eigenvalue energy the virtual coils keep')
    p.add_argument('--nvcoils', type=int,
                   help='exact number of virtual coils (overrides --cc-energy)')
    p.add_argument('--no-smaps', action='store_true')
    p.add_argument('--no-b0', action='store_true', help='skip the B0 map (needs julia)')
    p.add_argument('--no-r2star', action='store_true')
    p.add_argument('--no-calib', action='store_true', help='skip the calibration-region dataset')
    p.add_argument('--zero-pad-z', action='store_true',
                   help='zero-fill EPI slices outside the deGRE slab instead of failing')
    p.add_argument('--keep-cache', action='store_true', help='keep <seqname>_gridded.h5')
    a = p.parse_args(argv)
    cfg = PreprocessConfig(
        datdir=a.datdir, seqnames=a.seqnames, outdir=a.outdir, fn_gre=a.fn_gre, delay=a.delay,
        compress=not a.no_compress, cc_method=a.cc_method, cc_energy_thresh=a.cc_energy,
        Nvcoils=a.nvcoils, estimate_smaps=not a.no_smaps, estimate_b0=not a.no_b0,
        estimate_r2star=not a.no_r2star, extract_calib=not a.no_calib,
        zero_pad_z=a.zero_pad_z, keep_cache=a.keep_cache,
    )
    batch_preprocess(cfg)


if __name__ == '__main__':
    main()
