"""Stage 1 batch driver. Ports batch_preprocess.m.

Usage: edit the cfg = load_config(...) call below (or import and call
batch_preprocess(cfg) from your own script), then run this module.
"""

from preprocess.config import PreprocessingConfig, load_config, set_seq_paths
from preprocess.preprocess import preprocess


def batch_preprocess(cfg: PreprocessingConfig) -> None:
    print(f'Batch: {len(cfg.seqnames)} sequence(s) in {cfg.datdir}')
    for i, seqname in enumerate(cfg.seqnames, start=1):
        print(f'\n[{i}/{len(cfg.seqnames)}] {seqname}')
        paths = set_seq_paths(cfg, seqname)
        try:
            preprocess(cfg, paths)
        except Exception as e:  # noqa: BLE001 -- mirrors batch_preprocess.m's per-sequence try/catch
            print(f"ERROR in '{seqname}': {e}\nContinuing...")
    print('\nBatch complete.')


if __name__ == '__main__':
    cfg = load_config(datdir='/path/to/data/', seqnames=['caipi_ts'])
    batch_preprocess(cfg)
