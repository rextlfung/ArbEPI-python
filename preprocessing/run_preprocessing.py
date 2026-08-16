"""Stage 1 batch driver. Ports run_preprocessing.m.

Usage: edit the cfg = load_config(...) call below (or import and call
run_preprocessing(cfg) from your own script), then run this module.
"""

from preprocessing.config import PreprocessingConfig, load_config, set_seq_paths
from preprocessing.preprocess import preprocess


def run_preprocessing(cfg: PreprocessingConfig) -> None:
    print(f'Batch: {len(cfg.seqnames)} sequence(s) in {cfg.datdir}')
    for i, seqname in enumerate(cfg.seqnames, start=1):
        print(f'\n[{i}/{len(cfg.seqnames)}] {seqname}')
        paths = set_seq_paths(cfg, seqname)
        try:
            preprocess(cfg, paths)
        except Exception as e:  # noqa: BLE001 -- mirrors run_preprocessing.m's per-sequence try/catch
            print(f"ERROR in '{seqname}': {e}\nContinuing...")
    print('\nBatch complete.')


if __name__ == '__main__':
    cfg = load_config(datdir='/path/to/data/', seqnames=['caipi_ts'])
    run_preprocessing(cfg)
