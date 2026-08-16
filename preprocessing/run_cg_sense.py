"""Stage 2 batch driver: CG-SENSE reconstruction. Ports run_cg_sense.m.
No BART dependency -- cg_sense.py is plain numpy.
"""

import functools
import os

from preprocessing.cg_sense import cg_sense
from preprocessing.config import PreprocessingConfig, load_config, load_seq_params, set_seq_paths
from preprocessing.nifti_io import save_recon_nifti
from preprocessing.recon_frames import recon_frames


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
                threshold_mask=cfg.threshold_mask,
                do_sense=cfg.do_sense,
                cc_energy_thresh=cfg.cc_energy_thresh,
                seqname=seqname,
                runtime_s=runtime_s,
                **sp,
            )
        except Exception as e:  # noqa: BLE001 -- mirrors run_cg_sense.m's per-sequence try/catch
            print(f"ERROR [{seqname}]: {e}\nSkipping...")
    print('\nBatch complete.')


if __name__ == '__main__':
    cfg = load_config(datdir='/path/to/data/', seqnames=['caipi_ts'])
    run_cg_sense(cfg)
