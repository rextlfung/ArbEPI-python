"""Stage 2 batch driver: combined L1-wavelet + TV regularized reconstruction
via sigpy. Ports run_bart.m (BART's pics replaced by recon_sigpy.py -- see
that module's docstring for why and how).
"""

import functools
import os

from preprocessing.config import PreprocessingConfig, load_config, load_seq_params, set_seq_paths
from preprocessing.nifti_io import save_recon_nifti
from preprocessing.recon_frames import recon_frames
from preprocessing.recon_sigpy import wavelet_tv_recon


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
                threshold_mask=cfg.threshold_mask,
                do_sense=cfg.do_sense,
                cc_energy_thresh=cfg.cc_energy_thresh,
                seqname=seqname,
                runtime_s=runtime_s,
                **sp,
            )
        except Exception as e:  # noqa: BLE001 -- mirrors run_bart.m's per-sequence try/catch
            print(f"ERROR [{seqname}]: {e}\nSkipping...")
    print('\nBatch complete.')


if __name__ == '__main__':
    cfg = load_config(datdir='/path/to/data/', seqnames=['caipi_ts'])
    run_recon_sigpy(cfg)
