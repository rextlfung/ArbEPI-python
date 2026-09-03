"""Stage 2 batch driver: root-sum-of-squares reconstruction (no smaps, no
BART). Ports run_rss.m + the toppe.utils.ift3.m helper it depends on
(dozfft=true branch only -- the only one run_rss.m actually uses; ift3.m's
oversampling-trim/decimation options aren't exercised here either, since
run_rss.m calls it with no extra arguments).
"""

import os

import numpy as np

from preprocessing.config import PreprocessingConfig, load_config, load_seq_params, set_seq_paths
from preprocessing.nifti_io import save_recon_nifti
from preprocessing.recon_frames import recon_frames


def _ift3(d: np.ndarray) -> np.ndarray:
    """Centered inverse 3D FFT, batched over the trailing (coil) axis.
    Ports toppe.utils.ift3.m's sub_ift3: fftshift(ifftn(fftshift(D))) per
    coil -- note this is fftshift on *both* sides, not the more common
    ifftshift-before/fftshift-after pairing, replicated literally rather
    than switched to the conventional spelling. These are NOT shift-
    equivalent on an odd-length axis (a one-sample circular shift of the
    k-space input, i.e. a pure linear phase ramp in image space) --
    params.py's N_degre has Nz_degre=21 (odd), so this does bite on that
    grid. Safe here only because every consumer of this function takes a
    magnitude (`_rss_recon`) or a difference of two same-grid transforms
    (preprocessing/julia/b0map.jl's mirror of this convention), both of
    which cancel the ramp -- a future complex-valued consumer would
    inherit it silently."""
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(d, axes=axes), axes=axes), axes=axes)


def _rss_recon(data, _smaps):
    return np.sqrt(np.sum(np.abs(_ift3(data)) ** 2, axis=-1))


def run_rss(cfg: PreprocessingConfig) -> None:
    print(f'Batch: {len(cfg.seqnames)} sequence(s) in {cfg.datdir}')
    for i, seqname in enumerate(cfg.seqnames, start=1):
        print(f'\n[{i}/{len(cfg.seqnames)}] {seqname}')
        paths = set_seq_paths(cfg, seqname)
        seq_params = load_seq_params(paths)

        out_dir = os.path.join(cfg.datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)
        fn_recon = os.path.join(out_dir, f'{seqname}_recon_rss')

        try:
            img, sp, runtime_s = recon_frames(cfg, paths, seq_params, _rss_recon)

            print(f'Saving reconstruction to {fn_recon}.nii.gz')
            save_recon_nifti(fn_recon, img, seqname=seqname, runtime_s=runtime_s, **sp)
        except Exception as e:  # noqa: BLE001 -- mirrors run_rss.m's per-sequence try/catch
            print(f"ERROR [{seqname}]: {e}\nSkipping...")
    print('\nBatch complete.')


if __name__ == '__main__':
    cfg = load_config(datdir='/path/to/data/', seqnames=['caipi_ts'])
    run_rss(cfg)
