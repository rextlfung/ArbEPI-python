"""Stage 1.5 batch driver: B0 field map estimation via MRIFieldmaps.jl.

Unlike run_rss.py/run_cg_sense.py/run_recon_sigpy.py (pure-Python Stage 2
drivers), this shells out to Julia -- MRIFieldmaps.jl
(https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl) has no Python
port, and CLAUDE.md's preprocessing/ section previously described B0
estimation as "the external Julia package's job", left for a separate
consumer outside this repo. This module brings that step in-repo instead,
the same way raw_io.py isolates GE's proprietary GERecon SDK to one module
rather than porting it: preprocessing/julia/b0map.jl is a self-contained
Julia project (preprocessing/julia/Project.toml + Manifest.toml, pinned)
invoked as a subprocess, not embedded via PythonCall/juliacall -- there is
no other Julia dependency anywhere else in this pipeline to justify that
weight.

Depends on preprocess()'s STEP 2 output (`<seqname>_gre.h5`, whitened +
coil-compressed dual-echo GRE k-space + TE_degre), so run_preprocessing()
must have already completed for each seqname before this.
"""

import os
import shutil
import subprocess

import h5py

from preprocessing.config import PreprocessingConfig, load_config, load_seq_params, set_seq_paths
from preprocessing.nifti_io import save_recon_nifti

_JULIA_DIR = os.path.join(os.path.dirname(__file__), 'julia')
_JULIA_SCRIPT = os.path.join(_JULIA_DIR, 'b0map.jl')


def run_b0map(cfg: PreprocessingConfig) -> None:
    julia_bin = shutil.which('julia')
    if julia_bin is None:
        raise RuntimeError(
            'run_b0map: no `julia` executable on PATH -- install it via '
            'https://julialang.org/install/ (juliaup) first.'
        )

    print(f'Batch: {len(cfg.seqnames)} sequence(s) in {cfg.datdir}')
    for i, seqname in enumerate(cfg.seqnames, start=1):
        print(f'\n[{i}/{len(cfg.seqnames)}] {seqname}')
        gre_cache_path = os.path.join(cfg.datdir, 'recon', f'{seqname}_gre.h5')
        if not os.path.exists(gre_cache_path):
            print(
                f"ERROR [{seqname}]: '{gre_cache_path}' not found -- "
                'run preprocess() first. Skipping...'
            )
            continue
        output_path = os.path.join(cfg.datdir, 'recon', f'{seqname}_b0map.h5')

        try:
            subprocess.run(
                [
                    julia_bin, f'--project={_JULIA_DIR}', _JULIA_SCRIPT,
                    gre_cache_path, output_path, str(cfg.b0map_mask_thresh),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:  # noqa: BLE001 -- mirrors the other batch drivers' per-sequence try/catch
            print(f"ERROR [{seqname}]: {e}\nSkipping...")
            continue

        # NIfTI export for viewing in FSLeyes/etc. -- b0map_hz lives on the
        # deGRE acquisition grid, not the EPI grid, so its voxel size comes
        # from seq_params.fov_degre, not seq_params.fov (see nifti_io module
        # docstring for why this stays alongside, not instead of, the .h5
        # the Julia consumer above already wrote).
        paths = set_seq_paths(cfg, seqname)
        seq_params = load_seq_params(paths)
        with h5py.File(output_path, 'r') as f:
            b0map_hz = f['b0map_hz'][()]
        save_recon_nifti(
            output_path[: -len('.h5')], b0map_hz,
            fov=seq_params.fov_degre, seqname=seqname,
            mask_threshold=cfg.b0map_mask_thresh,
        )
    print('\nBatch complete.')


if __name__ == '__main__':
    cfg = load_config(datdir='/path/to/data/', seqnames=['caipi_ts'])
    run_b0map(cfg)
