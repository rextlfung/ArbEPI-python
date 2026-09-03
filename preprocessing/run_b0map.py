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

Also computes (or loads, if already cached by a prior recon_frames.py run)
sensitivity maps on the deGRE grid via smaps.load_smaps, and passes them to
b0map.jl as a `smap` argument -- see b0map.jl's own module docstring for
why this replaces MRIFieldmaps' phase-contrast coil-combine fallback (a
true matched-filter combine, expected to reduce field-map noise in this
pipeline's real low-per-coil-SNR object-center regions). This means
run_b0map() can now trigger ESPIRiT sensitivity-map estimation itself, not
only recon_frames.py -- both share the same
`<datdir>/recon/smaps_<seqname>_sigpy.h5` cache, so whichever stage runs
first pays the (one-time) ESPIRiT cost.

b0map.jl itself estimates entirely on the deGRE acquisition grid (that's
where the dual-echo images/phase data live). This driver then resizes that
result onto the *EPI* acquisition grid -- the shape/FOV a B0-corrected EPI
reconstruction (recon/) actually needs -- via grid_resize.py's
resize_to_epi_grid, the same crop+resize routine smaps.py's process_smaps
already uses to move ESPIRiT coil maps from the deGRE grid to the EPI grid.
`<seqname>_b0map.h5`'s 'b0map_hz'/'mask' keys hold the EPI-grid result (the
primary consumable); the native deGRE-grid arrays are kept alongside under
a '_degre' suffix for diagnostic/QC use.
"""

import os
import shutil
import subprocess

import h5py
import numpy as np

from preprocessing.config import PreprocessingConfig, load_config, load_seq_params, set_seq_paths
from preprocessing.grid_resize import resize_to_epi_grid
from preprocessing.nifti_io import save_recon_nifti
from preprocessing.smaps import load_smaps

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

        paths = set_seq_paths(cfg, seqname)
        seq_params = load_seq_params(paths)

        # Best-effort: ensures <datdir>/recon/smaps_<seqname>_sigpy.h5
        # exists with a deGRE-grid smaps_degre/emap_degre pair (estimating
        # fresh via ESPIRiT if this is the first stage to need them) -- see
        # smaps.load_smaps's docstring and b0map.jl's module docstring. Not
        # a hard requirement: run_b0map() only ever depended on
        # `ksp_gre_echoes` before this, and load_smaps additionally needs
        # `ksp_gre` (preprocess.py's STEP 2 always writes both together, so
        # this should never actually miss on real data) -- fall back to
        # b0map.jl's pre-existing no-smap behavior rather than failing the
        # whole field-map estimation over a missing/failed optional input.
        smaps_path = os.path.join(cfg.datdir, 'recon', f'smaps_{seqname}_sigpy.h5')
        try:
            load_smaps(cfg, paths, seq_params)
        except Exception as e:  # optional input, degrade gracefully
            print(f'WARNING [{seqname}]: could not load/estimate sensitivity maps ({e}) -- '
                  'falling back to no smap.')
            smaps_path = ''

        try:
            subprocess.run(
                [
                    julia_bin, f'--project={_JULIA_DIR}', _JULIA_SCRIPT,
                    gre_cache_path, output_path, smaps_path, str(cfg.threshold_mask),
                    str(cfg.b0map_mask_thresh), cfg.b0map_precon,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:  # mirrors the sibling batch drivers' try/catch
            print(f"ERROR [{seqname}]: {e}\nSkipping...")
            continue

        with h5py.File(output_path, 'r') as f:
            b0map_hz_degre = f['b0map_hz'][()]
            mask_degre = f['mask'][()].astype(bool)

        # b0map.jl (above) estimates on the deGRE grid -- that's where the
        # dual-echo images/phase data actually live -- but a B0-corrected
        # EPI reconstruction needs the field map on the *EPI* grid. Zero
        # outside the fitted mask before resizing (mirrors process_smaps'
        # own mask-before-crop/resize step in smaps.py) so cubic-spline
        # interpolation doesn't blend in MRIFieldmaps' embed() fill value
        # for unfit background voxels near the mask boundary; the mask
        # itself is resized with order=0 (nearest) so no fractional/
        # invented mask values appear. See grid_resize.py for the shared
        # crop+resize routine (same one process_smaps uses for coil maps).
        n_target = (seq_params.Nx, seq_params.Ny, seq_params.Nz)
        b0map_hz = resize_to_epi_grid(
            b0map_hz_degre * mask_degre, seq_params.fov_degre, seq_params.fov,
            n_target, order=3,
        ).astype(np.float32)
        mask = resize_to_epi_grid(
            mask_degre, seq_params.fov_degre, seq_params.fov, n_target, order=0,
        ).astype(bool)

        # Keep the native deGRE-grid arrays too (diagnostic/QC use, e.g.
        # comparing against the GRE magnitude image, which lives on that
        # grid) under an explicit '_degre' suffix; 'b0map_hz'/'mask' become
        # the EPI-grid versions -- the primary consumable, matching
        # smaps_<seqname>_sigpy.h5's 'smaps_raw' (small grid) vs. 'smaps'
        # (EPI grid) convention. finit_hz stays deGRE-grid only -- it's a
        # pure NCG-initialization diagnostic, never consumed downstream.
        with h5py.File(output_path, 'a') as f:
            f.move('b0map_hz', 'b0map_hz_degre')
            f.move('mask', 'mask_degre')
            f.create_dataset('b0map_hz', data=b0map_hz)
            f.create_dataset('mask', data=mask)

        # NIfTI export for viewing in FSLeyes/etc., now on the EPI grid/fov
        # like every other NIfTI this pipeline writes (see nifti_io module
        # docstring).
        save_recon_nifti(
            output_path[: -len('.h5')], b0map_hz,
            fov=seq_params.fov, seqname=seqname,
            mask_threshold=cfg.b0map_mask_thresh,
        )
    print('\nBatch complete.')


if __name__ == '__main__':
    cfg = load_config(datdir='/path/to/data/', seqnames=['caipi_ts'])
    run_b0map(cfg)
