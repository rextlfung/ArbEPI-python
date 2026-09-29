"""B0 field map from the dual-echo deGRE, via MRIFieldmaps.jl (Lin & Fessler,
IEEE TCI 2020) in julia/b0map.jl, run as a subprocess.

MRIFieldmaps.jl has no Python port, so julia/ is a small self-contained Julia
project (pinned Project.toml + Manifest.toml). First-time setup:
    julia --project=preprocess/julia -e 'import Pkg; Pkg.instantiate()'

b0map.jl estimates on the deGRE grid; resize_to_epi moves the result onto the
EPI grid, zeroing unfit voxels first so the cubic spline doesn't blend in
background. See b0map.jl's header for its choices (ROMEO-unwrapped
initialization, :diag preconditioner, the mandatory fit mask).
"""

import os
import shutil
import subprocess
import tempfile

import h5py
import numpy as np

from preprocess.grid_resize import resize_to_epi_grid

JULIA_DIR = os.path.join(os.path.dirname(__file__), 'julia')
JULIA_SCRIPT = os.path.join(JULIA_DIR, 'b0map.jl')


def julia_available() -> bool:
    return shutil.which('julia') is not None


def fit_mask(
    img_echo1: np.ndarray, mask_thresh: float, emap: np.ndarray | None = None, crop: float = 0.95
) -> np.ndarray:
    """The voxel mask b0map.jl fits within: first-echo RSS magnitude above
    mask_thresh x its peak (MRIFieldmaps' own b0init default is 0.1), ANDed
    with ESPIRiT's eigenvalue map above `crop` when given. Python copy of the
    rule, for when julia isn't run."""
    mask = img_echo1 > mask_thresh * img_echo1.max()
    if emap is not None:
        mask &= emap > crop
    return mask


def estimate_b0map(
    ksp_echoes: np.ndarray,
    te: np.ndarray,
    smaps_degre: np.ndarray | None = None,
    emap_degre: np.ndarray | None = None,
    crop: float = 0.95,
    mask_thresh: float = 0.1,
    precon: str = 'diag',
) -> dict[str, np.ndarray]:
    """Run julia/b0map.jl on deGRE k-space.

    ksp_echoes: [Nx, Ny, Nz, n_echoes, Nc] whitened deGRE k-space.
    te: [n_echoes] echo times (s).
    smaps_degre, emap_degre: ESPIRiT maps on the same grid and in the same coil
        basis as ksp_echoes ([Nx, Ny, Nz, Nc] and [Nx, Ny, Nz]). With them,
        b0map.jl combines coils with the maps (matched filter) and tightens the
        fit mask with emap > crop; without, it falls back to a phase-contrast
        combine and a magnitude-only mask.
    Returns deGRE-grid 'b0_map', 'finit_hz' (the ROMEO-unwrapped start) and
    'mask' (bool).
    """
    if not julia_available():
        raise RuntimeError(
            'estimate_b0map: no `julia` executable on PATH -- install it '
            '(https://julialang.org/install/) or set estimate_b0=False.'
        )
    with tempfile.TemporaryDirectory() as tmp:
        fn_in = os.path.join(tmp, 'gre.h5')
        fn_out = os.path.join(tmp, 'b0map.h5')
        with h5py.File(fn_in, 'w') as f:
            f.create_dataset('ksp_gre_echoes', data=ksp_echoes.astype(np.complex64))
            f.attrs['TE_degre'] = np.asarray(te, dtype=np.float64)
            if smaps_degre is not None:
                f.create_dataset('smaps_degre', data=smaps_degre.astype(np.complex64))
                f.create_dataset('emap_degre', data=emap_degre.astype(np.float32))
        subprocess.run(
            [
                shutil.which('julia'), f'--project={JULIA_DIR}', JULIA_SCRIPT,
                fn_in, fn_out, fn_in if smaps_degre is not None else '',
                str(crop), str(mask_thresh), precon,
            ],
            check=True,
        )
        with h5py.File(fn_out, 'r') as f:
            return {
                'b0_map': f['b0_map'][()],
                'finit_hz': f['finit_hz'][()],
                'mask': f['mask'][()].astype(bool),
            }


def resize_to_epi(
    b0_map: np.ndarray,
    mask: np.ndarray,
    fov_degre: tuple[float, float, float],
    fov: tuple[float, float, float],
    n_target: tuple[int, int, int],
    zero_pad_z: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """(b0_map, mask) on the EPI grid. The field map is zeroed outside the fit
    mask before the cubic-spline resize, and the mask is resized with nearest
    neighbor so it stays binary."""
    b0 = resize_to_epi_grid(
        b0_map * mask, fov_degre, fov, n_target, order=3, zero_pad_z=zero_pad_z
    ).astype(np.float32)
    m = resize_to_epi_grid(mask, fov_degre, fov, n_target, order=0, zero_pad_z=zero_pad_z) > 0.5
    return b0, m
