"""B0 field map from the dual-echo deGRE, via MRIFieldmaps.jl (Lin & Fessler,
IEEE TCI 2020) in julia/b0map.jl, run as a subprocess.

MRIFieldmaps.jl has no Python port, so julia/ is a small self-contained Julia
project (pinned Project.toml + Manifest.toml). First-time setup:
    julia --project=preprocess/julia -e 'import Pkg; Pkg.instantiate()'

b0map.jl estimates on the deGRE grid, within a fit mask (see its header for
its choices: ROMEO-unwrapped initialization, :diag preconditioner, the
mandatory fit mask and its small-component removal), and returns 0 Hz outside
it. resize_to_epi replaces those zeros with the harmonic extension of the
fitted field (extend_harmonic) before moving the map onto the EPI grid. The
recon applies b0_map wherever the sensitivity maps are nonzero, which reaches
past the fit mask (ESPIRiT's eigenvalue mask alone vs. that AND 10% of peak
magnitude; 18% more voxels on 20260930ballfat), so the value there matters: a
0 Hz background next to a fitted edge at tens of Hz is a field step the
forward model takes literally, and with laminar ordering (echo time linear in
ky) it is a fold along y that left the B0-SENSE inverse nearly singular there
(review item 261).
"""

import os
import shutil
import subprocess
import tempfile

import h5py
import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.linalg import cg

from preprocess.grid_resize import resize_to_epi_grid

JULIA_DIR = os.path.join(os.path.dirname(__file__), 'julia')
JULIA_SCRIPT = os.path.join(JULIA_DIR, 'b0map.jl')


def julia_available() -> bool:
    return shutil.which('julia') is not None


def drop_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """mask without its 6-connected components smaller than min_size voxels."""
    if min_size <= 1 or not mask.any():
        return mask.copy()
    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def fit_mask(
    img_echo1: np.ndarray,
    mask_thresh: float,
    emap: np.ndarray | None = None,
    crop: float = 0.95,
    min_component: int = 64,
) -> np.ndarray:
    """The voxel mask b0map.jl fits within: first-echo RSS magnitude above
    mask_thresh x its peak (MRIFieldmaps' own b0init default is 0.1), ANDed
    with ESPIRiT's eigenvalue map above `crop` when given, without components
    smaller than min_component voxels. Python copy of the rule, for when
    julia isn't run."""
    mask = img_echo1 > mask_thresh * img_echo1.max()
    if emap is not None:
        mask &= emap > crop
    return drop_small_components(mask, min_component)


def estimate_b0map(
    ksp_echoes: np.ndarray,
    te: np.ndarray,
    smaps_degre: np.ndarray | None = None,
    emap_degre: np.ndarray | None = None,
    crop: float = 0.95,
    mask_thresh: float = 0.1,
    precon: str = 'diag',
    min_component: int = 64,
    max_wraps: float = 2.0,
) -> dict[str, np.ndarray]:
    """Run julia/b0map.jl on deGRE k-space.

    ksp_echoes: [Nx, Ny, Nz, n_echoes, Nc] whitened deGRE k-space.
    te: [n_echoes] echo times (s).
    smaps_degre, emap_degre: ESPIRiT maps on the same grid and in the same coil
        basis as ksp_echoes ([Nx, Ny, Nz, Nc] and [Nx, Ny, Nz]). With them,
        b0map.jl combines coils with the maps (matched filter) and tightens the
        fit mask with emap > crop; without, it falls back to a phase-contrast
        combine and a magnitude-only mask.
    min_component: mask components smaller than this many voxels are dropped
        before the fit (isolated voxels can diverge; see b0map.jl's header).
    max_wraps: safety net against a diverged fit. Voxels the fit moved more
        than max_wraps phase wraps (max_wraps / dTE Hz) from finit are reset
        to finit. Not 0.5: the fit legitimately moves voxels by about one wrap
        where ROMEO unwrapped them wrong (measured on 20260922xiaokai and
        20260929ballfat: up to 566 Hz = 1.26 wraps, every such voxel agreeing
        with its neighbours to ~10-50 Hz while finit was a wrap off), whereas
        diverged voxels moved 16 to 10^4 wraps (review item 259).
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
                str(crop), str(mask_thresh), precon, str(int(min_component)),
            ],
            check=True,
        )
        with h5py.File(fn_out, 'r') as f:
            b0, finit = f['b0_map'][()], f['finit_hz'][()]
            mask = f['mask'][()].astype(bool)
    return {'b0_map': reset_diverged(b0, finit, mask, te, max_wraps), 'finit_hz': finit,
            'mask': mask}


def reset_diverged(
    b0: np.ndarray, finit: np.ndarray, mask: np.ndarray, te: np.ndarray, max_wraps: float = 2.0
) -> np.ndarray:
    """b0 with the mask voxels that moved more than max_wraps phase wraps
    (max_wraps / dTE Hz) from finit reset to finit (see estimate_b0map)."""
    te = np.asarray(te, dtype=np.float64)
    diverged = mask & (np.abs(b0 - finit) > max_wraps / abs(te[1] - te[0]))
    if not diverged.any():
        return b0
    print(f'  B0 fit: reset {int(diverged.sum())} voxel(s) that moved > {max_wraps:g} '
          f'wraps from the ROMEO start back to it')
    return np.where(diverged, finit, b0)


def extend_harmonic(f: np.ndarray, mask: np.ndarray, rtol: float = 1e-8) -> np.ndarray:
    """f with its values outside `mask` replaced by the discrete harmonic
    extension of f[mask]: the 7-point Laplacian is zero at every voxel outside
    the mask, with f fixed inside it and zero normal derivative at the volume
    boundary. The result is continuous across the mask edge and stays within
    the range of f[mask] (discrete maximum principle; on this grid -- a later
    cubic resize can still overshoot slightly), so it adds no field steps and
    no new extremes. Away from its sources the susceptibility field is
    harmonic too, though here the extension is a smoothness prior, not a
    physical model: it also fills low-signal object voxels.

    Nearest-value fill was the alternative considered: on 20260930ballfat it
    removed the mask-edge hotspots equally well, but its steps between
    neighbouring fill regions made the cubic resize overshoot the fitted range
    (-259..+122 Hz from a -218..+92 Hz fit), where this stays inside it.

    Solved by Jacobi-preconditioned CG (the system is a Dirichlet graph
    Laplacian, symmetric positive definite whenever the mask is nonempty):
    7-16 s on a 72x72x51 deGRE grid with 177k unfit voxels, vs 51 s for a
    sparse direct solve. An empty mask gives zeros.
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.where(mask, f, 0).astype(np.float64)
    if mask.all() or not mask.any():
        return out
    idx = np.arange(mask.size).reshape(mask.shape)
    pairs = []
    for ax in range(mask.ndim):
        lo = [slice(None)] * mask.ndim
        hi = [slice(None)] * mask.ndim
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        pairs.append((idx[tuple(lo)].ravel(), idx[tuple(hi)].ravel()))
    p = np.concatenate([a for a, b in pairs] + [b for a, b in pairs])
    q = np.concatenate([b for a, b in pairs] + [a for a, b in pairs])
    adj = sp.csr_matrix((np.ones(p.size), (p, q)), shape=(mask.size, mask.size))
    known = mask.ravel()
    unknown = ~known
    degree = np.asarray(adj.sum(axis=1)).ravel()[unknown]
    adj_u = adj[unknown]
    A = (sp.diags(degree) - adj_u[:, unknown]).tocsr()
    b = adj_u[:, known] @ out.ravel()[known]
    x, info = cg(A, b, rtol=rtol, maxiter=20 * max(mask.shape) ** 2,
                 M=sp.diags(1 / degree))
    if info != 0:
        raise RuntimeError(f'extend_harmonic: CG did not converge (info={info})')
    out.ravel()[unknown] = x
    return out


def resize_to_epi(
    b0_map: np.ndarray,
    mask: np.ndarray,
    fov_degre: tuple[float, float, float],
    fov: tuple[float, float, float],
    n_target: tuple[int, int, int],
    zero_pad_z: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """(b0_map, mask) on the EPI grid. The field map outside the fit mask is
    replaced by the harmonic extension of the fit (extend_harmonic; b0map.jl
    leaves 0 Hz there) before the cubic-spline resize, so there is no field
    step at the mask edge to blend. The mask is resized with nearest neighbor
    so it stays binary, and still marks the fitted voxels."""
    b0 = resize_to_epi_grid(
        extend_harmonic(b0_map, mask), fov_degre, fov, n_target, order=3,
        zero_pad_z=zero_pad_z,
    ).astype(np.float32)
    m = resize_to_epi_grid(mask, fov_degre, fov, n_target, order=0, zero_pad_z=zero_pad_z) > 0.5
    return b0, m
