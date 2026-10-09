"""Move a volume from the deGRE grid onto the EPI grid.

Sensitivity, B0 and R2* maps are all estimated on the deGRE grid and needed
on the EPI grid. The two acquisitions share an isocenter, and the deGRE FOV
covers the EPI FOV on every axis (params.py sizes it that way: at res_degre
spacing the FOV can only equal the EPI's when it divides evenly, so it is
usually larger). Each EPI voxel center is mapped to its continuous position
on the deGRE grid and the volume is interpolated there, which crops and
resamples in one step, on all three axes.

Voxel convention: centered FFT. Voxel i of an N-voxel axis sits at
(i - N // 2) * FOV/N, i.e. voxel N // 2 is isocenter, which is where
utils.ift3c (deGRE) and the EPI gridding plus recon's SENSE put it. For odd N
that is the FOV-centered grid; for even N it is half a voxel off it. An
earlier version assumed edge-aligned grids (voxel centers at
(i + 0.5) FOV/N - FOV/2, scipy.ndimage.zoom(grid_mode=True)'s convention),
which placed maps up to half an EPI voxel from the EPI's object: 1.2 mm in z
and 0.3 mm in x and y at the default protocol (review item 263). The crop is
exact, not whole deGRE voxels (item 258). mode='nearest'
holds the edge value where the target grid extends past the outermost
source voxel centers.
"""

import numpy as np
from scipy import ndimage


def resize_to_epi_grid(
    vol: np.ndarray,
    fov_src: tuple[float, float, float],
    fov: tuple[float, float, float],
    n_target: tuple[int, int, int],
    order: int = 3,
    zero_pad_z: bool = False,
) -> np.ndarray:
    """vol [Nx_src, Ny_src, Nz_src, ...] -> [Nx, Ny, Nz, ...] on the target
    grid n_target with FOV `fov` (m). Trailing axes pass through. The source
    FOV must cover the target FOV on every axis (both centered on isocenter).

    order: spline order; 3 (cubic, like MATLAB imresize3) for maps, 0 for
        masks so they stay binary.
    zero_pad_z: if the target z-FOV is larger than the source's (normally an
        error), zero the target slices the source doesn't fully cover (rounded
        inward, so a partly covered slice gets zero) instead of raising
        (review items 196, 203: an EPI resolution whose rounded z-FOV slightly
        exceeds the deGRE slab).
    """
    fov_src = np.asarray(fov_src, dtype=np.float64)
    fov = np.asarray(fov, dtype=np.float64)
    n_src = np.array(vol.shape[:3])
    n_tgt = np.array(n_target)
    short = fov_src < fov * (1 - 1e-6)
    if short[:2].any() or (short[2] and not zero_pad_z):
        axis = 'xyz'[int(np.argmax(short))]
        raise ValueError(
            f'resize_to_epi_grid: target {axis}-FOV ({fov[int(np.argmax(short))]:.4f} m) '
            f'exceeds source {axis}-FOV ({fov_src[int(np.argmax(short))]:.4f} m); '
            f'the deGRE must cover the EPI FOV (z only: pass zero_pad_z=True).'
        )

    # Target index j -> source continuous index s = scale * j + offset. Voxel i
    # sits at (i - N // 2) d (the centered-FFT convention), so
    # (j - Ne // 2) d_tgt = (s - Ns // 2) d_src.
    d_src, d_tgt = fov_src / n_src, fov / n_tgt
    scale = d_tgt / d_src
    offset = n_src // 2 - (n_tgt // 2) * scale
    extra = vol.ndim - 3
    matrix = np.concatenate([scale, np.ones(extra)])
    offsets = np.concatenate([offset, np.zeros(extra)])
    out_shape = tuple(n_tgt) + vol.shape[3:]
    kwargs = dict(matrix=matrix, offset=offsets, output_shape=out_shape, order=order,
                  mode='nearest')
    if np.iscomplexobj(vol):
        out = (ndimage.affine_transform(vol.real, **kwargs)
               + 1j * ndimage.affine_transform(vol.imag, **kwargs))
    else:
        out = ndimage.affine_transform(vol.astype(np.float64), **kwargs)

    if short[2]:
        # zero target slices not fully inside the source (edge to edge)
        s_c = scale[2] * np.arange(n_tgt[2]) + offset[2]
        covered = ((s_c - 0.5 * scale[2] >= -0.5 - 1e-9)
                   & (s_c + 0.5 * scale[2] <= n_src[2] - 0.5 + 1e-9))
        if not covered.any():
            raise ValueError('resize_to_epi_grid: zero-pad inner target range is empty.')
        out[:, :, ~covered] = 0
    return out
