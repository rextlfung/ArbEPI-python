"""Brain mask from the T1-weighted deGRE, applied to a (motion-corrected) EPI series.

The EPI mean has coil shading and no skull contrast, which is where `bet` fails (4184 cm3
on a 2.4 mm run); the deGRE's first echo has real anatomical contrast and is on the EPI
grid (preprocess/grid_resize.py), so the mask is estimated there and used on the EPI voxels
directly. The deGRE is acquired before the EPI, so head motion between the two would shift
the mask; `register_mask` removes it by rigidly registering the skull-stripped deGRE to the
mean of the motion-corrected EPI with FSL `flirt` and moving the mask with that transform.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import h5py
import numpy as np
from numpy.typing import NDArray

from analyze.mask import brain_mask, find_fsl_tool, fsl_affine, fsl_env


def load_gre(preprocessed: str, echo: int = 0):
    """(deGRE echo image (Nx, Ny, Nz) float32, voxel size mm) from a preprocess/ output file."""
    with h5py.File(preprocessed, "r") as f:
        gre = f["degre/img_echoes"][..., echo].astype(np.float32)
        vox = tuple(1e3 * float(fov) / n for fov, n in zip(f.attrs["fov"], gre.shape))
    return gre, vox


def gre_brain_mask(preprocessed: str, method: str = "auto", frac: float = 0.5, echo: int = 0):
    """(mask (Nx, Ny, Nz) bool, voxel size mm) from `degre/img_echoes[..., echo]` of a
    preprocess/ output file. method and frac are analyze.mask.brain_mask's."""
    gre, vox = load_gre(preprocessed, echo)
    return brain_mask(gre, vox, method=method, frac=frac), vox


def find_flirt() -> str | None:
    """Path of FSL's flirt (see motion.find_fsl_tool)."""
    return find_fsl_tool("flirt")


def register_mask(gre: NDArray, mask: NDArray, epi_mean: NDArray, voxel_size_mm,
                  max_rot_deg: float = 15.0):
    """(mask in the EPI's frame (bool), 4x4 FSL matrix in mm, GRE -> EPI) from a rigid (6 dof)
    flirt registration of the skull-stripped `gre` to `epi_mean` (normalized mutual
    information; both on the same grid), applied to `mask` with nearest-neighbour
    interpolation. The search is limited to +-max_rot_deg, since the head barely moves
    between the scans."""
    import nibabel as nib

    flirt = find_flirt()
    if flirt is None:
        raise RuntimeError("register_mask needs FSL's flirt (PATH, $FSLDIR/bin or ~/fsl/bin)")
    affine = fsl_affine(voxel_size_mm)
    env = fsl_env(flirt)
    with tempfile.TemporaryDirectory() as d:
        p = lambda n: os.path.join(d, n)  # noqa: E731
        for name, vol in (("gre", gre * mask), ("epi", epi_mean), ("mask", mask)):
            nib.save(nib.Nifti1Image(np.asarray(vol, np.float32), affine), p(name + ".nii.gz"))
        rng = [f"{-max_rot_deg:g}", f"{max_rot_deg:g}"]
        subprocess.run([flirt, "-in", p("gre"), "-ref", p("epi"), "-omat", p("m.mat"), "-out", p("o"),
                        "-dof", "6", "-cost", "normmi", "-searchrx", *rng, "-searchry", *rng,
                        "-searchrz", *rng], check=True, env=env, capture_output=True)
        subprocess.run([flirt, "-in", p("mask"), "-ref", p("epi"), "-applyxfm", "-init", p("m.mat"),
                        "-interp", "nearestneighbour", "-out", p("mo")], check=True, env=env, capture_output=True)
        reg = np.asanyarray(nib.load(p("mo.nii.gz")).dataobj) > 0
        return reg, np.loadtxt(p("m.mat"))


def mask_shift_mm(xfm: NDArray, mask: NDArray, voxel_size_mm) -> float:
    """How far `xfm` (register_mask's FSL matrix, voxel * size coordinates) moves the mask's
    centre of mass, in mm. The matrix's translation alone is not that: it is measured from the
    grid origin, so any rotation adds to it."""
    c = np.array(np.nonzero(mask)).mean(axis=1) * np.asarray(voxel_size_mm)
    return float(np.linalg.norm(xfm[:3, :3] @ c + xfm[:3, 3] - c))


def apply_brain_mask(series: NDArray, mask: NDArray) -> NDArray:
    """Zero every voxel outside `mask` in a (Nx, Ny, Nz, Nt[, Ns]) array."""
    if series.shape[:3] != mask.shape:
        raise ValueError(f"mask {mask.shape} does not match the series {series.shape[:3]}")
    return series * mask.reshape(mask.shape + (1,) * (series.ndim - 3))
