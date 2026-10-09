"""Brain mask of a 4-D series, by skull stripping its temporal mean.

method 'auto' uses FSL's `bet` when it is on PATH, else `brainextractor`, a
pure-Python re-implementation of the same algorithm (Smith, Hum Brain Mapp
2002), which is not voxel-identical to FSL's. A mask can also be passed in, which
is the way to go on a phantom (no skull for BET to find) or when a better mask
exists already.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
from numpy.typing import NDArray


def mean_image(series: NDArray) -> NDArray[np.float32]:
    """Temporal mean of a (Nx, Ny, Nz, Nt) magnitude series."""
    return np.asarray(series, dtype=np.float32).mean(axis=-1)


def _nifti(vol: NDArray, voxel_size_mm: tuple[float, float, float]):
    import nibabel as nib

    return nib.Nifti1Image(np.asarray(vol, dtype=np.float32), np.diag([*voxel_size_mm, 1.0]))


def bet_fsl(mean_vol: NDArray, voxel_size_mm, frac: float = 0.5) -> NDArray[np.bool_]:
    """FSL `bet` on the mean volume (`-m`, binary mask). Needs `bet` on PATH."""
    import nibabel as nib

    with tempfile.TemporaryDirectory() as d:
        nib.save(_nifti(mean_vol, voxel_size_mm), os.path.join(d, "mean.nii.gz"))
        env = dict(os.environ, FSLOUTPUTTYPE="NIFTI_GZ")
        subprocess.run(["bet", os.path.join(d, "mean.nii.gz"), os.path.join(d, "brain"),
                        "-m", "-f", str(frac)], check=True, env=env, capture_output=True)
        return np.asanyarray(nib.load(os.path.join(d, "brain_mask.nii.gz")).dataobj) > 0


def bet_python(mean_vol: NDArray, voxel_size_mm, frac: float = 0.5) -> NDArray[np.bool_]:
    """brainextractor (pure-Python BET) on the mean volume."""
    try:
        from brainextractor import BrainExtractor
    except ImportError as e:
        raise ImportError("brain extraction needs `brainextractor` "
                          "(uv sync --extra analyze) or FSL's bet on PATH, "
                          "or pass a mask") from e
    be = BrainExtractor(img=_nifti(mean_vol, voxel_size_mm), bt=frac)
    be.run()
    return np.asarray(be.compute_mask()) > 0


def brain_mask(series_or_mean: NDArray, voxel_size_mm=(1.0, 1.0, 1.0), method: str = "auto",
               frac: float = 0.5) -> NDArray[np.bool_]:
    """Binary brain mask (Nx, Ny, Nz). method: 'auto', 'fsl', 'python' or
    'threshold' (voxels above 20% of the 98th percentile; no skull stripping).
    frac: BET's fractional intensity threshold (smaller -> larger brain)."""
    vol = series_or_mean if series_or_mean.ndim == 3 else mean_image(series_or_mean)
    if method == "auto":
        method = "fsl" if shutil.which("bet") else "python"
    if method == "fsl":
        return bet_fsl(vol, voxel_size_mm, frac)
    if method == "python":
        return bet_python(vol, voxel_size_mm, frac)
    if method == "threshold":
        return vol > 0.2 * np.percentile(vol, 98)
    raise ValueError(f"unknown mask method {method!r}")


def load_mask(path: str) -> NDArray[np.bool_]:
    """A mask from a NIfTI file (any nonzero voxel)."""
    import nibabel as nib

    return np.asanyarray(nib.load(path).dataobj) > 0
