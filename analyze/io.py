"""Reading `recon/` output and writing statistic maps.

A reconstruction file (`recon.utils.save_result`'s `<name>_recon.h5`) holds the
complex image `X_recon` (Nx, Ny, Nz, Nt) and, for `--reg mslr`, its per-scale
components `X` (Nx, Ny, Nz, Nt, Nscales) with X_recon = sum over scales.

What the GLM is run on, per `scale`:
- 'sum': |X_recon|, the magnitude image an analysis normally uses.
- an int s: the component X[..., s] projected on the phase of the summed image's
  temporal mean, Re(X_s conj(phi)), phi = unit phasor of mean_t X_recon. A
  component is not a magnitude image: the dynamic (local) scale has nearly zero
  mean, and |X_s| would rectify its fluctuations and make a decrease look like
  an increase. The projection is linear, so the components' series add up to
  the summed image's wherever its phase is stable over time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import h5py
import numpy as np
from numpy.typing import NDArray

_FRAME_BLOCK = 16  # frames per read; the datasets are chunked along time


@dataclass
class Recon:
    series: dict[str, NDArray]  # label -> (Nx, Ny, Nz, Nt) float32, discard applied
    mean_signal: NDArray  # (Nx, Ny, Nz) mean of |X_recon|: the percent-signal-change base
    voxel_size_mm: tuple[float, float, float]
    n_scales: int


def _read_t(ds: h5py.Dataset, frames: slice) -> NDArray:
    """ds[..., frames] along the time axis (axis 3), a block of frames at a time."""
    idx = range(*frames.indices(ds.shape[3]))
    out = np.empty(ds.shape[:3] + (len(idx),) + ds.shape[4:], dtype=ds.dtype)
    for s in range(0, len(idx), _FRAME_BLOCK):
        blk = list(idx[s : s + _FRAME_BLOCK])
        out[:, :, :, s : s + len(blk)] = ds[:, :, :, blk[0] : blk[-1] + 1]
    return out


def voxel_size_from_sidecar(path_h5: str) -> tuple[float, float, float] | None:
    """Voxel size (mm) from the `<name>_recon.json` that recon.utils.save_result
    writes next to the .h5 (its `fov`, in m, over the image shape), or None."""
    fn = os.path.splitext(path_h5)[0] + ".json"
    if not os.path.exists(fn):
        return None
    with open(fn) as f:
        fov = json.load(f).get("fov")
    if fov is None:
        return None
    with h5py.File(path_h5, "r") as h:
        shape = h["X_recon"].shape[:3]
    return tuple(1000.0 * float(f) / n for f, n in zip(fov, shape))


def read_recon(path: str, scales: tuple = ("sum",), n_discard: int = 0,
               voxel_size_mm: tuple[float, float, float] | None = None) -> Recon:
    """Load the series to analyze. scales: 'sum', 'all' (the sum and every
    component), or any mix of 'sum' and component indices; labels are 'sum' and
    'scale<i>'."""
    with h5py.File(path, "r") as f:
        frames = slice(n_discard, None)
        X_recon = _read_t(f["X_recon"], frames)
        has_scales = "X" in f and f["X"].ndim == 5
        n_scales = f["X"].shape[4] if has_scales else 1
        want = list(scales)
        if "all" in want:
            want = ["sum", *range(n_scales)]
        series: dict[str, NDArray] = {}
        phi = comps = None
        for s in want:
            if s == "sum":
                series["sum"] = np.abs(X_recon).astype(np.float32)
                continue
            if not has_scales:
                raise ValueError(f"{path} has no per-scale components (not an mslr run)")
            s = int(s)
            if not 0 <= s < n_scales:
                raise ValueError(f"scale {s} out of range for {n_scales} scales")
            if phi is None:
                m = X_recon.mean(axis=3)
                phi = m / np.maximum(np.abs(m), 1e-30)
            if comps is None:
                comps = _read_t(f["X"], frames)
            series[f"scale{s}"] = (comps[..., s] * np.conj(phi)[..., None]).real.astype(np.float32)
    mean_signal = np.abs(X_recon).mean(axis=3).astype(np.float32)
    if voxel_size_mm is None:
        voxel_size_mm = voxel_size_from_sidecar(path) or (1.0, 1.0, 1.0)
    return Recon(series=series, mean_signal=mean_signal, voxel_size_mm=tuple(voxel_size_mm),
                 n_scales=n_scales)


def save_stat_nifti(fn: str, vol: NDArray, voxel_size_mm, **attrs) -> None:
    """Write `fn` (.nii.gz) and `<fn minus .nii.gz>.json` with `attrs`. Unlike
    preprocess.utils.save_recon_nifti this keeps the sign (t and z maps are
    signed). The affine is diagonal, as there: spacing, not orientation."""
    import nibabel as nib

    affine = np.diag([*voxel_size_mm, 1.0])
    nib.save(nib.Nifti1Image(np.asarray(vol, dtype=np.float32), affine), fn)
    base = fn[: -len(".nii.gz")] if fn.endswith(".nii.gz") else os.path.splitext(fn)[0]
    with open(base + ".json", "w") as f:
        json.dump(attrs, f, indent=2,
                  default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
