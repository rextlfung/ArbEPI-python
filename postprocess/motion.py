"""Inter-frame rigid-body motion correction of a reconstructed series.

Registers every frame of a (Nx, Ny, Nz, Nt) series to a reference frame with six
degrees of freedom (three rotations, three translations), then resamples. Two
backends estimate the motion:

- 'fsl': FSL's MCFLIRT (Jenkinson et al., NeuroImage 2002), when `mcflirt` is
  on PATH, under $FSLDIR/bin or under ~/fsl/bin. Only the transforms are taken
  from it; resampling is this module's, so both backends are applied the same
  way (and complex data, which MCFLIRT cannot read, is handled).
- 'python': a small multi-resolution least-squares registration on numpy/scipy,
  so a plain `uv sync --extra analyze` environment needs no FSL.

'auto' picks 'fsl' when it can find `mcflirt`, like `analyze.mask`'s `bet`.

Conventions (shared by both backends, so their outputs can be compared):

- A transform is a 4x4 matrix T that maps a point of the moving frame to the
  point of the reference it coincides with, in *scaled voxel coordinates*
  (millimeters from voxel index (0, 0, 0), no orientation: the affine this repo
  writes everywhere is diagonal). The corrected frame is g(T^-1 r).
- `params` are (rx, ry, rz [rad], tx, ty, tz [mm]) of T about the volume's
  center: T(x) = R (x - c) + c + t. These are the numbers to plot as a motion
  trace; MCFLIRT's own .par is about a different origin, so the translations
  differ from it by R c - c.

Motion is estimated on magnitudes; a complex series is resampled as real and
imaginary parts, which keeps the phase (B0 and coil phase are smooth).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation

from analyze.mask import find_fsl_tool, fsl_affine, fsl_env

_PYRAMID = (4, 2, 1)  # downsampling factors, coarse to fine
_MAX_THREADS = 16  # apply_transforms: each holds a few frame-sized temporaries
_MIN_DIM = 12  # a pyramid level whose smallest side would be below this is skipped


@dataclass
class MotionResult:
    series: NDArray  # corrected series, same shape and dtype as the input
    matrices: NDArray  # (Nt, 4, 4) float64, see the module docstring
    params: NDArray  # (Nt, 6): rx, ry, rz [rad], tx, ty, tz [mm], about the volume center
    ref: NDArray  # (Nx, Ny, Nz) reference magnitude the frames were registered to
    method: str  # 'fsl' or 'python'

    @property
    def framewise_displacement_mm(self) -> NDArray:
        """Power et al. (2012) framewise displacement: |d translation| plus
        |d rotation| times a 50 mm head radius, per frame (0 for the first)."""
        d = np.diff(self.params, axis=0, prepend=self.params[:1])
        return np.abs(d[:, 3:]).sum(1) + 50.0 * np.abs(d[:, :3]).sum(1)


# ---------------------------------------------------------------- transforms


def _center_mm(shape, vox) -> NDArray:
    return (np.asarray(shape, float) - 1.0) / 2.0 * np.asarray(vox, float)


def params_to_matrix(p, center_mm) -> NDArray:
    """4x4 T for p = (rx, ry, rz, tx, ty, tz) about `center_mm`."""
    R = Rotation.from_rotvec(p[:3]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center_mm - R @ center_mm + np.asarray(p[3:])
    return T


def matrix_to_params(T, center_mm) -> NDArray:
    R = T[:3, :3]
    t = T[:3, 3] - (center_mm - R @ center_mm)
    return np.concatenate([Rotation.from_matrix(R).as_rotvec(), t])


def _warp(vol: NDArray, T: NDArray, vox, order: int = 1, out_shape=None) -> NDArray:
    """vol resampled so that it lies on the reference: out(r) = vol(T^-1 r)."""
    vox = np.asarray(vox, float)
    Ti = np.linalg.inv(T)
    D = np.diag(vox)
    S = np.diag(1.0 / vox)
    return ndimage.affine_transform(
        vol, S @ Ti[:3, :3] @ D, offset=S @ Ti[:3, 3], output_shape=out_shape or vol.shape,
        order=order, mode="nearest", prefilter=order > 1,
    )


def apply_transforms(series: NDArray, matrices: NDArray, voxel_size_mm, order: int = 3) -> NDArray:
    """Resample every frame of `series` (Nx, Ny, Nz, Nt) by its matrix. Complex
    data is resampled as real and imaginary parts. Frames are independent and
    scipy.ndimage releases the GIL, so they run on a thread pool."""
    out = np.empty_like(series)

    def one(t):
        f = series[:, :, :, t]
        if np.allclose(matrices[t], np.eye(4)):
            out[:, :, :, t] = f
        elif np.iscomplexobj(f):
            out[:, :, :, t] = (_warp(f.real, matrices[t], voxel_size_mm, order)
                               + 1j * _warp(f.imag, matrices[t], voxel_size_mm, order))
        else:
            out[:, :, :, t] = _warp(f, matrices[t], voxel_size_mm, order)

    nt = series.shape[3]
    with ThreadPoolExecutor(max_workers=max(1, min(nt, os.cpu_count() or 1, _MAX_THREADS))) as pool:
        list(pool.map(one, range(nt)))  # list(): re-raise a worker's exception
    return out


# ------------------------------------------------------------- python backend


def _levels(shape):
    return [f for f in _PYRAMID if min(s // f for s in shape) >= _MIN_DIM] or [1]


def _downsample(vol: NDArray, f: int) -> NDArray:
    if f == 1:
        return vol
    sm = ndimage.gaussian_filter(vol, 0.5 * f)  # about the anti-alias width for stride f
    return np.ascontiguousarray(sm[::f, ::f, ::f])


def _reference_levels(ref: NDArray, voxel_size_mm) -> list:
    """What register_volume needs of the reference at each pyramid level:
    (stride, voxel size, mask, masked unit-norm values, parameter scale). Depends
    on `ref` alone, so a series registered to one reference computes it once."""
    vox = np.asarray(voxel_size_mm, float)
    out = []
    for f in _levels(ref.shape):
        r = _downsample(ref, f)
        vf = vox * f
        sel = r > 0.1 * np.percentile(r, 98)
        sel = ndimage.binary_dilation(sel, iterations=2)
        rv = r[sel]
        rv = rv / np.linalg.norm(rv)
        radius = 0.5 * float(np.min(np.asarray(r.shape) * vf)) / vf.mean()  # voxels, ~ half FOV
        # unknowns in about-equal units: rotation * radius (voxels), translation (voxels)
        scale = np.concatenate([np.full(3, radius), 1.0 / vf])
        out.append((f, vf, sel, rv, scale))
    return out


def register_volume(mov: NDArray, ref: NDArray, voxel_size_mm, p0=None, levels=None) -> NDArray:
    """Least-squares rigid registration of `mov` to `ref` (both float, same grid),
    coarse to fine. Returns the 6 params (about the volume center) of T.
    levels: `_reference_levels(ref, voxel_size_mm)`, when already computed."""
    vox = np.asarray(voxel_size_mm, float)
    p = np.zeros(6) if p0 is None else np.array(p0, float)
    c = _center_mm(ref.shape, vox)
    for f, vf, sel, rv, scale in levels or _reference_levels(ref, vox):
        m = _downsample(mov, f)

        def resid(q, m=m, sel=sel, rv=rv, vf=vf, c=c, scale=scale):
            T = params_to_matrix(q / scale, c)
            w = _warp(m, T, vf, order=3)[sel]
            n = np.linalg.norm(w)
            return rv - w / max(n, 1e-30)

        sol = optimize.least_squares(resid, p * scale, diff_step=0.005, x_scale=1.0,
                                     xtol=1e-8, ftol=1e-8, max_nfev=40)
        p = sol.x / scale
    return p


def _estimate_python(mag: NDArray, voxel_size_mm, ref_vol: NDArray, progress=None):
    nt = mag.shape[3]
    c = _center_mm(mag.shape[:3], voxel_size_mm)
    P = np.zeros((nt, 6))
    p_prev = None
    levels = _reference_levels(ref_vol, voxel_size_mm)
    for t in range(nt):
        P[t] = register_volume(mag[..., t], ref_vol, voxel_size_mm, p0=p_prev, levels=levels)
        p_prev = P[t]
        if progress:
            progress(t + 1, nt)
    return np.stack([params_to_matrix(p, c) for p in P])


# ---------------------------------------------------------------- FSL backend


def find_mcflirt() -> str | None:
    """Path of FSL's mcflirt (see find_fsl_tool)."""
    return find_fsl_tool("mcflirt")


def _estimate_fsl(mag: NDArray, voxel_size_mm, ref, mcflirt: str, cost: str = "normcorr"):
    import nibabel as nib

    affine = fsl_affine(voxel_size_mm)
    nt = mag.shape[3]
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, "in.nii.gz"), os.path.join(d, "mc")
        nib.save(nib.Nifti1Image(np.asarray(mag, np.float32), affine), inp)
        env = fsl_env(mcflirt)
        args = [mcflirt, "-in", inp, "-out", out, "-mats", "-cost", cost]
        args += ["-meanvol"] if ref == "mean" else ["-refvol", str(ref)]
        subprocess.run(args, check=True, env=env, capture_output=True)
        mats = [np.loadtxt(os.path.join(out + ".mat", f"MAT_{t:04d}")) for t in range(nt)]
        ref_vol = nib.load(out + "_meanvol.nii.gz").get_fdata() if ref == "mean" else None
    return np.stack(mats), ref_vol


# --------------------------------------------------------------------- driver


def correct_motion(series: NDArray, voxel_size_mm=(1.0, 1.0, 1.0), ref="middle",
                   method: str = "auto", order: int = 3, progress=None) -> MotionResult:
    """Register every frame of `series` (Nx, Ny, Nz, Nt; real or complex) to a
    reference and resample.

    ref: a frame index, 'middle' (MCFLIRT's default), or 'mean' (register to
        the middle frame, then to the mean of the aligned series: less
        dependent on one noisy frame; the python backend runs two passes, FSL
        its own -meanvol).
    method: 'auto', 'fsl' or 'python', see the module docstring.
    order: spline order of the final resampling (1 = trilinear, 3 = cubic).
    """
    if series.ndim != 4:
        raise ValueError("series must be (Nx, Ny, Nz, Nt)")
    mag = np.abs(series).astype(np.float32, copy=False)
    nt = mag.shape[3]
    if method == "auto":
        method = "fsl" if find_mcflirt() else "python"
    if method not in ("fsl", "python"):
        raise ValueError(f"unknown method {method!r}")
    first = nt // 2 if ref in ("middle", "mean") else int(ref)
    if not 0 <= first < nt:
        raise ValueError(f"reference frame {ref} out of range for {nt} frames")
    c = _center_mm(mag.shape[:3], voxel_size_mm)

    if method == "fsl":
        exe = find_mcflirt()
        if exe is None:
            raise FileNotFoundError("mcflirt not found (PATH, $FSLDIR/bin, ~/fsl/bin); "
                                    "use method='python'")
        mats, ref_vol = _estimate_fsl(mag, voxel_size_mm, "mean" if ref == "mean" else first, exe)
        if ref_vol is None:
            ref_vol = mag[..., first]
    else:
        mats = _estimate_python(mag, voxel_size_mm, mag[..., first], progress)
        ref_vol = mag[..., first]
        if ref == "mean":
            aligned = apply_transforms(mag, mats, voxel_size_mm, order=1)
            ref_vol = aligned.mean(axis=3)
            mats = _estimate_python(mag, voxel_size_mm, ref_vol, progress)

    out = apply_transforms(series, mats, voxel_size_mm, order=order)
    params = np.stack([matrix_to_params(T, c) for T in mats])
    return MotionResult(series=out, matrices=mats, params=params, ref=ref_vol, method=method)
