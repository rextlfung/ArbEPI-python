# postprocess/ -- operations on `recon/`'s output, before `analyze/`

Prototype. One operation so far: inter-frame rigid-body motion correction.

```
python -m postprocess --recon <name>_recon.h5 --out <name>_mc_recon.h5 \
    [--method auto|fsl|python] [--ref middle|mean|<frame>] [--n-discard N]
```

The output has the layout `analyze/` reads (`X_recon`, plus every per-scale
component `X` of an `mslr` run, all resampled with the same transforms), so
`python -m analyze --recon <name>_mc_recon.h5 ...` works unchanged. It also
holds `motion/{matrices,params,ref}`, and `<out>.par` has one line per frame:
rx ry rz (rad), tx ty tz (mm), about the volume center.

## Backends

| | needs | notes |
|---|---|---|
| `python` | numpy/scipy only (`uv sync --extra analyze`) | multi-resolution (4x, 2x, 1x) least-squares on a normalized-SSD cost, cubic interpolation inside the cost |
| `fsl` | `mcflirt` on PATH, `$FSLDIR/bin` or `~/fsl/bin`; `nibabel` | MCFLIRT estimates the transforms only; resampling is this package's, for both |

The reference frame defaults to the middle one (`Nt // 2`), as MCFLIRT does; `--ref mean` registers to the mean of the aligned series instead.

`auto` uses FSL when it finds `mcflirt`. Motion is estimated on magnitudes; a
complex series is resampled as real and imaginary parts.

Why the cost interpolates with cubic splines: with trilinear interpolation the
least-squares minimum is pulled toward whole-voxel positions (a known bias), and
the rotations of a synthetic phantom were off by 0.6-0.7 deg; cubic cut that to
<= 0.2 deg per volume.

## Checked

`tests/test_postprocess_motion.py` moves a synthetic head by known rotations and
translations. Both backends recover them to <= 0.65 deg and <= 0.16 mm (3 mm
voxels), with each other to the same tolerance. Not checked: real data (the
python backend's accuracy and runtime on a 90x90x60 x several-hundred-frame run).

## Not done

Slice-wise (intra-frame) motion, which matters for multi-shot EPI; applying
motion to `X_recon` *before* reconstruction (the operators assume a fixed
object); motion regressors into `analyze/`'s GLM (the `.par` file is there for
it); GPU acceleration.
