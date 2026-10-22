# postprocess/ -- operations on `recon/`'s output, before `analyze/`

Prototype. Two operations: inter-frame rigid-body motion correction, and a brain mask applied to the corrected series.

```
python -m postprocess --recon <name>_recon.h5 --out <name>_mc_recon.h5 \
    [--method auto|fsl|python] [--ref middle|mean|<frame>] [--n-discard N]
```

The output has the layout `analyze/` reads (`X_recon`, plus every per-scale
component `X` of an `mslr` run, all resampled with the same transforms), so
`python -m analyze --recon <name>_mc_recon.h5 ...` works unchanged. It also
holds `motion/{matrices,params,ref}`, and `<out>.par` has one line per frame:
rx ry rz (rad), tx ty tz (mm), about the volume center.

## Brain mask

`--brain-mask-from <seq>_preprocessed.h5` zeroes the corrected series outside a BET mask of
that file's T1-weighted deGRE (`degre/img_echoes`, echo 1, which is on the EPI grid and has
the anatomical contrast the EPI mean lacks: BET on an EPI mean gave 4184 cm3 at 2.4 mm).
`--bet-frac` is BET's threshold (default 0.5). The mask is stored as `brain_mask` in the
output and written as `<out minus .h5>_brain_mask.nii.gz`; pass that to `analyze/` as its
mask. The deGRE is acquired before the EPI, so the skull-stripped deGRE is first registered
to the corrected series' mean (rigid, FSL `flirt`, normalized mutual information, search
limited to +-15 deg) and the mask moved with that transform (`brain_mask_xfm`; the CLI
prints how far it moved the mask). `--no-mask-registration` skips it.
`brainmask.py` has the pieces; `tests/test_postprocess_brainmask.py` the checks.

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
