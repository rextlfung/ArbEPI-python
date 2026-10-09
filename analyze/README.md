# analyze/ -- task activation maps (GLM t- and z-scores)

Turns a reconstruction (`recon/`'s `<name>_recon.h5`) into statistic maps for a
block-design task: the t-score of a contrast per voxel, the equivalent z-score,
and the effect in percent of the voxel's mean signal, as NIfTI. Real data is
the point; `simulate_fmri/` uses the same GLM on simulated runs, and
`simulate_fmri/analysis.py` keeps only what needs a truth file.

## Setup

No torch, so the main environment is enough:

```
uv sync --extra analyze
```

`analyze` adds `h5py`, `nibabel` and `brainextractor` (Python >= 3.11). FSL's
own `bet`, if on `PATH`, is used instead.

## Usage

```
uv run python -m analyze --recon <datdir>/recon/sense_mslr_b0/<seq>_recon.h5 \
    --tr 0.506 --onsets 0 40 80 120 160 200 240 280 --duration 20 \
    --scales all --out <datdir>/recon/analysis/
```

- `--onsets` are seconds from the first frame kept; `--n-discard` drops
  further leading frames (the sequence's own warm-up frames are not in the
  file).
- `--scales`: `sum` (the summed image, default), component indices
  (`--reg mslr` runs), or `all` (the sum and every component).
- `--mask`: `auto` (FSL `bet` if on `PATH`, else `brainextractor`), `fsl`,
  `python`, `threshold` (no skull stripping), or a NIfTI mask. A phantom has no
  skull for BET to find: pass a mask or use `threshold`.
- `--no-ar1`: ordinary least squares. `--drift-cutoff 0`: a linear trend in
  place of the DCT drift terms.
- Several conditions or a custom contrast: the Python API,
  `analyze.run.analyze_recon(path, ExperimentParams(...))`.

Outputs in `--out`, per analyzed series `<label>` (`sum`, `scale0`, ...):
`<name>_<label>_{t,z,psc}.nii.gz` (+ `.json`), `<name>_mask.nii.gz` and
`<name>_summary.json` (design columns, dof, the AR(1) coefficient, the
FDR-surviving |t| threshold, voxel counts).

## Comparing reconstructions

```
uv run python -m analyze.compare --recon rss=<a>/rss_recon.h5 mslr=<b>/<seq>_recon.h5 \
    --scales all --tr 0.506 --onsets ... --duration 20 --out <dir>/compare/
```

Every file is analyzed with the same design and one brain mask (from the first
file unless `--mask` is given); `maps.png` shows axial slices of the statistic,
one row per series (a multi-scale file contributes `<label>` and
`<label>/scale<i>`), on one threshold (default p < 0.001 uncorrected, from the
first series' dof). `timecourses.png` has the peak voxel and the mean of the
top n% of voxels (`--top-percent`, selected on the first series so the lines
are the same voxels; `--peak-source per_series` selects on each), the task
blocks shaded, and the power spectrum with the task fundamental marked.
`summary.txt` is the table: dof, rho, extremes, the effect in the top n%, FDR
survivors. Python: `analyze.compare.compare_recons`, `plot_maps`,
`plot_timecourses`, `summary_table`.

## What it does

| file | contents |
|---|---|
| `design.py` | `ExperimentParams`/`Condition`, the canonical HRF, `build_design_matrix` (conditions, DCT drift, constant) |
| `glm.py` | `fit_glm` (OLS or AR(1)-prewhitened), `t_to_z`, FDR/Bonferroni, `low_band` |
| `mask.py` | `brain_mask` (BET), `load_mask` |
| `io.py` | `read_recon` (sum and per-scale series), `save_stat_nifti` |
| `run.py`, `__main__.py` | `analyze_recon` and the CLI |
| `compare.py` | side-by-side maps, time courses, spectra and a summary table for several series |

**Design.** Each condition is its blocks convolved with SPM's canonical HRF
(exactly, via the HRF's running integral), sampled at frame centers. Nuisance
terms are SPM's: a constant and the DCT cosines of period longer than
`--drift-cutoff` (128 s), `floor(2 T / 128)` of them for a run of T seconds.
Drift is far below a block task's frequency (a 20 s / 20 s task is 0.025 Hz;
the cutoff is 0.0078 Hz); the terms take up slow scanner and physiological
drift that would otherwise inflate the residual.

**Noise model.** Reconstructed fMRI residuals are autocorrelated, and a plain
t assumes they are not. AR(1) prewhitening estimates one coefficient pooled
over voxels, as SPM does, and whitens data and design by the exact AR(1)
filter before an ordinary fit; degrees of freedom are `nt - rank(X)`. Two
simplifications relative to SPM12: the coefficient is the pooled lag-1
autocorrelation of unit-variance residuals iterated to convergence
(Cochrane-Orcutt), not a ReML fit of AR(1) plus white noise. It is global, not
per voxel as in FSL's FILM. A recon with a temporal penalty can leave
residuals that are not AR(1); `simulate_fmri/` measured exactly that (see
CLAUDE.md), and its low-band GLM is available here as `glm.low_band` for
comparison.

**z-scores.** `t_to_z` is `sign(t) Phi^-1(F_t(|t|; dof))`, computed in log
space so it does not saturate at |z| = 8.3.

**Scales.** The summed image is analyzed as `|X_recon|`. A component of a
multi-scale low-rank reconstruction is not a magnitude image (the dynamic
scale has nearly zero mean), so it is projected on the phase of the summed
image's temporal mean, `Re(X_s conj(phi))`, which keeps signs and is linear in
the components. The percent change of every scale is relative to the summed
image's mean signal, so the scales' effects are on one axis. Reading the
result: a task response in the local scale and none in the global scale is
what a background/foreground separation predicts.

## Not here

Group-level statistics, motion regressors (this pipeline estimates no motion),
and cluster-level correction.
