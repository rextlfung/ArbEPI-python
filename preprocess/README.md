# preprocess/ — raw scanner data to reconstruction-ready k-space

Turns one sequence's raw GE ScanArchives (noise, calibration, EPI, plus the shared
dual-echo deGRE) into a single file, `<outdir>/<seq>_preprocessed.h5`, holding
everything `recon/` needs:

- zero-filled (kx, ky, kz) k-space for every frame, whitened and (optionally)
  coil-compressed, plus the fully sampled central calibration region;
- ESPIRiT coil sensitivity maps in the same coil space as the k-space;
- a B0 field map (MRIFieldmaps.jl) and an R2* map, on the EPI grid.

`demo.ipynb` walks through every step on a real dataset and is where the options
are documented and set. Start there. Ported from the MATLAB
[epi-preprocessing](https://github.com/rextlfung/epi-preprocessing).

## Layout

| File | What it holds |
|---|---|
| `preprocess.py` | `PreprocessConfig`, `preprocess()` and its stages (gridding, compression, maps, output) |
| `batch_preprocess.py` | `batch_preprocess()` and the `python -m preprocess.batch_preprocess` command line |
| `coils.py` | Noise whitening; coil compression (GCC and PCA) and the virtual-coil count rule |
| `epi_gridding.py` | Ramp-sampled readouts → Cartesian kx (1D NUFFT) |
| `oephase.py` | Odd/even (Nyquist ghost) phase estimation and correction |
| `smaps.py` | ESPIRiT sensitivity maps and their resize/mask/smooth/normalize |
| `b0map.py` + `julia/` | B0 field map: runs `julia/b0map.jl` (MRIFieldmaps.jl + ROMEO.jl) |
| `r2star.py` | R2* fit over the deGRE echoes (see [R2*](#r2-a-placeholder)) |
| `grid_resize.py` | deGRE grid → EPI grid (z crop + edge-aligned resample) |
| `calibrate_delay.py` | Sweep the readout k-space center offset (`delay`) |
| `utils.py` | ScanArchive, `scan_info.mat` and NIfTI I/O; small numerics; QA figures |
| `demo.ipynb` | Worked example on `20260915ball/2_6x_2.4mm` |

Tests: `tests/test_preprocess_<module>.py`, plus `test_preprocess_pipeline.py`
(end to end, with fake archives) and `test_preprocess_calib.py`.

## Setup

`preprocess/` runs in its own venv, `.venv-preprocessing`, because reading raw
ScanArchives needs GE's Orchestra SDK (`GERecon`), which is proprietary, not
pip-installable, and locked to Python 3.10 with numpy < 2. The SDK is distributed
on request as a GitHub release
([GEHC-External/MR-Orchestra-SDK-Python](https://github.com/GEHC-External/MR-Orchestra-SDK-Python/releases)).

```bash
uv venv --python 3.10 .venv-preprocessing
uv pip install --python .venv-preprocessing/bin/python -e ".[preprocessing,test]" ipykernel
uv pip install --python .venv-preprocessing/bin/python <SDK>/GERecon
```

The B0 map needs `julia` on `PATH` (e.g. via [juliaup](https://julialang.org/install/)).
Once, with network access:

```bash
julia --project=preprocess/julia -e 'import Pkg; Pkg.instantiate()'
```

Only `utils.ArchiveReader`/`read_archive` import GERecon, and only when called, so
the rest of the package, its tests, and `recon/` (which imports `preprocess.utils`)
work without the SDK. Run everything from the repository root.

## Inputs

```
<datdir>/scanarchives/<seq>_noise.h5   noise scan (optional: without it nothing is whitened)
<datdir>/scanarchives/<seq>_cal.h5     calibration scan (EPI with the blips off)
<datdir>/scanarchives/<seq>_epi.h5     EPI time series
<datdir>/scanarchives/gre.h5           dual-echo deGRE, shared by all sequences
<datdir>/seqs/<seq>/scan_info.mat      written by sequences/ArbEPI.py: matrix sizes, FOVs,
                                       kxo/kxe readout trajectories, the (ky, kz) schedules
                                       and echo times, deGRE echo times
```

## Quick start

```bash
PY=.venv-preprocessing/bin/python
DAT=/StorageRAID/rexfung/20260915ball

# Everything with the defaults, for several sequences
$PY -m preprocess.batch_preprocess $DAT 1_1x_5.4mm 2_6x_2.4mm

# Exactly 8 virtual coils, keep the gridding cache for later reruns
$PY -m preprocess.batch_preprocess $DAT 2_6x_2.4mm --nvcoils 8 --keep-cache

# All 32 physical coils, no B0 map (no julia)
$PY -m preprocess.batch_preprocess $DAT 2_6x_2.4mm --no-compress --no-b0
```

From Python:

```python
from preprocess.preprocess import PreprocessConfig, preprocess

cfg = PreprocessConfig(datdir=DAT, delay=-1.0, Nvcoils=None, keep_cache=True)
out = preprocess(cfg, "2_6x_2.4mm")  # -> <datdir>/recon/2_6x_2.4mm_preprocessed.h5
```

`demo.ipynb` has the table of every `PreprocessConfig` option. A failure in one
sequence of a batch is reported and the others continue.

## Outputs

`<outdir>/<seq>_preprocessed.h5` (default outdir `<datdir>/recon`, where `recon/`
looks), plain numpy-order HDF5:

| Dataset | Shape | Contents |
|---|---|---|
| `ksp_epi_zf` | (Nx, Ny, Nz, Nc, Nt) complex64 | zero-filled k-space, one frame per chunk |
| `omegas` | (Ny, Nz, Nt) bool | sampling mask (the same for every kx) |
| `echo_times` | (Ny, Nz, Nt) | time since excitation of each sample (s); 0 where unsampled |
| `ksp_calib` | (Nx, Ny_c, Nz_c, Nc, Nt) | fully sampled central region, full kx; attrs `calib_y_range`, `calib_z_range` |
| `smaps` | (Nx, Ny, Nz, Nc) complex64 | sensitivity maps, same coil space as `ksp_epi_zf`, unit RSS in the support |
| `b0map_hz`, `b0_mask` | (Nx, Ny, Nz) | B0 field map (Hz) and the voxels it was fit on |
| `r2star` | (Nx, Ny, Nz) | R2* (1/s) |
| `W` | (Ncoils, Ncoils) | whitening matrix |
| `cc_matrix`, `cc_evals` | (Nx, Nv, Ncoils), (Nx, Ncoils) | GCC matrices and per-x eigenvalues (PCA: (Nv, Ncoils), (Ncoils,)) |
| `degre/` | deGRE grid | QA volumes: `img_echoes`, `b0map_hz`, `finit_hz`, `mask`, `smaps`, `emap`, `r2star` |

Attributes: `noise_var` (thermal-noise variance of `ksp_epi_zf` per complex
sample), `whitened`, `coil_compressed`, `cc_method`, `Ncoils`, `Nc_out`,
`Nvcoils`, `Nvcoils_source` (`energy` or `user`), `cc_energy_kept`, `oephase_a`,
`delay`, `t_ref_s` (nominal TE), `TE_degre`, `fov`, `fov_degre`,
`n_frames_discard`, `r2star_method`.

Also written: `<seq>_smaps`, `<seq>_b0map` and `<seq>_r2star` as `.nii.gz` +
`.json` for viewing (magnitude only; voxel spacing is right but there is no patient
orientation). With `keep_cache=True`, `<seq>_gridded.h5` (whitened, all coils) is
kept too.

## The pipeline

### Stage A: gridding (always run; cached)

1. **Whitening.** `W` from the noise scan decorrelates the channels and gives each
   unit noise variance (Cholesky of the noise covariance). It is lossless (`W` is
   full rank and stored), and everything downstream assumes it: `noise_var` ≈ 1,
   recon's low-rank λ weights, PCA/GCC, ESPIRiT and SENSE (which has no noise
   covariance term). Without a noise scan, `W = I` and `whitened = False` (recon
   warns).
2. **Odd/even phase.** Opposite-direction readouts leave a phase difference
   between odd and even echoes that ghosts the image by FOV/2. From the calibration
   scan (no phase encoding) the phase between neighboring echo pairs is fit as
   `a[0] + a[1]·x` (`oephase.getoephase`) on whitened, uncompressed data, and
   `epiphasecorrect` removes it from every even echo. On `20260915ball` the coil
   basis used for the estimate (raw, whitened, compressed) moved `a` by less than
   the estimate's own noise (half-split of the calibration shots).
3. **Per frame**: regrid the ramp-sampled readouts onto Cartesian kx (1D NUFFT,
   density-compensated; separate trajectories for odd and even echoes, shifted by
   `delay`), apply the odd/even correction, and scatter each (shot, echo) into its
   (ky, kz) slot of the zero-filled grid.
4. **Noise**: the noise-scan readouts go through the same gridding and correction,
   so `noise_var` can later be measured in the output coil space.

The EPI archive is streamed frame by frame and the cache is checkpointed after
each frame; rerunning after a crash resumes where it stopped (the reader has no
seek, so it replays and discards the finished frames' readouts). A complete cache
is reused as is.

**Why compression can come after the cache.** Gridding, the odd/even correction
(one phase per echo and x, the same for every coil) and the scatter are linear and
act on each coil separately, so any coil-mixing matrix gives the same result
applied before or after them — GCC's per-x matrices too, because the sampling mask
depends only on (ky, kz). What does not commute is *estimating* things from coil
data (`a`, the compression, ESPIRiT), which is why those always use the fixed
whitened, uncompressed data.

### Coil compression

- **GCC** (`cc_method='gcc'`, default; Zhang, Pauly, Vasanawala & Lustig, MRM
  2013): the k-space is inverse-FFT'd along the fully sampled kx, and each x
  position gets its own [Nv, Ncoils] matrix from the top eigenvectors of that
  position's coil covariance (from the whitened deGRE, kx cropped/padded to the
  EPI Nx so the positions coincide; a central 24×24 (ky, kz) block, as the paper's
  ACS region). The matrices are then rotated within their subspaces to vary
  smoothly along x (the paper's alignment step). Only a few coils see any one x,
  so each position's spectrum is steep even though the whole volume's is flat.
- **PCA** (`cc_method='pca'`): one matrix for the volume (the previous behavior).
- **How many virtual coils**: the smallest number whose kept eigenvalue energy,
  summed over all x, reaches `cc_energy_thresh` (0.99). Summing over x weights
  each position by its signal, so no object mask is needed; taking the worst
  position instead swung between 11 and 31 coils depending on how the edge was
  defined. `Nvcoils` sets the count exactly. There is no floor tied to R.
- Maps are compressed with the same matrices as the k-space, which keeps the
  SENSE model exact; the compressed matrices have orthonormal rows, so whitened
  noise stays white.

Measured on the whitened `20260915ball` deGRE (32 coils), fraction of each
object voxel's SNR kept (matched-filter combine, R = 1), median / worst 5%:

| virtual coils | global PCA | GCC |
|---|---|---|
| 8 | 0.872 / 0.718 | 0.995 / 0.964 |
| 10 | 0.919 / 0.760 | 0.998 / 0.986 |
| 14 | 0.965 / 0.843 | 1.000 / 0.998 |

The previous rule (PCA, 90% energy, at least 2R) kept 14 coils and lost ~16% SNR
at the object's periphery; GCC at 99% keeps 10 coils and ~1.4%.

Those numbers are at R = 1. Under acceleration the loss is larger, because fewer
coils also means more noise amplification in the unfolding. Reconstructing
`2_6x_2.4mm` (R = 6) with unregularized CG-SENSE (30 iterations, 20 frames,
plain SENSE), compared with all 32 coils:

| virtual coils | mean-image difference | tSNR relative to 32 coils | CG time |
|---|---|---|---|
| GCC 6 | 15.1% | 0.70 | 2.7 s |
| GCC 8 | 7.2% | 0.80 | 3.5 s |
| GCC 10 (the 99% default) | 3.8% | 0.88 | 4.3 s |
| GCC 12 | 1.8% | 0.94 | 5.1 s |
| PCA 14 (previous rule) | 9.0% | 0.84 | 5.8 s |
| 32 (`compress=False`) | — | 1 | 13.9 s |

GCC with 10 coils beats the previous 14-coil PCA on both counts at ~3× less recon
time than 32 coils. Unregularized CG is the worst case for noise amplification
(regularized recon should narrow the gap, not measured here). Where SNR matters
more than recon time, raise `cc_energy_thresh` (0.995 gives ~12 coils here) or set
`Nvcoils`; `compress=False` keeps everything.

### Maps (from the dual-echo deGRE)

- **Sensitivity maps**: one ESPIRiT calibration (sigpy) on the whitened,
  uncompressed first echo at a 24³ calibration size, then resized to the EPI grid
  (`grid_resize`), masked where ESPIRiT's eigenvalue map exceeds `crop` (0.95),
  smoothed (6 mm Gaussian, mask-normalized), RSS-normalized and compressed. `crop`
  0.95 gives a support slightly larger than the object (43% vs 39% of the volume
  on `01_fullsamp_4p55mm`); tighter values start cutting into the object.
- **B0**: `julia/b0map.jl` fits MRIFieldmaps.jl's regularized field map on the
  deGRE grid from both echoes, combining coils with the sensitivity maps, starting
  from a ROMEO-unwrapped phase difference, with the `:diag` preconditioner (the
  default `:ichol` made the regularization ineffective and the map speckled; see
  the script's header). The map is zeroed outside the fit mask and resized to the
  EPI grid.
- **R2\***: see below. `t_ref_s` records the nominal-TE echo time the recon's
  R2* model is referenced to.

### R2\*: a placeholder

`r2star.py` fits ln|S(TE)| linearly over the deGRE echoes. The deGRE echo spacing
is set for B0 mapping — on `2_6x_2.4mm` 2.24 ms against T2* ≈ 47 ms, so the
echoes differ by ~5% — which makes the fit noise-dominated (precision
≈ √2 / (SNR·ΔTE), ~6 /s at SNR 100). A usable map needs a true multi-echo GRE: 3D
monopolar, ~4–8 echoes, the last near 1–2× T2*, thin slices. `r2star.py`'s
docstring has the details and references, and `docs/TODO.md` tracks it. The fit
already uses every echo, so such a scan needs no code change here.

### Calibration region

`find_calib_region` grows a rectangle outward from the k-space center, one edge at
a time, while the new line is sampled in every frame. For Poisson-disc masks
(`pd_calib_frac`) this recovers the pd calibration rectangle exactly (17×11 on
`2_6x_2.4mm`); CAIPI or random masks with no such region give none.

### Stage B: output

The cache is streamed frame by frame through the compression (hybrid space for
GCC: inverse FFT along kx, per-x matrix, FFT back), the calibration region is cut
out, `noise_var` is measured on the compressed noise, and the maps are written.
With `compress=False` the cache file itself becomes the output. The cache is
deleted unless `keep_cache`; keeping it lets a rerun with other compression
settings skip Stage A (see the end of `demo.ipynb`).

## Tuning the readout delay

A wrong `delay` shows up as a large, wrapping linear odd/even phase.
`calibrate_delay(paths)` sweeps delays over the calibration scan and picks the one
with no phase wraps and the smallest linear term:

```python
from preprocess.calibrate_delay import calibrate_delay
from preprocess.preprocess import PreprocessConfig, set_seq_paths

best, report = calibrate_delay(set_seq_paths(PreprocessConfig(datdir=DAT), "2_6x_2.4mm"))
```

## Performance

On `2_6x_2.4mm` (90×90×60, 60 frames × 900 readouts, 32 coils; deGRE 108×108×72):
the full default run takes 6–7 minutes (346 s alone on the machine), under 2 of
them for Stage A and the rest for ESPIRiT, the B0 fit and writing. Rerunning from
a kept cache with different compression settings takes about 1 minute without
the maps (62 s for 6 virtual coils) and about 4 with ESPIRiT. The whitened
32-coil cache is ~1.2 GB; the default output (10 virtual coils) ~0.75 GB, and
~1.8 GB uncompressed.

## Validation

- **Against the previous code, on real data**: with the previous choices (global
  PCA, 14 virtual coils, `a` estimated on whitened + compressed calibration data)
  the new pipeline reproduces frames computed by the previous `preprocessing/`
  code from the raw archives to 1e-7 (float32), with identical `noise_var`,
  `omegas` and `echo_times`. The default odd/even estimate changes the k-space by
  6.8e-4. The new B0 map correlates 0.9994 with the previous one and the
  sensitivity-map support agrees (Dice 0.991).
- **Earlier**: the gridding/correction/scatter chain matched a MATLAB/BART
  reconstruction of a real acquisition (`wb_2.4mm`) to 0.19% (RSS images, after a
  global scale); ESPIRiT, B0 and the odd/even estimate were checked against
  synthetic ground truth.
- **Tests** cover: whitening and compression (GCC recovers x-dependent subspaces
  PCA can't; compressing data per x equals the forward model of the compressed
  maps; alignment keeps each subspace; noise stays white), the odd/even estimate
  and correction sharing a pixel frame at odd Nx, gridding scale and trajectories,
  scatter placement, the deGRE unflattening, the calibration region against the pd
  rectangle, the R2* fit, the B0 fit on synthetic fields (including unwrapping
  beyond the naive ±1/(2ΔTE) range), and `preprocess()` end to end on fake
  archives (PCA output equals compressing before gridding; GCC output equals GCC
  of the uncompressed output; `Nvcoils`; no noise scan; cache reuse; resume after
  a crash).

Run them with the preprocessing extras installed, or they are skipped:

```bash
uv sync --extra test --extra preprocessing && uv run pytest -rs tests/test_preprocess_*.py
```
