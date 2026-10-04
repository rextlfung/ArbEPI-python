# recon/ — image reconstruction

Turns the zero-filled k-space written by `preprocess/` into images. There
are two kinds of reconstruction:

- **RSS** (`rss.py`): zero-filled inverse FFT per coil, then root-sum-of-squares
  over coils. Fast, no sensitivity maps, aliasing left in. A first look.
- **Iterative SENSE** (`sense.py`): solves

  $$\hat x = \arg\min_x \; \tfrac12 \|A x - y\|_2^2 + g(x)$$

  where $y$ is the acquired k-space, $A$ is the **encoding operator**
  (`operators.py`) and $g$ is a **regularizer** (`regularizers.py`). The
  regularizer determines which **solver** (`solvers.py`) is used.

`demo.ipynb` runs every reconstruction type on a real dataset and shows the
results. Start there if you want to see what each option does.

## Layout

| File | What it holds |
|---|---|
| `operators.py` | The encoding operators `SENSE`, `SENSE_B0`, `SENSE_B0_R2star` and their builders |
| `regularizers.py` | `MultiScaleLowRank`, `WaveletTV`, `SpatioTemporalWaveletTV` and `TemporalHighPass` (plus the pieces they're built from) |
| `solvers.py` | `pogm_restart` (POGM/FPGM/PGM), `pdhg` (primal-dual), `cg` |
| `sense.py` | `run_sense()` and the `python -m recon.sense` command line |
| `rss.py` | `run_rss()` and the `python -m recon.rss` command line |
| `utils.py` | File I/O, spectral-norm estimation, the tSNR report, and one-off analyses |
| `testbed.py` | A known-truth testbed (undersampled dynamic data with injected activation, built from a fully sampled run) and scoring of recons against it |
| `demo.ipynb` | Worked examples on `20260915ball/2_6x_2.4mm` |

Tests mirror this: `tests/test_recon_<module>.py`.

## Setup

Everything in `recon/` runs in its own venv, `.venv-recon`, installed from the
same `uv.lock` as the main environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-recon uv sync --extra recon --extra test
```

It is kept separate from `.venv` because `uv sync` is exact: it removes any
package the requested extras don't list, so the `uv sync --extra test` used
for sequence work would uninstall torch from a shared environment. For the
same reason, anything installed into `.venv-recon` by hand (`uv pip install`)
is removed by the next sync. torch comes from the lockfile's PyPI wheel; if
that build doesn't fit your GPU, pin a different one in `pyproject.toml`
(a `[[tool.uv.index]]` for the PyTorch CUDA index plus a `[tool.uv.sources]`
entry for `torch`) and re-lock, rather than installing it by hand.

Notebooks (`demo.ipynb`) use the plain `python3` kernel, and `ipykernel` is in
the `recon` extra (as is `nbclient`, for executing notebooks from a script), so an editor can use `.venv-recon/bin/python` as the kernel
directly. To run Jupyter itself, layer it on with `--with` rather than
installing it into the env (the next sync would remove it):

```bash
export UV_PROJECT_ENVIRONMENT=.venv-recon
uv run --extra recon --with jupyter jupyter lab recon/demo.ipynb
# or headless, e.g. to refresh the saved outputs
uv run --extra recon --with nbconvert jupyter nbconvert --to notebook --execute --inplace recon/demo.ipynb
```

Run everything from the repository root (`recon/` imports helpers from
`preprocess/`). Everything also runs on CPU, just much more slowly; the
device defaults to `cuda` when a GPU is available and `cpu` otherwise
(`--device` overrides it).

## Inputs

One file, `<datdir>/recon/<seq>_preprocessed.h5`, as written by `preprocess/`:

| Dataset / attr | Contents | Needed for |
|---|---|---|
| `ksp_epi_zf` | (Nx, Ny, Nz, Ncoils, Nframes) zero-filled k-space (whitened, and coil-compressed unless preprocessed with `compress=False`) | everything |
| `omegas`, `echo_times` | (Ny, Nz, Nframes) sampling mask, and each sample's time since excitation (s) | everything |
| attrs `noise_var`, `whitened` | thermal-noise variance of `ksp_epi_zf`; whether a noise scan whitened it | scaling |
| `smaps` | (Nx, Ny, Nz, Ncoils) ESPIRiT sensitivity maps, in the same coil space as `ksp_epi_zf` | SENSE |
| `b0_map` | (Nx, Ny, Nz) B0 field map in Hz on the EPI grid | `--B0` |
| `r2star_map`, attr `t_ref_s` | (Nx, Ny, Nz) R2* map in 1/s, and the nominal-TE reference time | `--R2star` |

It also holds `ksp_calib` (the fully sampled central calibration region), the
whitening/compression matrices and deGRE-grid QA volumes; see preprocess/demo.ipynb.

## Quick start

```bash
export UV_PROJECT_ENVIRONMENT=.venv-recon
PY="uv run --extra recon python"
DAT=/StorageRAID/rexfung/20260915ball
SEQ=2_6x_2.4mm

# Root-sum-of-squares
$PY -m recon.rss $DAT $SEQ

# Iterative SENSE with no regularizer (conjugate gradient), B0-corrected
$PY -m recon.sense $DAT $SEQ --reg none --B0

# Locally low rank (6x6x6 patches, stride 3), B0 + R2* corrected
$PY -m recon.sense $DAT $SEQ --reg mslr --B0 --R2star --patch 6 6 6 --stride 3 3 3

# Global + local low rank (two scales)
$PY -m recon.sense $DAT $SEQ --reg mslr --B0 \
    --patch full --stride full --patch 15 15 15 --stride 5 5 5

# L1-wavelet + TV on the first 10 frames
$PY -m recon.sense $DAT $SEQ --reg wavelet-tv --B0 --frames 0-9

# Temporal stability (tSNR, drift, fluctuation) of any saved reconstruction
$PY -m recon.utils tsnr $DAT/recon/rss/${SEQ}_recon.nii.gz --tr 1.0
```

Use `--frames` (e.g. `0-9` or `0,5,7`) and `--niter` to try settings quickly
before a full run. `--b0map <file.h5>` swaps in a different `b0_map` (and implies
`--B0`); its output goes to the same `sense_<reg>_b0/` directory, so it replaces a
run with the preprocessed map unless you move that first. From Python, `run_sense(...)` takes the same options and
returns the result in memory instead of saving it (see `demo.ipynb`).

## Outputs

| Command | Output |
|---|---|
| `recon.rss` | `<datdir>/recon/rss/<seq>_recon.{nii.gz,json}` |
| `recon.sense` | `<datdir>/recon/sense_<reg>[_b0\|_b0r2star]/<seq>_recon[_frames…].{h5,nii.gz,json}` |

- `.nii.gz` — magnitude image (Nx, Ny, Nz, Nframes), for viewing in FSLeyes,
  ITK-SNAP, etc.
- `.json` — every setting used (regularizer, λ, patch sizes, L, …), the
  acceleration factor R, $\sigma_1(A)$, runtime and final costs.
- `.h5` — the complex image `X_recon`, the per-scale components `X`
  (low rank only), the sampling mask `omega`, and the per-iteration traces
  `dc_costs` (for CG: relative residuals), `reg_costs`, `restarts`,
  `rel_changes`.

Image intensities are **not quantitative**: the data and operator scalings
described below are not undone.

## Encoding operators (`operators.py`)

Each operator maps one frame's image (Nx, Ny, Nz) to k-space at that frame's
**sampled locations only**, shape (K, Ncoils), rather than a zero-filled
grid. That is what keeps memory manageable: a dense k-space grid is R times
larger and ran out of GPU memory on real data. The builders
(`build_sense`, `build_sense_b0`, `build_sense_b0_r2star`) stack one operator
per frame into a mirtorch `BlockDiagonal` mapping (Nx, Ny, Nz, Nt) to
(K, Ncoils, Nt).

**`SENSE`** — multiply by each coil's sensitivity map, centered
ortho-normalized 3D FFT, keep the sampled points. With RSS-normalized
sensitivity maps it has $\sigma_1(A) \approx 1$.

**`SENSE_B0`** — also models off-resonance phase accrual. A k-space sample
acquired at time $t$ after excitation picks up phase
$e^{i 2\pi \Delta f(r) t}$ from the field map $\Delta f(r)$. Because $t$
varies along the echo train (~70 ms here), this is approximated with
**time segmentation**: $L$ segment images, each multiplied by a phasor
$e^{i 2\pi \Delta f(r) t_l}$, then transformed and combined per sample with
interpolation weights. The weights come from mirtorch's `mri_exp_approx`.
- `L = 32` segments. A sweep at the real echo-train length (`python -m
  recon.utils sweep`) shows a sharp transition around L ≈ 27–32. L = 6 barely
  helps; L = 32 is the smallest value with under 1% forward-model error.
- `nbins = 128` histogram bins for the fit. The mirtorch default of 20 made the
  fit ill-conditioned on real field maps, causing signal loss and speckle. A
  warning is printed if the fit looks ill-conditioned.
- The histogram spans the map's whole range, so the field map is first clipped
  to its 0.1–99.9 percentile range (`b0_clip_percentile`, `clip_b0_outliers`;
  0 disables). A few diverged voxels (up to −4.7 MHz in 10–50 of 486k voxels on
  20260922xiaokai, before preprocessing dropped mask islands) otherwise make
  every bin kilohertz wide and disable the correction everywhere.
- Sign convention (Sutton, Noll & Fessler 2003): the forward model multiplies
  the image by $e^{+i 2\pi \Delta f(r) t}$ before the FFT.
- Each segment is a full SENSE transform, so cost grows linearly with L: one
  forward + adjoint over 30 frames of a 240×240×45, 18-coil dataset takes
  5.8 s at L = 6 and 30.6 s at L = 32 (`python -m recon.utils benchmark`).

**`SENSE_B0_R2star`** — also models magnitude decay. The field becomes complex,
$\psi(r) = i 2\pi \Delta f(r) - R_2^*(r)$, and each segment's phasor is
$e^{\psi(r)(t_l - t_{\text{ref}})}$. $t_{\text{ref}}$ is the nominal TE, so the
reconstruction target is "the image at TE". The R2* map is fit by preprocessing
(`preprocess/r2star.py`) on the dual-echo deGRE, whose two closely spaced echoes
make it a placeholder: a trustworthy map needs a true multi-echo GRE (see that
module's docstring). The physics lives in
`SENSE_B0_R2star.segment_phasors`, which is computed once and shared by every
frame (a separate copy per frame doesn't fit in GPU memory).

## Regularizers, solvers and λ

| `--reg` | $g(x)$ | Solver |
|---|---|---|
| `none` | 0 | `cg`: conjugate gradient on $A^H A x = A^H y$ |
| `mslr` | multi-scale low rank | `pogm_restart` (default POGM; `--mom fpgm` or `pgm` also available) |
| `wavelet-tv` | $\lambda_{\ell_1}\|Wx\|_1 + \lambda_{TV}\|Dx\|_1$, per frame (jointly with `--lamb-ttv`/`--hp-weight`/`--joint`) | `pdhg` (primal-dual) |

POGM needs a closed-form proximal operator for $g$; the low-rank regularizer
has one (singular-value soft-thresholding), TV does not, hence the
primal-dual solver for `wavelet-tv`. With no regularizer, CG converges much
faster than a gradient method.

### Multi-scale low rank (`MultiScaleLowRank`)

After Ong & Lustig, "Beyond low rank + sparse: multiscale low rank matrix
decomposition" (arXiv 1507.08751). The image series is written as a sum of
components, one per patch scale, $X = \sum_k X_k$, and each component is
penalized by the nuclear norm of its space × time patches:

$$g(X_1,\dots,X_K) = \sum_k \lambda_k \sum_{\text{patches } b} \|P_b(X_k)\|_*$$

- One `--patch` gives a **locally low-rank** prior; a whole-volume patch
  (`--patch full`) gives a **globally low-rank** one; both together give the
  global + local decomposition.
- `--stride` controls patch overlap (defaults to the patch size, i.e. no
  overlap, except for the default 6 6 6 patch, which uses stride 3).
- The weights follow the paper's eq. 4,
  $\lambda_k = \lambda_{\text{global}} \big(\sqrt{p_k} + \sqrt{N_t} + \sqrt{\log(N_{\text{vox}} N_t / \max(p_k, N_t))}\big)$
  with $p_k$ the number of voxels per patch — the expected largest singular
  value of a unit-variance Gaussian noise patch.
- Early stopping: iterations stop when the relative change drops below
  `--conv-tol` (default 1e-5); `--conv-tol 0` always runs `--niter`.
- If the GPU runs out of memory, the command line falls back from POGM to FPGM
  to PGM (each keeps less solver state). It never switches to CPU mid-run.

### Scaling, so that λ means what the paper says

The paper's weights assume **unit-variance white noise** and, with an MRI
forward model, a **unit-norm operator**. `run_sense` arranges both:

1. **k-space noise.** `preprocess()` measures the thermal-noise variance of the
   final k-space by pushing the noise-scan readouts through the same whitening,
   coil compression and regridding as the EPI data, and records it as the
   `noise_var` attribute of `<seq>_preprocessed.h5`. `run_sense` divides the k-space
   by $\sqrt{\texttt{noise\_var}}$. Whitening already targets 1, so this is
   normally a few-percent correction (0.93–1.14 on the `20260915ball` and
   `20260918ball` datasets). Files without the attribute (no noise scan) are
   used as they are (a message says so).
   The noise level can't be estimated from the acquired k-space itself on
   these high-SNR datasets: every candidate region is dominated by signal
   leakage, not noise.
2. **Operator.** For `mslr`, $A$ is divided by its spectral norm
   $\sigma_1(A)$, estimated by power iteration. Plain SENSE already has
   $\sigma_1 \approx 1$; the B0 operators measure about 1.2–1.9.
3. **`--lambda-global` defaults to R**, the acceleration factor. The paper's
   formula calibrates the weights for noise alone; incoherent aliasing is
   noise-like but more structured, so it needs stronger regularization.
   `--lambda-global 1` gives the paper's noise-only weighting.

### L1-wavelet + TV (`WaveletTV`)

$W$ is a 3D orthogonal wavelet (`Wavelet3D`, default `db4`, 3 levels; each
axis is zero-padded to a multiple of $2^{\text{levels}}$) and $D$ is the
periodic finite difference along x, y and z (anisotropic TV). Each frame is
solved separately. The operator is divided by $\sigma_1(A)$ and each frame's
data by its 99th-percentile magnitude, so the default
`--lamb-l1 0.005 --lamb-tv 0.005` means the same on every dataset.

### Temporal regularization

Two optional terms couple the frames; any of them makes the solve joint
(all frames at once):

- **`--hp-weight MU`** (any `--reg`): $+\tfrac{\mu}{2}\|Px\|^2$, where $P$
  (`TemporalHighPass`) is the orthogonal projector onto temporal frequencies
  above `--hp-cutoff` (default 0.15 Hz), along t in the DCT-II basis. The
  penalty is exactly zero at and below the cutoff, so it can't bias a signal
  confined to that band (BOLD), and frequencies above it are shrunk by
  $1/(1+\mu)$ where the data don't constrain them. The DCT, not the DFT,
  because its even extension gives a slow drift no jump at the window edge,
  which would otherwise leak into the penalized band. $\mu$ is relative to the
  data term's curvature with $A$ normalized to unit norm, so it means the same
  for CG, MSLR and wavelet-TV. Needs the volume TR: `--volume-tr`, or the
  k-space file's `volume_tr` attr. How each solver takes it: CG adds $\mu P$
  to its normal equations (exact); wavelet-TV makes it a PDHG dual block with
  a closed-form prox, so the step size doesn't depend on $\mu$ (review item
  262: as a smooth term, $\mu = 30$ left 100 iterations far from converged);
  MSLR adds it to the gradient, which cuts POGM's step to
  $1/(N_{scales}(1+\mu))$, so keep $\mu$ modest there or raise `--niter`.
- **`--lamb-ttv L`** (`wavelet-tv`): $+L\|D_t x\|_1$, temporal TV with a
  non-periodic difference (`TemporalDiff`), in `SpatioTemporalWaveletTV`.
  It penalizes frame-to-frame jumps at every frequency, and shrinks the
  amplitude of real changes too.

With incoherent sampling (a new random mask every frame) the aliasing is
roughly white in time, so a temporal penalty above the cutoff removes only
the share of it outside the kept band; the in-band share looks exactly like a
slow signal to any temporal prior. `--joint` runs the joint wavelet-TV solver
without a temporal term (one data scale for all frames instead of one per
frame), the baseline for the temporal variants. `--tag` suffixes the output
directory so differently configured runs don't overwrite each other.

## Known-truth testbed (`testbed.py`)

```bash
# 80 frames: the object from a fully sampled run, masks from a 10x run
$PY -m recon.testbed build $DAT/recon/5_1x_radial_preprocessed.h5 \
    $DAT/recon/1_10x_radial_preprocessed.h5 $TB/recon/tb_preprocessed.h5 --volume-tr 0.4851
$PY -m recon.sense $TB tb --reg wavelet-tv --B0 --hp-weight 3 --tag hp3
$PY -m recon.testbed score $TB/recon/tb_preprocessed.h5 \
    $TB/recon/sense_wavelet-tv_b0_hp3/tb_recon.h5 --json hp3.json --png hp3.png
```

`build` makes a `<name>_preprocessed.h5` that `recon.sense` reads as is: the
object is the B0-SENSE CG reconstruction of the fully sampled run's mean
frame; each frame applies another run's real sampling mask to it with the
fully sampled run's echo times (the B0 model is then exactly the one the recon
uses -- an inverse crime by construction, so this isolates sampling, noise and
prior effects, not model mismatch) and adds fresh unit-variance noise. Four
2%-peak activations are injected in spherical ROIs: an HRF-convolved 10 s
on/off block at radius 3 and at radius 1.5 voxels, a 0.10 Hz sinusoid (in
band), and a 0.25 Hz sinusoid (out of band). `score` compares magnitudes after
one global scale: error of the mean image and per frame, temporal
fluctuation in non-activated voxels split into below/above 0.15 Hz, each
ROI's recovered amplitude (GLM; 1 = exact), t-score and leakage into a
surrounding shell, the false-positive rate (|t| > 3.29) of the block regressor
elsewhere, and edge sharpness relative to the truth. `--png` adds a panel
(truth and recon mean, temporal std, block t-map). The `_t_lowband` and
`false_pos_frac_lowband` entries repeat the GLM on the DCT components below
the cutoff only, with their own degrees of freedom: a temporal penalty leaves
the residual band-limited, which inflates the plain per-frame t (25-32% of
null voxels above |t| = 3.29 on `20260930ballfat`, vs 0.1% in the low band).

`score` also reads the files [`simulate_fmri/`](../simulate_fmri/README.md)
writes: a digital brain acquired by SNAKE-fMRI along a `scan_info.mat`
schedule, with one activation (`block_occipital`) in the same `truth` layout.
That truth comes from a forward model the recon does not share (T2* decay
along each echo train, BOLD updated at every excitation), so it is not an
inverse crime, at the price of a synthetic object and coils and no B0.

## Performance notes

- K-space is read and gathered one frame at a time, never as a dense
  (Nx, Ny, Nz, Ncoils, Nt) array; HDF5 datasets are read chunk by chunk
  (`utils.read_frames_cropped`), which is ~70× faster than a single read.
- Patch SVDs run in batches sized to a ~4 GB budget
  (`regularizers.patchSVST`), so fine patch sizes on large grids fit on a
  48 GB GPU with no measurable slowdown.
- On the demo dataset (90×90×60, 14 coils, R = 6, RTX A6000): RSS of 60 frames
  ≈ 10 s; CG-SENSE of 2 frames, 20 iterations ≈ 0.5 s (plain) or 6 s (B0);
  wavelet-TV of 1 frame, 50 iterations ≈ 2–35 s depending on the operator.
  For B0 operators, the one-off power iteration for $\sigma_1$ often costs
  more than the reconstruction itself.

## Validation

- The low-rank solver, patch SVD and SENSE operator were ported from the Julia
  `../mslr-recon` and matched it on real data to float32 precision, with the
  same iteration counts (`python -m recon.utils validate <julia .mat>`, which
  disables the operator normalization to reproduce Julia).
- The restructure into these modules was checked bit-for-bit against the
  previous code on a seeded synthetic set; CG and all operators are
  unchanged.
- Tests cover operator adjoints and spectral norms, the B0 operators against a
  brute-force time-varying forward model, the R2* sign convention, the
  wavelet's adjoint and isometry, solver convergence, and end-to-end recovery
  for each `--reg`.

## Adding a regularizer

A regularizer with a closed-form proximal operator (for POGM) provides
`prox(X, step)` and `cost(X)`, like `MultiScaleLowRank`. One without (for
PDHG) provides a linear operator `G`, a proximal operator `h_prox` for the
function applied to `Gx`, and a bound `G_norm_squared` on $\|G\|^2$, like
`WaveletTV`. Then add a `--reg` choice and a small `_solve_*` function in
`sense.py` that builds the data-consistency gradient and calls the solver.
