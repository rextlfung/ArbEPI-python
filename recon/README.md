# recon/ — image reconstruction

Turns the zero-filled k-space written by `preprocessing/` into images. There
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
| `regularizers.py` | `MultiScaleLowRank` and `WaveletTV` (plus the pieces they're built from) |
| `solvers.py` | `pogm_restart` (POGM/FPGM/PGM), `pdhg` (primal-dual), `cg` |
| `sense.py` | `run_sense()` and the `python -m recon.sense` command line |
| `rss.py` | `run_rss()` and the `python -m recon.rss` command line |
| `utils.py` | File I/O, spectral-norm estimation, the tSNR report, and one-off analyses |
| `demo.ipynb` | Worked examples on `20260915ball/2_6x_2.4mm` |

Tests mirror this: `tests/test_recon_<module>.py`.

## Setup

Everything in `recon/` runs in its own venv, `.venv-recon` (torch is a large,
GPU-specific install that has no reason to share a venv with the
GERecon-locked preprocessing environment):

```bash
uv venv .venv-recon
uv pip install --python .venv-recon/bin/python -e ".[recon,test]"
```

Run everything from the repository root (`recon/` imports helpers from
`preprocessing/`). Install the torch build matching your CUDA version first if
the default wheel doesn't fit your GPU. Everything also runs on CPU, just much more slowly; the
device defaults to `cuda` when a GPU is available and `cpu` otherwise
(`--device` overrides it).

## Inputs

All read from `<datdir>/recon/`, as written by `preprocessing/`:

| File | Contents | Needed for |
|---|---|---|
| `<seq>_epi_zf.h5` | `ksp_epi_zf` (Nx, Ny, Nz, Ncoils, Nframes), zero where not sampled; `omegas` (Ny, Nz, Nframes) sampling mask; `echo_times` (Ny, Nz, Nframes) in seconds; attr `noise_var` | everything |
| `smaps_<seq>_sigpy.h5` | `smaps` (Nx, Ny, Nz, Ncoils), ESPIRiT sensitivity maps | SENSE |
| `<seq>_b0map.h5` | `b0map_hz` (Nx, Ny, Nz), B0 field map in Hz on the EPI grid | `--B0` |
| `<seq>_gre.h5` + `<datdir>/seqs/<seq>/scan_info.mat` | dual-echo deGRE data (for the R2* map) and the nominal TE | `--R2star` |

## Quick start

```bash
PY=.venv-recon/bin/python
DAT=/StorageRAID/rexfung/20260915ball
SEQ=2_6x_2.4mm

# Root-sum-of-squares
$PY -m recon.rss $DAT $SEQ

# Iterative SENSE with no regularizer (conjugate gradient), B0-corrected
$PY -m recon.sense $DAT $SEQ --reg none --B0

# Locally low rank (6x6x6 patches, stride 3), B0 + R2* corrected
$PY -m recon.sense $DAT $SEQ --reg lowrank --B0 --R2star --patch 6 6 6 --stride 3 3 3

# Global + local low rank (two scales)
$PY -m recon.sense $DAT $SEQ --reg lowrank --B0 \
    --patch full --stride full --patch 15 15 15 --stride 5 5 5

# L1-wavelet + TV on the first 10 frames
$PY -m recon.sense $DAT $SEQ --reg wavelet-tv --B0 --frames 0-9

# Temporal stability (tSNR, drift, fluctuation) of any saved reconstruction
$PY -m recon.utils tsnr $DAT/recon/rss/${SEQ}_recon.nii.gz --tr 1.0
```

Use `--frames` (e.g. `0-9` or `0,5,7`) and `--niter` to try settings quickly
before a full run. From Python, `run_sense(...)` takes the same options and
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
- Sign convention (Sutton, Noll & Fessler 2003): the forward model multiplies
  the image by $e^{+i 2\pi \Delta f(r) t}$ before the FFT.
- Each segment is a full SENSE transform, so cost grows linearly with L: one
  forward + adjoint over 30 frames of a 240×240×45, 18-coil dataset takes
  5.8 s at L = 6 and 30.6 s at L = 32 (`python -m recon.utils benchmark`).

**`SENSE_B0_R2star`** — also models magnitude decay. The field becomes complex,
$\psi(r) = i 2\pi \Delta f(r) - R_2^*(r)$, and each segment's phasor is
$e^{\psi(r)(t_l - t_{\text{ref}})}$. $t_{\text{ref}}$ is the nominal TE, so the
reconstruction target is "the image at TE". The R2* map is estimated from the
dual-echo deGRE data (`preprocessing/r2star_map.py`). The physics lives in
`SENSE_B0_R2star.segment_phasors`, which is computed once and shared by every
frame (a separate copy per frame doesn't fit in GPU memory).

## Regularizers, solvers and λ

| `--reg` | $g(x)$ | Solver |
|---|---|---|
| `none` | 0 | `cg`: conjugate gradient on $A^H A x = A^H y$ |
| `lowrank` | multi-scale low rank | `pogm_restart` (default POGM; `--mom fpgm` or `pgm` also available) |
| `wavelet-tv` | $\lambda_{\ell_1}\|Wx\|_1 + \lambda_{TV}\|Dx\|_1$, per frame | `pdhg` (primal-dual) |

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
   `noise_var` attribute of `<seq>_epi_zf.h5`. `run_sense` divides the k-space
   by $\sqrt{\texttt{noise\_var}}$. Whitening already targets 1, so this is
   normally a few-percent correction (0.93–1.14 on the `20260915ball` and
   `20260918ball` datasets). Files written before the attribute existed are
   used as they are (a message says so);
   `preprocessing.preprocess.record_noise_var(cfg, paths)` adds it.
   The noise level can't be estimated from the acquired k-space itself on
   these high-SNR datasets: every candidate region is dominated by signal
   leakage, not noise.
2. **Operator.** For `lowrank`, $A$ is divided by its spectral norm
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
