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
  `rel_changes` (POGM), and under `trace/` the diagnostics of the other
  solvers: PDHG (`--reg wavelet-tv`) `iter`, `dc_cost`, `reg_cost` every
  `--trace-every` iterations (default 5; each evaluation costs one forward
  operator and one regularizer pass, about 6% of a run at 5) and `rel_change`
  (primal change per iteration), with a leading frame axis for the per-frame
  solver; CG `rel_residual`. The `.json` records `n_iters` (iterations run) and,
  where there is a stopping rule (POGM, CG), `converged`. PDHG always runs its
  `--niter` iterations, so read its trace for convergence: the primal objective
  is not monotone in a primal-dual method, only its trend is meaningful.

Image intensities are **not quantitative**: the data and operator scalings
described below are not undone.

## The optimization problem

Every iterative method here solves one problem,

$$
\hat{x} \;=\; \arg\min_{x}\; \underbrace{\tfrac12\,\|A x - y\|_2^2}_{f(x)\ \text{data fidelity}} \;+\; \underbrace{g(x)}_{\text{regularizer}},
$$

where $x \in \mathbb{C}^{N_x \times N_y \times N_z \times N_t}$ is the image time
series, $y$ the acquired multi-coil k-space at the sampled locations, and $A$
the encoding operator. The three pieces are chosen independently: the
operator $A$ ([below](#encoding-operators-operatorspy)), the regularizer
$g$ (`--reg`), and the temporal terms that may be added to $g$.

### Data fidelity and the encoding operator

Frame $t$ is acquired at its own sampling set $\Omega_t$ (a new mask every
frame), so $A$ is block diagonal, $A = \operatorname{blkdiag}(A_1, \dots, A_{N_t})$,
and $f(x) = \tfrac12 \sum_t \|A_t x_t - y_t\|_2^2$. For the plain SENSE
operator, coil $c$ at k-space location $k \in \Omega_t$ is

$$
(A_t x_t)[k, c] \;=\; \sum_{r} s_c(r)\, x_t(r)\, e^{-i 2\pi\, k \cdot r},
$$

with $s_c$ the coil sensitivity maps. With the field map $\Delta f(r)$ and
R2\* map $R_2^*(r)$, sample $k$ is acquired at time $\tau_k$ (its echo time
from `schedules`) and carries

$$
(A_t x_t)[k, c] \;=\; \sum_{r} s_c(r)\, x_t(r)\,
e^{\psi(r)\,(\tau_k - t_{\mathrm{ref}})}\, e^{-i 2\pi\, k \cdot r},
\qquad
\psi(r) = i\,2\pi\,\Delta f(r) - R_2^*(r),
$$

(phase only when $R_2^* \equiv 0$). Evaluating the exponential per sample is
as expensive as a non-uniform FFT, so it is approximated by **time
segmentation**: $e^{\psi(r)(\tau - t_{\mathrm{ref}})} \approx \sum_{l=1}^{L}
b_l(\tau)\, e^{\psi(r)(t_l - t_{\mathrm{ref}})}$, which turns $A_t$ into $L$
ordinary SENSE transforms of phased images, combined per sample with weights
$b_l$ (see [Encoding operators](#encoding-operators-operatorspy)).

### The regularizers $g$

| `--reg` | Cost $g$ | Solver |
|---|---|---|
| `none` | $0$ | CG |
| `mslr` | $\displaystyle\sum_{k=1}^{K} \lambda_k \sum_{b} \big\|P_b(X_k)\big\|_*$, with $x = \sum_k X_k$ | POGM / FPGM / PGM |
| `wavelet-tv` | $\displaystyle\sum_{t} \big(\lambda_{\ell_1}\|W x_t\|_1 + \lambda_{TV}\|D x_t\|_1\big)$ | PDHG |

**No regularizer.** $\hat{x} = \arg\min_x \tfrac12\|Ax-y\|^2$, the
least-squares (CG-SENSE) solution, found from the normal equations
$A^H A\, x = A^H y$.

**Multi-scale low rank** (Ong & Lustig). The unknown becomes $K$ components
$X_1, \dots, X_K$, one per patch size, that sum to the image, so the problem is

$$
\min_{X_1,\dots,X_K}\;
\tfrac12\Big\|A\Big(\sum_{k} X_k\Big) - y\Big\|_2^2
\;+\; \sum_{k=1}^{K} \lambda_k \sum_{b \in \mathcal{B}_k} \big\|P_b(X_k)\big\|_* .
$$

$P_b$ extracts patch $b$ of scale $k$ as a (voxels in patch) $\times$ $N_t$
matrix, and $\|\cdot\|_*$ is its nuclear norm (the sum of singular values).
One small patch gives a locally low-rank prior; a whole-volume patch a
globally low-rank one. The weights, with $p_k$ voxels per patch and $N_{\mathrm{vox}}$
voxels per frame, are

$$
\lambda_k = \lambda_{\mathrm{global}}
\Big(\sqrt{p_k} + \sqrt{N_t} + \sqrt{\log\!\big(N_{\mathrm{vox}} N_t / \max(p_k, N_t)\big)}\Big),
$$

the expected spectral norm of a $p_k \times N_t$ matrix of unit-variance
Gaussian noise ([scaling below](#scaling-so-that-λ-means-what-the-paper-says)).

**Wavelet + total variation.** Each frame is regularized with an $\ell_1$
penalty on orthogonal 3D wavelet coefficients $W x_t$ and an anisotropic total
variation $\|Dx_t\|_1 = \sum_{d \in \{x,y,z\}} \|D_d x_t\|_1$, with $D_d$
the periodic finite difference along axis $d$:

$$
\min_{x}\;\tfrac12\|Ax-y\|_2^2
+ \sum_t \Big(\lambda_{\ell_1}\|W x_t\|_1 + \lambda_{TV}\|D x_t\|_1\Big).
$$

### Temporal terms (optional, any `--reg`)

These couple the frames, so the problem is solved for all of them at once.

$$
g(x) \;\leftarrow\; g(x)
\;+\; \underbrace{\tfrac{\mu}{2}\,\|P x\|_2^2}_{\texttt{--hp-weight}}
\;+\; \underbrace{\lambda_{TT}\,\|D_t x\|_1}_{\texttt{--lamb-ttv}\ (\text{wavelet-tv})}
$$

- **Temporal high-pass**: $P$ is the orthogonal projector onto temporal
  frequencies above the cutoff (`--hp-cutoff`, 0.15 Hz) in the DCT-II basis,
  $P = C^{T}\operatorname{diag}(\mathbb{1}[f_j > f_c])\,C$ with $f_j = j / (2 N_t\,\mathrm{TR})$.
  The penalty is exactly zero for content at or below the cutoff, so it cannot
  bias a BOLD-band signal.
- **Temporal TV**: $D_t$ is the non-periodic difference along time,
  $(D_t x)_t = x_{t+1} - x_t$, summed over voxels.

### Scaling

The weights $\lambda_k$ assume unit-variance noise and a unit-norm operator, so
before solving, `run_sense` replaces

$$
y \leftarrow y / \sigma_n,
\qquad
A \leftarrow A / \sigma_1(A),
$$

with $\sigma_n^2$ the measured noise variance (`noise_var`) and
$\sigma_1(A)$ the spectral norm from power iteration. Both are applied for
`mslr`; the wavelet-TV solve normalizes $A$ the same way and each frame's data by
its 99th-percentile magnitude. Neither is undone on the output (this is not
quantitative imaging).

### Solvers

- **Proximal gradient (POGM / FPGM / PGM)** for `mslr`:
  $x^{+} = \operatorname{prox}_{g/L}\!\big(x - \tfrac1L \nabla f(x)\big)$ with
  momentum (and gradient restart), $L = \sigma_1(A)^2$. The prox of the
  nuclear norm is singular-value soft-thresholding,
  $\operatorname{prox}_{\beta\|\cdot\|_*}(M) = U \max(\Sigma - \beta, 0) V^H$.
- **Primal-dual (PDHG / Condat-Vu)** for `wavelet-tv`: the problem is written
  $\min_x f(x) + h(Gx)$ with $G = [W;\, D;\, P]$, since the sum of a wavelet
  $\ell_1$ and a TV term has no closed-form prox, but each block does
  (soft-thresholding; a closed-form shrinkage for the quadratic $P$ block).
  Temporal TV is the proximable term, a 1D TV denoising per voxel.
- **CG** for `none`, on $A^H A x = A^H y$ (with $+\mu P$ added when
  `--hp-weight` is set).

`sense.py` shows a `tqdm` progress bar for all three (stderr; POGM and CG with
cost / relative change or residual, PDHG with the iteration count, labelled
`frame i/Nt` in the per-frame mode). Each bar disappears when its solve
finishes, leaving the printed summary. The solver functions take
`progress=False` by default, so library use and tests stay quiet.

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
- `--cycle-spin` (with `--cycle-spin-seed`) moves every local scale's patch grid
  by a random shift in `[0, stride)` along each axis at every iteration
  (cycle spinning), so non-overlapping patches (`--stride` = `--patch`, `1/p^3`
  of the SVDs of stride 1, about `1/8` of stride `p/2` per axis) stand in for
  overlapping ones without a fixed block boundary. The shifted volume is rolled
  periodically, so patches at one edge wrap to the opposite edge (harmless while
  the object stays away from the field of view's edge). The penalty is a
  different one every iteration, so the cost trace is noisy and `--conv-tol`
  and POGM's restart rule do not apply: use `--mom pgm` (or `fpgm`) with
  `--conv-tol 0`. Global (whole-volume) scales are not shifted. The returned
  image is one prox output, so it carries the last iteration's grid.
- `--init RECON.h5` starts from that recon's `X` (same frames and scales) instead of
  A^H y, to run more iterations of an earlier result. PGM has no momentum, so a PGM
  continuation equals the longer run (tested); FPGM/POGM restart their momentum.
  With cycle spinning give a new `--cycle-spin-seed`, or the shifts repeat.
- `--dc-chunk FRAMES` (mslr) keeps the k-space in host memory and computes the
  data-consistency cost and gradient that many frames at a time
  (`solvers.ChunkedDataFidelity`), instead of holding the series' k-space, the
  prediction A x and the residual on the GPU: three tensors of Nc / R times an
  image series each (7.7 GB each for 516 frames of 23 coils at R = 10 on a
  136 x 136 x 44 grid). The result is the same up to the order the cost is
  summed (tested, with and without `--hp-weight`). A is applied frame by frame
  either way, so the chunk size changes neither the result nor the speed (8 frames
  of run C, B0, on a GPU shared with another job: 0.22 s/frame for the whole-series
  product and for chunks of 1, 2 and 4, 0.16 for 8): it only sets how many frames of
  k-space are on the device (15 MB each at that size), so any value from 1 to a few
  dozen will do. Measured on run C (516 frames, 23 coils, 136 x 136 x 44, one 9^3 local
  scale, PGM, a card with 38 GB free): without it the first iteration ran out of memory
  at 35.7 GB allocated, asking for 7.2 GB more; with `--dc-chunk 8` the k-space load left
  7.7 GB more free and a 100-iteration run fit, at a peak of 35.9 GB allocated over its
  first iterations. The rest of that peak is the solver's own image-sized buffers (3.4 GB
  each for 516 frames), which this does not touch: at this size it is a fit, not a
  comfortable margin.
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
  non-periodic difference, in `SpatioTemporalWaveletTV`. It penalizes
  frame-to-frame jumps at every frequency, and shrinks the amplitude of real
  changes too. It enters PDHG as the proximable term (`TemporalTVProx`: 1D TV
  denoising of each voxel's time series, a warm-started inner FISTA on its
  dual), not as a dual block of $G$: there its dual grows by
  $\sigma\|D_t x\|$ per iteration, tiny for a near-static image next to $L$,
  and in 100 iterations every $L$ above ~0.04 gave the same image (review item
  263).

With incoherent sampling (a new random mask every frame) the aliasing is
roughly white in time, so a temporal penalty above the cutoff removes only
the share of it outside the kept band; the in-band share looks exactly like a
slow signal to any temporal prior. `--joint` runs the joint wavelet-TV solver
without a temporal term (one data scale for all frames instead of one per
frame), the baseline for the temporal variants. `--tag` suffixes the output
directory so differently configured runs don't overwrite each other.

## Scoring a reconstruction against a known truth

Not part of `recon/`: `python -m analyze.score <truth.h5> <recon.h5>` (see
[analyze/README.md](../analyze/README.md#scoring-against-a-known-truth)) scores
a `<name>_recon.h5` against a file with a `truth` group, which
[`simulate_fmri/`](../simulate_fmri/README.md) writes -- from a simulated
session, or from a real fully sampled run with activation injected
(`python -m simulate_fmri.from_real`). This was `recon/testbed.py` until
2026-10-13.

## Performance notes

- K-space is read and gathered one frame at a time, never as a dense
  (Nx, Ny, Nz, Ncoils, Nt) array; HDF5 datasets are read chunk by chunk
  (`utils.read_frames_cropped`), which is ~70× faster than a single read.
- Patch SVDs run in batches (`regularizers.patchSVST`). On the CPU a thread pool
  of single-threaded LAPACK calls (torch's batched SVD does not parallelize over
  the batch; the pool size is `torch.get_num_threads()`, so `OMP_NUM_THREADS`
  bounds it on a shared machine). On the GPU, cuSOLVER's `gesvda` for batches of
  moderate matrices, with a fallback to the default driver when it raises
  (rank-deficient patches); its workspace is ~20x its input, so those calls
  carry at most 96 MB of patches. Patches with a Frobenius norm below the
  threshold skip the SVD. Measured on 90x90x60, 6^3 patches at stride 3 (16k
  patches), 119 frames: prox 79 s and 22 GB peak -> 8 s and 2.7 GB (GPU); 12.7x
  on the CPU.
- The encoding operator has no `fftshift`/`ifftshift` copies (shifts are a
  precomputed phase and index at the sampled locations), halving `SENSE` and
  `SENSE_B0` apply/adjoint time. `--reg mslr` with PGM/POGM reuses A x between
  the cost and the next gradient (one of three operator applications per
  iteration). The power iteration for sigma1 starts each frame from the previous
  frame's vector. K-space frames are inflated by a pool of 8 threads
  (`utils.iter_frames`), 3.7x faster than h5py on gzip-compressed files.
- Real data (`20260930ballfat/1_10x_radial`, 124 frames, 13 coils, B0, mslr POGM,
  6^3 patches stride 3, sigma1A fixed): 82.4 -> 62.5 s/iteration and 30.3 -> 10.1
  GB peak GPU memory, same costs, images equal to 1.2e-5. Stride 6, 10 frames:
  2.83 -> 1.46 s/iteration; the power iteration of that run 257 -> 87 s.
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
