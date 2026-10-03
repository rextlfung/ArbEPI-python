# simulate-fmri/ — simulated fMRI experiments acquired with ArbEPI

Simulates an fMRI scan of a brain phantom with a known BOLD activation, acquired
with exactly the sampling schedule of an ArbEPI sequence, and writes it in the
format `recon/` reads. Built on [SNAKE-fMRI](https://github.com/mind-inria/snake-fmri)
([documentation](https://mind-inria.github.io/snake-fmri/); Comby et al.,
[arXiv:2404.08282](https://arxiv.org/abs/2404.08282)).

```
scan_info.mat ──► simulate-fmri ──► <name>.mrd ──► recon/<name>_preprocessed.h5 ──► recon.rss / recon.sense
 (sequences/ArbEPI.py)   (SNAKE)                     + truth group                    recon.testbed score
```

It answers questions a scan cannot, because the truth is known: how well a
sampling pattern, echo-train ordering, acceleration factor or reconstruction
recovers an activation of known place, size and time course.

`recon/testbed.py` asks the same kind of question from the other side. Use
both; they fail differently.

| | `recon/testbed.py` | `simulate-fmri/` |
|---|---|---|
| Object | a real fully sampled scan (phantom or head) | BrainWeb digital brain, or an analytic phantom |
| Forward model | the recon's own B0-SENSE operator (an inverse crime by construction) | SNAKE: independent code, T2* decay along each echo train, BOLD updated at every excitation |
| Coils, B0, noise | measured | synthetic coils, no B0, white noise |
| Activation | 2% in four spheres (block, sinusoids) | SNAKE's BOLD model in occipital gray matter |
| Needs | two preprocessed scans | a `scan_info.mat` |

A simulation carries its ground truth in a `truth` group laid out like the
testbed's, so `python -m recon.testbed score` scores both.

## Layout

| File | What it holds |
|---|---|
| `simulate.py` | `simulate()`, `load_protocol()` and the `python -m simulate-fmri.simulate` command line |
| `sampler.py` | `ArbEPISampler`: a SNAKE sampler that plays `scan_info.mat`'s (ky, kz, echo time) schedule |
| `engine.py` | `ArbEPIAcquisitionEngine`: the per-shot signal model (a subclass of SNAKE's EPI engine) |
| `handlers.py` | `EllipsoidActivationHandler`: SNAKE's block-design activation, with the region placed in mm |
| `phantom.py` | BrainWeb at 3 T, the analytic phantom, field-of-view placement, coil sensitivities |
| `export.py` | SNAKE's `.mrd` → `recon/<name>_preprocessed.h5` with the `truth` group |
| `demo.ipynb` | The default protocol, simulated, reconstructed and scored |

Tests: `tests/test_simulate_fmri.py`.

The folder name has a hyphen, so Python's `import` statement cannot name it.
`python -m simulate-fmri.simulate` works; from other code, load the modules with
`importlib`:

```python
import importlib
simulate = importlib.import_module('simulate-fmri.simulate')
simulate.simulate('output/scan_info.mat', 'output/sim')
```

## Setup

One venv holds SNAKE and `recon/`, so a simulation can be reconstructed and
scored without switching environments:

```bash
UV_PROJECT_ENVIRONMENT=.venv-simulate uv sync --extra simulate --extra recon --extra test
```

It is separate from `.venv` and `.venv-recon` for the reason `.venv-recon` is:
`uv sync` is exact, so syncing a shared environment without these extras would
remove them. Three things about the `simulate` extra (see `pyproject.toml`):

- **snake-fmri is pinned to a GitHub commit.** The PyPI release (0.2.0) predates
  the `FOVConfig` API that the upstream documentation describes and that places
  ArbEPI's field of view on the phantom.
- **`ismrmrd` is held below 1.15.** That release made an `ismrmrdHeader`
  argument required, which SNAKE's MRD writer does not pass.
- **cupy is not installed.** SNAKE asks for `mri-nufft[finufft,cufinufft]`, and
  the `cufinufft` extra brings `cupy-cuda13x` and the CUDA 13 toolkit (several
  GB, one CUDA version). The Cartesian path used here needs neither, so
  `[tool.uv] override-dependencies` narrows it to `mri-nufft[finufft]`. SNAKE
  logs `Cupy not available, using CPU.`; that is expected.

SNAKE also logs `Existing <name>.mrd it will be overwritten` followed by
`[Errno 2] No such file or directory` whenever the output file does not exist
yet. That is its writer removing a file that is not there; nothing is wrong.

The BrainWeb phantom is downloaded on first use (about 40 s) and cached in
`~/.cache/brainweb` and `~/.cache/snake-fmri`. `--phantom ellipsoid` needs no
download.

## Quick start

Run everything from the repository root.

```bash
export UV_PROJECT_ENVIRONMENT=.venv-simulate
PY="uv run --extra simulate --extra recon python"
OUT=output/sim

# 1. The sequence: writes output/scan_info.mat (edit params.py to change the protocol)
$PY main.py

# 2. Simulate it
$PY -m simulate-fmri.simulate output/scan_info.mat $OUT --name sim

# 3. Reconstruct, as for a real scan
$PY -m recon.rss $OUT sim
$PY -m recon.sense $OUT sim --reg wavelet-tv --hp-weight 3 --tag hp3

# 4. Score against the truth
$PY -m recon.testbed score $OUT/recon/sim_preprocessed.h5 \
    $OUT/recon/sense_wavelet-tv_hp3/sim_recon.h5 --json hp3.json --png hp3.png
```

Step 1 only needs `scan_info.mat`; any `scan_info.mat` from a past session
works too, which simulates that session's acquisition.

Options of `simulate-fmri.simulate`:

| Option | Default | Meaning |
|---|---|---|
| `--name` | `sim` | output name; the `<seq>` to give `recon/` |
| `--phantom` | `brainweb` | or `ellipsoid` (analytic, no download) |
| `--coils`, `--coils-per-ring` | 16, 8 | receive array: rings of coils around z |
| `--model` | `T2s` | `T2s`: decay along each echo train. `simple`: every sample at TE |
| `--snr` | 1000 | SNAKE's noise level (see below); `inf` for none |
| `--block-on`, `--block-off` | 10, 10 | block design, s |
| `--delta-r2s` | 1000 | peak signal change = TE (ms) / this: 3% at TE = 30 ms |
| `--no-activation` | | a resting phantom |
| `--fa` | from `scan_info.mat` | flip angle, degrees |
| `--frames` | all | only the first N frames |
| `--workers` | half the CPUs | worker processes |
| `--seed` | 0 | noise seed |

## A first result

The default protocol (2.4 mm, 90 x 90 x 60, R = 10 Poisson-disc, radial
ordering, ETL 54, volume TR 0.506 s, 119 frames) on BrainWeb with the default
settings, reconstructed two ways and scored by `recon.testbed score`:

| | CG-SENSE (`--reg none`) | wavelet-TV + temporal high-pass (`--reg wavelet-tv --hp-weight 3`) |
|---|---|---|
| frame error, `nrmse_frame_pct` | 43.4 | 4.5 |
| fluctuation where nothing changes, `fluct_pct` | 37.2 | 0.37 |
| recovered amplitude, `amp_ratio` (1 = exact) | 0.38 | 0.52 |
| t-score of the activation, median (`_t`, `_t_lowband`) | 0.13, 0.13 | 17.3, 7.4 |
| region's mean time course vs truth, `corr` | 0.33 | 0.96 |
| false positives in the low band | 0.1% | 0.04% |
| edge sharpness vs truth | 0.26 | 0.93 |

Unregularized, each frame is inverted alone at R = 10 with 16 coils and the
noise swamps a 2% activation. The regularized reconstruction finds it, at about
half its true size. These are one simulation's numbers, there to show the
chain works; they are not a study of either method.

Timing on a 64-core machine with an RTX A6000 shared with another job: the
simulation takes about 100 s with 16 workers (`--workers 16`), some 60 s of it
the acquisition and the rest building the phantom and writing the output;
CG-SENSE 4.5 min; wavelet-TV 5 min. The `.mrd` is 0.9 GB and
the preprocessed file 0.8 GB.

Noise is seeded (`--seed`), but with more than one worker the shots finish in
a different order from run to run and SNAKE draws the noise in that order, so
two noisy runs differ in their noise. Noise-free runs are identical.

## What is simulated

**The acquisition** comes from `scan_info.mat`, nothing else: matrix size and
field of view, every echo's (ky, kz) location in every shot of every frame, the
echo times, the volume TR and the flip angle. So the sampling masks, the
`mask2epi` partitioning and ordering, R, ETL and TE are those of the sequence
that was generated.

| ArbEPI | SNAKE |
|---|---|
| one shot (one excitation, `ETL` echoes) | one repetition: `seq.TR` = volume TR / Nshots |
| one frame (`Nshots` shots) | one k-space frame |
| x (readout), y, z (the undersampled plane) | array axes 0, 1, 2 |
| echo times in `schedules[..., 2]` | each acquisition's `user_float[0]`, read by the engine |
| flip angle, nominal TE | `seq.FA`, `seq.TE` |

**The object**: BrainWeb subject 4's white matter, gray matter and CSF maps,
resampled to the acquisition grid, with approximate 3 T relaxation times
(`phantom.TISSUE_PROPS_3T`). The field of view is centered on the brain and
tissue outside it is cut off, as an ideal slab excitation would.

**The signal** of each tissue is the spoiled gradient-echo steady state at the
per-shot TR and flip angle, weighted by T2* decay at each sample's own time:
`exp(-t / T2*)` with `t` the echo time from the schedule plus the sample's place
in its readout. The center of k-space is acquired at the nominal TE with radial
ordering and wherever the ordering puts it otherwise, so ordering changes
contrast and blurring the way it does on the scanner.

**The activation** is SNAKE's: a copy of gray matter inside an ellipsoid in the
occipital cortex, weighted by the block design convolved with the Glover HRF and
scaled to peak at `TE / delta_r2s`. The phantom is updated at every excitation,
so the BOLD signal changes between the shots of one frame.

**Coils**: rings of `--coils-per-ring` coils around z (sigpy's birdcage model),
half a field of view apart, so sensitivities vary along both undersampled axes.
The recon is given these exact maps.

**Noise**: complex white noise on every coil's k-space sample, of variance
`2 * mean(image^2) / snr`, where `image` is the noise-free magnitude image over
the whole field of view. `noise_var` in the output is that variance; `recon.sense`
divides by its square root, as for a whitened scan.

## What is not

- **Off-resonance.** No B0 field, so no distortion, no signal dropout, and
  nothing for `--B0` or `--R2star` to correct: the output has no `b0_map`.
  T2* decay *is* simulated and the plain SENSE operator does not model it; that
  mismatch is real.
- **Readout imperfections.** No Nyquist ghost, no ramp sampling, no gradient
  delays. Samples sit on the Cartesian grid. `preprocess/` has nothing to do and
  is skipped.
- **Structured noise.** No physiological noise, drift or motion. (SNAKE has a
  motion handler; pass it through `simulate(handlers=...)`. It is not tested
  here.)
- **Fat and the excitation.** Three tissues, a perfect slab, no fat signal, no
  off-resonance loss of the water excitation.
- **Sensitivity estimation.** The recon gets the true maps, not ESPIRiT maps
  from a calibration scan.

## Output

`<outdir>/<name>.mrd` is SNAKE's own file (phantom, handlers, trajectory,
k-space). `<outdir>/recon/<name>_preprocessed.h5` holds:

| Dataset / attr | Contents |
|---|---|
| `ksp_epi_zf` | (Nx, Ny, Nz, Ncoils, Nframes) zero-filled k-space |
| `omegas`, `echo_times` | (Ny, Nz, Nframes) sampling mask and each sample's time since excitation (s) |
| `smaps` | (Nx, Ny, Nz, Ncoils) the sensitivities used |
| attrs `noise_var`, `whitened`, `t_ref_s`, `fov`, `volume_tr` | as `preprocess/` writes them |
| `truth/x0` | (Nx, Ny, Nz) noise-free magnitude image at TE, averaged over time |
| `truth/roi_masks`, `truth/waveforms` | (1, Nx, Ny, Nz) activated voxels; (1, Nframes) their time course, zero-mean, peak 1 |
| `truth` attrs `roi_names`, `amp` | `block_occipital`; the peak fractional signal change |
| `truth/amp_map` | (Nx, Ny, Nz) the fractional change voxel by voxel |
| `truth/image_rest`, `tissues`, `brain_mask`, `bold_shots`, `stimulus_shots` | the image without activation, tissue fractions, and the BOLD and stimulus time courses per excitation |

The true image of frame t is `x0 * (1 + amp_map * w(t))`. The activation adds
gray-matter signal on top of whatever tissue a voxel holds, so `amp_map` varies
with the tissue mix; `roi_masks` keeps the voxels that are at least 90% tissue
and at half the largest change among those or more (a voxel at the edge of the
brain has almost no signal to take a ratio against), and `amp` is their median,
which is what `recon.testbed score`'s `amp_ratio` divides by.

## What was changed relative to stock SNAKE, and why

Everything is done through SNAKE's own extension points (a sampler, an engine
and a handler subclass); SNAKE itself is not patched.

- **Sampler.** SNAKE's 3D-EPI sampler acquires full ky lines of one kz plane
  per shot. `ArbEPISampler` writes the schedule's arbitrary (ky, kz) list, with
  kx reversed on odd echoes.
- **Echo timing.** SNAKE's T2* model spaces all samples of a shot one dwell
  apart and takes TE at the sample nearest k = 0 in the first shot. The engine
  here uses the sequence's echo times, which the sampler stores per acquisition.
- **FFT centering.** SNAKE's FFT helper applies its two shifts the other way
  round from `recon/operators.py`. They agree for even matrix sizes and differ
  by one sample for odd ones. The engine uses the recon's convention.
- **Resampling.** SNAKE resamples the phantom at every shot. Without cupy that
  starts a process pool each time (about 1 s per shot) to copy an array already
  on the grid. The engine resamples only a phantom that is off the grid.
- **Activation region.** SNAKE's built-in occipital ellipsoid is defined in
  voxels of BrainWeb's full grid and lands elsewhere on another field of view.
  `EllipsoidActivationHandler` places it in mm. It also sets the HRF
  convolution's `oversampling` to 1 instead of 50: at a 50 ms shot TR, 50 means
  a 1 ms grid and minutes to hours per regressor, for a 1% difference.
- **Coils.** SNAKE's birdcage ring goes around the first array axis (x here),
  which leaves no variation along z, one of the two undersampled axes.
