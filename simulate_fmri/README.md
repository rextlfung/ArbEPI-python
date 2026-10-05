# simulate_fmri/ — simulated fMRI scans acquired with ArbEPI

Simulates an fMRI session of a brain phantom with a known BOLD activation,
acquired with exactly the sampling schedule of an ArbEPI sequence, and hands it
to the same `preprocess/` and `recon/` code a real scan goes through. Because
the truth is known, every stage can be checked against it: the calibrated
readout delay, the ghost correction, the sensitivity and field maps, and the
activation in the reconstructed series.

```
                         ┌─ session.py ─► scanarchives/*.h5 ─► preprocess ─► <name>_preprocessed.h5 ─┐
scan_info.mat ───────────┤   raw readouts of all four scans                                           ├─► recon.sense ─► recon.testbed score
 (sequences/ArbEPI.py)   └─ ideal.py ───────────────────────────────────► <name>_preprocessed.h5 ────┘        against <name>_truth.h5
```

There are two levels of fidelity.

| | `session.py` (raw data) | `ideal.py` |
|---|---|---|
| Writes | raw ADC readouts of the noise scan, EPIcal, deGRE and ArbEPI | k-space on the Cartesian grid, already in recon's format |
| Then | `preprocess/` runs on it, as on a scan | nothing; `recon/` reads it directly |
| Readout | ramp-sampled on `kxo`/`kxe`, with a readout delay and odd/even phase | ideal |
| Field | B0 map from the head's susceptibility; breathing | none |
| Decay and BOLD | at every sample's own time; BOLD is an R2* change | decay per echo; BOLD is an amplitude change |
| Coils | 32 loops, sensitivities estimated by ESPIRiT from the simulated deGRE | given to the recon |
| Noise | thermal with a coil covariance, plus physiological | thermal, white |
| Object grid | finer than the acquisition (intravoxel dephasing, partial volume) | the acquisition's |
| Built on | this repo's signal model (`forward.py`); SNAKE's phantom and BOLD regressor | [SNAKE-fMRI](https://github.com/mind-inria/snake-fmri)'s acquisition engine |
| Use it for | how the whole pipeline behaves on realistic data | isolating sampling, ordering and noise; an exact reference |

`recon/testbed.py` is a third option: a real object, coils and field, but
k-space synthesized by the recon's own operator. Conclusions should hold on
all of them, then on a scan.

## Layout

| File | What it holds |
|---|---|
| `session.py` | `simulate_session()`, `SessionConfig`, and `python -m simulate_fmri.session` |
| `forward.py` | The signal equation: ramp-sampled EPI readouts and the deGRE's k-space from spins on a fine grid |
| `b0.py` | Field map from the susceptibility of the head (dipole convolution, air cavities, shim) |
| `coils.py` | Receive array defined in space; coil noise covariance |
| `physio.py` | Physiological noise: BOLD-like, cardiac, respiratory, drift |
| `ideal.py` | `simulate()` and `python -m simulate_fmri.ideal`: SNAKE's engine, output in recon's format |
| `sampler.py`, `engine.py`, `handlers.py`, `export.py` | The ideal mode's SNAKE sampler, engine and activation handler, and its export |
| `protocol.py` | `load_protocol()`: the acquisition recorded in `scan_info.mat` |
| `phantom.py` | BrainWeb at 3 T, an analytic phantom, the head outline and air cavities |
| `demo.ipynb` | A session simulated, preprocessed, reconstructed with and without B0, and scored |

Tests: `tests/test_simulate_fmri_forward.py` (the signal model against a
brute-force sum), `..._models.py` (field, coils, physiology), `..._session.py`
(sessions through the real `preprocess()`), `test_simulate_fmri.py` (ideal mode).

## Setup

One venv holds SNAKE, `preprocess/`'s dependencies and `recon/`, so a session
can be simulated, preprocessed, reconstructed and scored without switching:

```bash
UV_PROJECT_ENVIRONMENT=.venv-simulate uv sync --extra simulate --extra preprocessing --extra recon --extra test
```

- **snake-fmri is pinned to a GitHub commit.** The PyPI release (0.2.0) predates
  the `FOVConfig` API the upstream documentation describes.
- **`ismrmrd` is held below 1.15**, which made a header argument required that
  SNAKE's MRD writer does not pass.
- **cupy is not installed**: `[tool.uv] override-dependencies` narrows SNAKE's
  `mri-nufft[finufft,cufinufft]` to `mri-nufft[finufft]`. SNAKE logs
  `Cupy not available, using CPU.`; that is expected.
- **GERecon is not needed.** `preprocess/` reads simulated archives with h5py
  (`preprocess.utils.ArchiveReader` recognizes them).
- **The B0 map needs julia**, as for real data (`preprocess/README.md`).
- The raw-data simulation uses torch, on the GPU when there is one.

BrainWeb is downloaded on first use (about 40 s) and cached in `~/.cache`.
`--phantom ellipsoid` needs no download.

## Quick start

Run everything from the repository root.

```bash
export UV_PROJECT_ENVIRONMENT=.venv-simulate
PY="uv run --extra simulate --extra preprocessing --extra recon python"
OUT=output/sim

# 1. The sequences: writes output/scan_info.mat (edit params.py to change the protocol)
$PY main.py

# 2. Simulate the session: raw readouts of all four scans, and the truth
$PY -m simulate_fmri.session output/scan_info.mat $OUT --name sim

# 3. Preprocess and reconstruct, as for a real scan
$PY -m preprocess.batch_preprocess $OUT sim
$PY -m recon.sense $OUT sim --reg wavelet-tv --B0 --hp-weight 3 --volume-tr 0.506 --tag hp3

# 4. Score against the truth
$PY -m recon.testbed score $OUT/sim_truth.h5 \
    $OUT/recon/sense_wavelet-tv_b0_hp3/sim_recon.h5 --json hp3.json --png hp3.png
```

Any `scan_info.mat` works in step 2, including one from a past session, which
simulates that session's acquisition. The ideal mode replaces steps 2 and 3's
first line: `$PY -m simulate_fmri.ideal output/scan_info.mat $OUT --name sim`
writes `$OUT/recon/sim_preprocessed.h5` directly (score against that file
instead of `sim_truth.h5`).

Options of `simulate_fmri.session`:

| Option | Default | Meaning |
|---|---|---|
| `--name` | `sim` | the `<seq>` of `preprocess/` and `recon/` |
| `--phantom` | `brainweb` | or `ellipsoid` (analytic, no download) |
| `--frames` | all | only the first N frames |
| `--coils` | 32 | receive coils, in rings of 8 |
| `--grid-factor` | 2 | spins per voxel per axis |
| `--noise` | 1 | thermal noise relative to the calibrated level; 0 = none |
| `--b0-scale`, `--shim-order` | 1, 1 | field map scale (0 = uniform) and the shim order removed |
| `--delay` | −0.3 | readout delay, samples |
| `--oe-phase` | −0.25 −0.32 | odd/even phase at the first and last echo pair, rad |
| `--physio` | 1 | physiological noise relative to the calibrated level; 0 = none |
| `--no-activation`, `--delta-r2s`, `--block-on`, `--block-off` | −0.98 1/s, 10 s, 10 s | the activation |
| `--preprocess` | | run `preprocess.batch_preprocess` afterwards |
| `--device`, `--seed`, `--fa` | | |

From Python, `simulate_session(scan_info, outdir, name, cfg=SessionConfig(...))`
takes the same settings and more (`physio=PhysioConfig(...)`, a custom
`Anatomy`).

## What the raw-data simulation models

Everything about the acquisition comes from `scan_info.mat`: matrix and field
of view, each echo's (ky, kz) and echo time in every shot of every frame, the
readout trajectories `kxo`/`kxe`, ADC dwell, flip angles, and the deGRE's
matrix, echo times and TR.

| | Model | Sized from |
|---|---|---|
| **Object** | BrainWeb subject 4's white matter, gray matter and CSF, on a grid `--grid-factor` times finer than the acquisition | T1: Wansapura 1999; T2*: Peters 2006/2007, 59.7 and 54.6 ms (see below) |
| **Excitation** | spoiled steady state at the per-shot TR and local flip angle; slab profile of 0.9 × the z field of view; water excitation's loss of flip off resonance | `lib/make_excitation_pulse.py`, `lib/make_water_excitation.py` |
| **B0** | dipole field of the head's susceptibility (tissue −9.05 ppm, air +0.36 ppm) with air cavities for the sinuses, mastoids and ear canals, minus a linear shim | a 3 T head field map: std 45 Hz, −267 to +190 Hz; the model gives 28 Hz, −144 to +272 Hz (0.1–99.9 percentiles over the brain, as for the scan) |
| **Decay** | exp(−t R2*) per tissue at each ADC sample's own time | same T2* values |
| **Readout** | ramp-sampled: an exact Fourier sum at `kxo`/`kxe`, shifted by a readout delay; a constant phase on every other echo, drifting along the train | real sessions: delay −0.3 samples, odd/even phase −0.25 to −0.32 rad |
| **Coils** | 32 loops in four rings around z, unit root-sum-of-squares | — |
| **Thermal noise** | complex Gaussian per raw sample, with a coil covariance; scales with voxel volume and sampling time | a 3 T 32-channel head deGRE: SNR 115 for a fully sampled volume of the default protocol; covariance from real `W` matrices (std ratio 1.1–1.2, correlations ≤ 0.02) |
| **Activation** | a change of R2* in the occipital gray matter, block design ⊛ Glover HRF, updated at every excitation | van der Zwaag 2009: −0.98 1/s at 3 T (2.9% at TE 30 ms) |
| **Physiological noise** | BOLD-like R2* fluctuations (smooth patterns, 1/f below 0.1 Hz), cardiac pulsation (CSF), respiration (signal and field), drift, all per excitation | Bodurka 2007: λ = 0.0128 (gray), 0.0085 (white), 0.021 (CSF); Van de Moortele 2002 for the breathing field |
| **deGRE** | fully sampled dual-echo 3D GRE at its own matrix, field of view and TR, with the same object, field and coils; leading receive-gain block | `sequences/deGRE.py` |
| **EPIcal, noise** | blip-free echo trains with the same readout; `ceil(20 Ncoils² / Nfid)` noise readouts | `sequences/EPIcal.py`, `sequences/noise.py` |

The signal model (`forward.py`) evaluates each scan from spins on the fine
grid, so a voxel dephases across its own field gradient and mixes tissues.
Since an echo train visits k-space at fixed echo times, everything spatial is
computed once per echo index and each shot is a weighted sum; the cost does not
grow with the length of the run. The default protocol (119 frames, 32 coils,
spins on 180 × 180 × 120) takes about 30 s on a GPU.

### 3 T T2* values

| Source | Gray matter | White matter | Notes |
|---|---|---|---|
| Peters et al., Proc ISMRM 14 (2006) 926; Magn Reson Imaging 2007;25:748 | 59.7 ms | 54.6 ms | cortical; through-slice dephasing fitted and removed. Without that correction: 47.1 and 44.0 ms. Caudate 46 ms, putamen 45 ms |
| Wansapura et al., J Magn Reson Imaging 1999;9:531 | 41.6 (occipital), 51.8 (frontal) ms | 48.4 (occipital), 44.7 (frontal) ms | FLASH, 14 echo times, no dephasing correction |
| van der Zwaag et al., NeuroImage 2009;47:1425 | 55 ms | | implied by ΔR2*/R2* = 0.054 and ΔR2* = 0.98 1/s in active motor cortex |

The simulation uses Peters' corrected values (`phantom.TISSUE_PROPS_3T`): the
field inhomogeneity that shortens an apparent T2* is simulated separately, so
the uncorrected values would count it twice.

## What it does not model

- **Motion**, including the field's dependence on head position.
- **Fat.** Three tissues; the scalp and skull give no signal, and there is no
  chemical shift.
- **Gradient imperfections beyond a delay**: no eddy currents, no trajectory
  errors along ky or kz, no concomitant fields.
- **Flow and inflow**, and T1 recovery between excitations other than the
  steady state (no approach to it at the start of a run).
- **B1**: the transmit field is uniform; receive sensitivities have unit
  root-sum-of-squares, so there is no image shading.
- **Perturbations beyond first order.** BOLD and physiological fluctuations are
  linearized (an error of 5 × 10⁻⁴ of the signal for a 3% change), and the
  evolution during a readout is a second-order Taylor expansion (2% at 300 Hz
  at the ends of a readout).
- **The air cavities are not anatomy.** BrainWeb's head model has no air inside
  it; the cavities are ellipsoids sized to give a realistic field.

## Output

`simulate_fmri.session` writes a session directory:

| File | Contents |
|---|---|
| `scanarchives/<name>_{noise,cal,epi}.h5`, `gre.h5` | raw readouts `[Nacq, Ncoils, Nfid]` in acquisition order (simulated-archive format, `preprocess/utils.py`) |
| `seqs/<name>/scan_info.mat` | a copy of the sequence's record |
| `<name>_truth.h5` | the `truth` group below, and `volume_tr` |

| `truth/` | |
|---|---|
| `x0` | (Nx, Ny, Nz) noise-free magnitude at the nominal TE, at the acquisition's resolution (the fine-grid image cut to the acquired k-space), time-averaged |
| `roi_masks`, `waveforms`; attrs `roi_names`, `amp` | the activated voxels, their time course (zero-mean, peak 1) and peak fractional change: `recon/testbed.py`'s layout |
| `amp_map`, `r2s_change` | the fractional change voxel by voxel (zero outside `brain_mask`); the R2* change per excitation |
| `b0_map`, `smaps`, `degre/b0_map` | the field each scan can measure (magnetization-weighted) and the coil maps, on the EPI and deGRE grids |
| `noise_covariance`, `oe_phase`; attrs `delay`, `snr0`, `sigma_raw` | what was injected |
| `physio/*` | the physiological time courses per excitation |
| `tissues`, `brain_mask`, `image_rest` | |

`simulate_fmri.ideal` writes `<name>.mrd` (SNAKE's file) and
`recon/<name>_preprocessed.h5` with the same `truth` group inside it.

## Checked against the real pipeline

On the default protocol (2.4 mm, 90 × 90 × 60, R = 10, 119 frames) with
BrainWeb, `preprocess/` recovered from the simulated raw data:

| | Injected | Recovered by `preprocess/` |
|---|---|---|
| Readout delay | −0.30 samples | −0.30 |
| Odd/even phase, first / middle / last echo pair | −0.250 / −0.285 / −0.320 rad | −0.263 / −0.300 / −0.338 |
| Noise variance after whitening | — | 0.95 |
| Coil compression | 32 coils | 19 virtual coils at 99.9% energy |
| Sensitivity maps | | agreement 0.998 per voxel (median) |
| B0 map | std 28 Hz, −144 to +272 Hz | correlation 0.95 |
| R2* (gray, white: 16.8, 18.3 1/s) | | median 15.8 1/s |

The tests do the same on small sessions, plus the exact case: with only the
object in the data, `preprocess()`'s k-space reconstructs the truth image to
within 1%.

Reconstructing that preprocessed session (`recon.sense --reg wavelet-tv
--hp-weight 3`, 100 iterations) and scoring it against the truth:

| | without `--B0` | with `--B0` |
|---|---|---|
| frame error, `nrmse_frame_pct` | 23.8 | 10.9 |
| fluctuation outside the activation, `fluct_pct` | 4.9 | 1.2 |
| edge sharpness vs truth | 0.70 | 0.90 |
| recovered amplitude, `amp_ratio` (1 = exact) | −0.03 | 0.64 |
| region's mean time course vs truth, `corr` | 0.12 | 0.79 |
| t-score in the region, median (`_t`, `_t_lowband`) | −0.1, −0.0 | 5.9, 2.3 |
| voxels elsewhere above threshold (`false_pos_frac_t3.29`, `_lowband`) | 34%, 0.3% | 37%, 0.3% |

With a head-like field the B0 model decides whether the region's time course
follows the task at all: correlation 0.12 without it, 0.79 with it, at 0.64 of
the true 1.5% amplitude.

Voxel by voxel, this 60 s run does not detect the activation, and the t-scores
should not be read as if it did. The plain GLM that gives the region a median t
of 5.9 also puts 37% of the rest of the brain above |t| = 3.29. Those are not
reconstruction errors: the simulated BOLD-like fluctuations (0.01–0.1 Hz, about
0.8% in gray matter) share the band of the task (0.05 Hz), and a GLM that
assumes white residuals cannot tell three task blocks from them (the same GLM
on random mixtures of this session's four BOLD-like time courses plus 0.26%
white noise, no reconstruction involved, flags 41%). In the low
band, where the degrees of freedom are counted, the region's median t is 2.3
against a threshold of 4.0, and 0.3% of other voxels pass. A longer run
(`params.py`'s `duration`) is what voxel-wise detection needs; `--physio 0`
removes the fluctuations.

For the same reason the fluctuation is not zero for a perfect reconstruction
here: the physiological noise is about 0.9% of the signal in gray matter at
this TE. One session, one seed; these numbers show the chain works and what the
simulation is sensitive to, not which method is best.

Times, on a 64-core machine with a shared RTX A6000: 27 s to simulate the
session, 6–20 min to preprocess it, depending on CPU load (ESPIRiT on 32 coils dominates), 3–5 min to
reconstruct without B0, and 40 min with it after a 75 min power iteration for
the B0 operator's norm (pass `--sigma1A` to skip it once known). The archives
take 1.8 GB.

Running `preprocess/` on a simulated session also found that
`preprocess/grid_resize.py` places the deGRE maps 1.2 mm from the EPI's frame
in z, and 0.3 mm in x and y (`docs/review-findings.md` item 263).
