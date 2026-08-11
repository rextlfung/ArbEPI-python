# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python/pypulseq port of [../ArbEPI](../ArbEPI) (MATLAB/Pulseq). Generates
fast, vendor-agnostic 3D-EPI MRI pulse sequences from arbitrary 2D
`(ky, kz)` sampling masks in the phase-encode-partition plane. Only entry
points and global config (`main.py`, `params.py`, `scanners.py`,
`ge_export.py`) sit at the repo root, mirroring `../ArbEPI` having
`params.m`/`main.m` directly at its own root — everything else lives under
`lib/`/`sequences/`/`sampling/`/`plotting/`, matching `../ArbEPI`'s
`src/`/`lib/` split (see README.md's Architecture section for the full
layout).

## Commands

```
uv sync --extra test               # install deps into .venv (see uv.lock)
uv run python main.py              # generate all 4 sequences into output/ using params.py defaults
uv run python main.py --ge         # also export each sequence to GE .pge via a local MATLAB install
uv run python main.py --plot       # also write diagnostic plots (see plotting/plot_last_run.py) into output/
uv run pytest tests/               # run the unit test suite
uv run pytest tests/test_mask2epi.py -v   # run a single test file
```

Dependency management is via `uv` (`pyproject.toml` + `uv.lock`), not
`pip`/`venv` directly. `pypulseq` is pulled from PyPI (`pypulseq>=1.5.0`),
not a local path — verify no local patch is needed there before ever
switching it back to a `file://` dependency.

There is no lint/format config in this repo. No MATLAB-based end-to-end
comparison suite exists either — no MATLAB install was available during the
initial port, so correctness is validated by: (a) unit tests on algorithm
invariants (exact sample counts, ky-non-decreasing ordering, full k-space
coverage, etc.), and (b) `tests/test_trajectory_matches_schedule.py`, which
reads k-space back out of an assembled Pulseq sequence and independently
confirms it matches the `schedules` array used to build it — this is the
test to extend when changing gradient/blip logic, since `seq.check_timing()`
passing only proves a sequence is well-formed, not that its gradients
encode the intended k-space locations.

## Architecture

### Data flow

```
params.py (load_params())  ──►  gen_sampling_masks(R, params)  ──►  omegas (Ny×Nz×Nframes bool)
                                                                        │
                                                                        ▼
                                                          sequences/arbepi.generate_arbepi(omegas, params)
                                                            │  mask2epi() called per frame
                                                            │  schedules: Nframes×Nshots×ETL×2
                                                            │  saved to output/samp_locs.mat
                                                            ▼
                                          sequences/epical.generate_epical() / sequences/noise.generate_noise()
                                          ← both load samp_locs.mat, so must run after generate_arbepi
```

`sequences/gre.generate_gre()` (coil sensitivity maps) is independent — it
doesn't touch `samp_locs.mat`.

### Index convention — read this before touching lib/mask2epi.py or the sequence files

Internal computation is **0-based** throughout (`mask2epi`'s `schedule`,
sampling masks, etc.) — a deliberate departure from the 1-based MATLAB
original. The single place this gets converted back is
`sequences/arbepi.py`, where `schedules` is written to `samp_locs.mat` as
`schedules + 1` so MATLAB-side reconstruction code sees the same convention
it always has. `parts` (the shot-label map) is already "1-based label, 0 =
unsampled" and needs no conversion either way.

### `.mat` file format

`output/samp_locs.mat` and `output/kxoe<Nx>.mat` are written via
`hdf5storage.savemat(..., fmt='7.3')`, matching the original MATLAB code's
`save(..., '-v7.3')`. **`scipy.io.loadmat`/`savemat` cannot read or write
v7.3 at all** — always use `hdf5storage.loadmat` (or raw `h5py`) when
touching these files, never `scipy.io`.

### Key design decisions (carried over from ../ArbEPI, still apply here)

**`mask2epi`** (`lib/mask2epi.py`) is the core algorithm. It partitions a 2D
`(Ny, Nz)` sampling mask into `Nshots` EPI trajectories, each of length
`ETL`. Ordering constraints: samples near ky = 0 are spread center-out
across shots (via `_center_out`, an `fftshift`-based interleave — MATLAB's
`fftshift` and `np.fft.fftshift` disagree for odd-length inputs, so
`_center_out` reimplements the left-rotate manually rather than calling
`np.fft.fftshift` directly; do not "simplify" this back to `np.fft.fftshift`
without re-deriving the odd-N case); ky is non-decreasing within each echo
train.

**`lib/make_readout_grads.py`** returns a `ReadoutGrads` dataclass with
pre-built gradient objects. Blips (`gy_blip`, `gz_blip`) are stored at *unit
amplitude* and scaled at assembly time via `pp.scale_grad(rg.gy_blip,
step_size)`. The readout trapezoid (`gro`) is circularly shifted so blips
fit within each Pulseq block boundary. `gro1`/`gro2` are the leading/trailing
half-trapezoids played outside/inside the echo loop respectively.

**`sequences/epical.py`** mirrors `sequences/arbepi.py`'s gradient design
exactly (same `make_readout_grads` call, same schedule-derived
`max_ky_step`/`max_kz_step`) but sets all blip scale factors to 0, so it
acquires unencoded lines at k-space center for EPI ghost correction.

**`lib/trap4ge.py`** (ported from `../PulCeq/matlab/trap4ge.m`) rounds every
gradient's rise/flat/fall times up to `params.crt` (20µs, the common raster
of Siemens' 10µs and GE's 4µs). Every gradient in the sequence passes
through it before being added to a block — this is a GE-hardware-timing
requirement, not optional cleanup.

**`params.py`**'s `load_params()` replaces MATLAB's `params.m` (which
injected variables into the caller's workspace — no Python equivalent).
Returns a single `Params` dataclass, passed explicitly to every function
that needs it. `Params.sys` (a pypulseq `Opts`) is a mutable object shared
across all four sequence-generation calls — anywhere the original MATLAB
did `systmp = sys; systmp.maxGrad = ...` (a value-semantics copy in MATLAB),
the Python port must `copy.deepcopy(sys)` first (see
`lib/make_readout_grads.py`, `sequences/arbepi.py`'s `sys_seq`) to avoid
mutating the shared system object.

**Hardware limits come from one place: `scanners.py`'s `ScannerSpec`.**
`load_params(scanner=...)` looks up a `ScannerSpec` (currently `'GE_MR750'`
or `'GE_UHP'`) and derives *both* `sys.max_grad`/`sys.max_slew` (T/m, T/m/s
— used for `.seq` generation) and `g_max`/`slew_max` (G/cm, G/cm/ms — used
only by `ge_export.py`'s `pge2.check`) from the same `spec.max_grad`/
`spec.max_slew` numbers, so they cannot drift out of sync the way they
used to (see git history for the bug this replaced: `slew_max` had been
hand-set to a value 25% higher than `sys.max_slew` actually specified).
`ScannerSpec.ge_coil` (e.g. `'xrm'`, `'hrmbuhp'`) is passed straight
through to MATLAB's `pge2.opts(...)` and `check_grad_acoustics(...)`,
which each carry their own more detailed per-coil tables (PNS SAFE-model
chronaxie/rheobase/alpha, and acoustic resonance frequencies respectively)
keyed off that same string — see `../PulCeq/matlab/+pge2/opts.m`'s header
comment for the authoritative table if adding a new scanner. `PNSwt`
stays a separate `Params` field (not part of `ScannerSpec`) since it's
scan-context — phantom vs. human — not a hardware constant; PNS is a
physiological limit, so don't silently raise `PNSwt` to make an error go
away — `PNSwt = [0, 0, 0]` is only valid for phantom/non-human scanning.
`ge_export.check_ge_feasibility()` runs MATLAB's hardware/PNS/acoustic
checks (`pge2.check` + `check_grad_acoustics`) without writing a `.pge`
file — `main.py --ge` calls it on all four sequences before exporting any
of them, so infeasibility surfaces immediately rather than after several
full exports have already run.

**`calc_te_tr_delays.py` only warns, never raises**, if the prescribed
`TE`/`TR` are unachievable — it silently falls back to zero padding delay,
so the sequence still builds with the *wrong* TE/TR baked in. Whether a
given `TE`/`TR`/`ETL` combination is achievable is a coupled, non-monotonic
function of `ETL` (shorter helps `min_te`, hurts `min_tr` since fewer
shots share the fixed `volume_tr` budget) and the actual per-shot ky/kz
blip requirements (which depend on the sampling mask, not just `ETL`) —
don't hand-derive feasibility, call `calc_te_tr_delays` directly (or scan
across candidate `ETL` values) to check.

**`mask2epi`'s `Nshots*ETL` must exactly equal the sampling mask's total
sample count** (`lib/mask2epi.py:71`'s assertion) — `Nshots = ceil(Ny*Nz/R/ETL)`
doesn't guarantee this holds for an arbitrary `ETL`; picking a new `ETL`
without also checking this divides-evenly constraint will crash `mask2epi`,
not just produce a suboptimal schedule.

### Scope: what's deliberately NOT ported

- **GE `.pge` export**: not reimplemented in Python. `ge_export.py`
  provides `export_to_ge(seq_path, out_path, params)`, which shells out to
  a local MATLAB install (`matlab -batch`) to run
  `matlab/write_to_ge_from_seq.m` — a thin adapter around
  `../ArbEPI/lib/write_to_ge.m` that reads a `.seq` file instead of an
  in-memory `mr.Sequence`, otherwise byte-identical logic. Requires
  `../pulseq`, `../toppe`, `../PulCeq`, and `../ArbEPI` (for
  `check_grad_acoustics.m`) checked out as sibling directories. MATLAB
  discovery checks `PATH` first, then globs
  `/Applications/MATLAB_*.app/bin/matlab` on macOS (the installer doesn't
  add it to `PATH` by default). Verified working end-to-end against a real
  MATLAB install, including at full default-params scale.
- **Fat-sat RF pulse**: the MATLAB original designs this via GE's
  `toppe.utils.rf.makeslr` (min-phase SLR), which has no Python equivalent.
  `lib/make_fatsat_rf.py` uses pypulseq's built-in `make_gauss_pulse`
  instead — simpler, less sharp spectral profile, no bit-exact-waveform
  requirement.
- **Plotting** (`plotting/plotting.py`): static, non-interactive matplotlib
  equivalents of the sampling-mask/trajectory/PSF plots. The interactive
  scroll/slider mask viewer from the MATLAB repo is not ported — no
  algorithmic content worth preserving there. `plot_one_tr` is an
  exception: it's a thin wrapper around pypulseq's own
  `Sequence.plot(stacked=True, time_range=...)` for a single-TR snippet
  (resized taller + `tight_layout()`'d, since the default stacked-plot
  figure size is too short for 6 rows and leaves labels overlapping) —
  not a reimplementation, just parameterized correctly. `plot_sampling_mask`/
  `plot_psf`/`plot_trajectory` all take an optional `frame_idx` to select
  one frame out of a multi-frame array; `plot_trajectory`'s per-frame case
  exact-slices `k_traj_adc` (`Sequence.calculate_kspace()` has no
  per-frame breakdown) rather than the fine continuous line — see that
  function's docstring for why (a sub-sequence block-extraction approach
  was tried and abandoned after finding an unexplained ~16-22 m^-1
  discrepancy on the readout axis). `plotting/plot_last_run.py` drives all four
  plotting functions against the most recent `output/` run and is wired
  into `main.py --plot`.
- **Poisson-disc sampling** (`sampling/pd_sample.py`): a local
  reimplementation of `sigpy.mri.poisson`'s algorithm, not a dependency on
  the `sigpy` package — see README's Scope section for the three
  independent bugs found (in both `../ArbEPI/lib/pd_sample.m` and real
  SigPy) that motivated this, and why `numba` (narrowly, for just this one
  function) is still a dependency.

See `README.md` for the getting-started walkthrough and the full
`Getting started` / `GE export` usage examples.
