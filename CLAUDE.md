# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python/pypulseq port of [../ArbEPI](../ArbEPI) (MATLAB/Pulseq). Generates
fast, vendor-agnostic 3D-EPI MRI pulse sequences from arbitrary 2D
`(ky, kz)` sampling masks in the phase-encode-partition plane. Only entry
points and global config (`main.py`, `params.py`, `scanners.py`) sit at
the repo root, mirroring `../ArbEPI` having `params.m`/`main.m` directly
at its own root — everything else lives under
`lib/`/`sequences/`/`sampling/`/`plotting/`/`seq2ge/`, matching
`../ArbEPI`'s `src/`/`lib/` split (see README.md's Architecture section
for the full layout).

## Commands

```
uv sync --extra test               # install deps into .venv (see uv.lock)
uv run python main.py              # generate all 4 sequences into output/ using params.py defaults
uv run python main.py --ge         # also export each sequence to GE .pge (pure Python, no MATLAB)
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
or `'GE_UHP'`) and stores it as `params.spec`; `sys.max_grad`/`sys.max_slew`
(used for `.seq` generation) and `seq2ge/check.py`/`seq2ge/writeceq.py`'s
hardware/PNS/acoustics checks (used by `seq2ge/ge_export.py`'s `--ge` path, pure
Python, see below) both read from that same `ScannerSpec` instance, so
they cannot drift out of sync the way they used to (see git history for
the bug this replaced: `slew_max` had been hand-set to a value 25% higher
than `sys.max_slew` actually specified). `ScannerSpec.ge_coil` (e.g.
`'xrm'`, `'hrmbuhp'`) keys `seq2ge/acoustics.py`'s per-coil forbidden-band
table and `ScannerSpec.chronaxie`/`rheobase`/`alpha` key `seq2ge/pns.py`'s
per-coil PNS coefficients — see `../PulCeq/matlab/+pge2/opts.m`'s header
comment for the authoritative table if adding a new scanner. `PNSwt`
stays a separate `Params` field (not part of `ScannerSpec`) since it's
scan-context — phantom vs. human — not a hardware constant; PNS is a
physiological limit, so don't silently lower `PNSwt` to make an error go
away — `[0, 0, 0]` is only valid for phantom/non-human scanning, and the
default is now the IEC 60601-2-33:2022-recommended `[0.8, 1.0, 0.7]` (see
the "Open finding" below for why it wasn't always). `seq2ge.ge_export.check_ge_feasibility()`
runs the hardware/PNS/acoustics check without writing a `.pge` file —
`main.py --ge` calls it on all four sequences before exporting any of
them, so infeasibility surfaces immediately rather than after several full
exports have already run.

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

### GE `.pge` export -- fully ported to Python, no MATLAB round trip

`seq2ge/` is a from-scratch Python port of PulCeq's `seq2ceq`/`writeceq`/
`pge2.pns`/`check_grad_acoustics` toolchain, and is now the operative path
for both `main.py --ge` and standalone use — `seq2ge/ge_export.py` calls straight
into `seq2ge.seq2ceq`/`seq2ge.writeceq`/`seq2ge.check`, with no `matlab -batch` shell-out
and no sibling `../pulseq`/`../toppe`/`../PulCeq`/`../ArbEPI` checkouts
required. MATLAB is only needed if re-validating this port against a fresh
`../PulCeq` checkout in the future (see `matlab_reference/dump_*.m` below).

`seq2ge/seq2ceq.py` + `seq2ge/blocks.py` port `seq2ceq.m`/`compareblocks.m`/
`getdynamics.m`/`getblocktype.m`; `seq2ge/writeceq.py` ports `writeceq.m`'s
binary `.pge` writer. Validated field-by-field against real MATLAB output
(`seq2ge/validate_against_matlab.py` + `matlab_reference/dump_ceq.m`, which dumps
MATLAB's `ceq` struct for comparison) on all four generated sequences:
`noise.pge`/`GRE.pge` come out byte-identical, `ArbEPI.pge`/`EPIcal.pge`
differ only in the `maxSlew` header float32's last bit (summation-order
noise, not a bug -- confirmed by matching every other field including the
full loop table exactly). This byte-for-byte/near-identical match was
re-confirmed after wiring `seq2ge/ge_export.py` end to end (fresh MATLAB
references regenerated from the exact `.seq` files `main.py --ge`
produces, `noise`/`GRE` byte-identical, `ArbEPI`/`EPIcal` 1-float32-ULP on
`maxSlew` only, all four `loop` tables matching exactly) -- not just on
`seq2ge/seq2ceq.py`/`seq2ge/writeceq.py` in isolation. Two real bugs were caught
and fixed by this validation: a sign error in `_write_grad`'s waveform
normalization, and a wrong assumption that the `.pge` header's `maxSlew`
field echoes the scanner's hardware limit rather than (as MATLAB actually
does) the sequence's realized peak slew.

One deliberate, disclosed deviation: `seq2ceq.m`'s gradient-heating
`Emax` calculation sums stale 1-indexed loop columns 11:13 (`energy_gz,
recphs, blockDuration`, not the three gradient energies its variable names
imply) -- `seq2ge/seq2ceq.py` sums the correct energy columns instead. This
produced identical `Emax_n` results on all four test sequences but is not
guaranteed to in general; consider reporting the indexing bug upstream to
PulCeq.

**PNS/acoustics/hardware checking is also ported**: `seq2ge/pns.py` (GE's
IEC 60601-2-33:2022 PNS model, from `pge2.pns.m` -- not the same model as
pypulseq's own `pypulseq.utils.safe_pns_prediction`, which is Siemens'
different SAFE parameterization), `seq2ge/acoustics.py` (`check_grad_acoustics.m`'s
FFT/forbidden-band check, per-coil tables copied verbatim), and `seq2ge/check.py`
(`check_ge_feasibility()`, combining both plus gradient/slew/B1 hardware
limits, run directly on `seq.get_gradients()` -- no MATLAB round trip).
Both `seq2ge/pns.py` and `seq2ge/acoustics.py` match real MATLAB output to
float64/float32 precision on identical input (`seq2ge/validate_pns.py` +
`matlab_reference/dump_pns_test.m`/`dump_acoustics_test.m`). `seq2ge/check.py`'s
whole-sequence PNS convolution matches MATLAB's real per-segment-instance
computation (via `matlab_reference/dump_pns_peak.m`, which replicates
`checksegment.m`'s pipeline without its fail-fast throw) to within 0.02
percentage points on the full ArbEPI/GE_UHP sequence (114.7% vs 114.72%).

Acoustics is checked over a bounded window rather than the whole sequence
-- its FFT cost scales with window length, and a full ~60s/15M-sample
sequence takes minutes for no benefit. The window is not an arbitrary
fixed duration: `seq2ge/check.py`'s `_blockrange_window_s` reproduces the
exact block-selection semantics of MATLAB's own check (`pge2.plot(ceq,
sys, 'blockRange', [1 10], ...)` inside `write_to_ge_from_seq.m` -- walk
`ceq.loop` rows in segment order starting at row 1, including whole
segments until the next segment's start row would be >= `block_range[1]`),
computed from this port's own `seq2ceq(seq)` rather than guessed at.
Reproduction against real MATLAB output (`matlab_reference/dump_acoustics_blockrange.m`)
on this repo's four sequences: window duration matches to within one 4us
raster sample (`GRE.seq`: 173766 vs MATLAB's 173767 samples; `ArbEPI.seq`:
25000 vs 25001), and the resulting acoustics number matches to <0.04%
relative error (`GRE.seq`: 0.4024 here vs MATLAB's 0.402213; `ArbEPI.seq`:
0.028146 here vs MATLAB's 0.02814424) -- the one earlier open question (a
fixed 1s window vs. MATLAB's shorter blockRange window giving different
magnitudes, `GRE.seq` 0.312 vs 0.402) is resolved: matching MATLAB's window
definition reproduces MATLAB's number directly, it was purely a
window-choice gap, not an algorithm mismatch. The residual gap (MATLAB's
per-segment `segment_dead_time`/`segment_ringdown_time` padding,
~117us/segment, a GE-Ceq-interpreter artifact with no Pulseq-timeline
equivalent) is deliberately not reproduced -- confirmed negligible above.
PNS and hardware limits are *not* windowed -- cheap enough to check in full.

**Real behavioral bug caught while wiring this in**: `check_grad_acoustics.m`
only ever calls MATLAB's `warning(...)` when `magb > threshold` (see its
source, line ~159 -- `if magb>threshold, warning(...); end`) -- it has
never once blocked a real MATLAB `--ge` export, unlike PNS
(`checksegment.m` really does `throw(MException(...))` above 80%, see
below). `seq2ge/check.py`'s `FeasibilityReport.ok` originally folded acoustics
into the same hard gate as PNS/hardware limits; this would have made the
Python `--ge` path *reject* `GRE.seq` (acoustics 0.402, over the 0.3
threshold) even though the exact same sequence has always exported
successfully via the real MATLAB path. Fixed: `.ok` now excludes
acoustics, `.summary()` reports it as `WARN` (non-blocking) rather than
`FAIL` when over threshold -- matching MATLAB's real behavior, not a
guess. `GRE.seq`'s acoustics number is still surfaced every run; whether
it's a real problem is a separate open question (see below).

**Open finding, still unresolved (sequence parameters, not code)**:
`params.py`'s `PNSwt` default was `[0, 0, 0]` for the entire lifetime of
this port until now, which meant PNS was never actually evaluated in any
`--ge` run to date -- weight zero makes `pge2.pns`'s per-channel
contribution zero regardless of the real waveform, and MATLAB's
`checksegment.m` really does *throw* on PNS > 80%
(`../PulCeq/matlab/+pge2/checksegment.m`: `if max(pt) > 80 ...
throw(MException('safety:pns', ...))`), so a real weight would have caught
this immediately if one had ever been used. `params.py`'s default is now
the IEC 60601-2-33:2022-recommended `wt = [0.8, 1.0, 0.7]`, and
`seq2ge/check.py` matches MATLAB's throw condition exactly
(`PNS_NORMAL_MODE_THRESHOLD = 80.0`) -- so **`main.py --ge` now correctly
fails by default** on three of the four sequences: `noise` 0% (no
gradients, passes), `EPIcal` 113.2%, `ArbEPI` 114.7%, `GRE` 100.6% -- all
three exceed MATLAB's 80% "normal mode" throw threshold, and exceed the
100% "first controlled mode" threshold outright. This is a safety-relevant
open item, not a code-correctness one: raising the weight surfaced a real
problem in the sequence design rather than solving it. The sequences this
repo generates by default have not been validated as PNS-safe for human
scanning, and `main.py --ge` will now say so instead of silently
succeeding. Revisit before any human scan: the sequence parameters need to
change (lower slew / longer blip rise times) to bring peak PNS under 80%.

`params.py`'s `Params` dataclass carries the selected `ScannerSpec` itself
as `params.spec` (rather than duplicating `max_grad`/`max_slew`/`b1_max`/
`ge_coil`/`pislquant`/etc. as separate derived fields, as it did when the
MATLAB path needed them formatted into a `pge2.opts(...)` snippet) --
`seq2ge/check.py`/`seq2ge/writeceq.py` read straight from `params.spec`.

`matlab_reference/` now holds only the one-off validation scripts used to produce
the MATLAB reference data cited throughout this section
(`dump_ceq.m`, `dump_pns_test.m`, `dump_pns_peak.m`, `dump_acoustics_test.m`,
`dump_acoustics_blockrange.m`) -- none are called by any Python code.
`write_to_ge_from_seq.m`/`ge_feasibility_check.m` (the former MATLAB
shell-out targets) have been removed now that `seq2ge/ge_export.py` no longer
calls them; their content is preserved in git history for anyone
re-deriving this port's validation record.
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
