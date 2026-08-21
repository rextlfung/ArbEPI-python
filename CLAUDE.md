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

## Commit conventions

Do not commit as Claude: no `Co-Authored-By: Claude ...` trailer and no
Claude/Anthropic identity in the commit author. Commits should read as the
user's own.

## Commands

Dependency management is via `uv` (`pyproject.toml` + `uv.lock`), not
`pip`/`venv` directly. `pypulseq` is pulled from PyPI (`pypulseq>=1.5.0`),
not a local path — verify no local patch is needed there before ever
switching it back to a `file://` dependency.

Linting is via Ruff, configured in `pyproject.toml`'s `[tool.ruff]`
(`select = ["E", "F", "I"]` — pycodestyle, pyflakes, import sorting;
`matlab_reference/` is excluded since it holds `.m` files, not Python).
No MATLAB-based end-to-end
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
                                                          sequences/ArbEPI.generate_arbepi(omegas, params)
                                                            │  mask2epi_{laminar,radial}() called per
                                                            │  frame (params.epi_trajectory selects)
                                                            │  schedules: Nframes×Nshots×ETL×2
                                                            │  saved to output/samp_locs.mat
                                                            ▼
                                          sequences/EPIcal.generate_epical() / sequences/noise.generate_noise()
                                          ← both load samp_locs.mat, so must run after generate_arbepi
```

`sequences/deGRE.generate_degre()` (deGRE: dual-echo GRE, written as
`deGRE.seq` -- coil sensitivity maps + B0 field map, see that module's
docstring) is independent — it doesn't touch `samp_locs.mat`.

### Index convention — read this before touching lib/mask2epi.py or the sequence files

Internal computation is **0-based** throughout (`mask2epi`'s `schedule`,
sampling masks, etc.) — a deliberate departure from the 1-based MATLAB
original. The single place this gets converted back is
`sequences/ArbEPI.py`, where `schedules` is written to `samp_locs.mat` as
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

**`lib/mask2epi.py`** holds the core partitioning algorithm, now as two
interchangeable variants that both turn a 2D `(Ny, Nz)` sampling mask into
`Nshots` EPI trajectories of length `ETL`: `mask2epi_laminar` (the
original) and `mask2epi_radial` (added later). `sequences/ArbEPI.py`
selects between them via `params.epi_trajectory` (`'laminar'` or
`'radial'`, default `'laminar'`, set in `params.py`'s "Sampling
trajectory" section) — a config choice, not a hardcoded call.
`mask2epi_laminar`'s ordering
constraints: samples near ky = 0 are spread center-out across shots (via
`_center_out`, an `fftshift`-based interleave — MATLAB's `fftshift` and
`np.fft.fftshift` disagree for odd-length inputs, so `_center_out`
reimplements the left-rotate manually rather than calling
`np.fft.fftshift` directly; do not "simplify" this back to
`np.fft.fftshift` without re-deriving the odd-N case); ky is non-decreasing
within each echo train. `mask2epi_radial` instead gives every shot a spoke
through k-space center (deliberately giving up ky-non-decreasing — see its
own docstring for why that's safe for the reference-scan-based Nyquist
ghost correction this repo relies on, and for how it forces the
k-space-center sample to land at echo index `(ETL - 1) // 2`, matching
`calc_te_tr_delays.py`'s definition of the nominal TE echo).

**Ordering within each shot** (both variants) is a two-pass optimization —
pass 1 minimizes total weighted path length (min-sum TSP), pass 2 refines
that tour to minimize the worst-case single step (bottleneck/min-max) —
plus a third crossing-cleanup pass for radial only. See
`lib/mask2epi.py`'s module docstring and `_bottleneck_2opt_order`'s /
`_euclidean_uncross_refine`'s own docstrings for the full derivation,
empirical results, and the literature this design is based on.

**`lib/make_readout_grads.py`** returns a `ReadoutGrads` dataclass with
pre-built gradient objects. Blips (`gy_blip`, `gz_blip`) are stored at *unit
amplitude* and scaled at assembly time via `pp.scale_grad(rg.gy_blip,
step_size)`. The readout trapezoid (`gro`) is circularly shifted so blips
fit within each Pulseq block boundary. `gro1`/`gro2` are the leading/trailing
half-trapezoids played outside/inside the echo loop respectively.

**`sequences/EPIcal.py`** mirrors `sequences/ArbEPI.py`'s gradient design
exactly (same `make_readout_grads` call, same schedule-derived
`max_ky_step`/`max_kz_step`) but sets all blip scale factors to 0, so it
acquires unencoded lines at k-space center for EPI ghost correction.

**`lib/trap4ge.py`** (ported from `../PulCeq/matlab/trap4ge.m`) rounds every
gradient's rise/flat/fall times up to `params.crt` via `math.ceil(t / crt -
1e-9) * crt`. Every gradient in the sequence passes through it before being
added to a block — this is a GE-hardware-timing requirement, not optional
cleanup. The `- 1e-9` epsilon before `ceil` was added after finding that
float64 division noise (`0.002 / 4e-6` evaluates to `500.00000000000006`,
not `500.0`) could make `ceil` silently pad an already-on-raster time by one
extra raster step; for the excitation slice-select gradient this decentered
the RF pulse in its (now-longer) flat top and broke `gz_ssr`'s area-
cancellation assumption, leaving a nonzero residual kz at readout — caught
by `test_epical_trajectory_is_centered`. `params.crt` is `4e-6` (GE's raster
only) rather than `20e-6` (lcm of Siemens' 10µs and GE's 4µs): the lcm value
was originally chosen so gradient boundaries stayed valid on both rasters,
a precaution this repo no longer needs since it only targets GE hardware —
revert to `20e-6` if Siemens-dual-raster compatibility becomes relevant
again (the epsilon fix applies at either value).

**`params.py`**'s `load_params()` replaces MATLAB's `params.m` (which
injected variables into the caller's workspace — no Python equivalent).
Returns a single `Params` dataclass, passed explicitly to every function
that needs it. `Params.sys` (a pypulseq `Opts`) is a mutable object shared
across all four sequence-generation calls — anywhere the original MATLAB
did `systmp = sys; systmp.maxGrad = ...` (a value-semantics copy in MATLAB),
the Python port must `copy.deepcopy(sys)` first (see
`lib/make_readout_grads.py`, `sequences/ArbEPI.py`'s `sys_seq`) to avoid
mutating the shared system object.

**Hardware limits come from one place: `scanners.py`'s `ScannerSpec`.**
`load_params()` looks up a `ScannerSpec` from its body-level `scanner`
variable (currently `'GE_MR750'` or `'GE_UHP'`, edit directly in
`params.py` to change) and stores it as `params.spec`; `sys.max_grad`/`sys.max_slew`
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

**`mask2epi_laminar`/`mask2epi_radial`'s `Nshots*ETL` must exactly equal
the sampling mask's total sample count** (asserted at the top of each) —
`Nshots = ceil(Ny*Nz/R/ETL)` doesn't guarantee this holds for an arbitrary
`ETL`; picking a new `ETL` without also checking this divides-evenly
constraint will crash either function, not just produce a suboptimal
schedule.

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
  discrepancy on the readout axis). `plot_trajectory`'s per-frame case also
  marks each shot's first ADC sample (a black-edged dot, same size as the
  rest) so where each echo train actually begins is visible at a glance —
  most informative for `mask2epi_radial`, where starts scatter around the
  spoke ends rather than clustering near one corner of k-space like
  `mask2epi_laminar`'s raster order. `plotting/plot_last_run.py` drives all
  four plotting functions against the most recent `output/` run and is
  wired into `main.py --plot`, which now runs *before* the `--ge` export
  step (both independently depend only on `samp_locs.mat`/`ArbEPI.seq`, not
  on each other) so the diagnostic plots are still written even if `--ge`
  fails. `docs/demo/` holds static copies of one `--plot` run's output,
  embedded in README's Demo section — not regenerated automatically, so
  re-copy from `output/` by hand if the plots change materially.
- **Poisson-disc sampling** (`sampling/pd_sample.py`): a local
  reimplementation of `sigpy.mri.poisson`'s algorithm, not a dependency on
  the `sigpy` package — see README's Scope section for the three
  independent bugs found (in both `../ArbEPI/lib/pd_sample.m` and real
  SigPy) that motivated this, and why `numba` (narrowly, for just this one
  function) is still a dependency.

### `preprocessing/` -- raw scanner data -> zero-filled k-space, ported from `../epi-preprocessing`

`preprocessing/` is a from-scratch Python port of the companion MATLAB repo
`../epi-preprocessing`: raw scanner data -> reconstructed images, the
consumer of `samp_locs.mat`/`kxoe<Nx>.mat` this repo's sequence-generation
side produces. It's a separate `pyproject.toml` optional-dependency group
(`preprocessing`), deliberately kept out of the main sequence-generation
dependency set -- see below for why it also needs its own venv.

**Two-stage pipeline, same structure as the MATLAB original**: Stage 1
(`preprocess.py`, ported from `preprocess.m`) reads raw ScanArchives,
whitens/coil-compresses/grids/phase-corrects, and writes a zero-filled
k-space volume. Stage 2 (`recon_frames.py` + `run_cg_sense.py`/`run_rss.py`/
`run_recon_sigpy.py`, ported from `recon_frames.m` + its three driver
scripts) reconstructs every frame via CG-SENSE, RSS, or combined
L1-wavelet+TV regularized SENSE -- all three are wired up as sanity checks
for validating Stage 1's output against real acquired data, not the final
production reconstruction (a separate, more advanced Julia pipeline).

**Raw ScanArchive reading needs GE's proprietary Orchestra SDK
(`GERecon`), isolated to one module (`raw_io.py`).** This is not optional --
`h5dump -H` on a real `.h5` ScanArchive shows the k-space payload is an
opaque `H5T_STD_U8LE` byte blob with no structured type info, and even the
MRI community's own `ge_to_ismrmrd` converter still links Orchestra's own
Boost/HDF5 libraries to decode it, so reimplementing this independently
isn't realistic. `GERecon` is *not* pip-installable and its compiled
extension is *not* committed to this repo (proprietary, ~100MB, and
ABI-locked to Python 3.10 with `numpy<2.0.0`) -- install it separately from
GE's SDK distribution (`pip install <SDK>/GERecon`) into its own venv
(`uv venv --python 3.10`), not the main sequence-generation environment,
which resolves `numpy>=2.0` and would conflict. Verified working end to end
against real project data: `GERecon.Archive(path).Metadata()` and
`.NextFrame()` both function correctly, and `.NextFrame()`'s exhaustion
(`RuntimeError` containing "No next frame available") is what `raw_io.
ArchiveReader` converts to a normal `StopIteration` for Python's iterator
protocol. Every other module in `preprocessing/` avoids importing
`raw_io`/`GERecon` at module level (`preprocess.py` and
`calibrate_delay.py` both import it lazily, inside the one function that
needs it) specifically so the rest of the package -- and its tests --
stay importable without the SDK installed.

**BART and hmriutils/MIRT are dropped entirely, replaced by plain numpy +
sigpy** (explicit user decision, not a fallback forced by unavailability --
BART itself is installed and working locally). Noise whitening and PCA
coil compression (`coils.py`) are a few lines of numpy (Cholesky
decorrelation, eigendecomposition) with no need for BART's `whiten`/`cc`/
`ccapply` at all. ESPIRiT sensitivity maps (`smaps.py`) use
`sigpy.mri.app.EspiritCalib` in place of `bart ecalib` (sigpy only supports
outputting one map set, unlike BART, which simplifies `process_smaps`'s
port slightly -- no `emaps(...,end)`-style selection among several map sets
is needed). The 1D NUFFT ramp-sample regridding (`epi_gridding.py`, ported
from hmriutils' `rampsampepi2cart.m`/`rampsamp2cart.m`/`reconecho.m`) uses
`sigpy.nufft_adjoint` with the same gradient-magnitude density
compensation (`dcf = |diff(kx)| / max(...)`) in place of MIRT's `Gmri`. The
combined L1-wavelet+TV regularized reconstruction (`recon_sigpy.py`,
replacing `run_bart.m`'s `bart pics -R W:... -R T:...`) is sigpy's standard
multi-regularizer pattern: `G = Vstack([Wavelet, FiniteDifference])`,
`proxg = prox.Stack([L1Reg(...,lamb_l1), L1Reg(...,lamb_tv)])`, solved via
`PrimalDualHybridGradient` -- the same structure `sigpy.mri.app.
L1WaveletRecon`/`TotalVariationRecon` each use individually, just combined.
**Accepted, disclosed tradeoff**: none of this claims numerical parity with
BART/MIRT (unlike `seq2ge/`'s float-ULP-level MATLAB validation) --
verification here is algorithm-invariant instead (round-trip/convergence
tests against synthetic data with known ground truth), the same philosophy
this repo already uses for sequence generation where no MATLAB comparison
is available (see Commands section above).

**Validated end to end against real acquired data**: `preprocess.py` ->
`run_rss.py` was run on a real acquisition (`wb_2.4mm`, GE_UHP hardware) and
compared against a real MATLAB/BART reference reconstruction
(`wb_2.4mm_recon_rss.mat`) -- 0.19% relative L2 error, Pearson r = 0.999997
(after fitting a single overall scale factor, since RSS's own coil
normalization differs by convention), with metadata (`Nvcoils`, `Nframes`,
matrix size, TR) matching exactly. This confirms the whole Stage 1 pipeline
(whitening, coil compression, EPI gridding, odd/even phase correction,
k-space scatter) end to end, not just its individual building blocks in
isolation. Every building block was also independently verified before
that: `raw_io.py` against real `ScanArchive` files (`Archive`/`NextFrame`
reading real cal/EPI data), `epi_gridding.py`/`oephase.py`/`cg_sense.py`/
`recon_sigpy.py` via synthetic round-trip/convergence tests, and the
trickiest piece of
`preprocess.py` -- the MATLAB column-major
`permute`/`reshape` chain that scatters gridded k-space into the correct
`(ky, kz)` zero-filled-volume slot -- has a dedicated location-encoding
test (`scatter_frame` in `test_preprocessing_preprocess.py`: each
(shot, echo) is given a unique decodable value, and the test asserts it
lands at exactly the schedule's location, catching any flatten-order
mismatch a shape-only check would miss). Every MATLAB `permute`/`reshape`
in this port is translated mechanically (`permute(A,[p...])` ->
`A.transpose(p_0based...)`, `reshape(A,dims)` -> `A.reshape(dims,
order='F')`, since MATLAB `reshape` is always column-major over the whole
array, exactly numpy's `order='F'`) -- not re-derived by hand, to avoid
introducing exactly this class of bug.

**Per-acquisition scan parameters travel as a `.mat` snapshot, not a copied
script.** MATLAB's `preprocess.m` gets `Nx`/`Ny`/`ETL`/`fov`/etc. by
`run()`-ing a per-acquisition `params.m` into its workspace, injecting
variables -- no Python equivalent of that pattern exists, and copying the
whole `params.py` module was considered and rejected (it would require
`pypulseq`/`scanners.py` in the GERecon-constrained preprocessing venv just
to read a handful of scalars, and ties a long-term data record to source
code that can change shape over time). Instead, `sequences/ArbEPI.py`
exports exactly the scalars `preprocess.py` needs to `params.mat`
(`hdf5storage.savemat(fmt='7.3')`, right next to its existing
`samp_locs.mat` write); `preprocessing/config.py`'s `load_seq_params` reads
it back with `h5py` (see below for why not `hdf5storage`). No new
dependency needed on either side: `hdf5storage` is already a main-repo
dependency (writer), plain `h5py` is already in the `preprocessing` extras
(reader).

**`preprocessing/matio.py` -- read hdf5storage `.mat` files with `h5py`,
correctly.** `hdf5storage` stores arrays *axis-reversed* on disk (MATLAB's
column-major convention); `h5py` reads the raw on-disk layout with no
correction. Verified empirically against a real `samp_locs.mat`:
`hdf5storage.loadmat`'s `schedules` is `(30, 20, 60, 2)` (matching this
repo's documented `Nframes x Nshots x ETL x 2` layout), while a bare
`h5py.File(...)['schedules'][()]` comes back `(2, 60, 20, 30)` -- exactly
the reverse, and exactly `raw.transpose()` recovers the correct array
(shape *and* values, checked element-by-element against
`hdf5storage.loadmat`'s output). `matio.read_mat_array`/`read_mat` apply
this transpose unconditionally; for vectors/scalars it's a no-op on the
values (only a singleton axis moves), so there's no need to special-case
shape. Use these for every hdf5storage-written `.mat` this pipeline reads
(`samp_locs.mat`, `kxoe<Nx>.mat`, `params.mat`) -- never `scipy.io` (can't
read v7.3 at all, see above), never a bare `h5py` read without the
transpose.

**Everything `preprocess.py`/`recon_frames.py` write for their own
internal use -- not for a human to open in a viewer -- stays `.h5`, not
`.mat`.** These are plain numpy-order `h5py` writes with no MATLAB
consumer (the zero-filled k-space output, the per-sequence GRE/smaps
caches) -- the opposite on-disk axis convention from the hdf5storage-written
files this pipeline *reads*. A `.mat` extension here would silently invite
reading it with `hdf5storage.loadmat`, which would return every multi-axis
array transposed; the `.h5` extension is a deliberate, visible signal that
these files follow this port's own convention, not MATLAB's. One related,
fixed inconsistency in the original: `recon_frames.m` looks for a *shared*
`<datdir>/recon/gre.mat`, but `preprocess.m` actually saves its
whitened+compressed `ksp_gre` to `<datdir>/scanarchives/gre.mat` -- those
paths don't match in the MATLAB original, and since the whitening matrix
comes from a per-sequence noise scan, a single shared cache is a latent
correctness bug (whichever sequence's `preprocess.m` ran last silently
wins for every other sequence's fallback smaps estimation). This port uses
one consistent per-sequence path, `<datdir>/recon/<seqname>_gre.h5`, on
both the writer (`preprocess.py`) and reader (`recon_frames.py`) side.

**The final reconstructed-image files are the one exception: `.nii.gz` +
JSON sidecar, not `.h5`.** `preprocessing/nifti_io.py`'s `save_recon_nifti`
is the shared writer `run_rss.py`/`run_cg_sense.py`/`run_recon_sigpy.py`
all call in place of their old direct `h5py.File(...)` writes. This is a
deliberate format split by *consumer*, not a blanket format change: the
intermediate files above (`ksp_epi_zf`, smaps, GRE cache) still feed the
downstream Julia advanced-recon pipeline and stay plain HDF5 (already
generic and directly readable via `HDF5.jl`, no format change needed
there), while the final magnitude images are for a human to look at, and
this repo's own `.h5` files have no viewer as good as ITK-SNAP/FSLeyes/3D
Slicer/etc., all of which expect NIfTI (or DICOM) rather than a bare
HDF5 array. NIfTI has no native complex dtype, so `save_recon_nifti` saves
magnitude only (`np.abs`) -- `run_cg_sense.py`/`run_rss.py`'s outputs were
already real-valued in practice (RSS/CG-SENSE magnitude combines), so this
loses nothing currently produced, but note it if a future recon driver
needs the complex-valued image itself; that consumer should keep reading
`recon_frames`' return value directly; rather than round-tripping it
through NIfTI. NIfTI also has no free-form attribute dict (unlike h5py's
`.attrs`), so per-recon parameters that used to live as `.h5` attrs
(`seqname`, `num_iter`, `lamb_l1`/`lamb_tv`, `runtime_s`, the `seq_params`
fields) are written to a `<fn_base>.json` sidecar instead -- the common
BIDS-style image+sidecar convention, not a repo-specific format. The
NIfTI affine is a plain diagonal voxel-size matrix derived from
`seq_params.fov`/the image shape; no patient-orientation information
exists anywhere in this pipeline (unlike a scanner-produced DICOM/NIfTI),
so voxel spacing is correct but radiological left/right or
anterior/posterior orientation is not guaranteed.

**`preprocessing/` now knows about deGRE's dual-echo acquisition, and hands
both echoes off to an external B0-mapping consumer.** `sequences/deGRE.py`
writes two full excitation/readout passes per phase encode (`TE_degre`, a
2-element array -- see `params.py`, echo innermost, then `iY`, then `iZ`,
including the `iZ=0` receive-gain-calibration pass -- see that module's
"Each (iY, iZ) phase-encode location is excited once per echo" docstring
note). `sequences/ArbEPI.py` now exports `n_echoes_degre = len(params.
TE_degre)` and `TE_degre` itself in its `params.mat` snapshot
(`preprocessing/config.py`'s `SeqParams.n_echoes_degre`/`TE_degre`, read by
`load_seq_params`; both default -- `1` and `None` respectively -- when
missing, since a `params.mat` snapshot written before this change is a
durable, non-regeneratable per-acquisition data record, not something to
raise `KeyError` on).

`preprocessing/preprocess.py`'s `unflatten_gre_echoes()` (STEP 2)
unflattens the raw archive against `n_echoes_degre`, returning
`[Nx_degre, Ny_degre, Nz_degre, n_echoes, Ncoils]` -- every echo, not just
one. Whitening and coil-compression-matrix estimation still use only
`PreprocessingConfig.gre_echo_idx` (default `0` = the shorter TE1, for
higher SNR -- either echo works for sensitivity maps, per `deGRE.py`'s
docstring) so `Nvcoils` selection and `smaps.py`'s `estimate_smaps`/
`process_smaps` behave exactly as before the dual-echo upgrade (the coil
subspace is echo-independent, so a compression matrix fit on one echo is
correctly applied to both); `apply_whitening`/`apply_coil_compression`
operate per-sample along the coil axis regardless of the extra echo axis,
so this doesn't change the selected echo's values at all. Verified with a
location-encoding test in the same style as `scatter_frame`'s
(`test_unflatten_gre_echoes_places_data_at_correct_indices` in
`tests/test_preprocessing_preprocess.py`): every acquisition is given a
value encoding its own `(echo, iY, iZ)`, and the test confirms every
echo's unflattened volume holds exactly the right value at each
`(iY, iZ)` and that the `iZ=0` calibration block is dropped entirely --
not just a shape-only check.

**The `<seqname>_gre.h5` cache is the handoff point to an external
B0-mapping consumer (a separate Julia package, not part of this repo) --
deliberately not GE-specific.** Alongside the existing `ksp_gre` dataset
(the selected echo, whitened + coil-compressed, unchanged key so
`recon_frames.py`'s cache reader needs no changes), STEP 2 now also
writes `ksp_gre_echoes` (`[Nx_degre, Ny_degre, Nz_degre, n_echoes,
Nvcoils]`, both echoes, whitened + coil-compressed) and a `TE_degre` attr
(seconds) whenever `seq_params.TE_degre` is available. Both are plain
numpy-order HDF5 (see the `.h5`-vs-`.mat` convention note above) -- a
consumer needs only `h5py`/`HDF5.jl`, never GERecon or the raw
ScanArchive. Actual B0 estimation (phase difference / `ΔTE`, 3D phase
unwrapping + fitting) is not implemented in this repo; that's the
external Julia package's job.

See `README.md` for the getting-started walkthrough and the full
`Getting started` / `GE export` usage examples.
