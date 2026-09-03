# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python/pypulseq port of [../ArbEPI](../ArbEPI) (MATLAB/Pulseq). Generates
fast, vendor-agnostic 3D-EPI MRI pulse sequences from arbitrary 2D
`(ky, kz)` sampling masks in the phase-encode-partition plane. Only entry
points and global config (`main.py`, `params.py`, `scanners.py`) sit at
the repo root, mirroring `../ArbEPI` having `params.m`/`main.m` directly
at its own root — everything else lives under
`lib/`/`sequences/`/`sampling/`/`plotting/`/`ge/`, matching
`../ArbEPI`'s `src/`/`lib/` split (see README.md's Architecture section
for the full layout).

## Review findings backlog

Open and resolved findings from periodic repo-wide correctness/consistency/
conciseness reviews live in [`docs/review-findings.md`](docs/review-findings.md),
not here -- it's a worklog, consulted occasionally, not an instruction a
session needs loaded by default. Numbering in that file is cumulative and
never reused: a few source files (`preprocessing/grid_resize.py`,
`tests/test_preprocessing_grid_resize.py`, `lib/make_prephasers.py`) cite
specific item numbers in comments, and items cross-reference each other by
number. Check it before a review pass, and add new findings there in the
same numbered format.

## Commit conventions

Do not commit as Claude: no `Co-Authored-By: Claude ...` trailer and no
Claude/Anthropic identity in the commit author. Commits should read as the
user's own.

## Commands

Dependency management is via `uv` (`pyproject.toml` + `uv.lock`), not
`pip`/`venv` directly. `pypulseq` is pulled from PyPI (`pypulseq>=1.5.0`),
not a local path — verify no local patch is needed there before ever
switching it back to a `file://` dependency.

`pytest` is an optional extra (`[project.optional-dependencies]`'s `test`
group), not a default dependency, matching `preprocessing`/`recon`/`lint`
all being separate opt-in extras rather than bloating every install — run
`uv sync --extra test` before `uv run pytest`. Skipping that sync isn't
just "tests aren't installed": `uv run pytest` on a bare `uv sync` still
resolves `pytest` into an ephemeral one-off environment that has `pytest`
itself but none of this repo's own dependencies, producing ~27 confusing
`ModuleNotFoundError: No module named 'numpy'` collection errors that look
like a broken repo rather than a missing extra.

Linting is via Ruff, configured in `pyproject.toml`'s `[tool.ruff]`
(`select = ["E", "F", "I"]` — pycodestyle, pyflakes, import sorting).
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
                                                            │  schedules: Nframes×Nshots×ETL×3
                                                            │  (ky, kz, echo time), kxo/kxe, and a scan-
                                                            │  scalar snapshot all saved to
                                                            │  output/scan_info.mat
                                                            ▼
                                          sequences/EPIcal.generate_epical() / sequences/noise.generate_noise()
                                          ← both load scan_info.mat, so must run after generate_arbepi
```

`sequences/deGRE.generate_degre()` (deGRE: dual-echo GRE, written as
`deGRE.seq` -- coil sensitivity maps + B0 field map, see that module's
docstring) doesn't *need* `scan_info.mat` to run (it can be called
standalone), but does patch its `TE_degre` field with the realized
(not prescribed) echo-time pair when the file already exists, since
`main.py`'s call order always runs `generate_arbepi` first -- see
`docs/review-findings.md` item 62 for why the realized pair, not
`params.TE_degre`, is what needs to reach `b0map.jl`'s ΔTE scaling.

### Index convention — read this before touching lib/mask2epi.py or the sequence files

Internal computation is **0-based** throughout (`mask2epi`'s `schedule`,
sampling masks, etc.) — a deliberate departure from the 1-based MATLAB
original. The single place this gets converted back is
`sequences/ArbEPI.py`, where `schedules` is written to `scan_info.mat` as
`schedules + 1` (on just its (ky, kz) channels, see below) so MATLAB-side
reconstruction code sees the same convention it always has. `parts` (the
shot-label map) is already "1-based label, 0 = unsampled" and needs no
conversion either way.

### `.mat` file format

`output/scan_info.mat` -- kxo/kxe (odd/even echo k-space trajectories for
ghost correction), schedules/parts (the sampling schedule), and a snapshot
of the scan scalars `preprocessing/` needs -- is written via
`hdf5storage.savemat(..., fmt='7.3')`, matching the original MATLAB code's
`save(..., '-v7.3')`. **`scipy.io.loadmat`/`savemat` cannot read or write
v7.3 at all** — always use `hdf5storage.loadmat` (or raw `h5py`) when
touching this file, never `scipy.io`. `schedules` itself is `Nframes ×
Nshots × ETL × 3`: `schedules[..., :2]` is the 1-based `(ky, kz)` index
pair, `schedules[..., 2]` is echo time in seconds since RF excitation for
that acquisition (uniform across every shot/frame, since readout timing
doesn't vary between them) -- not an index, so it's exempt from the 1-based
conversion above. This was a deliberate consolidation: `scan_info.mat`
replaces three previously-separate files (`samp_locs.mat`, `params.mat`,
`kxoe<Nx>.mat`), all written by `sequences/ArbEPI.py` at one point in the
pipeline now that nothing needs `kxoe<Nx>.mat`'s old two-stage,
Nx-dependent filename resolution (see `preprocessing/preprocess.py`'s
`load_kxoe`) or a separate `EPIcal.py`-computed copy of kxo/kxe (see
`sequences/EPIcal.py`'s module docstring for why that copy was redundant --
EPIcal's own kx trajectory is mathematically identical to ArbEPI's,
confirmed by `test_arbepi_kxoe_matches_epical`). The echo-time channel
exists so a future off-resonance-correction consumer (`recon/`) has, for
every k-space sample it already indexes, both a field-map value (see the
B0 field map paragraph below) and the acquisition time needed to convert
that field-map value into a phase-correction term -- `recon/` does not
consume it yet.

### Key design decisions (carried over from ../ArbEPI, still apply here)

**`lib/mask2epi.py`** holds the core partitioning algorithm, now as two
interchangeable variants that both turn a 2D `(Ny, Nz)` sampling mask into
`Nshots` EPI trajectories of length `ETL`: `mask2epi_laminar` (the
original) and `mask2epi_radial` (added later). `sequences/ArbEPI.py`
selects between them via `params.epi_trajectory` (`'laminar'` or
`'radial'`, default `'radial'`, set in `params.py`'s "Sampling
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

**The readout trapezoid is asymmetric (POPE)**: rise and fall slews are
independent (`params.ro_slew_rise`/`ro_slew_fall`, blips separately via
`params.blip_slew`), after Huber et al.'s "PNS Optimized Pulses for EPI"
(bioRxiv 2026, doi 10.64898/2026.07.22.739360) -- nerve-integration PNS
models peak at the *end* of each sustained slew event (fall of lobe n ->
blip window -> rise of lobe n+1), so the ramp-up is throttled while the
ramp-down runs faster. See `lib/make_readout_grads.py`'s module docstring
for the full geometry (a1/a_d/S notation, the -S/2 prephaser scaling via
`ReadoutGrads.gx_pre_scale`, and the parity-independent in-block kx = 0
crossing exported as `ReadoutGrads.echo_offset`), and the PNS section
below for the measured numbers and why the tuned asymmetry is milder than
the paper's. Two long-standing bugs (inherited from the MATLAB original)
were fixed as part of this change, both verified by measurement before and
by tests after: (1) the sampled kx window was off-center by `a1` (~21
lines at old defaults) because `gx_pre` pre-wound exactly -kmax while the
first sample sits `a1` past the wind -- an unintended one-sided partial
Fourier, now fixed by `gx_pre_scale` (test:
`test_arbepi_kx_coverage_and_nyquist`); (2) saved per-echo times
(`schedules[..., 2]`) and the realized TE ran ~0.6-0.7 ms late because
`calc_te_tr_delays`'s `min_te` omitted the `gro1` lead-in block, assumed
the echo sat at the composite block's center, and measured "RF center" as
the RF block's midpoint (dead-time/ringdown asymmetry included) -- now
anchored at the true kx = 0 crossing via `echo_offset` and
`pp.calc_rf_center` (test:
`test_arbepi_schedule_echo_times_match_measured_kx_zero_crossings`).

**`sequences/EPIcal.py`** mirrors `sequences/ArbEPI.py`'s gradient design
exactly (same readout-grads/derated-sys construction, via
`lib/readout_from_params.py` -- the single source of truth for
`params.slew_derate`/`ro_slew_rise`/`ro_slew_fall`/`blip_slew`, replacing
three formerly hand-copied `sys.max_slew = 100 * sys.gamma` derates; the
trajectory tests use the same factory) but sets all blip scale factors to
0, so it acquires unencoded lines at k-space center for EPI ghost
correction. Its `gx_pre` must carry the same `rg.gx_pre_scale` factor as
ArbEPI's (enforced by `test_arbepi_kxoe_matches_epical`).

**`lib/trap4ge.py`** (ported from `../PulCeq/matlab/trap4ge.m`) rounds every
gradient's rise/flat/fall times up to `params.crt` via `math.ceil(t / crt -
1e-9) * crt`. Every gradient in the sequence passes through it before being
added to a block — this is a GE-hardware-timing requirement, not optional
cleanup — with one deliberate exception: the POPE readout trapezoid
(`make_readout_grads`'s `gro`) computes its rise/flat/fall directly as crt
multiples instead, because `trap4ge`'s area-preserving amplitude rescale
would perturb the `a1`/`a_d` ramp-area geometry the prephaser scaling and
echo timing are derived from (same raster guarantee, different route). The `- 1e-9` epsilon before `ceil` was added after finding that
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
(used for `.seq` generation) and `ge/check.py`/`ge/writeceq.py`'s
hardware/PNS/acoustics checks (used by `ge/ge_export.py`'s `--ge` path, pure
Python, see below) both read from that same `ScannerSpec` instance, so
they cannot drift out of sync the way they used to (see git history for
the bug this replaced: `slew_max` had been hand-set to a value 25% higher
than `sys.max_slew` actually specified). `ScannerSpec.ge_coil` (e.g.
`'xrm'`, `'hrmbuhp'`) keys `ge/acoustics.py`'s per-coil forbidden-band
table and `ScannerSpec.chronaxie`/`rheobase`/`alpha` key `ge/pns.py`'s
per-coil PNS coefficients — see `../PulCeq/matlab/+pge2/opts.m`'s header
comment for the authoritative table if adding a new scanner. `PNSwt`
stays a separate `Params` field (not part of `ScannerSpec`) since it's
scan-context — phantom vs. human — not a hardware constant; PNS is a
physiological limit, so don't silently lower `PNSwt` to make an error go
away — `[0, 0, 0]` is only valid for phantom/non-human scanning, and the
default is now the IEC 60601-2-33:2022-recommended `[0.8, 1.0, 0.7]` (see
the "PNS finding history" below for why it wasn't always). `ge.ge_export.check_ge_feasibility()`
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
across candidate `ETL` values) to check. Its `echo_offset` parameter
(pass `ReadoutGrads.echo_offset`) anchors the prescribed TE at the true
kx = 0 crossing — see the POPE paragraph above for the TE-accuracy bugs
this fixed.

**`mask2epi_laminar`/`mask2epi_radial`'s `Nshots*ETL` must exactly equal
the sampling mask's total sample count** (asserted at the top of each) —
`Nshots = ceil(Ny*Nz/R/ETL)` doesn't guarantee this holds for an arbitrary
`ETL`; picking a new `ETL` without also checking this divides-evenly
constraint will crash either function, not just produce a suboptimal
schedule.

### GE `.pge` export -- fully ported to Python, no MATLAB round trip

`ge/` is a from-scratch Python port of PulCeq's `seq2ceq`/`writeceq`/
`pge2.pns`/`check_grad_acoustics` toolchain, and is now the operative path
for both `main.py --ge` and standalone use — `ge/ge_export.py` calls straight
into `ge.seq2ceq`/`ge.writeceq`/`ge.check`, with no `matlab -batch` shell-out
and no sibling `../pulseq`/`../toppe`/`../PulCeq`/`../ArbEPI` checkouts
required. MATLAB is only needed if re-validating this port against a fresh
`../PulCeq` checkout in the future (the one-off `dump_*.m` scripts used to
produce the reference data cited below no longer live in this repo — see
"matlab_reference/ removed" below).

`ge/seq2ceq.py` + `ge/blocks.py` port `seq2ceq.m`/`compareblocks.m`/
`getdynamics.m`/`getblocktype.m`; `ge/writeceq.py` ports `writeceq.m`'s
binary `.pge` writer. Validated field-by-field against real MATLAB output
(`ge/validate_against_matlab.py` + the since-removed `dump_ceq.m`, which
dumped MATLAB's `ceq` struct for comparison) on all four generated sequences
**as they existed at validation time** (`GRE.pge`, the single-echo
predecessor to today's dual-echo `deGRE.pge` -- not re-validated against
MATLAB since that rename/upgrade):
`noise.pge`/`GRE.pge` come out byte-identical, `ArbEPI.pge`/`EPIcal.pge`
differ only in the `maxSlew` header float32's last bit (summation-order
noise, not a bug -- confirmed by matching every other field including the
full loop table exactly). This byte-for-byte/near-identical match was
re-confirmed after wiring `ge/ge_export.py` end to end (fresh MATLAB
references regenerated from the exact `.seq` files `main.py --ge`
produces, `noise`/`GRE` byte-identical, `ArbEPI`/`EPIcal` 1-float32-ULP on
`maxSlew` only, all four `loop` tables matching exactly) -- not just on
`ge/seq2ceq.py`/`ge/writeceq.py` in isolation. Two real bugs were caught
and fixed by this validation: a sign error in `_write_grad`'s waveform
normalization, and a wrong assumption that the `.pge` header's `maxSlew`
field echoes the scanner's hardware limit rather than (as MATLAB actually
does) the sequence's realized peak slew.

One deliberate, disclosed deviation: `seq2ceq.m`'s gradient-heating
`Emax` calculation sums stale 1-indexed loop columns 11:13 (`energy_gz,
recphs, blockDuration`, not the three gradient energies its variable names
imply) -- `ge/seq2ceq.py` sums the correct energy columns instead. This
produced identical `Emax_n` results on all four test sequences but is not
guaranteed to in general; consider reporting the indexing bug upstream to
PulCeq.

**PNS/acoustics/hardware checking is also ported**: `ge/pns.py` (GE's
IEC 60601-2-33:2022 PNS model, from `pge2.pns.m` -- not the same model as
pypulseq's own `pypulseq.utils.safe_pns_prediction`, which is Siemens'
different SAFE parameterization), `ge/acoustics.py` (`check_grad_acoustics.m`'s
FFT/forbidden-band check, per-coil tables copied verbatim), and `ge/check.py`
(`check_ge_feasibility()`, combining both plus gradient/slew/B1 hardware
limits, run directly on `seq.get_gradients()` -- no MATLAB round trip).
Both `ge/pns.py` and `ge/acoustics.py` match real MATLAB output to
float64/float32 precision on identical input (`ge/validate_pns.py` +
the since-removed `dump_pns_test.m`/`dump_acoustics_test.m`). `ge/check.py`'s
whole-sequence PNS convolution matches MATLAB's real per-segment-instance
computation (via the since-removed `dump_pns_peak.m`, which replicated
`checksegment.m`'s pipeline without its fail-fast throw) to within 0.02
percentage points on the full ArbEPI/GE_UHP sequence (114.7% vs 114.72%).

Acoustics is checked over a bounded window rather than the whole sequence
-- its FFT cost scales with window length, and a full ~60s/15M-sample
sequence takes minutes for no benefit. The window is not an arbitrary
fixed duration: `ge/check.py`'s `_blockrange_window_s` reproduces the
exact block-selection semantics of MATLAB's own check (`pge2.plot(ceq,
sys, 'blockRange', [1 10], ...)` inside `write_to_ge_from_seq.m` -- walk
`ceq.loop` rows in segment order starting at row 1, including whole
segments until the next segment's start row would be >= `block_range[1]`),
computed from this port's own `seq2ceq(seq)` rather than guessed at.
Reproduction against real MATLAB output (via the since-removed
`dump_acoustics_blockrange.m`) on this repo's four sequences **as they
existed at validation time** (`GRE.seq`, the single-echo predecessor to
today's dual-echo `deGRE.seq` -- not re-measured against MATLAB since that
rename/upgrade, so these numbers are a historical record of the method's
accuracy, not a current claim about `deGRE.seq`): window duration matches
to within one 4us raster sample (`GRE.seq`: 173766 vs MATLAB's 173767
samples; `ArbEPI.seq`: 25000 vs 25001), and the resulting acoustics number
matches to <0.04% relative error (`GRE.seq`: 0.4024 here vs MATLAB's
0.402213; `ArbEPI.seq`: 0.028146 here vs MATLAB's 0.02814424) -- the one
earlier open question (a fixed 1s window vs. MATLAB's shorter blockRange
window giving different magnitudes, `GRE.seq` 0.312 vs 0.402) is resolved:
matching MATLAB's window definition reproduces MATLAB's number directly,
it was purely a window-choice gap, not an algorithm mismatch. **The
window-duration half of this reproduction still holds exactly today; the
`ArbEPI.seq` acoustics *magnitude* does not** -- the switch to the
asymmetric POPE readout changed it 5.3x, to 0.1484 (see
`docs/review-findings.md`'s "Current baseline" table for the current
whole-sequence numbers), so `0.028146`/`0.02814424` here are historical
only, on both counts (the `GRE.seq` -> `deGRE.seq` rename *and* the POPE
readout change) -- not just the rename this paragraph originally flagged.
The residual gap (MATLAB's
per-segment `segment_dead_time`/`segment_ringdown_time` padding,
~117us/segment, a GE-Ceq-interpreter artifact with no Pulseq-timeline
equivalent) is deliberately not reproduced -- confirmed negligible above.
PNS and hardware limits are *not* windowed -- cheap enough to check in full.

**Real behavioral bug caught while wiring this in**: `check_grad_acoustics.m`
only ever calls MATLAB's `warning(...)` when `magb > threshold` (see its
source, line ~159 -- `if magb>threshold, warning(...); end`) -- it has
never once blocked a real MATLAB `--ge` export, unlike PNS
(`checksegment.m` really does `throw(MException(...))` above 80%, see
below). `ge/check.py`'s `FeasibilityReport.ok` originally folded acoustics
into the same hard gate as PNS/hardware limits; this would have made the
Python `--ge` path *reject* `GRE.seq` (acoustics 0.402, over the 0.3
threshold) even though the exact same sequence has always exported
successfully via the real MATLAB path. Fixed: `.ok` now excludes
acoustics, `.summary()` reports it as `WARN` (non-blocking) rather than
`FAIL` when over threshold -- matching MATLAB's real behavior, not a
guess. This gate-design decision is permanent regardless of any one
sequence's number (see `ge/check.py`'s `PNS_NORMAL_MODE_THRESHOLD`
comment for the same reasoning applied to PNS's 80/100% split). The
acoustics number that motivated it is now stale, though, since `GRE.seq`
became the dual-echo `deGRE.seq`: today's `deGRE.seq` measures acoustics
0.2456 -- *under* the 0.3 threshold, so the WARN-not-FAIL distinction is
no longer even live for this repo's current default sequences (re-check
after any sequence-timing change, since acoustics is scan-parameter-
dependent the same way PNS is).

**PNS finding history (resolved 2026-08-27; kept because the numbers below
are cited elsewhere)**: `params.py`'s `PNSwt` default was `[0, 0, 0]` for
the entire lifetime of this port until 2026-08, which meant PNS was never
actually evaluated in any `--ge` run to date -- weight zero makes
`pge2.pns`'s per-channel contribution zero regardless of the real
waveform, and MATLAB's `checksegment.m` really does *throw* on PNS > 80%
(`../PulCeq/matlab/+pge2/checksegment.m`). With the IEC
60601-2-33:2022-recommended `wt = [0.8, 1.0, 0.7]` the original
full-hardware-slew sequences measured `EPIcal` 113.2%, `ArbEPI` 114.7%,
`GRE` 100.6% (on GE_UHP); an interim symmetric derate to 100 T/m/s
(hardcoded in the sequence files at the time) brought ArbEPI to ~84.3% --
still over the 80% normal-mode line. The resolution is the POPE
asymmetric readout (see `lib/make_readout_grads.py`'s paragraph above)
plus an empirical slew sweep (2026-08-27, ~600 rise/fall/blip candidates,
full-dims worst-frame ArbEPI builds scored by `ge/pns.py`'s RSS-combined
total): the tuned defaults in `params.py`
(`slew_derate=100`, `ro_slew_rise=100`, `ro_slew_fall=120`,
`blip_slew=105`) measure **79.8% peak on the full ArbEPI build (GE_MR750,
seed=0) at min TE 34.86 ms**, vs 77.4% at min TE ~35.8 ms for the
symmetric-100 design through the same code -- POPE spends ~2.4% of PNS
margin to shorten TE by ~0.9 ms. `blip_slew=105` is a deliberate
ride-the-line choice (explicit user decision) leaving only ~0.2% margin
to the 80% limit -- thinner than observed mask-to-mask variation, so
re-verify after any seed/mask/`R`/`ETL`/resolution change and drop back
to `blip_slew=100` (78.3% at min TE 35.10 ms) if a new mask pushes it
over. Two sweep lessons worth keeping: (a) the
per-channel PNS maxima are badly misleading here -- the y/z blips play
centered on the kx turnaround, exactly where the readout fall ramp ends,
so aggressive fall/blip slews RSS-combine into a 3-channel hotspot (e.g.
rise/fall/blip 95/200/170 looks fine per-channel but totals 106%), which
is why the tuned fall/rise ratio is far milder than the POPE paper's
hardware-limit fall; (b) a prescribed TE of 30 ms (a considered target) is
unreachable under 80% -- every swept config at min TE <= 35.2 ms exceeded
the line, ~33 ms costs >85%, and ~30 ms well over 100%.
`test_arbepi_default_params_peak_pns_under_normal_mode_limit`
(tests/test_ge_check.py) now regression-guards the <80% property on every
test run, and `plotting/compare_readout_pns.py` (`uv run python -m
plotting.compare_readout_pns`) rebuilds the symmetric-vs-POPE comparison
-- two full ArbEPI sequences from the same seed=0 masks and identical
nominal parameters, per-variant PNS-over-one-TR figures plus a combined
overlay (`output/compare_pope/`), and a printed table (peak PNS, echo
spacing, realized TE). PNS remains scan-context-dependent: these numbers
are for `PNSwt = [0.8, 1.0, 0.7]` on GE_MR750 with the seed=0 mask, and
any change to scanner, mask seed, `R`/`ETL`, or resolution needs the
check re-run (the regression test and `main.py --ge` both do).

**Investigated and closed (2026-09-03): the POPE paper's *adaptive*
ramp (hardware-max slew until predicted PNS hits 99%, then throttle to a
constant "steady-state" slew for the rest of the ramp) does not help
here, and should not be retried without addressing the reason below
first.** The paper's own description (Methods, Fig. 2B) is a genuinely
different design from what this repo implements: `lib/make_readout_grads.py`
throttles the *entire* rise ramp to one constant slew
(`ro_slew_rise=100`), whereas the paper only throttles the *tail* of the
ramp and lets the early part run at full hardware slew (250 T/m/s on
their Siemens system) -- worth trying since a nerve-integration PNS
kernel weights recent slew history much more than distant history, so
front-loading fast slew where it's furthest from the instant being
evaluated should, naively, lower PNS at that instant. A prototype tested
this directly against the real PNS model: took the actual full-dims
seed=0 `ArbEPI` build at shipped params (rise=100, fall=120, blip=105,
baseline peak PNS 78.9%, matching the 79.84% figure above), located every
one of the 1200 real rise ramps in the sampled `gx` waveform (uniformly
496us/124 samples, confirmed by `diff(gx)/dt`), and replaced each one --
at *fixed total duration and fixed endpoints*, so nothing downstream
(flat top, fall, blips) retimes -- with a two-piece ramp: GE_MR750's
200 T/m/s hardware-max slew for a leading fraction tau, then whatever
constant slew finishes the same delta-G in the remainder, i.e. exactly
the paper's shape. Result: peak PNS (evaluated as the true global max
over the whole sequence via `ge/pns.py`'s real IEC 60601-2-33 model, not
just at the original ramp-end instant) **increases monotonically for
every tau > 0 tested**, with no minimum away from tau=0 -- +0.16pp at
tau=25us, +0.33pp at 50us, a sharp knee to +11.3pp by 100us, +35pp by
200us. Mechanism: the naive "recent slew dominates" intuition is correct
*at the original evaluation instant*, but it says nothing about the rest
of the timeline -- concentrating any nontrivial duration of 200 T/m/s
slew creates a *new* competing PNS peak right at the end of that burst,
and in this repo's train that new peak wins immediately. The reason it
wins so fast is timing density: `chronaxie` (334us, GE_MR750) is
comparable to or longer than the entire rise/fall/blip cycle here (rise
496us ~= 1.5 chronaxie, fall 416us ~= 1.25 chronaxie, separated by only a
~54-58us blip window) -- there is no PNS-quiet stretch within one cycle
long enough to hide a hardware-max burst without it landing immediately
next to the already-hot fall+blip turnaround (the same RSS-hotspot
coupling the sweep lessons above already document). This is the likely
resolution of the discrepancy between this repo's own POPE gain (~3%
echo-spacing reduction, symmetric-100 1108us -> POPE 1076us) and the
paper's headlined 7-38.8% (their Fig. 2E, resolution-dependent): their
benefit comes from long flat-tops/readouts relative to ramp duration and
chronaxie, which this repo's ETL=60 tightly-packed 3D-EPI train does not
have. **Conclusion: do not implement the adaptive/bang-then-throttle
ramp shape as a standalone change** -- it is strictly worse than the
current constant-slew rise in this design's PNS/timing regime, not just
unproven. It could conceivably become viable if the cycle geometry
changes enough to open up a genuine quiet stretch (materially longer
flat top/lower ETL, or a much shorter chronaxie scanner profile) --
re-run this same prototype methodology against the real `ge/pns.py`
model and real generated sequence before trying again, rather than
re-deriving it from the paper's numbers alone, since those numbers come
from a different PNS model (Siemens SAFE) and a different timing regime.

`params.py`'s `Params` dataclass carries the selected `ScannerSpec` itself
as `params.spec` (rather than duplicating `max_grad`/`max_slew`/`b1_max`/
`ge_coil`/`pislquant`/etc. as separate derived fields, as it did when the
MATLAB path needed them formatted into a `pge2.opts(...)` snippet) --
`ge/check.py`/`ge/writeceq.py` read straight from `params.spec`.

**`matlab_reference/` removed** -- it held only the one-off validation
scripts used to produce the MATLAB reference data cited throughout this
section (`dump_ceq.m`, `dump_pns_test.m`, `dump_pns_peak.m`,
`dump_acoustics_test.m`, `dump_acoustics_blockrange.m`), none of which were
ever called by any Python code; their content is preserved in git history
for anyone re-deriving this port's validation record against a fresh
`../PulCeq` checkout. `write_to_ge_from_seq.m`/`ge_feasibility_check.m` (the
former MATLAB shell-out targets) were removed earlier, once `ge/ge_export.py`
no longer called them, for the same reason.
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
  step (both independently depend only on `scan_info.mat`/`ArbEPI.seq`, not
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
- **`ge/coppe.py`** SSH-copies a folder of `.pge` files (e.g. `output/*.pge`)
  to the scanner, auto-allocating unused `pge2`/v7 entry numbers (0-9999)
  and writing each file's `pgeN.entry` — the pull-based two-hop transfer
  TOPPEpsdSourceCode's UserGuide documents doing by hand. University of
  Michigan fMRI lab-internal (server names are lab-specific); not part of
  `main.py --ge`'s own export path, and not needed to generate or validate
  `.pge` files. See `ge/README.md` for usage and SSH key setup.
- **`sampling/external_mask.py`**'s `load_external_mask` is a deliberate
  manual escape hatch, not a fifth `params.sampling_method` --
  `gen_sampling_masks` has no `'external'` branch for it. Loads a
  precomputed 2D `(ky, kz)` or 3D `(ky, kz, t)` mask from an outside
  collaborator's own v5 `.mat` file (`scipy.io.loadmat`, not
  `hdf5storage.loadmat` -- v5, not this repo's own v7.3 convention) for a
  sample-selection method this repo doesn't itself implement. Using it
  means calling `sequences.ArbEPI.generate_arbepi(omegas, params)`
  directly with the loaded array in place of `gen_sampling_masks`'s
  output, bypassing `main.py`'s documented entry point by hand.

### `preprocessing/` -- raw scanner data -> zero-filled k-space, ported from `../epi-preprocessing`

`preprocessing/` is a from-scratch Python port of the companion MATLAB repo
`../epi-preprocessing`: raw scanner data -> reconstructed images, the
consumer of `scan_info.mat` this repo's sequence-generation side produces.
It's a separate `pyproject.toml` optional-dependency group
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
BART/MIRT (unlike `ge/`'s float-ULP-level MATLAB validation) --
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
exports exactly the scalars `preprocess.py` needs into `scan_info.mat`
(`hdf5storage.savemat(fmt='7.3')`, alongside the kxo/kxe/schedules arrays it
also writes there -- see the ".mat file format" section above);
`preprocessing/config.py`'s `load_seq_params` reads it back with `h5py`
(see below for why not `hdf5storage`). No new dependency needed on either
side: `hdf5storage` is already a main-repo dependency (writer), plain
`h5py` is already in the `preprocessing` extras (reader).

**`preprocessing/matio.py` -- read hdf5storage `.mat` files with `h5py`,
correctly.** `hdf5storage` stores arrays *axis-reversed* on disk (MATLAB's
column-major convention); `h5py` reads the raw on-disk layout with no
correction. Verified empirically against a real `scan_info.mat`:
`hdf5storage.loadmat`'s `schedules` is `(30, 20, 60, 3)` (matching this
repo's documented `Nframes x Nshots x ETL x 3` layout), while a bare
`h5py.File(...)['schedules'][()]` comes back `(3, 60, 20, 30)` -- exactly
the reverse, and exactly `raw.transpose()` recovers the correct array
(shape *and* values, checked element-by-element against
`hdf5storage.loadmat`'s output). `matio.read_mat_array`/`read_mat` apply
this transpose unconditionally; for vectors/scalars it's a no-op on the
values (only a singleton axis moves), so there's no need to special-case
shape. Use these for every hdf5storage-written `.mat` this pipeline reads
(`scan_info.mat`) -- never `scipy.io` (can't read v7.3 at all, see above),
never a bare `h5py` read without the transpose.

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
TE_degre)` and `TE_degre` itself in its scan-scalar snapshot (now part of
`scan_info.mat`, formerly its own `params.mat` -- see the ".mat file
format" section above) (`preprocessing/config.py`'s
`SeqParams.n_echoes_degre`/`TE_degre`, read by `load_seq_params`; both
default -- `1` and `None` respectively -- when missing, since a snapshot
written before this change is a durable, non-regeneratable per-acquisition
data record, not something to raise `KeyError` on).

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

**The `<seqname>_gre.h5` cache is the handoff point to B0 field map
estimation -- deliberately not GE-specific.** Alongside the existing
`ksp_gre` dataset (the selected echo, whitened + coil-compressed,
unchanged key so `recon_frames.py`'s cache reader needs no changes), STEP
2 now also writes `ksp_gre_echoes` (`[Nx_degre, Ny_degre, Nz_degre,
n_echoes, Nvcoils]`, both echoes, whitened + coil-compressed) and a
`TE_degre` attr (seconds) whenever `seq_params.TE_degre` is available.
Both are plain numpy-order HDF5 (see the `.h5`-vs-`.mat` convention note
above) -- a consumer needs only `h5py`/`HDF5.jl`, never GERecon or the raw
ScanArchive.

**B0 field map estimation is implemented, via
[MRIFieldmaps.jl](https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl)
(Lin & Fessler, "Efficient Regularized Field Map Estimation in 3D MRI",
IEEE TCI 2020) -- not ported to Python, since no such port exists and
MRIFieldmaps.jl's regularized NCG solver is the actual state of the art
here, not boilerplate worth reimplementing.** `preprocessing/julia/` is a
small, self-contained Julia project (`Project.toml` + a pinned
`Manifest.toml`, both committed) holding one script, `b0map.jl`, invoked
as a subprocess by `preprocessing/run_b0map.py` (`julia
--project=preprocessing/julia preprocessing/julia/b0map.jl <gre_h5>
<output_h5> [smaps_h5] [eig_mask_threshold] [mask_threshold] [precon]`) --
not embedded via PythonCall/juliacall,
since there is no other Julia dependency anywhere in this pipeline to
justify that weight, and a subprocess boundary mirrors how `raw_io.py`
already isolates GE's proprietary GERecon SDK to one module rather than
embedding it more deeply. First-time setup needs `julia
--project=preprocessing/julia -e 'import Pkg; Pkg.instantiate()'` (network
required once, to populate the local package depot); after that, no
network access is needed to run it.

`b0map.jl` reads `ksp_gre_echoes`/`TE_degre` back from the cache, IFFTs
each echo/coil to image space with the same centered-FFT convention as
`run_rss.py`'s `_ift3` (`fftshift(ifft(fftshift(.)))` per axis), and calls
`MRIFieldmaps.b0map(finit, images, echotime; smap, mask, precon)`.
`precon=:diag` (not `b0map`'s own default, `:ichol`) -- see the dedicated
paragraph below for why. `smap`, when `preprocessing/smaps.py`'s
`load_smaps` has a deGRE-grid sensitivity-map cache available (see that
module's docstring: it now resizes the same `cal_size`-cropped ESPIRiT
calibration onto *this* deGRE grid, not just the EPI grid), replaces
MRIFieldmaps' own phase-contrast coil-combine fallback with a true
matched-filter combine -- `run_b0map.py` computes/loads this best-effort
(falls back to no `smap` if unavailable, e.g. no `ksp_gre` in the GRE
cache, rather than failing field-map estimation over an optional input).
An explicit magnitude-threshold `mask` (default `0.1` x peak first-echo
magnitude, matching `MRIFieldmaps.b0init`'s own default), ANDed with an
ESPIRiT-eigenvalue-based mask when `smap` is available, is *not* optional
here despite `b0map`'s own `mask` keyword defaulting to "every voxel": its
no-`smap` coil-combine path divides by each voxel's coil sum-of-squares,
and a synthetic all-zero-background test volume confirmed this produces a
background 0/0 = NaN that poisons Julia's `maximum()` and returns an
all-NaN field map end to end (real scanner data has thermal noise
everywhere so an exactly-zero voxel won't occur, but masking out
background is standard practice for this package regardless, so the mask
stays mandatory here rather than becoming a latent footgun).

**`precon=:diag`, not MRIFieldmaps' own default `:ichol` -- the actual fix
for a real, measured reconstruction artifact.** A B0-corrected
reconstruction (see `recon/`'s "B0 off-resonance correction" subsection)
kept showing salt-and-pepper speckle after landing on `L=32`; the field
map itself turned out to be the cause, and specifically its optimizer, not
its regularization weight. `l2b` (`b0map`'s log2 roughness-regularization
weight) was swept across `[-6, 28]` -- a 16384x-to-270-million-x range in
the actual `beta = 2^l2b` -- and had essentially no effect on the fitted
field map's smoothness (`mean|Laplacian|` roughness flat at ~8.2-8.4
throughout, confirmed both under-converged and via a full 2000-iteration
NCG run at `l2b=4`, which was still measurably reducing cost yet converged
to the same roughness as `l2b=-6`). Traced to `:ichol`'s preconditioner
(`H = spdiagm(hcurv) + CC`) being built from the *same* `CC = beta * C'C`
operator that appears in the gradient (`grad = hderiv + CC*w`) -- as
`beta` grows, both numerator and denominator become `CC`-dominated and the
preconditioned NCG step collapses toward `-w`, independent of `beta`.
Switching to `precon=:diag` (`Hdiag = hcurv + diag(CC)`, no such
numerator/denominator cancellation since only the diagonal of `CC` enters
the preconditioner) fixed it directly: field-map roughness dropped ~4x
under `:diag` vs `:ichol` at identical `l2b`/`niter`, and re-reconstructing
with the `:diag` field map confirmed the fix at the image level too --
B0-correction-induced excess image roughness (over an uncorrected
baseline) dropped from +61.1 (`:ichol`) to +19.5 (`:diag`). This isn't
data-specific: MRIFieldmaps.jl's own
[`02-b0map.jl` example](https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl/blob/main/docs/lit/examples/02-b0map.jl)
independently notes "the diagonal preconditioner seems to be as effective
as the incomplete Cholesky preconditioner" in this Julia port (unlike the
original MATLAB implementation, where `ichol` is the expected/reliable
win) -- though that comparison is about final-RMSE-vs-wall-time on its own
canonical test case, not `l2b` sensitivity specifically, so it corroborates
the outcome without independently confirming the mechanism above (traced
directly in this pipeline's own code instead). With `precon=:diag` fixing
the actual problem, `l2b`/`niter` were *removed* from `b0map.jl`'s exposed
CLI arguments (they'd been added specifically to chase this bug) and
MRIFieldmaps' own call is no longer passed them at all -- both silently
revert to library defaults (`-6.0`/`30`), which is all that's needed once
the preconditioner is right.

**`finit` is built via [ROMEO.jl](https://github.com/korbinian90/ROMEO.jl)
(Dymerska et al., "Phase unwrapping with a rapid opensource minimum
spanning tree algorithm (ROMEO)", MRM 2021) rather than left to `b0map`'s
own default (a plain, possibly-aliased two-point phase difference).**
`b0map`'s NCG solve has a data-fit term that is itself periodic (see the
cost function referenced below), so it doesn't strictly need unwrapped
phase to converge -- but that periodicity only guarantees a *locally*
consistent optimum, not that NCG finds its way to the globally correct 2π
branch starting from a badly-aliased `finit`. This was measured, not
theoretical: `b0map.jl`'s own dedicated test
(`test_run_b0map_unwraps_a_field_map_beyond_the_naive_unambiguous_range`
in `tests/test_preprocessing_run_b0map.py`) uses a synthetic field map
exceeding the naive `finit`'s +-1/(2 dTE) unambiguous range (dTE = 2 ms =>
+-250 Hz); `b0map` fed the ROMEO-unwrapped `finit` recovers the true field
to well under 60 Hz RMSE, fed the plain wrapped `finit` it converges to a
local minimum that reproduces the aliasing instead of correcting it
(~207 Hz RMSE, confirmed in scratch testing during development).

Which array actually needs unwrapping is *not* obvious, and got it wrong
on the first attempt: `MRIFieldmaps.coil_combine`'s phase-contrast formula
(`zdata_e = sum_c conj(y_{c,1}/sos) * y_{c,e}`) makes the *reference*
echo's own combined phase (`zdata[...,1]`) identically zero by
construction for every voxel (`conj(y_{c,1}) * y_{c,1} / sos` sums to a
real positive number) -- confirmed empirically when a first attempt at
spatially unwrapping `zdata[...,1]` changed exactly zero voxels on data
that plainly needed it. The physically meaningful signal is
`zdata[...,2]`, which for this two-echo case reduces to exactly `y2 *
conj(y1) / sos` -- the same wrapped phase *difference* `MRIFieldmaps.
b0init` itself computes (`angle.(y2 .* conj(y1))`). So `b0map.jl`'s
`romeo_finit` unwraps that one 3D volume via `ROMEO.unwrap`, weighted by
its own magnitude (a coherence-like quantity in `[0, 1]` from the
phase-contrast combine -- naturally lower wherever the two echoes
disagree, i.e. exactly where `unwrap` should trust the local phase less,
not the raw image amplitude), then divides by `2π * dTE` to get Hz. Only
the first two echoes are used, matching `b0init`'s own restriction to
two-point phase difference in the non-water-fat case -- consistent with
this pipeline only ever acquiring a two-echo deGRE.

**Axis order, both directions, verified empirically rather than assumed:**
HDF5.jl reads/writes arrays reversed relative to h5py/numpy, the mirror
image of the `hdf5storage`-vs-`h5py` gotcha `preprocessing/matio.py`
documents for the opposite (MATLAB-writer) direction. Confirmed against a
real Python-written `(Nx, Ny, Nz, n_echoes, Ncoils)` dataset: Julia's
`read` returns it as `(Ncoils, n_echoes, Nz, Ny, Nx)`, and
`permutedims(raw, reverse(1:ndims(raw)))` recovers the correct array
(`b0map.jl`'s `read_numpy_array`). The same reversal is applied on
*write* (`write_numpy_array`), so `<seqname>_b0map.h5`'s
`b0map_hz`/`finit_hz`/`mask` datasets land on disk already in numpy axis
order and need no correction
read back from Python -- confirmed both ways with a synthetic dual-echo
GRE volume carrying a known spatial field-map ramp (`tests/
test_preprocessing_run_b0map.py`): shape matches exactly, and the
recovered field map correlates > 0.98 with the injected ground truth.
That test (and the whole `preprocessing/julia/` path) is skipped
whenever no `julia` executable is on `PATH` -- the same tolerance this
repo already extends to MATLAB-based comparisons (no MATLAB install was
available during this port either, see the Commands section) and to
`raw_io.py`'s GERecon dependency.

### `recon/` -- Multi-Scale Low-Rank (MSLR) fMRI reconstruction, ported from `../mslr-recon` onto `mirtorch`

`recon/` is a Python/PyTorch port of the companion Julia repo `../mslr-recon`
(multi-scale locally-low-rank fMRI reconstruction, Ong & Lustig 2016 --
SENSE forward model per time frame + a nuclear-norm patch regularizer at
several spatial scales, solved via FISTA/POGM), built on
[mirtorch](https://github.com/guanhuaw/MIRTorch) in place of `../mslr-recon`'s
MIRT.jl + LinearMapsAA dependency. It consumes this repo's own
`preprocessing/` output directly (`<seqname>_epi_zf.h5`'s `ksp_epi_zf` +
`smaps_<seqname>_sigpy.h5`'s `smaps`) -- no format bridging needed, since
mslr-recon already reads exactly this key/shape convention from ArbEPI-python's
sigpy exports (see `../mslr-recon`'s own docs).

Like `preprocessing/`, this is a separate `pyproject.toml` optional-dependency
group (`recon`: `torch`, `mirtorch`, `h5py`, `nibabel`) kept out of the core
and `preprocessing` dependency sets -- torch is a large, often
GPU-index-specific install with no reason to share a venv with GERecon's
Python-3.10/numpy<2.0-locked `preprocessing` extra, so it gets its own
dedicated venv (`.venv-recon`) the same way `preprocessing` gets
`.venv-preprocessing`. Unlike `preprocessing/julia/`'s subprocess boundary to
Julia, `recon/` imports mirtorch/torch directly at module level (mirtorch is
pure Python, pip-installable, no cross-language embedding problem to solve)
-- tests gate on it via `pytest.importorskip("torch")`/`("mirtorch")`, not a
`shutil.which` subprocess check, so the main test suite still collects
without the `recon` venv active.

**Module layout**: `recon/lowrank.py` (patch extraction/recombination +
singular-value soft-thresholding, ported from `../mslr-recon/src/recon.jl`)
batches every patch into one tensor and calls a single `torch.linalg.svd`
rather than looping per-patch like the Julia original's `@threads`/streaming
CUSOLVER dispatch -- PyTorch's batched SVD already parallelizes internally,
so the loop-based Array/CuArray dispatch split in `recon.jl` has no Python
equivalent needed. The one piece of Julia's `SVST` that *is* still needed is
the exact-zero shortcut (patches with `‖X‖_F <= beta` are forced to exact
zero before the SVD, not just after) -- a correctness safeguard against
subnormal-magnitude matrices producing NaN, not a Julia-GPU-only concern.
`recon/solvers.py` ports `../mslr-recon/src/mirt_mod.jl`'s `pogm_restart`
(PGM/FPGM/POGM with gradient restart and early stopping via `conv_tol`)
faithfully in the momentum/restart math, but drops its GPU-memory-specific
mechanics (manual buffer aliasing, forced `GC.gc()`/`CUDA.reclaim()`) since
those exist only to fit Julia's broadcast-allocates-a-new-array semantics
under a 48GB budget -- PyTorch's caching allocator and Python-float-times-
complex64-tensor weak-type promotion don't have that problem.

`recon/b0_correction.py`/`recon/operators_b0.py` add B0 off-resonance
correction on top of the plain `GatheredSense` encoding operator above --
see the dedicated "B0 off-resonance correction" subsection below for the
full design and investigation history. `recon/save_result.py` persists a
`ReconResult` to `.h5`/`.nii.gz`/`.json` (the "next piece" the rest of this
section used to say was missing); `recon/run_b0_recon.py` is the driver
that runs a real (not validation-only) B0-corrected reconstruction end to
end. `recon/sweep_time_segments.py` and `recon/benchmark_b0_cost.py` are
one-off analysis scripts (not part of the production path) that produced
the numbers cited in that subsection.

**`recon/operators.py`'s `GatheredSense`** is a custom `mirtorch.linear.
linearmaps.LinearMap` subclass, not `mirtorch.linear.mri.Sense` directly --
deliberately, despite `Sense` (with `norm='ortho'`) implementing
mathematically the exact same convention as `../mslr-recon/src/sense_gpu.jl`'s
`Asense_gpu` (verified by adjoint self-consistency in
`tests/test_recon_operators.py`: both apply
`fftshift(fftn(ifftshift(.)), norm='ortho')` forward and the mirror-image
adjoint, which -- since `fftshift`/`ifftshift` are permutation matrices,
`P^T = P^-1`, and ortho-normalized `fftn`/`ifftn` are mutually adjoint -- is
exactly the true adjoint for any grid size, odd or even, not a naively-
expected swapped-shift version). The reason for the custom subclass: mirtorch's
`Sense` returns a *dense* masked `(Nc,Nx,Ny,Nz)` k-space grid per frame, and
on this repo's real acquisition scale (240x240x45, 18 coils, 30 frames, R~9)
that OOMs a 49GB GPU on the very first gradient evaluation (measured: an
~11GB dense k-space tensor per intermediate, versus mslr-recon's own
memory budget which assumes the *gathered* `(K,Nc)` representation
`Asense_gpu` already uses, `K = Nx*Ny*Nz/R`). `GatheredSense` implements the
identical forward/adjoint math but gathers to the `K` sampled locations,
cutting every k-space-shaped tensor by the acceleration factor `R` -- this
was a real, measured fix, not a preemptive optimization (`build_encoding_
operator`+`gather_ksp` are the two entry points; `.A[it].idx` on the
returned `mirtorch.linear.BlockDiagonal` exposes each frame's own flat
spatial sample indices for gathering a matching k-space target array).

**`recon/reconstruct.py`'s `_load_array` reads chunked HDF5 datasets
chunk-by-chunk along the last axis, not via a single `d[()]` call.** Measured
on real `ArbEPI_epi_zf.h5` data (chunked one time-frame per chunk, ~373MB
each): a bare `d[()]` full-dataset read ran at ~7 MB/s (838M+ read syscalls
for 7.5GB -- an h5py/HDF5 chunk-cache pathology when the default cache
doesn't fit even one chunk), versus ~500 MB/s reading one same-sized chunk
slice at a time -- a two-orders-of-magnitude difference that made an 11GB
file's load alone take 15+ minutes before this fix (23s after).

**Validated against real `../mslr-recon` (Julia) output on real scanner
data** (2026-08-25, RTX A6000, `20260822ball_radial` and `20260822ball_laminar`
datasets -- the two `mask2epi` trajectory variants from the same acquisition,
see `../mslr-recon/experiments/20260822ball.jl`'s header -- both
`Nx,Ny,Nz,Nvc,Nt = 240,240,45,18,30`, `R~9`, all six run via
`recon/validate_against_mslr.py`, which re-derives every reconstruction
parameter from the Julia `.mat`'s own saved fields rather than
re-specifying them, so it always replicates exactly what the reference run
used):

| dataset | config | n_iters | dc_cost max rel diff | reg_cost max rel diff | X_recon rel L2 err | Pearson r | runtime (python vs julia) |
|---|---|---|---|---|---|---|---|
| radial | L (local only, `[15,15,15]`) | 55 | 5.9e-7 | 3.0e-6 | 1.6e-5 | 0.9999999998 | 309s vs 405s |
| radial | G (global only, whole volume) | 56 | 1.6e-6 | 8.1e-6 | 3.8e-5 | 0.9999999989 | 96s vs 134s |
| radial | G+L (both scales) | 101 | 1.6e-6 | 2.1e-4 | 2.1e-5 | 0.9999999997 | 597s vs 785s |
| laminar | L | 54 | 5.0e-7 | 2.7e-6 | 1.5e-5 | 0.9999999998 | 304s vs (not re-timed) |
| laminar | G | 55 | 1.6e-6 | 1.1e-5 | 3.3e-5 | 0.9999999993 | 95s vs (not re-timed) |
| laminar | G+L | 101 | 8.1e-7 | 1.2e-4 | 2.0e-5 | 0.9999999997 | 600s vs (not re-timed) |

All six configs converge to the *same* iteration count as the corresponding
Julia run (confirming `pogm_restart`'s gradient-restart and early-stopping
logic matches exactly, not just the final answer) and match to float32
summation-order noise -- the same class of ~1-ULP difference
`seq2ge/validate_against_matlab.py` already documents against real MATLAB
output, just for a very different (iterative, GPU-batched-SVD-heavy)
numerical pipeline. `reg_cost[-1]`'s check gets a looser tolerance
(`rtol=5e-4` vs the default `1e-4`) specifically for the two-scale `G+L`
configs -- summing nuclear norms across scales, including a giant
whole-volume SVD, measurably accumulates more floating-point noise
(~1-2e-4) than any single-scale config (~1e-6 on both datasets), a real,
consistent pattern rather than a threshold picked to paper over one
failing run. Python also ran consistently faster than Julia on every
radial config (laminar wasn't independently re-timed, since it exercises
identical code paths to radial).

**Why Python is faster despite Julia's raw kernels being faster per call
(measured 2026-08-25, in-situ instrumentation of the real `scripts/
reconstruct.jl` on real data, not isolated microbenchmarks)**: two
overheads specific to `mslr-recon`'s Julia implementation, not a general
Julia-vs-Python effect --
1. **`GC.gc(true); CUDA.reclaim()`**, called once per iteration as the
   first line of `g_prox` (`mirt_mod.jl`'s docstring point 5: added to fit
   Julia's broadcast-allocates-a-new-array semantics under a 48GB VRAM
   budget), costs **0.86-1.85s per iteration** in steady state --
   comparable to or larger than the entire FFT forward+adjoint call
   (~0.67s) and several times the SVD cost for the `G` config (~0.28s).
   PyTorch's caching allocator needs no such call.
2. **Sequential, not batched, per-patch GPU SVD** for the `L`/`G+L`
   configs -- `recon.jl`'s own GPU dispatch path is deliberately serial
   ("sequential CUSOLVER calls, no `@threads`", since `CuArray`s can't use
   `@threads`): measured **5.2s for 4500 sequential 3375x30 SVDs** in situ,
   vs. `recon/lowrank.py`'s single batched `torch.linalg.svd` call over the
   same 4500 patches at **3.8s** -- cuSOLVER's batched routine amortizes
   per-call launch overhead that 4500 individual calls each pay.

Both were confirmed directly, not inferred: a scratch-instrumented copy of
`reconstruct.jl` (`@elapsed`-style timing wrapped around `dc_cost_grad`'s
`A' * (A * image_sum(X) - ksp)` and around `g_prox`'s `GC.gc`/`CUDA.reclaim`
and `patchSVST` sections) run for a few real iterations on the real
`20260822ball_radial` data reproduced the full per-iteration wall-clock
almost exactly by summing these pieces. Meanwhile isolated single-call
benchmarks confirm Fessler's expectation holds at the kernel level: Julia's
`Asense_gpu` forward/adjoint (0.298s / 0.361s) is genuinely faster than
`recon/operators.py`'s equivalent (0.480s / 0.549s) and the two configs'
whole-volume SVD costs are comparable (Julia 0.158s vs Python 0.170s for
the same `2592000x30` matrix) -- it's the two overheads above, not the
underlying linear algebra, that flip the net result. (Aside, found while
instrumenting: Julia's own bulk `h5read()` on `ArbEPI_epi_zf.h5` hits the
same HDF5 chunk-cache pathology `recon/reconstruct.py`'s `_load_array` docs
above -- tiny-read-storm, not disk-speed-bound -- confirming that gotcha is
an HDF5-tooling issue in general, not Python/h5py-specific; irrelevant to
the runtime comparison above since `runtime_s` on both sides only measures
the solver loop, not data loading.)

Not yet ported from `../mslr-recon`: `src/activation.jl` (a standalone
GLM task-activation module, not wired into the main pipeline even in the
original) and `src/metrics.jl`/`scripts/report.jl` (tSNR maps and
convergence-plot reporting) -- both are QA/visualization, not required for
a working reconstruction path. `recon/save_result.py` now does write
`ReconResult` to disk (`.h5` full-precision complex + solver trace,
`.nii.gz`+`.json` magnitude image + metadata, reusing
`preprocessing/nifti_io.py`'s `save_recon_nifti`) -- `recon/run_b0_recon.py`
is the first real (not `validate_against_mslr.py`-style comparison-only)
consumer of it.

### B0 off-resonance correction

Two-stage B0 correction on top of the plain `GatheredSense` encoding
operator, consuming `preprocessing/run_b0map.py`'s field map
(`<seqname>_b0map.h5`) and `preprocessing/preprocess.py`'s per-sample
`echo_times`:

- **`recon/b0_correction.py`'s `demodulate_smaps`** -- static, single-
  segment correction: a per-voxel conjugate-phase phasor (evaluated at
  the nominal TE) pre-multiplied into `smaps` before the encoding operator
  is built, zero added per-iteration cost. Validated against a brute-force
  synthetic ground truth (`tests/test_recon_b0_correction.py`): corrects
  the dominant geometric-shift component well in a small-phase-excursion
  regime (~98% forward-model error reduction), but only partially at this
  pipeline's *real* scale (B0 up to +-300-350 Hz over an ETL=60, ~72ms
  echo train -- tens of radians of phase drift, not a fraction of one):
  ~5% error reduction, since a single TE-centered phase term can't track
  how off-resonance phase keeps accruing differently across the echo
  train. That gap is why time-segmented correction exists as a second
  stage, not a redundant one.
- **`recon/operators_b0.py`'s `GatheredSenseB0`** -- the fuller,
  time-segmented stage, via `mirtorch.linear.mri.mri_exp_approx` (the same
  min-max frequency-segmentation fit `mirtorch`'s own NUFFT-based
  `Gmri`/`GmriGram` use). `build_encoding_operator_b0(smaps, omega,
  b0map_hz, echo_times_yz, L, nbins)` solves the segmentation fit once
  against this pipeline's real per-frame-invariant ETL distinct echo
  times (not once per frame -- an earlier version did, which also OOM'd a
  real reconstruction by storing an independent per-frame copy of the
  shared `(L,*N)` `c_phasors` tensor). `estimate_spectral_norm` (power
  iteration) is required alongside it -- the B0-corrected operator's
  spectral norm has no closed form the way the plain SENSE operator's does.
  `GatheredSenseB0` subclasses `recon/operators.py`'s `GatheredSense`,
  delegating to its `_apply`/`_apply_adjoint` for the per-segment
  FFT/gather and scatter/IFFT/coil-combine rather than duplicating that
  math a second time.

**Sign convention and `mri_exp_approx`'s Hz/ms calling convention.**
Both stages share one convention, derived from Sutton, Noll, Fessler
("Fast, iterative image reconstruction for MRI in the presence of field
inhomogeneities," IEEE TMI 2003) and cross-checked against
`mirtorch.linear.mri.Gmri`'s own demo notebook, not just re-derived: the
forward operator needs `exp(+i 2*pi*b0map_hz(r)*t)` multiplied into the
image before the spatial-encoding FFT -- see `recon/b0_correction.py`'s
module docstring for the full sign derivation. `mri_exp_approx(b0, bins,
lseg, t)` (read directly from mirtorch 0.3.1's own source, the pinned
dependency) expects `b0` in **Hz** and `t` in **milliseconds** (it divides
by 1000 internally, twice), and returns `tl` already in **seconds** -- so
`operators_b0.py` passing `echo_times_s * 1000` against an unscaled-Hz
field map is correct, not a units bug, and `-b0map_hz` (not `+`) is what
composes correctly with `mri_exp_approx`'s own internal sign to reproduce
the physically-correct convention above (matching `Gmri`'s own
`zmap=-b0` call, which exists for the same reason).

**`L` (segment count): swept, not guessed.** The mirtorch-matching default
`L=6` was never validated against this pipeline's real bandwidth-time
product (`BT = Δf_range * T_readout ≈ 370 Hz * 0.072 s ≈ 27`) -- the one
test that seemed to validate a small `L` (`test_more_segments_reduces_
error_in_the_realistic_regime`'s "L=16 is ~exact" result) turned out to be
an artifact of its own synthetic grid having only 12 distinct echo times,
not evidence about the real `ETL=60` scale. `recon/sweep_time_segments.py`
reproduces the ground-truth construction at the real scale (`Ny=ETL=60`,
real field-map range/echo spacing) and sweeps `L` directly: there's a
sharp, Nyquist-like phase transition around `L≈27-32` (matching the
computed `BT`), not a gradual improvement curve -- `L=6` gives only ~35%
error reduction (barely better than no correction at all), while `L≈31-32`
is needed to get relative forward-model error under 1%. **Chose `L=32`**
as the production value (the smallest swept `L` clearing that 1% bar) --
`operators_b0.py`'s `build_encoding_operator_b0`, `reconstruct.py`'s
`run_recon`, and `run_b0_recon.py`'s `main`/`--L` all default to `L=32`
directly now, not overridden at each call site.

**`nbins`: the histogram width `mri_exp_approx` fits its segmentation
against, and a real signal-loss-plus-incoherent-noise bug in its own
right, independent of `L`.** `mri_exp_approx` builds its fit from an
*equal-width*, plain voxel-count histogram (mirtorch 0.3.1's
`_uniform_histogram` -- no magnitude weighting, unlike MIRT's original,
whose weight-vector argument mirtorch's port dropped) of `b0map_hz`'s
*entire* range, background included and unmasked. At this pipeline's real
scale (~half the volume near-zero background, in-object range wide and
asymmetric -- roughly -300 to +70 Hz, not symmetric around 0), mirtorch's
own `Gmri` default `nbins=20` puts nearly all the histogram's mass into
1-3 bins near zero, making the `(nbins, L)` least-squares fit severely
ill-conditioned everywhere else: measured per-sample `b_weights` row sums
(`operators_b0.py`'s `_check_b_weight_row_sums` -- each row should sum to
~1.0 when well-conditioned) ranged `[0.12, 2.89]` at `nbins=20` vs.
`[0.9985, 1.0022]` at `nbins=100` on the same real data. `nbins=128`
(comfortably past that threshold) is the production default; raise it
further before lowering it.

**Cost of that choice, measured not extrapolated**
(`recon/benchmark_b0_cost.py`, synthetic tensors at this repo's real scale
-- 240x240x45, 18 coils, 30 frames, `K≈288,000` samples/frame, on a free
RTX A6000): one forward+adjoint pass costs `L=6`: 5.78s, `L=32`: 30.63s,
`L=60` (`=ETL`, the accuracy ceiling): 57.40s -- essentially linear in `L`
(the encoding operator loops over segments rather than batching them, by
design, to keep peak memory at the single-segment footprint regardless of
`L`; memory is not the bottleneck at any of these `L`, topping out around
12.65GB at `L=60` on a 48GB GPU). Extrapolated to a full 101-iteration G+L
reconstruction using this repo's own measured non-encoding overhead
(~4.9s/iter): `L=6` ≈ 18 min, `L=32` ≈ 60 min, `L=60` ≈ 105 min --
`L=60` costs ~1.75x `L=32` for no measurable accuracy gain past the
`L≈32` threshold, so `L=32` is both the accuracy floor and the practical
sweet spot, not a compromise between two competing costs.

**Real reconstruction run**: `recon/run_b0_recon.py` (`--L 32`) reproduces
the existing G+L (multi-scale) config already validated against
`../mslr-recon` for the uncorrected case, saving to
`<datdir>/recon/mslr_b0/G+L_L<L>/<name>_recon.*` (one directory per `L`,
since `L` was under active investigation). `sigma1A` is measured fresh via
power iteration for the B0-corrected operator, not reused from the
uncorrected reference -- its spectral norm has no guaranteed relationship
to the uncorrected operator's.

**Field-map noise, not `L`, was the actual remaining artifact.** After
landing on `L=32`, the B0-corrected reconstruction still showed a
persistent salt-and-pepper speckle texture, distinguishable from this
phantom's *real* air-bubble signal voids (confirmed by
`preprocessing/gre_diagnostics.py`, which reconstructs the dual-echo deGRE
images to NIfTI/PNG for direct visual comparison against the field map --
the raw GRE images themselves are clean). Diagnosing this led to
`preprocessing/julia/b0map.jl`'s `precon` fix below -- see that module's
docstring for the full mechanism (`:ichol`'s preconditioner shares the
same `CC` operator as the `l2b` regularization term, which numerically
cancels `l2b`'s effect on the NCG descent direction once `CC` dominates;
switching to `:diag` fixes it directly and is independently corroborated
by MRIFieldmaps.jl's own example docs). Re-running the L=32 reconstruction
with the `:diag`-preconditioned field map confirmed the fix at the image
level, not just the field-map level: image-domain roughness
(`mean|Laplacian(magnitude)|` over the object mask) excess *above* the
uncorrected baseline dropped from +61.1 (`:ichol`) to +19.5 (`:diag`) --
roughly a 3x reduction in B0-correction-induced speckle, while the
legitimate boundary-sharpening effect of B0 correction is preserved (both
still differ from the uncorrected baseline by a similar relative L2
amount, ~13-16%, since correcting real geometric distortion is supposed to
change the image).

**Sensitivity-map rewiring (`preprocessing/smaps.py`'s `load_smaps`)**: a
real methodological improvement made available at the same time, but *not*
the fix for the above -- see `preprocessing/`'s section below for the
detail. Measured to make no difference to field-map roughness on this
phantom (whole-mask and by radial zone, including the low-SNR object
center this change specifically targets) once `precon=:diag` is already in
place, so it's infrastructure for future/noisier datasets, not something
this dataset's own results depend on.

**`grid_resize.py`'s `grid_mode=True` alignment fix (see
`preprocessing/`'s section below) is directly load-bearing here.**
`GatheredSenseB0.c_phasors` and `demodulate_smaps`'s phasor are both
per-voxel functions of `b0map_hz`, which reaches the encoding operator
already resized onto the EPI grid by that same code path -- a
pixel-center-vs-edge alignment error there would silently mis-register
the field map against `smaps`/the image grid the operator actually
applies to, not just against the deGRE-grid diagnostics `grid_resize.py`'s
own docstring measures. Any future change to that resize convention needs
re-checking against a real B0-corrected reconstruction, not just the
grid-alignment unit test.

See `README.md` for the getting-started walkthrough and the full
`Getting started` / `GE export` usage examples.
