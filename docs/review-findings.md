# Review findings backlog

Working to-do list of open code-review findings for this repo, meant to be
worked through locally with Claude Code. This file is the canonical home
for the backlog; it was split out from CLAUDE.md's "Open TODOs" section
(items 1-102, five review passes between 2026-08-31 and 2026-09-02) so the
list can grow without CLAUDE.md itself becoming unwieldy. CLAUDE.md's
Architecture section still has the authoritative design documentation this
list assumes as background.

**Conventions** (carried over from the CLAUDE.md history): **[measured]**
= reproduced by running the code (or by arithmetic on shapes the code
fixes), with the number quoted. **[verify]** = suspicious, but needs a
judgement call against the reference implementation before acting.

**Numbering**: items are never renumbered or reused, even once fixed or
closed -- source files and this document cross-reference each other by
item number. New findings continue from the highest number below. When an
item is fixed, mark it `[x]` in place (don't delete it) so the reference
stays resolvable; a closed-as-not-a-bug item stays listed with a note
explaining the disposition.

**Provenance**: items 1-102 were found across five prior review passes
(2026-08-31 through 2026-09-02, recorded in CLAUDE.md's git history) and
migrated here unchanged in substance -- only the pass-by-pass narrative
framing ("re-confirmed this pass", baseline deltas per pass, etc.) was
trimmed, since this file tracks current status rather than a chronological
log. Every migrated item was re-verified against the tree at `8baadb1`
(this file's origin commit) before migration: `git diff 25457b8 HEAD
--stat` shows only CLAUDE.md itself changed since the last source commit
(`f00e2ee`), so every still-open item's cited code is exactly as
described below. Items 103+ are new findings from the review pass that
created this file (2026-09-03, against `8baadb1`).

## Current baseline (2026-09-03, against `8baadb1`)

- `uv run ruff check .` (after `uv sync --extra test --extra lint`): **31
  errors** -- 29 `E501` + 1 `F401` + 1 `F841` (see item 86 for the
  breakdown by file).
- `uv run pytest` (plain main venv, no `output/` present): **120 passed,
  20 skipped**.
- `uv run pytest` (same venv, with `output/*.seq`/`*.pge` present from a
  `main.py --ge` build): **125 passed, 15 skipped**. Not independently
  re-measured with the `preprocessing` extras added; see item 70's table
  for that combination as last recorded (149/154 passed).
- Whole-sequence feasibility (`uv run python main.py --ge`, full
  default-params build, GE_MR750, `PNSwt = [0.8, 1.0, 0.7]`, seed 0) --
  all four sequences `.ok`:

  | sequence | peak PNS | acoustics | max grad | max slew |
  |---|---|---|---|---|
  | `ArbEPI.seq` | 79.8% | 0.1484 | 50.00 mT/m | 119.0 T/m/s |
  | `EPIcal.seq` | 78.1% | 0.1484 | 50.00 mT/m | 119.0 T/m/s |
  | `deGRE.seq` | 77.4% | 0.2456 | 49.76 mT/m | 174.3 T/m/s |
  | `noise.seq` | 0.0% | 0.0000 | 0.00 mT/m | 0.0 T/m/s |

  All four numbers match the previously-recorded baseline exactly --
  confirms the source tree really hasn't changed since the last pass.

## Correctness

- [x] **8.** Closed as not-a-bug: `check_grad_acoustics`'s axis
  cross-product is a faithful port of `../ArbEPI/lib/check_grad_acoustics.m`'s
  identical loop structure, already documented with a comment in
  `ge/acoustics.py`. No code change.
- [x] **13.** Resolved by `1ebb2bb`: `sequences/noise.py` now captures
  `sys.adc_dead_time` before zeroing `sys_seq`'s copy, so `pad_duration`
  adds the real dead time instead of always adding zero.
- [x] **36.** Resolved by `1ebb2bb`: `sequences/deGRE.py`'s `tr_min` now
  takes `max()` over `gx_pre`/`gy_pre`/`gz_pre` and over
  `gx_spoil`/`gy_pre`/`gz_pre`, instead of charging the prephase/spoiler
  blocks only their x-axis gradient's duration.
- [x] **37.** Resolved by `1ebb2bb`: `sequences/deGRE.py`'s `te_min` now
  uses the same `max(calc_duration(rf), calc_duration(gz_ss)) - (rf.delay
  + pp.calc_rf_center(rf)[0])` formula `lib/calc_te_tr_delays.py` uses,
  instead of the RF block's midpoint. Item 62 (ΔTE export precision) is a
  separate, not-yet-fixed follow-on that needs cross-module coordination
  with `sequences/ArbEPI.py`'s `scan_info.mat` writer.
- [x] **38.** Resolved: `test_arbepi_default_params_peak_pns_under_normal_mode_limit`
  now builds the full default-`Nframes` (30) sequence instead of
  `Nframes=1`, so it measures the real worst frame rather than frame 0.
  Chose "accept the full-build cost" over the other two options the item
  offered (picking `argmax` of per-frame blip steps, or parametrizing over
  several frames) -- simplest and matches the "real worst frame" guarantee
  exactly, and the cost is modest (~11s measured, not the ~5x pessimistic
  estimate the item guessed). Verified: peak PNS now measures 79.84%
  (frame 10), exactly matching CLAUDE.md's/this doc's recorded worst-frame
  number, and the test still passes under the 80% limit.
- [x] **39.** Resolved by `1ebb2bb`: `resize_to_epi_grid` now raises
  (`np.allclose(fov_src[:2], fov[:2], rtol=1e-6, atol=1e-6)`) on an x/y FOV
  mismatch, matching the existing z check's strictness.
- [x] **40.** Resolved by `1ebb2bb`: `preprocess()` now opens `mf` and
  calls `resume_start_frame(...)` inside the `try/finally`, so a short
  archive's `StopIteration` during resume both closes the handle and gets
  converted to the friendly `RuntimeError`.
- [x] **41.** Resolved by `1ebb2bb`: `smaps.load_smaps` now compares the
  cached `Nvcoils` attr against the current `<seqname>_gre.h5`'s
  `ksp_gre.shape[-1]` before trusting the smaps cache, re-estimating on
  mismatch (falls back to trusting the cache only if the GRE file isn't
  available to check against).
- [x] **42.** Resolved by `1ebb2bb`: `recon_frames.py` now wraps
  `min(cfg.Nframes, nframes_avail)` in `int()`.
- [x] **43.** Resolved by `1ebb2bb`: `lib/trap4ge.py` now sets
  `gout.flat_area = gout.amplitude * gout.flat_time` after rescaling,
  alongside the existing `gout.area` update. Also resolves item 71's
  prerequisite for reverting item 17's `crt`.
- [x] **44.** Resolved: both `preprocessing/oephase.py`'s `epiphasecorrect`
  and `preprocessing/preprocess.py`'s `compute_oephase` now use the same
  standard `fftshift(ifft(ifftshift(.)))` / `fftshift(fft(ifftshift(.)))`
  centered-FFT pairing on axis 0 (ifftshift *before* the transform,
  fftshift after -- the textbook-correct one, not `epiphasecorrect`'s old
  fftshift-on-both-sides spelling or `compute_oephase`'s old mixed
  fftshift-in/ifftshift-out). Verified: for even `nx` this is numerically
  identical to the old code on both functions (fftshift == ifftshift
  there), so this repo's current `Nx=240` production behavior is
  unchanged. Added odd-`nx` coverage:
  `tests/test_preprocessing_oephase.py::test_epiphasecorrect_removes_odd_even_mismatch`
  is now parametrized over `nx in [64, 63]` (both pass) -- and its
  `_img_to_kspace`/`_kspace_to_img` test helpers were updated to the same
  standard convention, since the previous helpers only round-tripped
  correctly for even `nx` themselves. `compute_oephase`'s deliberate
  whole-array (not just axis-0) shift is unaffected by this change for any
  axis whose length isn't guaranteed even, since the other two axes are
  either averaged (`np.mean` over cal shots) or summed
  (`getoephase`'s per-coil accumulation) downstream -- both operations
  invariant to a circular reorder. `_center_out`'s odd-length trap in
  `lib/mask2epi.py` is unrelated (a different function, not touched).
  Items 64/91 (the other two spellings of this same convention question,
  in `run_rss.py`/`gre_diagnostics.py`) are tracked separately.
- [x] **45.** Closed as not live today: `check_seq_feasibility`'s
  bin-center `max_slew` sampling only under-reports ramps shorter than ~2
  gradient rasters, and this repo's POPE readout ramps are ~50 rasters
  (accurate today). Matches pypulseq's own `calc_pns.py` sampling
  convention, so not a plain bug either. Revisit if a future ramp design
  ever approaches the 1-2 raster range. No code change.
- [x] **61.** Resolved by `1ebb2bb`: `sequences/deGRE.py` now calls
  `seq.set_definition('FOV', params.fov_degre)`. Confirmed in a fresh
  build: `output/deGRE.seq`'s `[DEFINITIONS]` now reads `FOV 0.216 0.216
  0.042`, matching the sequence's actual encoding.
- [x] **62.** Resolved: `sequences/deGRE.py` now derives `delay_te[1]`
  from `delay_te[0]` plus `round(dTE_prescribed / raster) * raster`
  (nearest raster multiple of the prescribed ΔTE) instead of ceiling each
  echo's delay independently against `te_min` -- guaranteeing the
  realized ΔTE is within half a raster step (2 us) of prescribed, vs. up
  to a full raster step (4 us) of drift possible with two independent
  ceils. `generate_degre` now also patches `scan_info.mat`'s `TE_degre`
  field with the realized (not prescribed) pair after
  `sequences/ArbEPI.py` writes it (`main.py` always runs `generate_arbepi`
  first; the patch is a no-op skip, not an error, if `scan_info.mat`
  doesn't exist yet, so `generate_degre` is still callable standalone) --
  fixes the 0.040% scale error `b0map.jl`'s ΔTE-based Hz conversion was
  carrying. CLAUDE.md's stale "deGRE doesn't touch scan_info.mat" claim
  updated to match. Verified end to end (`main.py --ge`, full build):
  `scan_info.mat`'s `TE_degre` reads `[3.040, 5.276]` ms (realized) after
  `deGRE.seq` builds, not `[3.0369, 5.2738]` ms (prescribed); all four
  sequences still build, pass timing checks, and pass GE feasibility.
  Caveat found while verifying: at this repo's actual current
  `te_min`/`TE_degre` values (post items 36/37's fixes), the old
  independent-ceil method happens to land on the same nearest-raster ΔTE
  as the new method by coincidence -- so this fix doesn't change the
  *currently exported* numbers for the default config, only the general
  case (confirmed by a 200k-trial sweep: ~25% of random `te_min`/`TE_degre`
  combinations diverge, and the new method is always at least as close to
  the prescribed ΔTE, provably within half a raster step vs. the old
  method's up-to-a-full-raster-step worst case).
- [x] **63.** Partially resolved by `1ebb2bb`: `lib/make_readout_grads.py`'s
  comment now states what actually happens (blips can start up to 2
  samples before the ADC window closes) instead of the opposite. The
  functional rounding itself is unchanged (still `round`, not `floor`) --
  that's a real timing/coverage tradeoff (a slightly larger flat top),
  left as a deliberate choice for whoever wants to spend that margin, not
  applied here.
- [x] **64.** Resolved: `preprocessing/run_rss.py`'s `_ift3` docstring now
  states the convention is magnitude-/difference-safe (not
  shift-equivalent), names the odd `Nz_degre=21` case where it actually
  bites, and explains why both current consumers (`_rss_recon`'s `np.abs`,
  `b0map.jl`'s echo-difference) are immune. Left the FFT-shift spelling
  itself unchanged (switching to `ifftshift` was the other option offered
  by this item, but changing behavior wasn't necessary once the docstring
  is honest about it, and `_ift3` is a literal port of
  `toppe.utils.ift3.m`). `ge/acoustics.py:77`'s `ifftshift` remains
  provably equivalent to the MATLAB original there (`n1 + ZF_FAC*n1` is
  always even) -- not touched. See item 91 for the still-open
  `gre_diagnostics.py` copy of this same function.
- [x] **74.** Resolved by `1ebb2bb`: `recon/run_b0_recon.py`'s
  `_load_omega` now reads the authoritative `omegas` dataset directly via
  `h5py` (mirroring `reconstruct._load_omega`'s non-fallback path),
  falling back to the coil-0 `!= 0` derivation (with a printed warning)
  only for a recon file written before `omegas` existed -- removing both
  the mask-correctness bug and the redundant full-archive read in the
  common case.
- [x] **75.** Resolved: `GatheredSenseB0` now stores `pos` (int64, `(K,)`,
  2.3 MB/frame, 69 MB total) plus one `b_by_echo` table shared across every
  frame's instance, gathering `self.b_by_echo[self.pos, il:il+1]` lazily
  inside `_apply`/`_apply_adjoint` instead of precomputing a materialized
  `(K,L)` tensor per frame (was 2.21 GB total at `L=32`). Constructor
  signature changed (`b_weights` -> `pos, b_by_echo`); updated all three
  other call sites (`tests/test_recon_operators_b0.py` x2,
  `recon/sweep_time_segments.py`) to pass `pos=torch.arange(K)` alongside
  their existing `(K,L)` tensor, an identity gather that reproduces the
  old behavior exactly.
- [x] **76.** Resolved by `1ebb2bb`: `operators_b0.py`'s
  `estimate_spectral_norm` now delegates to `recon/solvers.py`'s
  `poweriter(A.apply, A.adjoint, x0, niter=niter, tol=tol)` (defaults
  changed from a fixed `niter=30` to `niter=200, tol=1e-6`, matching
  `poweriter`'s own defaults) instead of its own fixed-30-iteration loop
  with no convergence check. Also resolves half of item 89's duplication.
  `run_b0_recon.py`'s call site no longer pins `niter=30`.
- [x] **77.** Resolved: `operators_b0.py`'s `nbins` docstring now says
  `mri_exp_approx` fits from a plain *voxel-count* histogram (with the
  `_uniform_histogram` scatter-add cited), not a "magnitude-weighted" one,
  and explains background dominates by sheer count instead. Also added
  the equal-width-range note (`b0.amin()`/`amax()` over the whole volume
  is what makes an asymmetric in-object range expensive in bins).
- [x] **78.** Resolved by `1ebb2bb`: `gre_diagnostics.py` now raises a
  clear `ValueError` naming the cause (pre-dual-echo cache) when
  `TE_degre` is missing from the GRE cache's attrs, and asserts
  `n_echoes == 2` before the echo2/ratio panels that assumed it -- instead
  of a bare `KeyError`/`IndexError`.
- [x] **93.** Resolved by `1ebb2bb`: `run_recon`'s `sigma1A` now defaults
  to `None`; when `None` and `fn_b0map` is set, it's measured
  automatically via `estimate_spectral_norm` on the operator actually
  built. Calling without `fn_b0map` and without `sigma1A` now raises a
  clear `ValueError` instead of silently needing a value with no
  auto-estimate path. `run_b0_recon.py`'s own pre-measured `sigma1A` is
  still passed explicitly, so its behavior is unchanged.
- [x] **96.** Resolved by `1ebb2bb`: `plotting/plotting.py`'s `plot_psf`
  now computes `np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(omega)))`.
  Verified: for an all-ones `(240, 45)` mask the PSF magnitude now peaks
  at exactly `(Ny//2, Nz//2) = (120, 22)`, matching the analytic delta.
- [x] **97.** Resolved by `1ebb2bb`: `ge/seq2ceq.py` now seeds each
  `Segment`'s `Emax_n` from its own first row (`row0`) at construction
  time, instead of leaving it at the dataclass's cross-segment default of
  `1` for a segment whose energy never exceeds the initial `Emax_val=0.0`.
- [x] **98.** Resolved by `1ebb2bb`: the same `nBlocksInSegment` bounds
  guard the consistency-check loop already had is now applied to the
  other three block-walking loops in `ge/seq2ceq.py` (variable-delay
  detection, loop-table construction, gradient-heating) -- each `break`s
  before reading past the final, possibly-truncated segment instance.
  Verified: full test suite (`uv run pytest`) still passes, including
  `test_seq2ceq.py`'s whole-sequence smoke tests.
- [x] **99.** Resolved by `1ebb2bb`: `cg_sense.py`'s mask is now RSS
  across all coils (`np.sqrt(np.sum(np.abs(kdata_zf) ** 2, axis=coil_dim,
  keepdims=True)) > 0`), matching `recon_sigpy.py`'s `sp.rss(...) > 0`
  approach, instead of a single coil's exact-zero check.
- [x] **103.** Resolved by `1ebb2bb`: `ticaipi_sample` now raises a
  `ValueError` when `Ny % Ry != 0 or Nz % Rz != 0`, naming the offending
  `(Ny,Nz)`/`(Ry,Rz)`/`R`, instead of silently double-sampling/missing
  locations. Verified against the item's own measured repro:
  `(240,45,4)` and `(240,45,6)` now raise; `(240,45,9)` (this repo's
  actual production `R`) still passes. Existing
  `tests/test_ticaipi_sample.py` cases all use evenly-dividing configs and
  still pass unchanged. Chose "raise" over reworking `caipi_sample` to
  pass a per-frame `shift_offset` (the item's other option, which would
  make the guarantee hold generally) -- raising is the smaller, safer
  change and this repo's own shipped config never hits it.
- [x] **104.** Resolved by `1ebb2bb`: `ge/blocks.py`'s `_compare_gradients`
  non-trap branch now also checks `g1.delay == g2.delay`, matching the
  trap branch. Verified: full test suite still passes, including
  `test_seq2ceq.py`'s whole-sequence smoke tests over `ArbEPI.seq`/
  `EPIcal.seq` (the only sequences with 'grad'-type events).

## Consistency & documentation

- [ ] **17. `trap4ge` is currently a no-op. [measured]** `params.py`
  sets `crt = 4e-6` and every `ScannerSpec` sets `grad_raster_time =
  4e-6`; pypulseq already puts every trapezoid on the gradient raster, so
  the round-up can never change anything. Instrumented across all real
  call sites: 11 calls, 0 changed any rise/flat/fall time or amplitude.
  So `lib/trap4ge.py`'s "both the Siemens (10us) and GE (4us) raster"
  docstring is inaccurate (`crt` is GE-only now), as is the "hardware
  requirement, not optional cleanup" framing (true in intent, vacuous in
  effect). Keep it (it's the right net if `crt` ever returns to `20e-6`,
  see item 43's coupling) but say so; consider a test pinning the no-op
  property so the situation stays visible.
- [ ] **32. [investigated, accepted as-is] `plot_trajectory(frame_idx=...)`
  does a whole-sequence `calculate_kspace()`.** Item 29 (resolved)
  already eliminated the actual duplicate computation this worried about;
  `plot_trajectory` is called only once per run, so there's no redundant
  *repeated* call to cache away, and `calculate_kspace()` is monolithic
  (no way to compute `k_traj_adc` without `k_traj` through pypulseq's
  public API). The one remaining angle -- a sub-`Sequence` extraction for
  just the target frame -- is the approach `plotting/plotting.py`'s own
  module docstring documents as tried and abandoned (an unexplained
  ~16-22 m^-1 kx offset). Left as an accepted, documented residual cost.
- [ ] **33. [kept deliberately, not a bug] `ReadoutGrads.max_blip_area` is
  dead under the current POPE geometry.** Set in both branches, returned
  in the dataclass, never read elsewhere -- but a comment in
  `lib/make_readout_grads.py` explains it was load-bearing pre-POPE
  (symmetric-ramp case, `S = Nx*deltak + max_blip_area`) and stays in
  case of a future revert to symmetric-slew readout.
- [ ] **50. `preprocessing/preprocess.py`'s module docstring contradicts
  the rest of the repo.** Its first paragraph still says "this module has
  NOT been run end-to-end against real data" -- while the
  `preprocessing/` section of CLAUDE.md records exactly that run:
  `preprocess.py` -> `run_rss.py` on `wb_2.4mm` (GE_UHP) against a real
  MATLAB/BART RSS reference, 0.19% relative L2 error, Pearson r=0.999997,
  metadata matching exactly. Rewrite the docstring to point at that
  validation instead of disclaiming it.
- [ ] **51. `ge/coppe.py` and `ge/README.md` are invisible in CLAUDE.md.**
  Zero mentions of "coppe" even though the GE export section enumerates
  every other `ge/` module by name, and `coppe.py` is the largest file in
  the directory (582 lines, plus its own README and test file). Add a
  sentence: SSH-copies a folder of `.pge` files to the scanner
  (auto-allocating `pge2` entry numbers), UM-lab-internal, not part of the
  `main.py --ge` path.
- [ ] **52. `sampling/external_mask.py` is unreferenced and undocumented
  on both sides.** Zero occurrences in CLAUDE.md; README's architecture
  tree summarizes `sampling/` without it. `gen_sampling_masks` has no
  `'external'` branch, so using it means bypassing the documented entry
  point and calling `generate_arbepi(omegas, ...)` by hand. Decide which
  it is and document it: wire in as a fifth `params.sampling_method` (needs
  a path parameter on `Params`), or document it in both files as a
  deliberate manual escape hatch for collaborator-supplied masks.
- [ ] **53. [needs retargeting] The item-12-era warning in CLAUDE.md
  points at a file that has since been created; the pointer is now stale
  in the other direction.** Historical: item 12's caveat said
  `recon/operators_b0.py` didn't exist yet; it now does (`recon/` holds
  twelve modules), and the `grid_mode=True` alignment change it warned
  about is live again with a concrete target (`GatheredSenseB0`'s
  `c_phasors`). Worth a line in CLAUDE.md's `recon/`/B0 section
  connecting the two rather than leaving the pointer dangling.
- [ ] **54. `params.py`'s `seed` comment still contradicts the default it
  sits on.** `params.py:290-294` says "None = a fresh unseeded rng each
  run (current default behavior)" immediately above `seed = 0`. Same
  contradiction `main.py`'s copy of the comment was already fixed for
  (item 20), left in place at the definition site.
- [ ] **55. `params.py`'s `PNSwt` comment points the wrong direction.**
  `:274-275` says "see the 'PNS-driven slew limits' comment below"; that
  comment is above it, not below.
- [x] **56.** Resolved by `1ebb2bb`: `noise.py`'s `sys_seq` now builds
  from `copy.deepcopy(params.sys)` (full hardware), matching
  `ArbEPI.py`/`EPIcal.py`, with a comment explaining the choice has no
  observable effect here (no gradients) but avoids the gratuitous
  divergence. Folded in with item 13's fix in the same commit.
- [x] **65.** Resolved by `1ebb2bb`, folded in with items 13/56:
  `noise.py` now calls `seq.set_definition('FOV', params.fov)` +
  `seq.set_definition('Name', seqname)` before `seq.write(...)`. Verified
  in a fresh build: `output/noise.seq`'s `[DEFINITIONS]` block now has
  both keys.
- [ ] **66. [measured] The ArbEPI acoustics numbers in `ge/check.py` and
  CLAUDE.md are stale by 5x since the POPE switch.** Both historically
  cited `ArbEPI.seq: 0.028146 here vs MATLAB's 0.02814424`. Today's
  `ArbEPI.seq` measures **0.1484** through the same check (see the
  Current Baseline table above) -- a 5.3x change from the asymmetric
  readout ramps. The disclaimer around these numbers names only the
  `GRE.seq` -> `deGRE.seq` rename, not the POPE change. The *window* half
  of the reproduction record still holds exactly (25000 samples at the 4
  us raster, matching MATLAB's 25001). Re-record only the magnitude in
  CLAUDE.md, and extend the disclaimer to cover the POPE change too.
- [ ] **67. `preprocessing/nifti_io.py`'s module docstring describes
  `run_b0map`'s pre-resize behavior.** It says the field map is written
  "on the deGRE grid, so its voxel size comes from `fov_degre`, not
  `fov`". `run_b0map.py:119-123` actually passes the *EPI*-grid
  `b0map_hz` with `fov=seq_params.fov`, and its own adjacent comment says
  "now on the EPI grid/fov like every other NIfTI this pipeline writes".
  `nifti_io`'s docstring is the stale half, and names the exact wrong
  value someone would "restore" it to.
- [ ] **68. `tests/test_ge_check.py`'s two smoke tests check against a
  different scanner than the sequences were built for.**
  `test_check_seq_feasibility_runs` and
  `test_check_seq_feasibility_noise_has_no_gradients` both hardcode
  `SCANNERS['GE_UHP']`, but the `output/*.seq` files they read were built
  under `params.py`'s `GE_MR750` -- different `max_grad` (100 vs 50 mT/m)
  and a different PNS coefficient triple. Assertions are only `>= 0` today
  so nothing is wrong yet, but any strengthening would silently measure
  the wrong hardware. Use `load_params().spec` here too, matching the
  sibling PNS regression test in the same file. Interacts with item 46/70
  (these tests never run on a fresh checkout anyway).
- [ ] **69. `preprocessing/smaps.py`'s `cal_size` "crop" is a zero-*pad*
  on z at this repo's real dimensions.** `estimate_smaps` resizes
  k-space to `(ncoils, cal_size, cal_size, cal_size)` = 24^3, but
  `Nz_degre = 21 < 24`, so sigpy pads z rather than cropping it. Not a
  geometry bug (k-space zero-padding interpolates while preserving FOV),
  but two docstring claims are then wrong: "center-crop" is true only for
  x/y, and "EspiritCalib's own crop becomes a no-op" now means a
  24-wide calibration region on z with 3 synthetic-zero rows, a small real
  effect on the fit. Either clamp per axis (`min(cal_size, n)`) or state
  explicitly that z is padded and why that's acceptable.
- [x] **79. `b0map.jl`'s `l2b`/`niter` recording gap.** Resolved by
  `f00e2ee`: those CLI arguments were removed outright, the writer no
  longer emits an `l2b` attribute, and `config.py` documents the choice
  as deliberate (backed by an actual sweep, `precon=:diag` identified as
  the real fix). No further action.
- [x] **80. `l2b`/`niter` unreachable from `run_b0map.py`.** Resolved by
  the same `f00e2ee` commit that fixed item 79 -- nothing is unreachable
  because nothing is exposed to reach.
- [ ] **81. Three docstrings still call time-segmented B0 correction
  "not-yet-implemented" in the commit that implements it.**
  `recon/b0_correction.py`'s module docstring says the residual blur
  "needs full time-segmented correction (a separate, not-yet-implemented
  stage)"; `tests/test_recon_b0_correction.py`'s
  `test_realistic_regime_only_partly_corrects` docstring closes with the
  same phrase; `b0_correction.py`'s sign-convention paragraph says it
  "has not yet been verified against a real reconstruction" while
  `run_b0_recon.py` exists precisely to do that. `recon/operators_b0.py`
  landed in the same commit and correctly describes itself as "the actual
  fix that regime needs" -- the repo now states both things. Wording fix,
  not a deletion (the static stage is still worth keeping as the cheap
  sign/scale check).
- [ ] **82. `L=6` is still the default in four places after the sweep
  concluded 32.** `operators_b0.py:144`, `reconstruct.py:132`,
  `run_b0_recon.py:58` and `:140`'s CLI default all still say `L=6`, and
  three docstrings assert the sweep never happened
  (`operators_b0.py`/`run_b0_recon.py`'s "not swept against a real error
  bound"/"see CLAUDE.md's recon/ section for that open item") while
  `recon/sweep_time_segments.py` sits in the same directory doing exactly
  that sweep. Decide the default, set it in all four places, and replace
  the docstring paragraphs with the sweep's actual numbers (recorded only
  in a commit message today, not in the repo). Interacts with item 75 (at
  L=32 the per-frame `b_weights` redundancy is 2.2 GB).
- [ ] **83. Every "see CLAUDE.md's `recon/` section" pointer the B0 code
  adds is dangling, and two pre-existing CLAUDE.md claims about `recon/`
  are now false.** Four cross-references point at content CLAUDE.md
  doesn't contain (an L-sweep open item that doesn't exist as such; a
  `b0map.jl` tuning comparison that isn't recorded; a real-scale benchmark
  table that doesn't exist). In the other direction, CLAUDE.md's item 53
  and item 49 both describe `recon/` state that the B0 commit has since
  answered (`recon/operators_b0.py` now exists; `recon/save_result.py` now
  exists). Retarget the four pointers, correct 53/49 (see item 53 above),
  and give CLAUDE.md's `recon/` section a B0 subsection covering the sign
  convention, the ms/Hz calling convention, the `nbins` finding, and
  whatever item 82 settles on for `L`.
- [ ] **84. [narrowed] `pyproject.toml`'s dangling "see recon/README"
  comment is the one remaining piece of this item; the README
  architecture-tree gap it originally named was already fixed by
  `f00e2ee`.** `f00e2ee` added `gre_diagnostics.py` under `preprocessing/`
  and `operators_b0.py`/`b0_correction.py`/`save_result.py`/
  `run_b0_recon.py`/`sweep_time_segments.py`/`benchmark_b0_cost.py` under
  `recon/` to README's architecture tree -- confirmed current against the
  real directory listing. Only `pyproject.toml`'s `recon` extra comment
  ("see recon/README or top-level docs for the CUDA wheel selection this
  machine needs") is still stale: no `recon/README*` exists. Fix that one
  line.
- [ ] **85. `tests/test_recon_operators_b0.py`'s "realistic regime" is
  the toy grid `sweep_time_segments.py` was written to replace, and its
  tests pin the `nbins` value the code documents as broken.**
  `test_more_segments_reduces_error_in_the_realistic_regime` asserts
  `err_l16 < 0.1 * err_l8`, commented "L16 >= the 12 distinct ky times in
  this small synthetic grid, so it's essentially exact" --
  `sweep_time_segments.py`'s own module docstring calls this test's
  finding "an artifact of its toy grid... it says nothing about whether
  L=6 is adequate at the real ETL=60 scale." Separately, `_build_b0_
  operator` defaults `nbins=20` and every test uses that or 10 -- exactly
  the setting `operators_b0.py`'s docstring identifies as the actual root
  cause of a real signal-loss failure. Fix: rename the test to say "toy
  grid", fold `sweep_time_segments.py`'s real-scale fixture in as the
  actual realistic-regime test, and add one case at production `nbins=128`
  asserting no row-sum warning fires.
- [ ] **92. Four committed source files name an AI model ("Fable") as
  the authority for a design decision.** `recon/operators_b0.py:28`,
  `recon/run_b0_recon.py:18`, `recon/operators_b0.py:55`, and
  `tests/test_recon_b0_correction.py:137` all attribute a technical
  decision to "Fable's staged plan" with no way for a reader to resolve
  that reference against anything in the repository. All four sentences
  already state the real technical reason alongside the name -- delete
  the attribution, or cite the actual reference (Sutton/Noll/Fessler for
  the signal model, `sweep_time_segments.py` for the L bound). Fixing
  item 82's docstrings covers three of the four sites; the test file's is
  separate. This repo's own CLAUDE.md "Commit conventions" section
  already directs that commits carry no AI identity; source shipped by
  those commits is the same question.
- [ ] **95. `run_b0map.py`'s two `# noqa: BLE001` codes are inert, and
  one is doubly so.** `preprocessing/run_b0map.py:94`/`:108` both carry
  `BLE001` suppressions, but `BLE` isn't in `pyproject.toml`'s `select =
  ["E", "F", "I"]` (item 48's territory), so neither does anything today
  -- and `:108` catches a *narrow* exception
  (`subprocess.CalledProcessError`), so `BLE001` wouldn't fire there even
  if `BLE` were enabled. The comments are worth keeping; the codes are
  cargo-culted from the five sibling batch drivers whose `except
  Exception` form does match the rule. Fold into item 48, whose proposed
  `B` ruleset would newly flag `:108`'s suppression as unused.
- [ ] **100. Item 84's README-architecture-tree claim was already fixed
  by `f00e2ee`; only the `pyproject.toml` sub-claim (see item 84 above,
  now narrowed) is still open.** Recorded here only as a pointer so a
  future reader doesn't re-derive this from scratch -- item 84 above is
  already the corrected, narrowed version.
- [ ] **101. Two small doc gaps, worth folding into the next general docs
  pass.** (a) `preprocessing/matio.py:7-9`'s module docstring still
  quotes `schedules`' pre-echo-time shape (`h5py raw (2, 60, 20, 30)` ->
  logical `(30, 20, 60, 2)`), stale since the dual-echo upgrade made it
  `x3` -- confirmed against `sequences/ArbEPI.py:299-301`'s 3-channel
  concatenation. `read_mat_array`'s `.transpose()` is shape-agnostic, so
  no functional bug, just a stale worked example -- update the two shape
  tuples and channel count. (Not the same array as
  `lib/mask2epi.py:75`'s own, still-correct "ETL x 2" comment, which
  describes the array *before* `ArbEPI.py` appends the echo-time column.)
  (b) `README.md:22`'s core-dependency sentence omits `tqdm`, which
  `pyproject.toml:19` lists as a real core dependency and which
  `sequences/ArbEPI.py:23`/`sequences/deGRE.py:48` both actually import.
  Add it to the list.
- [ ] **106. `preprocessing/calibrate_delay.py:31`'s `_matlab_round`
  docstring cites the wrong file for the third copy of the same
  helper.** It says "also duplicated in smaps.py", but `smaps.py` has no
  such function -- the real third copy is in `preprocessing/
  grid_resize.py:56-58`, whose own docstring correctly points back at
  `oephase.py`'s original ("see preprocessing/oephase.py's own copy,
  which handles negatives too, for the general case"). Confirmed:
  `grep -rn "_matlab_round" .` finds exactly four hits --
  `oephase.py` (original), `grid_resize.py` (non-negative copy, correctly
  cross-referenced), `calibrate_delay.py` (non-negative copy, this
  finding's wrong cross-reference), and its own test file. Doc-only,
  one-line fix: `smaps.py` -> `grid_resize.py`.

## Test & tooling health

- [ ] **15. [superseded by item 86's updated count] `uv run ruff check .`
  lint debt.** Originally 20 errors, all `E501`; item 86 below has the
  current, larger count (31, no longer all `E501`). Kept only as a
  pointer -- act on item 86's breakdown instead.
- [ ] **46. The tests that gate on `output/*.seq` silently skip on a
  fresh checkout.** `tests/test_ge_check.py`/`test_seq2ceq.py`
  `pytest.skip` when `output/noise.seq`/`output/ArbEPI.seq` are missing --
  which is every clone and every CI run, since `output/` is gitignored.
  Item 70 below corrects this item's own test *count* (5 cases across 3
  functions, not 3 tests); the underlying gate is the same. Fix
  demonstrated by the PNS test in the same file: build a small sequence
  into `tmp_path` inside the test.
- [ ] **47. [superseded by the Current Baseline table above] This
  document's recorded pytest baseline should always be read from the
  Current Baseline section, not from this item's original historical
  numbers.** Kept only as a pointer for the "why does this item exist"
  question a future reader might have.
- [ ] **48. Ruff's `B` (flake8-bugbear) ruleset finds 45 issues the
  current `select = ["E", "F", "I"]` can't see; two are worth acting on
  regardless of whether `B` is ever enabled.** `B023` at
  `lib/mask2epi.py:294` -- `max_excl` closes over `top_idx`/`top_vals`,
  rebound on every pass of the enclosing loop; currently safe (defined
  and consumed within one pass) but a genuine footgun in the hottest
  function in the file. `ARG001` at `ge/writeceq.py:103` -- a
  `parent_by_id` parameter never used in the body. The rest is `B905`
  (12x `zip` without `strict=`), `B028` (3x `warnings.warn` with no
  `stacklevel`, including both of `calc_te_tr_delays`' TE/TR warnings,
  exactly where a caller-side line number would help), `ARG005`/`ARG001`
  in test doubles/callbacks, and one `B011` `assert False` in
  `tests/test_gen_sampling_masks.py:47`. Consider adding `B` to
  `[tool.ruff.lint] select` after clearing the two above; item 95 would
  add one more finding (an inert `noqa`) once `B` is on.
- [ ] **49. [superseded -- see items 89 and 93] `recon/solvers.py`'s
  `poweriter` was dead, and `run_recon` had no wired-up non-validation
  entry point.** Original finding: `poweriter` had zero callers, while
  `run_recon`'s `sigma1A` was a required kwarg with nothing to supply it
  outside `validate_against_mslr.py`. Since then, `recon/save_result.py`
  and `recon/run_b0_recon.py` (added by the B0 commit) answer the "no
  wired-up driver" half; the `poweriter`-vs-`estimate_spectral_norm`
  duplication and the missing-fallback problem are now tracked precisely
  by items 89 and 93 respectively. Kept only as a pointer.
- [ ] **70. [measured, supersedes the original counts in items 43/46/47]
  Item 46 undercounted the `output/`-gated tests (5 cases across 3
  functions, not 3 tests).** `test_check_seq_feasibility_runs` and
  `test_seq2ceq_self_consistency` are each parametrized over
  `['noise.seq', 'ArbEPI.seq']`. Current numbers are in the Current
  Baseline section above (120/20 without `output/`, 125/15 with it, plain
  venv); a fuller preprocessing-extras breakdown was last measured at
  149/154 passed and is not re-verified here.
- [ ] **86. [measured] Lint debt is 31 errors, no longer all `E501`.**
  `uv run ruff check .` reports 31 errors (see Current Baseline). All 11
  non-baseline-`E501` ones are in files the B0 commit added --
  `recon/benchmark_b0_cost.py` (4x `E501`, plus the repo's first-ever
  `F401` and `F841`), `recon/sweep_time_segments.py` (3x `E501`),
  `preprocessing/gre_diagnostics.py` (2x `E501`). The two non-`E501`
  findings are real dead code: `benchmark_b0_cost.py:29` imports
  `GatheredSense` and never uses it (only `build_encoding_operator` is
  called), and `:86`'s `y = A.apply(x0)` is assigned but never read (the
  adjoint on the next line times against the independent `y0` instead).
  `ruff --fix` clears the `F401`; the rest is manual.
- [ ] **87. [measured] The entire B0 feature is invisible to the default
  test command.** The 20 skips in the plain-venv baseline include
  `tests/test_recon_b0_correction.py` and `tests/test_recon_operators_
  b0.py`, both whole-module `pytest.importorskip("torch")` -- so all
  ~1400 lines the B0 commit added, including every sign-convention and
  adjoint check written specifically to guard them, run zero times in the
  environment CLAUDE.md's Commands section tells a contributor to use.
  Documented pattern for `recon/`, not wrong on its own, but the skip
  ledger is now the majority of this repo's test surface by count. Worth
  deciding once (CI installs the extras, or this doc says plainly which
  fraction of the suite a plain `uv run pytest` exercises) rather than
  per-module.
- [x] **88.** Resolved: `benchmark_b0_cost.py`'s docstring now says
  `build_mem` is dominated by `c_phasors` (linear in L, 664 MB at L=32),
  with only the small per-frame `pos` index arrays (item 75's fix, 69 MB
  total) being genuinely L-independent -- not "static, L-independent"
  overall.

## Conciseness & performance

- [ ] **57. [measured] Pass 3 (`_euclidean_uncross_refine`) is now ~90%
  of `mask2epi_radial` and the single largest remaining hot spot in
  sequence generation.** Full 30-frame build is ~18 s on 4 cores, of
  which `_compute_schedules` is ~7.6 s. Profiling one frame (0.94 s
  total): `_euclidean_uncross_refine` 1.22 s cumulative vs.
  `_sum_2opt_refine` 0.09 s and `_bottleneck_2opt_order` 0.03 s. Two
  pure-Python inner loops dominate: `_segments_cross`/`ccw` at 289,428
  scalar calls (0.47 s) and `span_has_pinned` at 108,530 generator
  constructions over a <=3-element frozenset (0.15 s). Two cheap,
  low-risk wins: (a) vectorize `_count_crossings`' all-pairs orientation
  test with numpy (a fixed O(m^2) cross-product over an `(m,2)` array,
  four broadcasted `ccw` evaluations, no Python loop needed); (b) replace
  `span_has_pinned`'s `any(i <= p <= j for p in pinned)` with a
  precomputed sorted array plus `bisect`, or hoist `min(pinned)`/
  `max(pinned)` out of the loop. Neither changes any result -- both are
  exact-predicate rewrites, so `tests/test_mask2epi.py`'s existing
  crossing-count assertions are the regression guard.
- [ ] **58. `preprocessing/preprocess.py`'s `_build_omegas` and
  `_build_echo_times` are the same loop written twice** -- same
  `schedules[frame, :, :, 0/1].ravel()` indices, same `(Ny, Nz, Nframes)`
  scatter, differing only in what they write. One function returning both
  arrays would halve the index arithmetic and make it structurally
  impossible for the two grids to fall out of alignment.
- [x] **59.** Resolved by `1ebb2bb`: `rampsampepi2cart` now allocates `dc`
  with `dtype=np.result_type(dco, dce)` instead of hardcoded `complex`
  (complex128).
- [x] **60.** Resolved by `1ebb2bb`: `EPIcal.py`'s shot loop is now
  `range(-params.Ndummyshots, params.Nshots)` with `is_dummy = shot < 0`,
  0-based like the rest of the repo. Confirmed `shot` is used for nothing
  else in the loop body before making the change.
- [x] **71.** Resolved by item 43's fix in `1ebb2bb`: `trap4ge.flat_area`
  is now always kept correct, so this is no longer a live prerequisite
  gating a future item-17 `crt` revert -- it's simply fixed.
- [x] **72(a).** Resolved by `1ebb2bb`: `sequences/deGRE.py` now reads
  `pe2_steps[iZ - 1]`, dropping the dead `max(0, ...)`.
- [ ] **73. `recon/lowrank.py`'s `pcount` buffer-reuse parameter is dead
  in production.** `patchSVST` accepts `pcount` and forwards it to
  `patches2img`, which zeroes and reuses it instead of allocating -- but
  the only production caller, `recon/reconstruct.py`'s `g_prox`, never
  passes one, so every call allocates a fresh `(Nx,Ny,Nz)` float32 count
  array (10.4 MB at real dims, once per scale per iteration). PyTorch's
  caching allocator makes this nearly free (tidiness, not a measured
  cost), but it's a parameter that exists solely to be passed and never
  is. Either thread a persistent buffer through from `run_recon`, or
  delete the parameter from both functions. Same disposition question as
  item 89's `poweriter`.
- [ ] **89. `GatheredSenseB0` is `GatheredSense` copied verbatim plus two
  factors, and `estimate_spectral_norm` is `poweriter` written a second
  time.** `operators_b0.py:83-107`'s `_apply`/`_apply_adjoint` reproduce
  `operators.py:49-66` line for line -- same gather, same scatter, same
  `fftshift(fftn(ifftshift(.)))` convention `operators.py`'s docstring
  justifies at length -- with `c_phasors[il]`/`b_weights[:,il]` inserted.
  `GatheredSenseB0`'s own docstring states the L=1 reduction and a test
  proves it numerically, so the collapse is already justified: make
  `GatheredSense` the L=1 case, or subclass/delegate.
  **[partially resolved by `1ebb2bb`, item 76]** `estimate_spectral_norm`
  no longer duplicates `poweriter` -- it now calls
  `recon/solvers.py`'s `poweriter` directly. The remaining, still-open
  half is the `GatheredSenseB0`/`GatheredSense` `_apply`/`_apply_adjoint`
  duplication. Minor, same file: `:127`'s function-body `import warnings`
  is the only one in the repo not at module level.
- [ ] **90. [measured] `echo_times_s` is materialized 240x redundantly on
  the GPU, in both callers.** `reconstruct.py:181` and
  `run_b0_recon.py:88` both do `echo_times_2d.to(device).unsqueeze(0)
  .expand(Nx,-1,-1,-1).contiguous()`, turning a `(Ny,Nz,Nt)` array into a
  dense `(Nx,Ny,Nz,Nt)` one: 311 MB of float32 at real dims, expanded
  from 1.30 MB of distinct values -- and `build_encoding_operator_b0`
  reads it only at each frame's sampled flat indices. The `.contiguous()`
  is what forces the materialization (`expand` alone is a free stride-0
  view). Cheaper: index the `(Ny,Nz,Nt)` array with `idx // Nz`-style
  arithmetic and skip the broadcast, or at minimum share one helper
  between the two identical call sites (same duplication class as item
  74's `_load_omega`).
- [ ] **91. `_ift3` now exists three times, and the newest copy inherits
  item 64's false justification on the one grid where it bites.**
  `preprocessing/gre_diagnostics.py:29` is a verbatim copy of
  `preprocessing/run_rss.py`'s `_ift3`, and `preprocessing/julia/
  b0map.jl` implements the same convention a third time in Julia.
  `gre_diagnostics.py` runs on the deGRE grid, where `Nz_degre=21` is
  odd -- still safe today (magnitude-only, `np.sqrt(np.sum(np.abs(...)
  **2))`), but now the third place a future complex-valued consumer could
  pick the pattern up from. Import `run_rss._ift3` rather than copying
  it, and fix item 64's docstring once, in one place.
- [ ] **94. `recon/run_b0_recon.py` re-implements `run_recon`'s smaps
  load + RSS normalization verbatim.** `:79-81` is a line-for-line copy
  of `recon/reconstruct.py:154-156`. More dangerous than item 89's
  duplication: `run_b0_recon`'s whole preamble exists to measure
  `sigma1A` for the operator `run_recon` builds moments later, so any
  future change to smaps normalization silently desynchronizes the
  estimate from the reconstruction it's estimated for -- no error, just a
  wrong step size (same bug class as item 74's omega divergence, in the
  same function, for the same reason). Factor the load+normalize into one
  helper both call.
- [ ] **102. `preprocessing/epi_gridding.py`'s `reconecho()` is dead
  code.** Defined and documented (including a docstring cross-reference
  from `rampsamp2cart`'s own docstring), but zero callers anywhere in the
  repo or its tests. Everything that actually grids EPI data goes through
  `rampsampepi2cart`/`rampsamp2cart`, which reimplement the same per-echo
  NUFFT+DCF logic inline instead of calling it. Not the same class as
  items 33/73 (those are struct fields/parameters, not a whole unused
  function). Either delete it, or wire `rampsamp2cart`'s inner loop to
  call it (needs batching to be useful as-is).
- [ ] **105. [measured] `recon_frames.py`'s `use_parfor=True` path
  re-pickles/re-transmits the full `smaps` array once per frame, not once
  total.** `preprocessing/recon_frames.py:73-75` binds `smaps` into
  `functools.partial(_recon_one_frame, recon_fn=recon_fn, smaps=smaps)`
  and passes the resulting `worker` to `executor.map(worker,
  frame_data)`. `ProcessPoolExecutor.map` pickles each dispatched task
  (callable + bound args) independently onto its internal call queue --
  binding a large array via `partial` does not send it once; it gets
  re-serialized per task. Verified with a minimal reproduction
  (a module-level class wrapping an array, instrumented `__reduce__`
  call-counting, run through `ProcessPoolExecutor.map` exactly as this
  code does): confirmed one pickle of the bound array per dispatched
  task, not one for the pool's lifetime. At this repo's own documented
  production scale (`Nx,Ny,Nz,Nvcoils=240,240,45,18`, `Nframes=30`),
  `smaps` is `240*240*45*18*8` bytes ~= 356 MiB (complex64); with
  `use_parfor=True` that's re-pickled and piped to a worker process on
  every one of the 30 frames -- roughly 10.4 GiB of redundant IPC
  serialization for data that never changes across frames, on a module
  whose surrounding design (the frame-by-frame HDF5 streaming a few
  lines above, explicitly commented against exceeding physical RAM) is
  otherwise careful about exactly this kind of cost. No test coverage of
  the `use_parfor=True` path at all (`grep -rn "use_parfor" tests/` ->
  nothing) -- it's an opt-in feature (`cfg.use_parfor`, default `False`),
  so nothing in this repo's own pipeline is affected until someone
  enables it. Fix: pass `smaps` via `ProcessPoolExecutor(initializer=...,
  initargs=(smaps,))` (set once per worker process) or a shared-memory
  array, rather than binding it into the per-task callable.
