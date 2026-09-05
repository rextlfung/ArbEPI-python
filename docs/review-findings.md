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
created this file (2026-09-03, against `8baadb1`). Items 107-118 are new
findings from a later pass (2026-09-04, against `119a6e9`) that also
re-verified every item above still marked `[x]`: all 106 were confirmed
still resolved against the current tree (no source file changed between
`8baadb1` and `119a6e9` except CLAUDE.md's own doc cleanup and one
docs-plus-11-line-comment addition, `119a6e9`), so nothing needed
reopening. Items 119-124 are new findings from a later pass (2026-09-05,
against `0d6d821`) that also re-verified every item 107-118 against the
current tree: no source file changed between `119a6e9` and `0d6d821`
except this doc itself, so all twelve were confirmed still open exactly
as described (none needed closing).

## Current baseline (2026-09-05, against `0d6d821`)

- `uv run ruff check .` (after `uv sync --extra test --extra lint`): **29
  errors**, all `E501` -- unchanged from the previous pass.
- `uv run pytest` (plain main venv): **126 passed, 15 skipped**, re-run
  after `rm -rf output` -- unchanged from the previous pass. Not
  re-measured this pass with the `preprocessing`/`recon` extras (see the
  previous baseline entries in git history for those counts, both of
  which were themselves unchanged from their own prior runs).
- Whole-sequence feasibility (`uv run python main.py --ge`, full
  default-params build, GE_MR750, `PNSwt = [0.8, 1.0, 0.7]`, seed 0) --
  all four sequences `.ok`, re-measured fresh this pass (`rm -rf output`
  first) and **unchanged from the previous baseline in every figure**:

  | sequence | peak PNS | acoustics | max grad | max slew |
  |---|---|---|---|---|
  | `ArbEPI.seq` | 79.8% | 0.1484 | 50.00 mT/m | 119.0 T/m/s |
  | `EPIcal.seq` | 78.1% | 0.1484 | 50.00 mT/m | 119.0 T/m/s |
  | `deGRE.seq` | 77.4% | 0.2556 | 49.76 mT/m | 174.3 T/m/s |
  | `noise.seq` | 0.0% | 0.0000 | 0.00 mT/m | 0.0 T/m/s |

  `deGRE.seq`'s acoustics is **0.2556** (matching this file's own
  already-corrected table, not the stale 0.2456 still quoted in
  `ge/check.py`'s module docstring -- see item 123 below, a third,
  previously-unflagged occurrence of the same stale figure item 111
  already tracked in CLAUDE.md and this file's own table).

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
  old behavior exactly. Later confirmed with the real `torch`/`mirtorch`
  extras: all 32 `tests/test_recon_*.py` cases pass.
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
- [ ] **107. `ge/seq2ceq.py`'s consistency-check and gradient-heating loops
  silently skip the sequence's final segment instance whenever it's a
  complete (non-truncated) fit.** [measured] Two of the four
  `nBlocksInSegment`-bounded block-walking loops added by item 98
  (`seq2ceq.py:154` and `:175`, both `if n + seg.nBlocksInSegment >
  ceq.nMax: break`) use a different, off-by-one formula from the other
  two (`:83` and `:131`, both the correct `if n + seg.nBlocksInSegment - 1
  > ceq.nMax: break`). For a segment instance starting at row `n` with
  `nb` blocks, the last block it touches is row `n + nb - 1`; the instance
  is complete iff `n + nb - 1 <= ceq.nMax`. The `:154`/`:175` formula
  breaks one raster too early -- it treats a perfect, complete instance
  ending exactly at `ceq.nMax` as if it were truncated, and never
  processes it. Confirmed with a synthetic 4-TR, one-segment repro (8
  blocks total, no trailing rows, gradient amplitude increasing each TR so
  the true worst instance is the last one): the buggy gradient-heating
  loop reports `Emax_n = 5` when the true worst instance starts at row 7.
  Also confirmed against the real committed `output/ArbEPI.seq` (41400
  blocks, one segment, `nb=69`, 600 perfectly-tiled instances, no
  truncation anywhere): the buggy loop visits only 599 of the 600
  instances (never reaching row 41332, the true last instance's start).
  In that particular build the true global max (row 4900) isn't the last
  instance, so today's exported `Emax_n` happens to still be correct by
  coincidence -- but `seg.Emax_n` is written directly into the `.pge`
  binary (`ge/writeceq.py:242`, a field GE's scanner-side gradient-heating
  logic reads), so this under-reports the true worst-case instance
  whenever a sequence's last TR genuinely has peak gradient/blip energy.
  The consistency-check omission (`:154`) is lower-stakes (only emits a
  `warnings.warn`) but means a real segment-definition bug occurring
  specifically in the sequence's last TR goes completely undetected --
  and `tests/test_seq2ceq.py::test_seq2ceq_self_consistency` only asserts
  *no* warnings fire, which trivially passes whether or not the last
  instance was even checked. Item 98's own description ("Added the
  missing `nBlocksInSegment` bounds guard... the same... bounds guard the
  consistency-check loop already had") is itself slightly inaccurate: only
  the gradient-heating loop got the literal (buggy) formula the
  consistency-check loop already had; the variable-delay and loop-table
  loops in the same commit got the different, correct `-1` formula --
  nobody noticed the consistency-check loop's own formula was off by one,
  or that copying it verbatim reproduced the defect in the gradient-
  heating loop. Not covered by CLAUDE.md's disclosed `Emax_n` deviation
  (that's about the stale *column* indices 11:13, not about skipping an
  entire *instance*). Fix direction: change both breaks to `if n +
  seg.nBlocksInSegment - 1 > ceq.nMax:`, matching the other two loops, and
  re-run `ge/validate_against_matlab.py` against a fresh MATLAB reference
  to confirm the `-1` formula matches `seq2ceq.m`'s own (no MATLAB
  reference available in this environment to check directly).
- [ ] **108. `preprocessing/calibrate_delay.py`'s inline oephase
  computation is a third, unfixed copy of the FFT-shift-convention bug
  item 44 was supposed to have fixed everywhere.** [measured]
  `calibrate_delay.py:93` reads
  `np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(oephase_data), n=Nx,
  axis=0))` -- `fftshift`-in / `ifftshift`-out. Item 44 explicitly fixed
  this exact spelling in both `preprocess.py`'s `compute_oephase`
  (`:134`, now `fftshift(ifft(ifftshift(.)))`, the textbook-correct
  pairing) and `oephase.py`'s `epiphasecorrect`, citing "`compute_oephase`'s
  old mixed fftshift-in/ifftshift-out" as the bug -- but
  `calibrate_delay.py` hand-duplicates the same odd/even-phase-estimation
  math inline instead of calling `compute_oephase` (it already imports
  `apply_delay`/`load_kxoe` from `preprocess.py`, so importing
  `compute_oephase` too would be a direct drop-in), and its copy still
  carries the pre-fix spelling. Item 44's own writeup states this
  discrepancy is numerically inert only for even `nx` (this repo's real
  `Nx=240`) and a real difference for odd `nx` -- so this is currently
  silent but latent, and directly contradicts item 44's "both functions
  now use the same convention" resolution, since there were really three
  copies of this computation, not two, and only two got fixed. No test
  exercises `calibrate_delay()` end-to-end
  (`tests/test_preprocessing_calibrate_delay.py` only covers
  `select_best_delay`/`_matlab_round`), so nothing caught the miss. Fix:
  replace `calibrate_delay.py:92-94` with a call to
  `preprocess.compute_oephase(ksp_cal, kxo, kxe, Nx, fov[0]*100)`, which
  also removes the duplication (see item 117 for the same "should call the
  shared helper instead of duplicating" pattern elsewhere in
  `preprocess.py`).
- [ ] **109. `recon/benchmark_b0_cost.py` crashes on its own stated usage
  -- stale `Nx`-expanded `echo_times` shape left behind by item 90.**
  [measured] `_build_inputs()` (`benchmark_b0_cost.py:84`) builds
  `echo_times_s = t_yz_s.reshape(1, Ny, Nz,
  1).expand(Nx, Ny, Nz, Nt).contiguous()` -- a dense `(Nx,Ny,Nz,Nt)`
  tensor -- and passes it into `build_encoding_operator_b0(smaps, omega,
  b0map_hz, echo_times_s, L=L, nbins=NBINS)` at `:126`. But item 90
  changed `build_encoding_operator_b0` to expect the compact `(Ny,Nz,Nt)`
  shape with no `Nx` broadcast (its body does `echo_times_flat =
  echo_times_yz.reshape(n_yz, Nt)` where `n_yz = Ny*Nz`, only valid for
  exactly `Ny*Nz*Nt` elements). `reconstruct.py`/`run_b0_recon.py` were
  both updated to the new contract via the shared `_load_echo_times`
  helper item 90 added, but this one-off script's own tensor construction
  was missed. Reproduced directly: running `python -m
  recon.benchmark_b0_cost` (the script's own documented usage) raises
  `RuntimeError: shape '[18, 2]' is invalid for input of size 144` at the
  very first swept `L` value. Fix: build `echo_times_s` at `(Ny, Nz, Nt)`
  (drop the `Nx` expand), matching `reconstruct.py`'s/`run_b0_recon.py`'s
  convention.
- [ ] **110. `preprocessing/preprocess.py`'s `n_frames_discard` is
  computed and written but has no reader anywhere in the repo.** [verify]
  `preprocess.py:274` computes `NframesDiscard =
  round(seq_params.discard_duration / seq_params.volume_tr)` and `:413`
  writes it as `mf.attrs['n_frames_discard']`; a repo-wide grep for
  `n_frames_discard` finds only this write site -- no reader in
  `recon_frames.py`, `run_rss.py`, `run_cg_sense.py`,
  `run_recon_sigpy.py`, or anywhere in `recon/`. `params.py` computes
  `Nframes = round((duration + discard_duration) / volume_tr)`, i.e.
  `Nframes` (and hence the sampling `schedules`/`omegas` and the
  `ksp_epi_zf` volume `preprocess()` writes) already includes any
  discard/steady-state frames as ordinary frames `0..N-1`, and every
  Stage-2 driver reconstructs `range(nframes)` from frame 0 with no skip
  logic. So if `discard_duration` is ever set > 0 (a real, documented
  field, just defaulting to 0 today, so this is inert in the shipped
  config), the non-steady-state frames would land in every reconstructed
  time series unfiltered -- `n_frames_discard` looks like it was meant to
  let a consumer trim them, but nothing does. Either wire a Stage-2
  consumer to skip the first `n_frames_discard` frames, or document
  explicitly that this attr is metadata-only for a human/future consumer
  to act on by hand.
- [ ] **119. `lib/mask2epi.py`'s `mask2epi_radial` crashes with `ETL=1`
  (`.max()` on a zero-size array), while `mask2epi_laminar` handles the
  same input fine.** [measured] The pass-3 uncrossing-cleanup step's
  "achieved worst-case step" computation (`mask2epi.py:1126-1128`) is:
  ```python
  shot_max = _pairwise_weighted_dist(shot_coords, deltak)[
      np.arange(ETL - 1), np.arange(1, ETL)
  ].max()
  ```
  For `ETL == 1`, both `np.arange(ETL - 1)` and `np.arange(1, ETL)` are
  empty, so the fancy-indexed selection is a zero-size array and `.max()`
  raises `ValueError: zero-size array to reduction operation maximum which
  has no identity`. Reproduced directly: `mask2epi_radial(mask, ETL=1,
  Nshots=4)` on an 8x8 mask with 4 sample points raises this exact error;
  `mask2epi_laminar` on the identical input succeeds and returns a correct
  schedule, confirming `ETL=1` is a legitimately-supported input in
  general -- the module's own docstring calls the two functions
  "interchangeable," and `max_blip_steps` in this same file explicitly
  special-cases `ETL == 1` for precisely this failure mode ("ETL == 1 has
  no consecutive samples within a shot to diff... handled explicitly here
  rather than at each of this function's three call sites"). Every other
  helper in this file that could face a single-point tour already guards
  this case (`_sum_optimized_order`/`_bottleneck_2opt_order`: `if m <= 2:
  return ...`; `_mst_bottleneck`: `if m <= 1: return 0.0`;
  `_euclidean_uncross_refine`: `if m <= 2: return order`) -- only this one
  inline computation lacks an equivalent guard. `ETL=2`/`ETL=3` both work
  correctly through `mask2epi_radial`, so the failure is specific to
  `ETL=1`. Inert in the shipped default config (`ETL=60`), but a real crash
  for any caller who scans `ETL` down toward 1 -- exactly the kind of sweep
  CLAUDE.md itself recommends for checking `calc_te_tr_delays` feasibility
  ("don't hand-derive feasibility... or scan across candidate `ETL`
  values"). `tests/test_mask2epi.py` has no `ETL=1` case for
  `mask2epi_radial` (only implicitly for `mask2epi_laminar`). Fix: guard
  the `.max()` the same way `max_blip_steps` does, e.g. `shot_max = 0.0 if
  ETL <= 1 else _pairwise_weighted_dist(...)[...].max()`, and add a
  parametrized `ETL=1` case to `tests/test_mask2epi.py` covering
  `mask2epi_radial`.
- [ ] **120. `preprocessing/epi_gridding.py`'s `rampsamp2cart` is a fourth,
  untracked copy of the FFT-shift-pairing bug items 44/64/91/108 already
  cover elsewhere -- and this copy can cause a real image-domain shift, not
  just an inert phase artifact.** [measured] `rampsamp2cart:53` computes
  ```python
  dc = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ximg, axes=0), axis=0), axes=0)
  ```
  -- `fftshift`-in / `fftshift`-out, the same non-canonical pairing item 44
  fixed everywhere it had already been found (`preprocess.py`'s
  `compute_oephase`, `oephase.py`'s `epiphasecorrect`, both now
  `ifftshift`-in / `fftshift`-out, e.g. `oephase.py:123,130`) and that item
  108 (still open) flags as un-fixed in `calibrate_delay.py:93`.
  `epi_gridding.py` itself is not cited anywhere in this file (grep only
  finds items 59 and 102, about unrelated things in the same module), so
  this is a genuinely new instance, not a re-report. `fftshift`/`ifftshift`
  only disagree for odd-length axes; `ximg`'s axis 0 has length `nx`
  (this repo's real `Nx=240`, even), so this is currently a no-op
  difference -- inert today, the same "silent but latent" framing item 108
  uses. No test would catch it either way: `tests/
  test_preprocessing_epi_gridding.py` only uses even `nx` (64, 48, 48), and
  its own oracle (`test_rampsamp2cart_recovers_object_location_and_shape`)
  independently reproduces the same non-canonical pairing to invert `dc`
  back to image space, so it self-consistently can't detect the mismatch
  even in principle. **Why this instance is worse than the already-tracked
  ones**: items 44/64/91/108's mismatched shift acts on *k-space* data
  right before the terminal inverse transform to image space, so by the
  Fourier shift theorem it only bakes in a linear *phase* ramp in the
  image -- invisible to every real consumer (magnitude, or a phase
  *difference*). Here the shift acts the other way: it circularly rotates
  `ximg` (image-space, from `nufft_adjoint`) before the *forward* FFT that
  produces `dc` (k-space) -- for odd `nx` this bakes a linear phase ramp
  into k-space along kx, which the terminal inverse FFT in Stage 2
  (`_ift3`/`_ifftc` etc.) turns into a genuine one-voxel *circular shift of
  the reconstructed image* along the readout axis, visible in magnitude,
  plus direct corruption of `compute_oephase`/`epiphasecorrect`'s
  phase-based ghost-correction fit (which consumes `rampsampepi2cart`'s
  complex output, not just its magnitude). Fix: change
  `epi_gridding.py:53` to `np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ximg,
  axes=0), axis=0), axes=0)`, matching `oephase.py`'s canonical pairing,
  and parametrize `tests/test_preprocessing_epi_gridding.py` over an odd
  `nx` the way item 44's fix parametrized
  `test_epiphasecorrect_removes_odd_even_mismatch` over `[64, 63]`.
- [ ] **121. `plotting/plotting.py`'s `plot_pns_one_tr` loses gradient
  history before the window start, contradicting its own docstring's claim
  of exact parity with `check_seq_feasibility`'s PNS number for any
  `shot_index > 0`.** [verify] `plot_pns_one_tr(seq, params, shot_index)`
  (`plotting.py:270-326`) calls `sample_gradients_tesla_per_m(seq,
  time_range=(t0, t0 + params.TR))` for `t0 = shot_index * params.TR`, then
  feeds that window straight into `ge/pns.py`'s `pns()`. `pns()` computes
  its result via `fftconvolve(s[ch], f)` on `s = np.diff(g, axis=1)/dt`,
  which implicitly treats everything before the start of the passed-in
  array as zero gradient. `check_seq_feasibility` (`ge/check.py:231`) calls
  `sample_gradients_tesla_per_m(seq)` with no `time_range`, i.e. the whole
  sequence from t=0, so its convolution correctly carries forward the tail
  of every prior TR's slew activity into the next. For `shot_index > 0`,
  `plot_pns_one_tr`'s windowed call has no memory of the previous shot's
  trailing gradients (readout ramp-down, blips, spoilers), so it reads
  artificially low for roughly `20 * chronaxie` (~6.68 ms for GE_MR750's
  `chronaxie=334e-6`) into a TR that's only ~100 ms long
  (`volume_tr=2s / Nshots=20`) -- directly contradicting the docstring's
  "this is a decomposition of the same peak number [`check_seq_feasibility`]
  reports, not an independent estimate." Currently masked: both call sites
  in this repo always pass `shot_index=0` (`plot_last_run.py`'s
  `frame_idx * params.Nshots` with default `frame_idx=0`;
  `compare_readout_pns.py`'s call with no override), so today's numbers are
  unaffected, since there's genuinely no prior history to miss at t0=0. But
  `plot_last_run`'s `frame_idx` is a documented, user-facing parameter
  meant to select any frame -- calling it with `frame_idx > 0` (a
  legitimate, supported use) would silently understate PNS near the
  window's start despite the docstring's parity claim. Fix direction:
  either sample gradients from t=0 through the window end and pass the
  full history into `pns()` (windowing only the plotted/reported region
  afterward), or make the docstring explicit that `shot_index > 0` is an
  approximation that omits inter-shot PNS memory.
- [ ] **122. `ge/seq2ceq.py`'s two loops item 107 flags also use a
  stricter outer `while` bound than the two already-correct loops, a
  distinct root cause item 107's own proposed fix doesn't address.**
  [verify, not live today] The two already-correct block-walking loops use
  `while n <= ceq.nMax:` (`seq2ceq.py:81,126`); the two loops item 107
  flags for a missing `-1` in their inner break condition use `while n <
  ceq.nMax:` instead (`:150,171`). For a segment with `nBlocksInSegment ==
  1` whose final instance starts exactly at row `n == ceq.nMax` (a
  complete, non-truncated single-block instance), `n <= ceq.nMax` enters
  the loop body correctly but `n < ceq.nMax` is `False` and the body never
  runs -- so even after applying item 107's fix verbatim (which only
  touches the inner `if n + seg.nBlocksInSegment [- 1] > ceq.nMax: break`
  condition), the consistency-check and gradient-heating loops would still
  skip that final instance whenever its segment happens to have exactly
  one block. Not reachable in this repo's actual sequences today (every
  real segment spans many blocks -- e.g. `ArbEPI.seq`'s documented `nb=69`
  -- since TRID is set once per shot, not once per block), so this is in
  the same "not live today" category as item 45. Flagging it as a separate
  item because it's a distinct root cause from item 107's break-formula
  bug, and item 107's own stated fix direction would leave it unfixed --
  worth changing both outer bounds to `<=` in the same pass as item 107's
  fix.

## Consistency & documentation

- [x] **17.** Resolved: `lib/trap4ge.py`'s docstring no longer claims a
  "both Siemens (10us) and GE (4us) raster" dual-raster rationale --
  states plainly that `crt` is GE-only now, that the round-up is a
  measured no-op at `crt == grad_raster_time` (today's setting), and
  cross-references CLAUDE.md's own `trap4ge` paragraph for the fuller
  story. Kept the function itself unchanged (right net if `crt` ever
  reverts to `20e-6`). Did not add a dedicated no-op-pinning test -- the
  claim is now stated as measured fact in the docstring rather than
  needing a regression guard of its own.
- [x] **32.** Closed as investigated, accepted as-is -- no code change.
  Item 29 already eliminated the actual duplicate computation this
  worried about; the one remaining angle (sub-`Sequence` extraction) is
  the approach `plotting/plotting.py`'s own module docstring already
  documents as tried and abandoned. Nothing further to do.
- [x] **33.** Closed as kept deliberately, not a bug -- no code change.
  `lib/make_readout_grads.py` already carries the comment explaining
  `ReadoutGrads.max_blip_area`'s pre-POPE load-bearing use and why it
  stays for a possible future revert.
- [x] **50.** Resolved: `preprocessing/preprocess.py`'s module docstring
  now leads with the real end-to-end validation (`wb_2.4mm`, GE_UHP,
  0.19% rel. L2 error, r=0.999997 against a MATLAB/BART reference)
  instead of disclaiming it as never run. Also fixed a stale test-name
  typo found while rewriting it: the docstring cited
  `test_preprocess_scatter_frame_places_data_at_correct_indices`, which
  doesn't exist -- the real name is
  `test_scatter_frame_places_data_at_correct_indices`.
- [x] **51.** Resolved: added a bullet to CLAUDE.md's GE `.pge` export
  section describing `ge/coppe.py` (SSH-copies `.pge` files to the
  scanner, auto-allocates `pge2`/v7 entry numbers, UM-lab-internal, not
  part of `main.py --ge`) and pointing at `ge/README.md` for usage/SSH
  setup.
- [x] **52.** Resolved via the documentation option (not wired in as a
  fifth `sampling_method` -- that would need a path parameter on `Params`
  and is a larger change than this item's scope): added a bullet to
  CLAUDE.md's architecture notes and a line in README's directory tree
  describing `sampling/external_mask.py` as a deliberate manual escape
  hatch for a collaborator-supplied mask, used by calling
  `generate_arbepi(omegas, ...)` directly rather than through
  `gen_sampling_masks`.
- [x] **53.** Resolved: added a paragraph to CLAUDE.md's B0
  off-resonance correction subsection connecting `grid_resize.py`'s
  `grid_mode=True` alignment fix to its concrete target --
  `GatheredSenseB0.c_phasors`/`demodulate_smaps`'s phasor are both
  per-voxel functions of `b0map_hz` on the (resized) EPI grid, so a future
  alignment regression there would silently mis-register the field map
  against the encoding operator, not just against the diagnostics
  `grid_resize.py`'s own docstring measures.
- [x] **54.** Resolved: `params.py`'s `seed` comment now correctly states
  that `seed = 0` (int, reproducible) is the actual default, and `None`
  (unseeded, fresh mask each run) is the alternative -- matching
  `main.py`'s already-fixed copy (item 20).
- [x] **55.** Resolved: `params.py`'s `PNSwt` comment now says "see the
  'PNS-driven slew limits' comment above", matching where it actually is.
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
- [x] **66.** Resolved: both `ge/check.py`'s module docstring and
  CLAUDE.md's matching paragraph now extend the disclaimer around
  `ArbEPI.seq: 0.028146 here vs MATLAB's 0.02814424` to cover the POPE
  readout change (5.3x, to 0.1484), not just the `GRE.seq` -> `deGRE.seq`
  rename -- both now point at `docs/review-findings.md`'s "Current
  baseline" table for the current number, and state the window-duration
  half of the reproduction still holds exactly.
- [x] **67.** Resolved: `preprocessing/nifti_io.py`'s module docstring now
  says the field map is written on the EPI grid (`fov`, not `fov_degre`),
  matching what `run_b0map.py` actually does and its own adjacent comment.
- [x] **68.** Resolved alongside item 46: both smoke tests now call
  `check_seq_feasibility(seq, load_params().spec)` instead of hardcoding
  `SCANNERS['GE_UHP']`, matching the scanner the fixture-built sequences
  actually use (and the sibling PNS regression test's own pattern).
- [x] **69.** Resolved via the docstring option (not a functional change):
  `estimate_smaps`'s `cal_size` docstring now says the resize is a crop
  only on axes where the source is larger, explicitly names that z is
  zero-padded at this repo's real dims (`Nz_degre=21 < 24`), and
  qualifies the "EspiritCalib's own crop becomes a no-op" claim as
  holding only for x/y. No functional change (still not a geometry bug --
  k-space zero-padding preserves FOV).
- [x] **79. `b0map.jl`'s `l2b`/`niter` recording gap.** Resolved by
  `f00e2ee`: those CLI arguments were removed outright, the writer no
  longer emits an `l2b` attribute, and `config.py` documents the choice
  as deliberate (backed by an actual sweep, `precon=:diag` identified as
  the real fix). No further action.
- [x] **80. `l2b`/`niter` unreachable from `run_b0map.py`.** Resolved by
  the same `f00e2ee` commit that fixed item 79 -- nothing is unreachable
  because nothing is exposed to reach.
- [x] **81.** Resolved: all three sites fixed. `recon/b0_correction.py`'s
  module docstring now names the time-segmented stage as implemented
  (`recon/operators_b0.py`'s `GatheredSenseB0`) rather than
  "not-yet-implemented"; its sign-convention paragraph now says the
  convention *was* verified against a real reconstruction (via
  `run_b0_recon.py`'s real runs, see CLAUDE.md's recon/ B0 subsection --
  item 83) instead of claiming it wasn't;
  `tests/test_recon_b0_correction.py`'s matching phrase was fixed in the
  same edit as item 92's Fable removal (same paragraph). Kept the static
  stage itself unchanged -- still worth keeping as the cheap sign/scale
  check.
- [x] **82.** Resolved: `L=32` is now the default in all four places
  (`operators_b0.py`'s `build_encoding_operator_b0`, `reconstruct.py`'s
  `run_recon`'s `L_b0`, `run_b0_recon.py`'s `main`'s `L_b0`, and its `--L`
  CLI default). Replaced the three docstrings that asserted the sweep
  never happened with the sweep's actual numbers (BT~=27, sharp
  phase-transition at L=27-32, L=6 only ~35% error reduction vs. L=32's
  <1% forward-model error) and a pointer to `recon/sweep_time_segments.py`
  instead of a dangling CLAUDE.md reference. Verified with the real
  `torch`/`mirtorch` extras: all 32 `tests/test_recon_*.py` cases still
  pass with the new default (including the operator-construction and
  adjoint-consistency checks). Interacts with item 75 (already fixed --
  the per-frame `b_weights` redundancy this would have cost at L=32 no
  longer applies, since that item removed the materialization entirely).
- [x] **83.** Resolved: CLAUDE.md's pre-existing "B0 off-resonance
  correction" subsection (it already covered the two-stage design and the
  `L` sweep in detail) now also covers the `nbins` finding (the real
  root-cause of the signal-loss/incoherent-noise failure, previously
  undocumented in CLAUDE.md despite two source-code pointers to it) and a
  sign-convention/`mri_exp_approx` Hz-vs-milliseconds-calling-convention
  paragraph. Fixed the stale "overriding `operators_b0.py`'s own `L=6`
  default" phrasing now that item 82 made `L=32` the direct default in
  all four places, not an override. All source-code "see CLAUDE.md's
  recon/ section" pointers (`operators_b0.py` x4, `run_b0_recon.py`,
  `b0_correction.py`, `benchmark_b0_cost.py`) now resolve to real content
  in that subsection -- confirmed by re-reading each one against the
  updated section. Also fixed a stale `echo_times_s` parameter name in
  the subsection's own prose (item 90 renamed it to `echo_times_yz`).
- [x] **84.** Resolved: `pyproject.toml`'s `recon` extra comment now
  points only at CLAUDE.md's `recon/` section (and says explicitly that
  no `recon/README` exists), dropping the dangling "see recon/README"
  half.
- [x] **85.** Resolved, all three parts. Renamed
  `test_more_segments_reduces_error_in_the_realistic_regime` to
  `test_more_segments_reduces_error_in_a_toy_grid`, with a docstring that
  explicitly says it's not the realistic regime and points at the new
  real-scale test. Added `test_more_segments_reduces_error_at_real_scale`,
  which imports `recon/sweep_time_segments.py`'s own
  `_setup_real_scale`/`_build_operator` helpers directly (real ETL=60,
  real field-map range -300 to +70 Hz -- not a fourth copy of that ground
  truth) and asserts `L=32` keeps forward-model error under 1% while
  `L=6` is at least 5x worse, turning the sweep's one-off finding into a
  regression guard. Added `test_production_nbins_avoids_row_sum_warning`,
  asserting `nbins=128` (production default) doesn't trip
  `_check_b_weight_row_sums`' ill-conditioning warning at real scale
  (using `recwarn`, not just eyeballing stdout). Verified with the real
  `torch`/`mirtorch` extras: all 34 `tests/test_recon_*.py` cases pass
  (was 32 -- the two new tests both pass on first try, no flakiness
  observed).
- [x] **92.** Resolved: all four "Fable" citations removed. Three were
  covered by item 82's docstring rewrites (`operators_b0.py`'s module
  docstring and its `GatheredSenseB0` docstring -- the latter rewritten
  already by item 89's subclass/delegate change -- and `run_b0_recon.py`'s
  module docstring), each replaced with the real reasoning
  (`recon/sweep_time_segments.py`'s sweep for the L bound). The fourth,
  `tests/test_recon_b0_correction.py:137`, was separate -- fixed alongside
  item 81 in the same docstring edit (both were the same paragraph).
- [x] **95.** Resolved: removed both inert `# noqa: BLE001` codes from
  `preprocessing/run_b0map.py`, keeping the explanatory comments they were
  attached to (`:94`'s "optional input, degrade gracefully" and `:108`'s
  "mirrors the sibling batch drivers' try/catch"). `BLE` was never added
  to `pyproject.toml`'s `select` (item 48's ruleset decision stayed
  narrow, only the two concretely-actionable `B023`/`ARG001` findings),
  so there was nothing left for these codes to suppress.
- [x] **100.** Closed as superseded -- item 84 above is the corrected,
  now-resolved version; nothing further to point at.
- [x] **101.** Resolved, both parts. (a) `preprocessing/matio.py`'s module
  docstring now quotes the current shapes (`h5py raw (3, 60, 20, 30)` ->
  logical `(30, 20, 60, 3)`) and explains the third channel is
  `sequences/ArbEPI.py`'s appended echo-time column -- `lib/mask2epi.py:75`'s
  own "ETL x 2" comment is untouched, since it correctly describes the
  array before that channel is appended. (b) README's core-dependency
  sentence now lists `tqdm` alongside pypulseq/numpy/scipy/matplotlib/
  hdf5storage/numba.
- [x] **106.** Resolved: `preprocessing/calibrate_delay.py`'s
  `_matlab_round` docstring now says "also duplicated in grid_resize.py",
  the real third copy, instead of `smaps.py` (which has no such
  function).
- [ ] **111. `deGRE.seq`'s acoustics number is stale in both CLAUDE.md and
  this file's own "Current baseline" table -- real is 0.2556, not
  0.2456.** [measured] This pass's fresh `main.py --ge` build (reproduced
  twice, deterministic under the fixed `seed=0`) measures `deGRE.seq`
  acoustics as **0.2556**, not the 0.2456 the previous "Current baseline"
  table (and CLAUDE.md's matching "today's `deGRE.seq` measures acoustics
  0.2456" claim in its GE-export section) both cited. This isn't a
  regression: item 57's own re-verification text elsewhere in this file
  already recorded "0.1484/0.1484/0.2556/0.0000" for exactly this
  four-sequence build after landing that item's vectorization fix -- so
  0.2556 has been the real number since at least item 57's commit, and
  0.2456 was a stale/typo'd figure that the "Current baseline" table
  (created afterward) and CLAUDE.md both independently carried forward
  without cross-checking against item 57's own text. This file's baseline
  table above is now corrected to 0.2556; CLAUDE.md's copy is out of this
  run's scope to edit (only `docs/review-findings.md` may be modified this
  pass) but should be updated to match the next time CLAUDE.md itself is
  touched.
- [ ] **112. `recon/sweep_time_segments.py` still describes `L=6` as "the
  current production default" and cites a test name item 85 renamed.**
  [measured] The module docstring (`:9-11`) says "...L=6 (the current
  production default, params.py-adjacent choice in
  operators_b0.py/run_b0_recon.py)..." and the sweep table's own printed
  marker (`:148`, `marker = "  <- current default" if L == 6 else ""`)
  labels the `L==6` row as current -- but item 82 changed the default to
  **`L=32`** in all four places (`operators_b0.py`'s
  `build_encoding_operator_b0`, `reconstruct.py`'s `run_recon`,
  `run_b0_recon.py`'s `main`/`--L`), confirmed still the case by reading
  the current code, and `operators_b0.py`'s own module docstring was
  updated accordingly by that item. Separately, the same file's docstring
  (`:6-7`) still points at
  `tests/test_recon_operators_b0.py::test_more_segments_reduces_error_in_the_realistic_regime`,
  which item 85 renamed to
  `test_more_segments_reduces_error_in_a_toy_grid` (confirmed: no test of
  the old name exists anywhere in the repo). Both drifts look like this
  script was simply missed when items 82 and 85 updated every other
  doc/code reference -- worth one pass over `sweep_time_segments.py` to
  bring its docstring and printed marker in line with both.
- [ ] **113. Dangling `docs/review-findings.md` item-number
  cross-references in source comments: item 28 (and, previously flagged
  but still unresolved, item 12) don't exist in this file.**
  [measured] `lib/make_prephasers.py:10`'s module docstring says "a real,
  if not previously live, consistency bug in this port -- see
  `docs/review-findings.md` item 28" -- but this file's item numbers run
  8, 13, 15, 17, 32, 33, 36-106 (now extending to 117); 28 is simply
  absent. The fix itself is real and correct (confirmed `make_prephasers`
  does share one duration across all three axes, matching the comment's
  description), so this is a broken citation, not a live bug. Root cause,
  traced through git history: when the review backlog was split out of
  CLAUDE.md (`c49712d`), only still-open items were migrated into this
  file -- items already closed beforehand in CLAUDE.md's own history
  (28 among them, and also 12) were dropped rather than carried forward as
  resolved, contradicting both CLAUDE.md's "Numbering... never reused"
  claim and this file's own header ("a closed-as-not-a-bug item stays
  listed... so the reference stays resolvable"). The same gap affects
  `preprocessing/grid_resize.py`'s and
  `tests/test_preprocessing_grid_resize.py`'s "item 12" citations, and
  this file's own internal "item 20" cross-reference (in item 54's entry
  above) -- none of items 12/20/28 can be looked up here. Fix direction:
  either restore stub entries for the dropped-but-cited numbers (`[x]
  12.`, `[x] 20.`, `[x] 28.`, each with a one-line "closed pre-migration,
  see git history at <commit>" note) so every source-code citation
  resolves, or replace the four source-code citations with a description
  of the fix in prose instead of a dangling item number.
- [ ] **114. `README.md`'s `--plot` file list is missing `PNS_one_tr.png`.**
  [measured] `README.md:48` (Getting Started step 3) says `--plot` writes
  "diagnostic plots (`mask.png`, `psf.png`, `trajectory.png`,
  `one_tr.png`)" -- four files. But `plotting/plot_last_run.py:27-58`
  (which `main.py --plot` calls) writes a fifth: `PNS_one_tr.png`, from
  `plot_pns_one_tr` (added alongside the PNS-driven slew-limit work
  documented in CLAUDE.md's "PNS finding history"). Confirmed by grepping
  the whole README: `PNS_one_tr.png` is never mentioned anywhere in it,
  even though it's part of every `--plot` run's actual output and part of
  `plot_last_run`'s own printed confirmation message.
- [ ] **123. `ge/check.py`'s module docstring quotes the same stale
  `deGRE.seq` acoustics figure (0.2456) that item 111 already found and
  corrected in CLAUDE.md and this file's own baseline table -- a third,
  previously-unflagged occurrence.** [measured] `ge/check.py`'s module
  docstring (around line 39) says "today's `deGRE.seq` measures acoustics
  0.2456 (under the 0.3 threshold...)". Item 111 already established the
  real, reproducible number is **0.2556** (this file's own "Current
  baseline" table now reflects that), and that 0.2456 was a stale/typo'd
  figure independently carried by CLAUDE.md's GE-export section -- but
  item 111's text never mentions `ge/check.py`, and this docstring is a
  third, distinct occurrence of the same wrong number, this time inside
  the source tree rather than in docs. Fix: update `ge/check.py`'s
  docstring to 0.2556, or better, point at this file's "Current baseline"
  table the way the surrounding paragraph already does for the `ArbEPI`
  number, so it can't drift out of sync again.
- [ ] **124. `recon/reconstruct.py`'s `run_recon` docstring still claims
  `echo_times` gets "broadcast across Nx here," directly contradicting the
  actual post-item-90 implementation in the same file.** [measured]
  `run_recon`'s docstring (`reconstruct.py:169-171`) says the `echo_times`
  dataset is "`(Ny,Nz,Nt)`, broadcast across Nx here since kx doesn't
  affect echo time." But `run_recon` actually gets `echo_times` via
  `_load_echo_times(fn_ksp, device)` (same file, ~line 106), whose own
  docstring says the opposite: it exists precisely so neither call site
  duplicates "the broadcast-to-`(Nx,Ny,Nz,Nt)` pattern
  `build_encoding_operator_b0` no longer needs" -- and its body just
  returns the native `(Ny,Nz,Nt)` array with no broadcast. That array is
  passed straight into `build_encoding_operator_b0`, whose own docstring
  (`operators_b0.py:159-166`) is explicit that `echo_times_yz` is read
  directly at `(Ny,Nz,Nt)` "rather than broadcast to a dense
  `(Nx,Ny,Nz,Nt)` tensor first ... see docs/review-findings.md item 90" --
  item 90 is exactly the fix that *removed* the Nx broadcast this stale
  sentence in `run_recon`'s own docstring still describes as current
  behavior. So there are three descriptions of the same data in one
  codebase, two consistent (`_load_echo_times`, `build_encoding_operator_b0`)
  and one stale (`run_recon`, in the very same file as the first) -- a
  maintainer reading only `run_recon`'s docstring would believe a dense
  `(Nx,Ny,Nz,Nt)` echo-time tensor is materialized inside it, which is
  exactly the memory blowup item 90 fixed and no longer happens. Items 90's
  and 83's (CLAUDE.md) writeups both mention updating docstrings that
  referenced the old broadcast, but neither touched this specific sentence.
  Fix: reword `reconstruct.py:169-171` to match `_load_echo_times`'s/
  `build_encoding_operator_b0`'s accurate phrasing.

## Test & tooling health

- [x] **15.** Closed as superseded -- see item 86's breakdown instead.
- [x] **46.** Resolved: added `tests/conftest.py`'s session-scoped
  `built_seq_dir` fixture, which builds `ArbEPI.seq`/`noise.seq`
  (`Nframes=1` for speed) into a tmp dir once and is shared across all 5
  previously-gated cases in `test_ge_check.py`/`test_seq2ceq.py`, instead
  of each test independently reading (and `pytest.skip`ping on) `output/`.
  Folded in item 68's fix in the same edit (both files now pass
  `load_params().spec` instead of hardcoding `SCANNERS['GE_UHP']`).
  Verified: `rm -rf output && uv run pytest` -> same 126 passed/15 skipped
  as with `output/` present -- the 5 cases now always run instead of
  being environment-dependent, at a total cost of ~11s for all of them
  (the fixture's build is amortized across every dependent test).
- [x] **47.** Closed as superseded -- read the pytest baseline from the
  Current Baseline section, not this item.
- [x] **48.** Resolved the two actionable findings: `lib/mask2epi.py`'s
  `max_excl` now binds `top_idx`/`top_vals` as default arguments instead
  of relying on the enclosing loop's closure (B023 -- was safe today,
  fixed the footgun anyway); `ge/writeceq.py`'s `_max_realized_slew` no
  longer takes the unused `parent_by_id` parameter (ARG001), and its one
  call site updated to match. Left the rest (45 total `B` findings minus
  these two) as-is -- `B905`/`B028`/test-double `ARG`s -- since adding `B`
  to `[tool.ruff.lint] select` wholesale is a separate decision this item
  didn't ask for. Verified: full test suite unaffected, and a live
  `main.py --ge` run (which exercises `_max_realized_slew` via
  `writeceq`) still writes all four `.pge` files correctly.
- [x] **49.** Closed as superseded -- both halves resolved: item 93 (`run_
  recon`'s `sigma1A` fallback) is fixed, item 89's `poweriter` duplication
  is fixed (only the `GatheredSenseB0`/`GatheredSense` half remains open
  under item 89 itself).
- [x] **70.** Closed as superseded/informational -- the 5-cases-across-3-
  functions count is folded into item 46's fix below; read baselines from
  the Current Baseline section.
- [x] **86.** Resolved the two real (non-`E501`) findings:
  `recon/benchmark_b0_cost.py` no longer imports unused `GatheredSense`,
  and the discarded `A.apply(x0)` timing result is now named `_` (matching
  the adjoint call's own `_ = A.adjoint(y0)` two lines down) instead of
  a silently-unused `y`. `E501` debt (29 errors, unchanged in count and
  files) is untouched -- pure line-length style across many files, out of
  this item's scope. Verified: `uv run ruff check .` -> 29 errors, all
  `E501`, matching the pre-B0-commit baseline exactly (`F401`/`F841` both
  gone).
- [x] **87.** Closed: this item asked for a decision between "CI installs
  the extras" (out of scope for a local code-review pass -- no CI config
  exists in this repo to change) and "this doc says plainly which
  fraction of the suite a plain `uv run pytest` exercises" -- the latter
  is already done, in this file's own "Current baseline" section above
  ("Skip composition at full richness: 6 torch-gated ... = most of the
  skip count is opt-in extras working as intended, not broken tests").
  No further action.
- [x] **88.** Resolved: `benchmark_b0_cost.py`'s docstring now says
  `build_mem` is dominated by `c_phasors` (linear in L, 664 MB at L=32),
  with only the small per-frame `pos` index arrays (item 75's fix, 69 MB
  total) being genuinely L-independent -- not "static, L-independent"
  overall.
- [ ] **115. `plotting/` has zero test coverage -- including no regression
  guard for item 96's real, previously-shipped PSF bug.** [measured] A
  repo-wide grep confirms no file under `tests/` references `plot_psf`,
  `plot_trajectory`, `plot_sampling_mask`, `plot_one_tr`,
  `plot_pns_one_tr`, or imports `plotting.plotting` at all. This matters
  concretely because item 96 documents a real, previously-shipped
  correctness bug in `plot_psf` (wrong FFT-shift convention, fixed by
  switching to `fftshift(ifft2(ifftshift(omega)))`,
  `plotting/plotting.py:197`) that was verified only by a one-off manual
  measurement ("PSF magnitude now peaks at exactly `(Ny//2, Nz//2)`"), not
  captured as a regression test -- nothing in the suite would catch that
  bug coming back (e.g. a future edit that "simplifies" the shift calls
  back to a single `fftshift`). A cheap, high-value addition: a
  `tests/test_plotting.py` asserting `plot_psf`'s PSF peaks at the DC
  location for a synthetic all-ones mask, plus basic smoke tests (a
  figure is produced, right title/`frame_idx` handling) for the other
  plotting functions.
- [ ] **116. `tests/test_ticaipi_sample.py` has no regression test for the
  `ValueError` guard item 103 added.** [measured]
  `sampling/ticaipi_sample.py:39-46` raises `ValueError` when `Ny % Ry !=
  0 or Nz % Rz != 0` -- a real, previously-fixed correctness bug (silent
  double-sampling/missing k-space locations, per item 103's own measured
  ~44%-of-swept-grid failure rate). Confirmed the raise still fires
  correctly today (e.g. `ticaipi_sample([240,45], 4, 0)` raises with a
  clear message). But `tests/test_ticaipi_sample.py` contains only two
  tests (`test_ticaipi_full_coverage_over_R_frames`,
  `test_ticaipi_cycles_with_period_R`), both using evenly-dividing `(N,
  R)` configs -- neither exercises the raise path. A one-line
  `pytest.raises(ValueError)` test (using item 103's own cited repro,
  `ticaipi_sample([240, 45], 4, 0)`) would close this gap and guard
  against the check being silently weakened or removed later.

## Conciseness & performance

- [x] **57.** Resolved, with a correction to this item's own attribution
  along the way. Both suggested wins landed: (b) `span_has_pinned` now
  does a single `bisect.bisect_left` lookup against a `pinned` array
  sorted once outside the hot loop, instead of building/iterating a fresh
  `any(...)` generator on every call. (a) The orientation test itself is
  vectorized -- but profiling first (`cProfile` on
  `_compute_one_frame_schedule` for the real worst frame, params.py
  defaults) showed **`_count_crossings` was not the dominant caller this
  item's own text named** -- it's called only ~26 times per frame
  (negligible). The actual source of the 289k-call figure is
  `_euclidean_uncross_refine`'s Stage-2 crossing-*detection* loop (a
  separate `for i: for j: if _segments_cross(...): break` scan that reran
  on every one of up to 40 passes) -- confirmed by re-profiling after
  vectorizing only `_count_crossings`, which left the scalar `ccw` call
  count essentially unchanged (133,400 -> still triggered by the
  detection loop). Fix: factored the vectorized all-pairs orientation
  test into a shared `_crossing_matrix(pts)` helper, used by both
  `_count_crossings` (`np.count_nonzero`) and a new
  `_find_first_crossing(pts)` (`np.argmax` on the flattened boolean
  matrix returns the same first-(i,j)-in-row-major-order result the
  scalar early-break scan found) -- the latter now replaces the Stage-2
  detection loop directly. `_segments_cross`/scalar `ccw` are unchanged
  and still used by `tests/test_mask2epi.py` as an independent oracle
  (kept, not deleted). Verified both vectorized functions equivalent to
  the original scalar loop on 5000+ random trials (small-integer grids
  with shared endpoints/collinear points, plus continuous coordinates up
  to m=80) before touching production code. All 37
  `tests/test_mask2epi.py` cases pass, and a full `main.py --ge` build
  reproduces the exact recorded PNS/acoustics baseline (79.8/78.1/77.4/
  0.0% peak PNS, 0.1484/0.1484/0.2556/0.0000 acoustics) -- the schedule
  computation is unchanged in output, only faster. Measured on the real
  worst frame (frame 10, default params): single-frame cost 0.474s ->
  0.332s (-30%); full 30-frame `_compute_schedules` (parallel, 4 workers):
  0.976s -> 0.565s (-42%) -- both real before/after timings on this
  environment's current hardware, not the item's original (now stale)
  "~18s/7.6s" figures, which predate several other optimizations already
  landed by earlier passes.
- [x] **58.** Resolved, via a smaller change than "merge into one
  function": factored the duplicated `schedules[frame, :, :,
  0/1].ravel()` indexing out into a shared `_iy_iz(schedules, frame)`
  helper both `_build_omegas` and `_build_echo_times` now call, rather
  than fully merging the two into a single function. Kept the two public
  functions and their independent call sites/tests unchanged (both are
  separately unit-tested with different fixtures, and the call site
  writes `omegas`/`echo_times` as two separate HDF5 datasets) -- the
  actual duplication this item flagged (the index computation, not the
  scatter-target dtype/shape) is now single-sourced, which is what makes
  the two grids structurally unable to drift apart. Verified against the
  real `preprocessing` extras venv (`uv sync --extra test --extra
  preprocessing`): both `test_build_omegas_marks_scheduled_locations` and
  `test_build_echo_times_places_values_at_scheduled_locations` pass.
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
- [x] **73.** Resolved: deleted the `pcount` parameter from both
  `patches2img` and `patchSVST` (confirmed no caller -- production or
  test -- ever passed one; `patches2img` now always allocates its own
  count buffer, same as the old default-`None` path). Chose deletion over
  threading a persistent buffer through `run_recon`, since nothing
  measured the allocation as a real cost (PyTorch's caching allocator
  already makes it near-free) -- unlike item 89's `poweriter`, where the
  duplication was a second *implementation*, not just an unused
  parameter.
- [x] **89.** Fully resolved (the `estimate_spectral_norm`/`poweriter`
  half was already fixed by `1ebb2bb`, item 76). `GatheredSenseB0` now
  `class GatheredSenseB0(GatheredSense)`, delegating to
  `GatheredSense._apply`/`_apply_adjoint` (via `super()._apply(x *
  self.c_phasors[il])` / `super()._apply_adjoint(y * b_il)`) for the
  per-segment FFT/gather and scatter/IFFT/coil-combine, instead of a
  second copy of that code with `c_phasors[il]`/`b_by_echo` inserted --
  `__init__` also delegates (`super().__init__(smaps, samp)`) rather than
  duplicating the `idx`/`N`/`Nc`/`dims`/`LinearMap.__init__` setup. Also
  moved the function-body `import warnings` to module level (item 89's
  minor note). Verified with the real `torch`/`mirtorch` extras (`uv sync
  --extra test --extra recon`): all 32 `tests/test_recon_*.py` cases pass,
  including `test_build_encoding_operator_b0_matches_manual_per_frame_
  construction` (the exact check that the delegated math matches the
  original inline math) and `test_adjoint_is_self_consistent`.
- [x] **90.** Resolved: `build_encoding_operator_b0` now takes
  `echo_times_yz` at its native `(Ny,Nz,Nt)` shape and looks up each
  sampled location via `echo_times_flat[idx % (Ny*Nz), it]` (torch's
  C-order flatten of the `(Nx,Ny,Nz)` mask means `idx % (Ny*Nz)` is
  exactly the `(Ny,Nz)` index, since echo time is constant across `ix`) --
  no more `.expand(Nx,-1,-1,-1).contiguous()` materialization. Both call
  sites (`reconstruct.py`, `run_b0_recon.py`) now share one
  `_load_echo_times(fn_ksp, device)` helper (added next to `_load_omega`,
  same duplication class as item 74) instead of duplicating the load+
  broadcast. Verified the indexing identity numerically (`idx %
  (Ny*Nz)`-based lookup vs. the old dense-broadcast-then-index result,
  exact match on synthetic data) and updated
  `tests/test_recon_operators_b0.py`'s fixture/assertions to match the
  new signature. Later confirmed with the real `torch`/`mirtorch` extras
  (`uv sync --extra test --extra recon`, installed successfully in this
  environment after all): `test_build_encoding_operator_b0_matches_
  manual_per_frame_construction` passes, along with all 32
  `tests/test_recon_*.py` cases.
- [x] **91.** Resolved (the Python half -- `preprocessing/julia/b0map.jl`'s
  Julia copy is a separate language, not mergeable): `gre_diagnostics.py`
  now does `from preprocessing.run_rss import _ift3` instead of keeping a
  verbatim copy, so item 64's docstring fix (already applied to
  `run_rss.py`) now covers this consumer too automatically. Verified no
  circular import (`run_rss.py` doesn't import `gre_diagnostics`) and no
  new lint/test regressions.
- [x] **94.** Resolved: added `recon/reconstruct.py`'s
  `_load_normalized_smaps(fn_smaps, device) -> (smaps, smaps_chw)` helper
  (next to `_load_omega`/`_load_echo_times`), called by both `run_recon`
  (here) and `run_b0_recon.py` instead of each duplicating the load +
  RSS-normalize + permute. Removed `run_recon`'s now-redundant second
  `smaps_chw = smaps.permute(...)` computation further down (it was
  computed twice in the same function even before this fix). Also fixed
  an E501 regression from item 90's rename (`echo_times_s` ->
  `echo_times_yz` pushed one line over 100 chars in
  `run_b0_recon.py`) noticed while re-running ruff here. Verified: `uv
  run ruff check .` back to 29 errors, `uv run pytest` unchanged at
  126/15.
- [x] **102.** Resolved: deleted `reconecho()` from
  `preprocessing/epi_gridding.py` (confirmed zero callers anywhere in the
  repo or its tests before removing). `_density_compensation` and
  `rampsamp2cart`/`rampsampepi2cart` -- the functions actually used --
  are untouched; the module docstring's "ports ... reconecho.m" and
  `_density_compensation`'s own "ports reconecho.m" both still refer to
  the *original MATLAB* file this port's DCF logic came from, not the
  deleted Python function, so left as-is.
- [x] **105.** Resolved: `preprocessing/recon_frames.py`'s `use_parfor=True`
  path now passes `recon_fn`/`smaps` via
  `ProcessPoolExecutor(initializer=_init_worker, initargs=(recon_fn,
  smaps))` -- set once per worker process at pool startup, stored in a
  module-level `_worker_state` dict, read back by
  `_recon_one_frame_worker` -- instead of binding them into the per-task
  callable via `functools.partial` (which `executor.map` re-pickled once
  per dispatched frame). The unused `functools` import was removed.
  Verified functionally with a standalone `ProcessPoolExecutor` +
  `_init_worker`/`_recon_one_frame_worker` reproduction (real
  multiprocessing, not just imports): results match a plain serial
  computation exactly across 5 dispatched frames. Serial
  (`use_parfor=False`) path unchanged.
- [ ] **117. `preprocessing/preprocess.py`'s STEP 3 duplicates
  `smaps.py`'s `load_smaps()` caching logic instead of calling it, and the
  duplicate is already narrower and drifting.** [verify] `preprocess()`'s
  STEP 3 (`preprocess.py:323-353`) hand-rolls the same "check cached
  `Nvcoils` attr, load-or-estimate-and-cache" pattern
  `smaps.load_smaps()` (used by `recon_frames.py:76`) already implements
  -- but narrower: it never computes/writes `smaps_degre`/`emap_degre`
  (the deGRE-grid maps `run_b0map.py` needs), leaving that to
  `load_smaps()`'s documented backfill path the first time
  `recon_frames.py` or `run_b0map.py` runs later. Not a correctness bug
  today (the backfill path is real and tested), but it's duplicated
  cache-validity logic in two places that can already drift: item 41's
  fix made `smaps.py:170-173` compare `int(f.attrs['Nvcoils'])` against a
  freshly-read `ksp_gre.shape[-1]`, while `preprocess.py:330` still
  compares `f.attrs.get('Nvcoils') == Nvcoils` -- similar but not the same
  check, with no test pinning them to identical behavior. Since
  `paths.recon`'s GRE cache (read by `load_smaps`) is the very file STEP 2
  just wrote moments earlier, `preprocess.py` could call `load_smaps(cfg,
  paths, seq_params)` directly instead -- which would also produce a
  complete cache (with `smaps_degre`/`emap_degre`) on the very first run
  rather than deferring that to a later backfill.
- [ ] **118. `sampling/pd_sample.py`'s `dtype` parameter
  (`'logical'`/`'double'`/`'complex'`) is dead in production and
  untested.** [measured] `pd_sample`'s `dtype` branch (`:295-300`) is only
  ever called with the default `'logical'` throughout the codebase
  (`gen_sampling_masks.py` never passes `dtype=`), and
  `tests/test_pd_sample.py` never exercises the `'double'`/`'complex'`
  paths either. Low severity -- this is MATLAB-parity surface carried over
  from the port, not a wrong result -- flagged only because it's untested
  code that could silently break without anyone noticing were it ever
  used. Either add a couple of parametrized `dtype=` cases to
  `test_pd_sample.py`, or drop the untested branches if nothing is
  expected to ever pass a non-default `dtype`.
