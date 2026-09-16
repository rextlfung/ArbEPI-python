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
as described (none needed closing). Items 125-130 are new findings from
a later pass (2026-09-06, against `bbd8266`) that also re-verified every
item 107-124 against the current tree: no source file changed between
`0d6d821` and `bbd8266` except this doc itself, so all eighteen were
confirmed still open exactly as described (none needed closing). Items
131-135 are new findings from a later pass (2026-09-07, against
`639345e`) that also re-verified every item 107-130 against the current
tree: no source file changed between `bbd8266` and `639345e` except this
doc itself, so all twenty-four were confirmed still open exactly as
described (none needed closing). Items 136-145 are new findings from a
later pass (2026-09-08, against `3930358`) that also re-verified every
item 107-135 against the current tree: `git diff 639345e HEAD --stat`
shows only this doc itself changed between `639345e` and `3930358`
(items 131-135 were added in that span, doc-only), so all thirty-five
were confirmed still open exactly as described (none needed closing).
This pass split the review across six parallel subagents (`ge/`;
`lib/`+`sequences/`+`params.py`/`scanners.py`/`main.py`;
`sampling/`+`plotting/`; `preprocessing/`; `recon/`; a dedicated
docs-vs-code consistency sweep), each briefed on the open items in its
scope to avoid duplication; every new finding below was independently
re-verified against the live tree (not just trusted from the subagent's
report) before being recorded here, including a from-scratch
reproduction of item 136's sampling bug and item 137's trigger-detection
bug. Items 147-159 are new findings from a later pass (2026-09-09,
against `eab904b`) that also re-verified every item 107-146 against the
current tree: `git diff 3930358 HEAD --stat` shows `CLAUDE.md`,
`README.md`, `main.py`, `params.py`, `plotting/plotting.py`,
`sampling/caipi_sample.py`, `sampling/external_mask.py`,
`sampling/gen_sampling_masks.py`, and several test files changed in that
span (a "Simplify user-facing config" reorg of `params.py`/README, a
generalization of the top-level sampling-mask flow to accept
collaborator-provided masks, and a rewrite of `caipi_sample.py`'s
`balanced_factors`), alongside this doc itself (items 136-146 were added
there) -- none of those changed files are cited by any item 107-146's own
file:line citations, so all forty were confirmed still open exactly as
described (none needed closing). This pass split the review across four
parallel subagents (`sampling/`+`plotting/`;
`params.py`/`main.py`/`scanners.py`+README/CLAUDE.md consistency;
`ge/`+`lib/`+`sequences/`; `preprocessing/`+`recon/`), each briefed on
the open items already tracked in its scope plus an explicit note to
scrutinize the recently-changed, least-reviewed code first; every new
finding below was independently re-verified against the live tree (not
just trusted from the subagent's report) before being recorded here,
including a from-scratch reproduction of item 147's `caipi_sample`
regression and item 150's `noise.py` timing shortfall. Items 160-165 are
new findings from a later pass (2026-09-10, against `dac8252`) that also
re-verified every item 107-145 against the current tree: `git diff
eab904b HEAD --stat` shows `ge/read_pge.py`, `ge/seq2ceq.py`, `params.py`,
`plotting/compare_readout_pns.py`, `plotting/plotting.py`,
`preprocessing/gre_diagnostics.py`, `preprocessing/run_b0map.py`,
`sampling/caipi_sample.py`, `sampling/ticaipi_sample.py`, `scanners.py`,
`sequences/noise.py`, and several test files changed in that span (items
127/147-159's fixes), alongside this doc itself -- of the still-open
107-145 items, all were confirmed still open with unchanged substance
except: item 116 (closed, superseded -- item 147's `balanced_factors`
restriction made `ticaipi_sample`'s own divisibility guard unreachable via
the public API, so the regression test it asked for can no longer
exercise that code path; the equivalent invariant is already covered by
item 147's own `test_balanced_factors_raises_when_no_factor_pair_divides`),
item 143 (closed -- its own baseline-correction action was already
complete when logged, and this pass confirmed no other file, including
CLAUDE.md, carries the misattributed claim it warned a future reader
about), item 121 (still open, citation updated -- `plot_pns_one_tr` now
spans `plotting.py:286-342` after `plot_one_tr`'s item 127/149 fix added
lines above it), item 130 (still open, citation updated --
`run_b0map.py`'s per-sequence try/except now spans lines 105-162 after
item 151's fix widened it), and item 133 (still open, citation updated --
`gre_diagnostics.py`'s `fn_gre` path line moved from `:34` to `:39` after
item 152's docstring expansion). This pass split the review across four
parallel subagents with the same scope split as the previous pass
(`sampling/`+`plotting/`; `ge/`+`lib/`+`sequences/`+`params.py`/`main.py`/
`scanners.py`+docs; `preprocessing/`; `recon/`), each re-verifying its
assigned open items against the live tree (not just re-reading this file)
before reporting, and independently hunting for new findings in its
scope; six survived independent re-verification and are recorded below,
the rest (several test-coverage-gap candidates that overlapped existing
items, and one low-confidence/unmeasured hypothesis about
`run_b0_recon.py` rebuilding its encoding operator twice) were judged
either duplicates or not solid enough to record. Items 166-168 are new
findings from a later pass (2026-09-11, against `046ef61`) that also
re-verified the state of the tree since the previous pass: `git diff
dac8252 HEAD --stat` shows only `docs/review-findings.md` (items 160-165,
added by the previous pass) and one new test,
`tests/test_arbepi_kx_oversamples_when_nyquist_rate_exceeds_max_grad`
(`tests/test_trajectory_matches_schedule.py`, a pure regression-test
addition with no production-code change) changed in that span -- so every
still-open item's cited code is exactly as it was when last verified, and
none needed re-checking line-by-line; this pass instead spent its budget
entirely on hunting for new findings. Split across four parallel
subagents with the same scope as the previous two passes (`sampling/`+
`plotting/`; `ge/`+`lib/`+`sequences/`; `preprocessing/`; `recon/`+
`params.py`/`main.py`/`scanners.py`+docs), each independently re-verifying
its assigned scope's open items were still accurately described (not just
trusting this file) before hunting for anything new; three findings
survived independent reproduction against the live tree (recorded as
items 166-168 below) and are recorded in the section their content best
matches (all three landed in Correctness) rather than strictly by which
subagent found them. One subagent (`recon/`+top-level) found nothing new
after checking several specific hypotheses that all turned out to already
be covered or to check out as correct; one candidate from another
subagent (a latent `gz_ss.delay` negative-delay risk, item 168) was kept
despite being "not live today" -- the same disposition as items 45/107's
sibling 122 -- since it's a real, reproducible code gap with no test or
guard, not a stylistic nitpick. Items 169-180 are new findings from a
later pass (2026-09-12, against `ecb8f2f`) that also re-verified every
item 107-168 against the current tree: `git diff 046ef61 HEAD --stat`
shows `ge/coppe.py`, `lib/make_spoilers.py`, `lib/mask2epi.py`,
`params.py`, `preprocessing/config.py`, `preprocessing/preprocess.py`,
`preprocessing/smaps.py`, `sequences/ArbEPI.py`, `sequences/EPIcal.py`,
and two test files changed in that span -- a gradient-spoiler redesign
(area now expressed as cycles/voxel, varied per shot, with a new
gx-residual cancellation term), `mask2epi_radial`'s golden-angle-based
echo-train start-direction flip, an RF-spoiling phase increment change
from 117 to 115.4 degrees, a `ge/coppe.py` fix for a silently-failing SSH
hop, and `preprocessing/smaps.py`'s new sensitivity-map edge smoothing --
alongside this doc itself. Every item citing an unchanged file was
confirmed still accurate; items 117, 132, 133, 138, 139, 140, 141, and 142
needed citation and/or substance updates (applied in place below) since
their cited code moved or its surrounding behavior changed by these
commits. This pass split the review across four parallel subagents
(`lib/`+`sequences/`+`params.py`'s spoiler/golden-angle changes;
`preprocessing/`'s smaps changes; `ge/coppe.py`'s SSH fix; `recon/`,
untouched this round, a lighter re-verification-plus-hunt pass), each
re-verifying its assigned open items against the live tree before hunting
for anything new; eleven of their findings survived independent
verification and are recorded as items 170-180 below. Item 169 (a real TE-
feasibility regression in the shipped default config, caused by the
golden-angle change increasing the worst-case ky blip step from 37 to 39
samples) was found and verified directly in this pass's own synthesis
step, not by a subagent, after a fresh `main.py --ge` build surfaced a new
`calc_te_tr_delays` warning no prior baseline had reported; confirmed by
reproducing `max_blip_steps` with `_golden_angle_flip_start` forced off
(37) vs. on (39) against the same seed-0 schedules. Items 181-187 are new
findings from a later pass (2026-09-13, against `5e263f3`) that also
re-verified the state of the tree since the previous pass: `git diff
ecb8f2f HEAD --stat` shows only `docs/review-findings.md` itself changed
in that span (items 169-180, added by the previous pass) -- the source
tree is byte-for-byte identical to what that pass reviewed, confirmed
again by a fresh `uv run ruff check .` (29 errors, all `E501`), `uv run
pytest` (143 passed/15 skipped plain, 209 passed/5 skipped with
`preprocessing`+`recon` extras, 34 passed for `tests/test_recon_*.py`
alone), and a fresh `main.py --ge` build (identical peak-PNS/acoustics
numbers and the identical item-169 TE-feasibility warning) all matching
the previous pass's baseline exactly -- so no still-open item needed
re-verification against changed code this pass; the entire budget went
into hunting for new findings. Split across four parallel subagents with
the same scope as the previous several passes (`sampling/`+`plotting/`;
`ge/`+`lib/`+`sequences/`+`params.py`/`main.py`/`scanners.py`;
`preprocessing/`; `recon/`), each confirming its assigned scope's open
items were unchanged before hunting for anything new; seven findings
survived independent re-verification against the live tree (not just
trusted from each subagent's own report -- every one below was re-checked
by direct code reading and, where executable, a fresh reproduction) and
are recorded as items 181-187 below. Also sharpened item 115's own
citation: one of its supporting sub-claims ("no test file... imports
`plotting.plotting` at all") went stale after `tests/test_plotting.py` was
added for items 127/149, though that item's substantive finding -- none of
the actual plotting *functions* are tested -- remains fully open. Item 188
is a new finding from a later pass (2026-09-14, against `a6759be`) that
also re-verified every item 107-187 against the current tree: `git diff
5e263f3 HEAD --stat` shows only `docs/review-findings.md` itself changed in
that span (items 181-187, added by the previous pass) -- the source tree is
byte-for-byte identical to what that pass reviewed, confirmed again by a
fresh `uv run ruff check .` (29 errors, all `E501`), `uv run pytest` (143
passed/15 skipped plain, 209 passed/5 skipped with `preprocessing`+`recon`
extras, 34 passed for `tests/test_recon_*.py` alone), and a fresh `main.py
--ge` build (identical peak-PNS/acoustics numbers and the identical
item-169 TE-feasibility warning) all matching the previous pass's baseline
exactly. Split across four parallel subagents with the same scope as the
previous several passes (`sampling/`+`plotting/`; `ge/`+`lib/`+`sequences/`+
`params.py`/`main.py`/`scanners.py`; `preprocessing/`; `recon/`), each
independently re-verifying its assigned open items against the live tree
(not just trusting this file) before hunting for anything new. Three of the
four subagents (`ge/`+`lib/`+`sequences/`+`params.py`/`main.py`/
`scanners.py`; `preprocessing/`; `recon/`) confirmed every one of their
assigned open items unchanged and found nothing new that survived their own
verification bar -- expected at this point given how many prior passes have
already combed this exact scope at high rigor (matching the 2026-09-11
pass's own precedent for a clean "nothing new" result). The `preprocessing/`
subagent's one candidate finding, a third untracked instance of item 133's
cache-path-duplication pattern (`<seqname>_b0map.h5`, independently
hand-built in `run_b0map.py:76`/`gre_diagnostics.py:40`), survived
independent re-verification and is recorded as item 188 below. The
`sampling/`+`plotting/` subagent confirmed all eight of its assigned open
items unchanged, found no new findings, but while re-running item 136's own
reproduction methodology found that one of its two reported mechanisms
needed a substantive correction: mechanism (b) (the calibration-disc
seed-stall) reproduced precisely, with a sharper root-cause confirmation
than originally reported (frame 3's raw pre-crop mask sums to exactly
`n_calib`, confirming zero points were placed beyond the calibration disc),
but mechanism (a) (the claimed "~2% every single run" unfiltered-fill leak)
did not reproduce across 100 independent trials in this environment and has
been downgraded to `[verify]`/unconfirmed in place below, rather than
re-reported as newly measured -- this is a correction to an existing item,
not a new finding, so it isn't separately numbered. Item 110 also needed a
one-line citation update (write site shifted from `:413` to `:414`).
Items 189-191 are new findings from a later pass (2026-09-15, against
`b701489`) that also re-verified every item 107-188 against the current
tree: `git diff a6759be HEAD --stat` shows `preprocessing/config.py`,
`preprocessing/julia/b0map.jl`, `preprocessing/preprocess.py`,
`preprocessing/run_b0map.py`, `preprocessing/run_cg_sense.py`,
`preprocessing/run_recon_sigpy.py`, `preprocessing/smaps.py` (heavily
rewritten, +208/-lines), and two test files changed in that span --
`PreprocessingConfig.threshold_mask` (default 0.2) was renamed to `crop`
(default 0.95, now a single eigenvalue threshold shared between sigpy's
EspiritCalib and `process_smaps`'s object mask, itself the fix for a
previously-tracked two-thresholds-fighting bug), `preprocess.py` gained
gzip compression on `ksp_epi_zf`, and `smaps.py` gained masked Gaussian
re-smoothing helpers, GPU device auto-selection, and a post-resize
re-mask step while dropping a now-redundant pre-resize mask -- alongside
this doc itself. Every item citing only the unchanged files (ge/, lib/,
sequences/, sampling/, plotting/, recon/, params.py, scanners.py,
main.py -- byte-identical since the last several passes, reconfirmed
again this pass by a fresh `uv run ruff check .` (29 errors, all
`E501`), `uv run pytest` (143 passed/15 skipped plain, 209 passed/5
skipped with `preprocessing`+`recon` extras, 34 passed for
`tests/test_recon_*.py` alone), and a fresh `main.py --ge` build
(identical peak-PNS/acoustics numbers and the identical item-169
TE-feasibility warning)) needed no re-verification. Items 110, 117, 139,
141, 175, and 178 needed citation-only updates (line numbers shifted by
the `smaps.py`/`config.py`/`preprocess.py` changes, substance unchanged
in every case); items 161, 174, 183, and 188 were confirmed fully
unchanged (their cited lines fall outside the diff's actual edits).
Item 177 was closed as superseded -- the specific numbered comments it
quoted (`# 1. Eigenvalue support mask...`, `# 2+3. Crop
z...interpolate...`) no longer exist in `smaps.py` after the crop/mask
rewrite removed the step they partly labeled -- and item 192 records the
differently-shaped inconsistency the rewrite left behind (a single
orphaned `# 4. Normalize` with no `1`/`2`/`3` above it). This pass split
its budget across two parallel subagents: one focused entirely on
re-verifying the items above against the actual `preprocessing/` diff and
hunting for new findings there (items 189-190 plus the item 177/192
disposition), the other doing a lighter fresh-eyes hunt across the
unchanged areas per the established rotation (found one new
documentation gap, item 191, plus confirmation that several other
specific hypotheses it traced -- a README `--plot` file-list omission, a
spoiler off-by-one, a `calc_te_tr_delays` missing term, missing
`write_ceq`/`read_pge` coverage -- were all already-tracked duplicates,
not new). Every finding below was independently re-verified against the
live tree (not just trusted from either subagent's report) before being
recorded. Items 196-202 are new findings from a later pass (2026-09-16,
against `de3d535`) that also re-verified every item 107-195 against the
current tree: `git diff b701489 HEAD --stat` shows `CLAUDE.md`,
`lib/readout_from_params.py`, `main.py`, `params.py`, two brand-new files
(`preprocessing/lowres_calib_recon.py`, `preprocessing/r2star_map.py`),
`recon/operators_b0.py`, `recon/reconstruct.py`, `recon/run_b0_recon.py`,
`sampling/pd_sample.py`, `sequences/ArbEPI.py`, `sequences/deGRE.py`, and
three test files changed in that span (item 194/195's acoustic-resonance
dwell-selection and calibration-sizing fixes, a default-protocol switch to
"ABCD" at 2.4mm iso/90x90x60/R=6/TE=30ms, deGRE now generated last, and a
generalization of B0 correction to a complex off-resonance+T2* field),
alongside this doc itself (items 193-195, added in the same commit range
as the code fixes they close). Items 119, 124, 133, 135, 138, 140, 163,
164, and 167 needed citation-only updates (line numbers shifted by these
changes, substance unchanged in every case); item 136's mechanism (b) is
now resolved (closed as 136(b), fixed by `8efa7dd`'s seed-rejection fix)
while mechanism (a) remains open/unconfirmed as 136(a); item 169 was
closed as no longer live -- not a code fix, but the default-protocol
switch to a much smaller matrix means today's shipped default no longer
triggers the TE-feasibility warning the item described, though the
underlying golden-angle mechanism it flagged is confirmed still present
in a smaller, currently-harmless form. This pass split its budget across
four parallel subagents with the same scope as the previous several
passes (`sampling/`+`plotting/`; `ge/`+`lib/`+`sequences/`+`params.py`/
`main.py`/`scanners.py`; `preprocessing/`; `recon/`), each re-verifying
its assigned open items against the live tree before hunting for anything
new, with explicit instruction to give the newly-changed/newly-added code
(especially the two new `preprocessing/` files and the rewritten
`recon/operators_b0.py` complex-field generalization) the heaviest
scrutiny. Seven findings survived independent re-verification against the
live tree (not just trusted from each subagent's report) and are recorded
as items 196-202 below; the `recon/` subagent's pass over the rewritten
B0/R2* code found it consistent with CLAUDE.md's documented sign
convention and end-to-end parameter threading, so no new findings were
recorded there despite the size of that rewrite.

## Current baseline (2026-09-16, against `de3d535`)

- `uv run ruff check .` (after `uv sync --extra test --extra lint`): **29
  errors**, all `E501` -- unchanged from the previous pass.
- `uv run pytest` (plain main venv, fresh `.venv`, `rm -rf output` first):
  **147 passed, 15 skipped** (up from 143/15 -- `sampling/pd_sample.py`'s
  item 194/195 fixes added new test cases), same skip composition (**9**
  `sigpy`/`nibabel`-gated `preprocessing` files, **6** `could not import
  'torch'` `recon` files). With `--extra preprocessing --extra recon` also
  synced: **216 passed, 5 skipped** (up from 209/5), all five `julia
  executable not found on PATH` (`tests/test_preprocessing_run_b0map.py`),
  **zero** GERecon-gated -- confirming every `preprocessing`/`recon`-gated
  skip from the plain-venv run is addressable by syncing extras, none is a
  real failure. `tests/test_recon_*.py` alone: **37 passed**, 0 failed (up
  from 34 -- `test_recon_operators_b0.py`'s r2star-generalization tests).
- Whole-sequence feasibility (`uv run python main.py --ge`, full
  default-params build) -- all four sequences `.ok`, freshly measured this
  pass (`rm -rf output` first). **Numbers below are a new baseline, not
  comparable to prior passes**: `params.py`'s default protocol changed
  from the old 240x240x45/R=9/GE_MR750/TE=34.9ms config to "ABCD" (2.4mm
  isotropic, `N=[90,90,60]`, R=6, TE=30ms, commit `0b9c25f`) since the last
  baseline. No `calc_te_tr_delays` TE-feasibility warning fires anywhere
  in the build log under the new defaults (see item 169's closure above):

  | sequence | peak PNS | acoustics | max grad | max slew |
  |---|---|---|---|---|
  | `ArbEPI.seq` | 69.3% | 0.0231 | 28.82 mT/m | 119.2 T/m/s |
  | `EPIcal.seq` | 64.7% | 0.0231 | 28.71 mT/m | 119.2 T/m/s |
  | `deGRE.seq` | 77.4% | 0.2556 | 49.76 mT/m | 174.3 T/m/s |
  | `noise.seq` | 0.0% | 0.0000 | 0.00 mT/m | 0.0 T/m/s |

  `deGRE.seq`'s own resolution/`N` didn't change with the ABCD protocol
  switch (`fov_degre`/`N_degre` are separate params), so its PNS/acoustics/
  grad/slew numbers are unchanged from every prior pass, and item 123's
  finding (`ge/check.py`'s docstring still quoting the stale 0.2456 instead
  of 0.2556) remains open and unchanged. `ArbEPI.seq`/`EPIcal.seq`'s
  numbers dropped substantially (peak PNS 79.9%->69.3%, acoustics
  0.1764->0.0231) purely as a consequence of the smaller matrix/coarser
  resolution in the new default protocol, not a code change -- CLAUDE.md's
  "PNS finding history" section (which documents the *old* protocol's
  tuned-slew numbers, min TE 34.86 ms, 79.8% peak) is now describing a
  superseded default config, not a stale number within the same config;
  out of scope to fix here (only `docs/review-findings.md` may be modified
  this pass) -- carried forward for the next pass that touches CLAUDE.md.

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
  computed and written but has no reader anywhere in the repo.** [verify;
  citation updated 2026-09-15 against `b701489` -- the write site shifted
  from `:414` to `:416` after `b701489` added `compression='gzip',
  compression_opts=4` (2 lines) to the `ksp_epi_zf` dataset creation just
  above it; substance unchanged] `preprocess.py:274`
  computes `NframesDiscard =
  round(seq_params.discard_duration / seq_params.volume_tr)` and `:416`
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
  same input fine.** [measured; citation updated 2026-09-16 against
  `de3d535` -- write site shifted from `:1126-1128` to `:1186-1188`,
  substance and crash unchanged, re-reproduced directly this pass] The
  pass-3 uncrossing-cleanup step's "achieved worst-case step" computation
  (`mask2epi.py:1186-1188`) is:
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
  (`plotting.py:286-342` -- shifted down from `:270-326` by items 127/149's
  `nominal_te_value`/`plot_one_tr` additions above it in the same file;
  this function itself is untouched) calls `sample_gradients_tesla_per_m(seq,
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
- [ ] **125. `sequences/deGRE.py` excites with the EPI sequence's flip angle
  and RF duration instead of the deGRE-specific values `params.py` computes
  for exactly this purpose and that are never read anywhere.** [measured]
  `deGRE.py:72-74` calls
  ```python
  rf, gz_ss, gz_ssr = pp.make_sinc_pulse(
      params.fa / 180 * math.pi,
      duration=params.rf_dur,
      ...
  ```
  -- `params.fa` and `params.rf_dur` are the *EPI* sequence's Ernst angle
  (derived from the EPI `TR`, `params.py:215`) and RF duration
  (`params.py:216`, `2e-3`). But `params.py:97-98` declares
  `alpha_degre`/`rf_dur_degre` on `Params`, computed at `:264,266`
  (`alpha_degre = 180/pi * acos(exp(-TR_degre/T1))`, `rf_dur_degre =
  0.4e-3`) and threaded into the `Params(...)` constructor at `:347-348`
  -- named and structured exactly in parallel with `fa`/`rf_dur`, clearly
  intended as deGRE's own Ernst angle/RF duration for its much shorter
  `TR_degre` (8 ms vs. the EPI sequence's ~100 ms per-shot TR). A
  repo-wide grep confirms `alpha_degre`/`rf_dur_degre` have zero read
  sites anywhere outside `params.py` itself. Measured directly against
  this repo's shipped defaults (`load_params()`): `params.fa` = 22.19°,
  `params.rf_dur` = 2.0 ms, vs. the unused `alpha_degre` = 6.35°,
  `rf_dur_degre` = 0.4 ms -- a >3x difference in both flip angle and RF
  duration. Using 22° instead of the Ernst-optimal 6.35° for an 8 ms TR
  causes substantial extra saturation and lower steady-state SNR in the
  actual `deGRE.seq` build that feeds coil-sensitivity-map estimation and
  B0 field mapping; using a 2 ms RF pulse instead of 0.4 ms also eats far
  more of the already-tight `TR_degre` budget than necessary. Not a
  recent regression: traced through git history to the original
  single-echo `sequences/gre.py` (`a3df8fc^:sequences/gre.py:52-53`,
  before the dual-echo deGRE upgrade), which already used
  `params.fa`/`params.rf_dur` with an unused `alpha_gre`/`rf_dur_gre`
  sitting in `params.py` at the time -- the deGRE rename (`a3df8fc`)
  carried the same dead fields/bug forward unchanged, just renaming the
  suffix. No test references `alpha_degre`/`rf_dur_degre`, so nothing
  catches this. Fix: change `deGRE.py:73-74` to use `params.alpha_degre`
  and `params.rf_dur_degre`, then re-verify `te_min`/`tr_min`/`TR_degre`
  still clear (the much shorter 0.4 ms RF should shrink the timing
  budget, not break it) and re-check
  `test_degre_excitation_is_centered`/the PNS and timing regression tests
  after the change.
- [ ] **126. `ge/writeceq.py`'s sliding-window gradient/RF heating-check
  block count undercounts by exactly one segment instance's block count
  whenever a segment's block count evenly divides
  `NMAXBLOCKSFORGRADHEATCHECK` (40000).** [measured] The header field
  computation (`ge/writeceq.py:93-100`):
  ```python
  n = 1
  while n < min(ceq.nMax, NMAXBLOCKSFORGRADHEATCHECK):
      seg = segment_by_id[int(ceq.loop[n - 1, 0])]
      n += seg.nBlocksInSegment
      if n > NMAXBLOCKSFORGRADHEATCHECK:
          n -= seg.nBlocksInSegment
          break
  _w(fid, 'i', n - 1)
  ```
  rolls back and discards the *entire* last segment instance whenever that
  instance ends exactly at the cap (`n` becomes `CAP + 1`, which is `>
  CAP`, even though the instance itself is complete, not truncated) --
  the same "off-by-one at an exact boundary" class of bug as items
  107/122 in the sibling `ge/seq2ceq.py`. Reproduced directly with a
  synthetic single-segment case (`nb=40`, `nMax=40200`, matching the
  code above verbatim): the written value comes out **39960** instead of
  the correct **40000** -- an exact undercount of one instance's `nb=40`
  blocks. This isn't specific to `nb=40`: any `nb` dividing 40000 evenly
  (40, 50, 80, 100, 125, 160, 200, 250, ...) reproduces the same-size
  undercount. Inert for this repo's shipped sequences today (checked all
  four `output/*.seq` files via `seq2ceq()`: `ArbEPI.seq` has `nb=69`,
  `40000 % 69 = 31 != 0`; `EPIcal.seq`/`deGRE.seq`/`noise.seq` all have
  `nMax` well under 40000) -- but this field is written directly into the
  `.pge` binary (`_w(fid, 'i', n - 1)`, immediately below the code
  quoted above), a value GE's scanner-side gradient/RF-heating logic
  reads, so a future sequence whose per-segment block count happens to
  divide 40000 evenly would silently get a wrong (short) heating-check
  window. No test exercises this field at all -- `tests/` has zero
  references to `NMAXBLOCKSFORGRADHEATCHECK` or a synthetic `nMax > 40000`
  case. Fix: compare against the instance's *last row* rather than the
  next instance's start row, e.g. `if n - 1 > NMAXBLOCKSFORGRADHEATCHECK:
  n -= seg.nBlocksInSegment; break` (mirroring the correct `n +
  nBlocksInSegment - 1 > ceq.nMax` form items 107/122 already identify in
  `ge/seq2ceq.py`), and add a regression test with a synthetic `Ceq`
  whose segment size divides 40000 exactly.
- [x] **127.** Resolved together with item 149 (the mirror-image bug in
  `plotting/plotting.py`'s `plot_one_tr`): added a shared, parity-aware
  `plotting.plotting.nominal_te_value(per_echo_values, ETL)` helper
  (exact echo `ETL//2` for odd `ETL`, average of `ETL//2-1`/`ETL//2` for
  even `ETL`, matching `calc_te_tr_delays.py`'s continuous `ETL/2 - 0.5`
  definition) and wired `compare_readout_pns.py:65`'s `te_realized` to it.
  Unit tests added in `tests/test_plotting.py` covering both parities.
- [ ] **131. `recon/reconstruct.py`'s `_reg_weights` computes each scale's
  regularization weight from the *declared* patch size, but
  `recon/lowrank.py`'s `img2patches`/`patchSVST` (the functions that
  actually consume `patch_sizes[k]`) silently clip each axis to the image
  dimension first -- so a "whole-volume" scale declared at or above the
  image size gets the wrong weight.** [measured] `_reg_weights`
  (`reconstruct.py:133-145`) computes `p_k = math.prod(ps)` directly from
  the caller-supplied `patch_sizes[k]`, with no clamping against the
  actual `(Nx,Ny,Nz)`. But `img2patches`/`patchSVST`
  (`lowrank.py:21-44`, called one line below/above with that same
  `patch_sizes[k]` value at `reconstruct.py:239,257,266`) both clip every
  axis via `psx, psy, psz = (min(p, n) for p, n in zip(patch_size,
  (Nx,Ny,Nz)))` before ever extracting a patch -- clipping exists
  specifically so a scale can be declared "as large as possible" without
  hardcoding the exact grid dims, a natural way to spell a whole-volume
  scale. Reproduced directly: for a `(12,12,8)` volume, `Nt=5`, declaring
  a whole-volume scale as `(16,16,16)` gives `_reg_weights` `p_k=4096`
  (`lambda_k~=66.8`) while `patchSVST` actually operates on the clipped
  `p_k=1152` patch (correct weight `~=37.4`) -- a ~1.8x miscalibration of
  that scale's regularization strength, silently, with no assertion
  anywhere that `patch_sizes[k] <= (Nx,Ny,Nz)`. This directly undermines
  the module's own docstring claim ("lambda_k set by the Ong & Lustig
  (2016) closed-form formula ... no tuning needed beyond
  lambda_global") -- the whole point of that closed-form weight is that
  it matches the *actual* patch geometry. Currently inert in this repo's
  own driver scripts (`run_b0_recon.py`/`validate_against_mslr.py` both
  read `patch_sizes` from a reference file whose "global" scale already
  matches the image dims exactly, confirmed by checking the validated
  configs cited in CLAUDE.md's MSLR table), and no test in
  `tests/test_recon_lowrank.py`/`tests/test_recon_reconstruct.py` passes
  an oversized `patch_sizes` entry -- but any future caller relying on
  the clipping behavior `img2patches`/`patchSVST` were clearly built to
  support (e.g. specifying a round-number "big" patch instead of the
  exact grid dims) gets a silently wrong regularization weight, not an
  error. Fix: clip each axis the same way inside `_reg_weights` (or have
  it call a shared helper with `img2patches`/`patchSVST`) before computing
  `p_k`, and add a test with `patch_sizes` exceeding the image on at
  least one axis, asserting `_reg_weights`'s implied `p_k` matches what
  `patchSVST` actually used.
- [ ] **132. `preprocessing/recon_frames.py`'s per-frame failure message
  never says which frame failed.** [measured; citations updated 2026-09-12
  against `ecb8f2f` -- these had already drifted before the previous
  (2026-09-11) baseline was cut, from commit `c4d1794`'s
  `_worker_state`/`_init_worker`/`_recon_one_frame_worker` insertion; not
  a change from this pass's own diff, just never previously corrected]
  `_recon_one_frame`
  (`recon_frames.py:33-38`) catches any exception from `recon_fn` and
  prints `f'recon_frames: reconstruction failed on a frame -- skipping.
  {e}'` -- both call sites (the serial list comprehension and
  `_recon_one_frame_worker`, `recon_frames.py:98,100`, was `:94,96`)
  dispatch over
  `frame_data = (f['ksp_epi_zf'][...,frame] for frame in
  range(nframes))` but never thread `frame` into `_recon_one_frame`
  itself, so the printed message carries no frame index. With the
  default `Nframes` up to 30, two or more failing frames print
  indistinguishable lines, and the only way to identify which frame(s)
  actually failed is to notice which slices of the returned `img` array
  are all-zero after the fact (the existing "all output frames are zero"
  check at `:109-112`, was `:104-107`, only catches the all-frames-failed
  case, not a
  partial failure). Low severity -- doesn't change any computed output,
  only debuggability when `recon_fn` raises on a subset of frames -- but
  cheap to fix: thread `frame` through `_recon_one_frame`'s signature and
  into both call sites' generator/worker so the printed message names the
  failing frame index.
- [x] **136(b).** Resolved by `8efa7dd` ("redefine calibration region as a
  per-axis kmax-fraction rectangle"): `_poisson_disc_core_jit`'s initial
  active point is now rejection-sampled against `calib_mask`
  (`pd_sample.py:170-184`, module docstring "point 4"), eliminating the
  seed-stall described in mechanism (b) below. Re-verified 2026-09-16
  against `de3d535`: 200 seeds at a ~13%-area calibration region and 200
  more at `calib_frac=0.3`/`Ny=240,Nz=45,R=9` gave **0 stalls**, where the
  pre-fix code would have stalled a substantial fraction of the time. The
  calibration region's *shape* changed from ellipse to rectangle in the
  same commit range, which independently introduced a different
  corner-stripping bug under `crop_corner=True` -- see item 196.
- [ ] **136(a). `sampling/pd_sample.py`'s exact-count fill step is
  unguarded against `crop_corner=True`'s ellipse, but no reproduction of a
  routine leak has succeeded across two independent passes.** [\[verify\],
  unconfirmed -- re-verified 2026-09-16 against `de3d535`, still
  unconfirmed] Originally two distinct, compounding mechanisms were
  reported under one item number; mechanism (b) above is now fixed, this
  sub-item (a) remains open and unconfirmed:

  The exact-count enforcement step (`pd_sample.py`'s fill branch) fills any
  shortfall between the binary-search mask and `target_samples` from
  `np.flatnonzero(~mask)` -- every currently-unsampled pixel in the full
  rectangular `(ny, nx)` grid, with **no** `rho <= 1` filter -- even though
  `crop_corner=True`'s earlier step (`mask = mask * (rho <= 1)`) is
  supposed to confine every sample to the centered inscribed region. This
  code path is real and still genuinely unguarded as described. But the
  claim that it leaks on "every single run" (originally: 5 seeded runs at
  `Ny=240, Nz=45, R=9, decay=1.4, calib_frac=0`, each landing 20-28/1200
  (~2%) samples outside the ellipse) has now failed to reproduce across
  two independent passes: the 2026-09-14 pass got 0/100 across a 5x20
  sweep, and this pass (2026-09-16, against `de3d535`, using the doc's own
  sweep methodology across 5 diverse configs) got **0 leaks** across 300+200
  further seeds. Root cause of the original claim not holding (per the
  2026-09-14 pass): the binary search's `accel_search = accel * 0.95`
  biases toward a *higher*-density target than requested, so the post-crop
  sample count tends to converge from the **overshoot** side, which routes
  into the *prune* branch (`mask & ~calib_mask`, already filtered to
  inside the region, safe) rather than the unguarded *fill* branch this
  item depends on. Whether an undershoot (and hence this leak) is ever
  reachable in practice still needs a fresh, explicit repro before being
  cited as measured again -- until then, treat this as a real but
  unconfirmed code-level gap, not a reproduced leak. Fix direction (still
  worth doing defensively even unconfirmed, since mechanism (b) above used
  to fall through to this same unguarded path as a fallback of last
  resort): restrict the exact-count fill step's candidate pool to
  `np.flatnonzero(~mask & (rho <= 1))` when `crop_corner=True`, and add a
  regression test with `calib_frac > 0`/`crop_corner=True` covering both an
  overshoot and (if a repro is ever found) an undershoot seed.
- [ ] **137. `ge/blocks.py`'s `get_block_type` reads a nonexistent `.trig`
  attribute instead of pypulseq's real `.trigger` dict, so physio-trigger
  blocks are never detected.** [measured] `ge/blocks.py:38-39`:
  ```python
  trig = getattr(block, 'trig', None)
  has_trigger = trig is not None and trig.channel == 'physio1'
  ```
  pypulseq 1.5.0.post1's `Sequence.get_block()` never sets any `.trig`
  attribute on the returned block namespace -- trigger extensions are
  attached under `.trigger`, a *dict* (`{0: trigger_obj, ...}`, to allow
  more than one trigger per block), not a single object with its own
  `.channel` field directly on the block. Reproduced directly:
  `seq.add_block(pp.make_trigger(channel='physio1', duration=100e-6))`,
  then `hasattr(seq.get_block(1), 'trig')` is `False` while
  `hasattr(..., 'trigger')` is `True`
  (`{0: namespace(type='trigger', channel='physio1', ...)}`), so
  `get_block_type(...).has_trigger` comes back `False` for a block that
  genuinely carries a `physio1` trigger event. Confirmed through the full
  pipeline too: a synthetic 2-block sequence with one trigger block
  produces `seq2ceq(seq).loop[:, 13]` (the `physioTrigger` loop column,
  per `ge/ceq.py`'s `LOOP_COLUMNS`) as `[0., 0.]` for both instances.
  `has_trigger` is the *only* signal `get_dynamics`
  (`ge/blocks.py:108-183`) uses to populate the loop table's
  per-block-instance `physioTrigger` column -- the dynamic trigger-gating
  mechanism, distinct from `writeceq.py`'s separately-and-deliberately-
  always-0 per-parent-block trigger field (which carries an explicit
  comment disclosing it as an intentional MATLAB-quirk match; this one
  has no such disclosure and is a plain wrong-attribute-name bug).
  Currently 100% inert -- a repo-wide grep confirms no sequence-generation
  code (`sequences/`, `lib/`, `params.py`) anywhere calls
  `pp.make_trigger`/adds a trigger block today -- but it would silently
  break any future cardiac/respiratory-gated sequence added to this repo:
  the exported `.pge`'s loop table would never flag a single block as
  needing a physio trigger, regardless of how many trigger blocks the
  source `.seq` actually contains. No test exercises this
  (`tests/test_seq2ceq.py`/`tests/test_ge_check.py` have zero
  `trig`/`physio` references). Fix: read `block.trigger` (a dict,
  possibly absent) instead of `block.trig`, e.g.
  `trig_dict = getattr(block, 'trigger', None) or {}; has_trigger = any(t.channel == 'physio1' for t in trig_dict.values())`,
  and add a regression test exercising a synthetic trigger block through
  `get_block_type`/`get_dynamics`/`seq2ceq` -- the same untested-bug
  pattern item 134 already flags for `write_ceq`/`read_pge`, one level up
  the pipeline.
- [ ] **138. `sequences/ArbEPI.py`'s post-readout spoiler scaling has a
  quantifiable off-by-one against this repo's 0-based indexing
  convention, already flagged in an inline comment but untracked in this
  backlog.** [measured, low severity; re-verified/updated 2026-09-12
  against `ecb8f2f` -- the buggy formula, its cause, and its severity are
  unchanged, but the surrounding code was rewritten by commit `8448ff3`
  ("Vary gradient spoiler per shot, switch RF phase increment to 115.4
  deg", introducing `params.py`'s `spoil_cycles_min`/`max`), so the quoted
  code/numbers/citation below replace the now-stale originals, and the
  "(and identically EPIcal.py's)" claim in
  this item's own title was wrong even before that rewrite and is
  corrected below; citation re-updated 2026-09-16 against `de3d535` --
  shifted from `:276-284` to `:281-289`, substance unchanged]
  `sequences/ArbEPI.py:281-289`:
  ```python
  seq.add_block(
      pp.scale_grad(gx_spoil, (x_scale * gx_spoil.area - gx_residual) / gx_spoil.area),
      pp.scale_grad(
          gy_spoil,
          (y_scale * gy_spoil.area - (y_locs[-1] + 1 - Ny / 2) * rg.deltak[1]) / gy_spoil.area,
      ),
      pp.scale_grad(
          gz_spoil,
          (z_scale * gz_spoil.area - (z_locs[-1] + 1 - Nz / 2) * rg.deltak[2]) / gz_spoil.area,
      ),
  )
  ```
  The `+ 1` in the y/z terms still doesn't match the 0-based `y_locs`/
  `z_locs` convention used everywhere else in this file -- CLAUDE.md's
  "Index convention" section documents 0-based as the deliberate,
  repo-wide internal convention. **Correcting this item's own title: this
  bug is `ArbEPI.py`-only, not "identically EPIcal.py's".**
  `sequences/EPIcal.py`'s post-readout spoiler block (`:150-157`, both
  before and after the per-shot-randomization rewrite) has never contained
  a `y_locs`/`z_locs`-based rewind term at all -- EPIcal applies no ky/kz
  encoding, so there's nothing to rewind on those axes (its own comment
  says exactly this: "no rewind term is needed on y/z here since, unlike
  ArbEPI, no ky/kz encoding was ever applied"); only its x-axis
  `gx_residual` cancellation is shared logic with ArbEPI, and that term
  isn't schedule-index-based, so it's unaffected by this `+1` issue.

  **Framing correction**: the previous "ends at ky = -deltak[1] instead of
  0" description is now stale -- that was accurate when the y-axis
  spoiler was a pure rewind-to-center term (pre-`8448ff3`:
  `pp.scale_grad(gy_spoil, -((y_locs[-1]+1-Ny/2)*deltak[1])/gy_spoil.area)`,
  no additive spoiling). Post-rewrite, y legitimately carries its own
  per-shot-randomized spoiling moment (`y_scale * gy_spoil.area`) on top
  of the rewind, so the correct framing is: the spoiler under-delivers by
  exactly `deltak[1]` relative to the intended `y_scale * gy_spoil.area`
  target, not "ends at -deltak[1] instead of 0."

  **Updated magnitude** (seed-0 default-params schedule, measured against
  the current `spoil_cycles_max=4.0` build, which replaced the old fixed
  `n_cycles_spoil=2`): the shortfall is still exactly `deltak[1] = 4.6296
  m^-1` (y) / `deltak[2] = 24.6914 m^-1` (z) in absolute terms (depends
  only on `Ny`/`Nz`/`deltak`, confirmed identical across shots 0-2), but
  as a *fraction* of the actually-delivered z-spoil area it's now
  **~0.56%-0.74%** (down from the old fixed ~1.1%), since `gz_spoil.area`
  moved from 2222.2 to a per-shot-randomized value in `[3333.3, 4444.4]`
  at the new `spoil_cycles_min/max = [3.0, 4.0]` range. Severity remains
  low (well under the spoiler's own dephasing margin). Fix direction
  unchanged: drop the `+ 1` to match the 0-based convention (or confirm
  via a fresh MATLAB comparison that the `+1` is intentional and update
  the comment instead), then re-verify against
  `tests/test_trajectory_matches_schedule.py`'s existing k-space coverage
  checks.
- [ ] **139. `preprocessing/config.py`'s `load_seq_params` reads
  `scan_info.mat` via a bare `h5py.File`, not `matio.read_mat`,
  contradicting `matio.py`'s own unconditional stated rule.** [verify,
  not live today; citation updated 2026-09-15 against `b701489` --
  `config.py`'s `threshold_mask` field was renamed to `crop` with a 9-line
  docstring expansion above `load_seq_params`, shifting it from
  `config.py:169-200` (body `:177-199`) to `config.py:178-209` (body
  `:186-209`); substance unchanged] `config.py:186-208` opens
  `paths.scan_info` directly
  and reads every field with plain `f[name][()]`
  (`.item()`/`.ravel()`), never calling
  `preprocessing.matio.read_mat`/`read_mat_array`. `matio.py`'s module
  docstring is explicit: "Use these for every hdf5storage-written `.mat`
  this pipeline reads (`scan_info.mat`) -- never `scipy.io`... never a
  bare `h5py` read without the transpose." Every field `load_seq_params`
  reads today is a scalar or a short 1D vector (`Nx`, `TR`, `TE_degre`,
  etc.), and `matio.py`'s own docstring confirms the missing transpose is
  a values no-op for those shapes (only a singleton axis moves) -- so
  this isn't a live correctness bug today, and
  `test_load_seq_params_round_trips_a_scan_info_fixture` (which writes
  its fixture pre-transposed exactly like real hdf5storage output)
  passes. But it's a direct deviation from the stated convention and a
  latent trap: if a future field added to the `scan_info.mat` snapshot
  (or to `SeqParams`) is 2D+ (a matrix, not a scalar/vector), reading it
  through this bare-`h5py` path instead of `matio.read_mat_array` would
  silently return it axis-reversed -- exactly the class of bug
  `matio.py` exists to prevent, and the same class items 44/64/91/108/120
  already document elsewhere in this codebase for the sibling
  FFT-shift-convention mistake. `preprocess.py`'s own
  `load_kxoe`/`load_schedules` (reading the same file) already go through
  `matio.read_mat`/`read_mat_array` correctly, so `config.py` is the
  outlier, not the rule. Fix: route `load_seq_params` through
  `matio.read_mat`/`read_mat_array` like every other `scan_info.mat`
  reader in this repo.
- [x] **146. `lib/make_readout_grads.py`'s POPE readout can hard-crash
  (`AssertionError`) on a legitimate small-readout-FOV/sparse-mask
  combination, with no fallback and no actionable guidance toward the fix.**
  [measured] `make_readout_grads.py:245-260` sizes the flat-top readout
  amplitude from Nyquist-critical sampling alone --
  `A = min(deltak[0] / dwell, sys.max_grad)`, i.e. purely a function of
  `dwell` and readout FOV (`deltak[0] = 1/fov[0]`) -- with no awareness of
  how much k-space area the phase-encode blip turnaround (`M =
  max(a1, a_d)`, driven by the largest consecutive-sample ky/kz step in the
  schedule) is about to consume from the same lobe. When `A` is too large
  relative to the target x-resolution (`S = Nx*deltak[0] = 1/res[0]`) and
  `M`, the required flat-top duration goes negative and line 258's
  `assert flat >= 0` fires: `'Readout lobe would be triangular (ramps alone
  exceed the required area) -- unsupported; would need A =
  sqrt(2*S/(1/slew_rise + 1/slew_fall)).'` -- a hard crash mid-build, not a
  warn-and-fall-back like `calc_te_tr_delays` (item 144) uses for its own
  unachievable-parameter case.

  Reproduced against a real collaborator-supplied custom sampling mask
  (109x91, 70 samples, R~142 -- see the "Using custom ky-kz-t sampling
  masks" README section / PR #3): `Nx=91`, `res=[2,2,2]mm`
  (`fov_x=182mm`), `ETL=70`, `Nshots=1` gives `max_ky_step=19`,
  `max_kz_step=17` from `mask2epi_radial`'s ordering -- large jumps,
  since a 70-point mask this sparse forced into one 70-echo train has few
  short hops available -- and at the repo's default `dwell=2e-6` this
  assertion fires immediately; `dwell=4e-6` (or, equivalently, oversampling
  the readout FOV to `Nx~160+` at the same `res_x=2mm`, i.e. `fov_x` up to
  ~320mm+, leaving `dwell` untouched) both clear it, confirming `A`'s
  `1/(fov_x * dwell)` dependence is exactly the lever (doubling either
  factor halves `A` and produces the same `Nfid`/`Tread`, ~0.45ms, either
  way). This is a real, reproducible failure mode independent of the
  custom-mask machinery itself (PR #3's `custom_mask_path` loading/
  validation all worked correctly here; the crash is purely downstream in
  gradient design) -- any sufficiently sparse/scattered mask at a
  small-enough readout FOV would trigger it, custom or built-in
  `sampling_method`. Not investigated further / not fixed here per
  explicit instruction to treat as a separate follow-up. Fix direction:
  at minimum, name both remedies (`dwell`, readout-FOV oversampling) in the
  assertion message so a user hitting this isn't left to reverse-engineer
  the `A`/`S`/`M` algebra themselves; a fuller fix would have
  `make_readout_grads` fall back to solving for a smaller `A` (per the
  message's own `sqrt(2*S/(1/slew_rise + 1/slew_fall))` formula) rather
  than requiring the caller to hand-tune `dwell`/`Nx` until it happens to
  fit.

  Resolved 2026-09-15: reproduced independently against the repo's own
  *default* (non-custom-mask) params after an in-progress, uncommitted
  resolution change (`res` 0.9mm -> 2.4mm, `N` 240x240x45 -> 90x90x60,
  `fov` unchanged at 216mm) -- confirming this is purely
  `1/res[0]`-vs-ramp-area structural, not mask-sparsity-driven as
  originally written above: at the shipped `dwell=2e-6`, the POPE ramps
  alone (`A*(r+d)/2`, with `A` pinned at `sys.max_grad` since
  `1/(fov[0]*dwell)` already exceeds it) consume more k-space area than
  `Nx*deltak[0]` needs even with zero blip contribution (`M=0`), so no
  mask/ETL/trajectory choice could have avoided it at that `Nx`/`dwell`.
  Fix direction 1 (`dwell`, named in the assertion message above) is what
  got implemented, per explicit user request rather than the "fuller fix"
  direction 2 (solving for a smaller `A`): `dwell` is no longer a fixed
  `params.py` field at all -- `lib/readout_from_params.py`'s new
  `find_min_feasible_dwell` linearly searches ADC-raster-quantum
  multiples of `dwell` (the required flat-top duration is monotonically
  non-decreasing in `dwell`, so the first feasible one found is also the
  fastest) and `make_readout_grads_from_params` calls it internally --
  every existing 3-arg call site (`ArbEPI.py`/`EPIcal.py`/`noise.py`,
  `plotting/compare_readout_pns.py`, and the test suite) needed no
  signature change. At the reproducing config this landed on `dwell=4e-6`
  (one raster step up from the old fixed `2e-6`), which also happens to
  drop the readout out of the hardware-clamped regime entirely (echo
  spacing 392us/line); full ArbEPI/EPIcal/noise/deGRE builds all pass
  `ge/check.py` feasibility with peak PNS well under the 80% normal-mode
  line (69.6%/64.5%/0%/77.4%) at this lower resolution. One test
  (`test_arbepi_kx_oversamples_when_nyquist_rate_exceeds_max_grad`) had
  to pin a local `Nx=240` override so it keeps exercising the
  hardware-clamped/oversampled branch regardless of the global default's
  own `Nx`, since auto-search now actively steers away from that regime
  whenever a larger dwell both fixes feasibility and clears the clamp.
  `sequences/deGRE.py`'s unrelated `dwell_degre` floor (previously
  `params.dwell`, i.e. always exactly `sys.adc_raster_time`) now reads
  `sys.adc_raster_time` directly -- behavior-preserving, no coupling to
  the EPI search. Not addressed: `dwell` is not surfaced into
  `scan_info.mat`'s scalar snapshot (it never was, even when fixed), so a
  `preprocessing/`-side consumer keying off the realized dwell (e.g. a
  future `calibrate_delay.py` refinement) still has no direct read path --
  only a printed console line at `ArbEPI.py` generation time.
- [x] **147.** Resolved: `balanced_factors` now restricts its candidate
  search to `(Ry, Rz)` pairs that evenly divide `(Ny, Nz)` (raising a
  clear `ValueError` if none exist, e.g. `balanced_factors([64,64], 9)`,
  rather than silently returning a non-dividing pair) instead of picking
  purely by closest FOV-weighted log-ratio. At the shipped default
  `(240, 45, 9)`, the restricted search returns `(3, 3)` -- same as the
  pre-`eab904b` behavior -- so the crash is gone; the docstring's worked
  example was rewritten to `(240, 60, 4) -> (4, 1)`, which does survive
  the restriction and still demonstrates non-square reweighting. Updated
  `tests/test_caipi_sample.py`'s affected cases accordingly and added
  regression tests: `test_balanced_factors_raises_when_no_factor_pair_divides`,
  and shipped-`(240,45,9)`-config tests in both `test_caipi_sample.py` and
  `test_ticaipi_sample.py` asserting the real `params.Nshots*ETL` sample
  count and no `ticaipi_sample` raise.
- [x] **148.** Resolved: `plotting/compare_readout_pns.py` now imports and
  calls `resolve_omegas(p0, rng=np.random.default_rng(p0.seed))` instead
  of `gen_sampling_masks(p0.R, p0, ...)` directly. Also broadened the
  `assert p0.seed is not None` a few lines above to `assert p0.seed is not
  None or p0.custom_omegas is not None`, since `seed` is `None` by design
  on the custom-mask path.
- [x] **149.** Resolved together with item 127: `plot_one_tr` now builds
  `echo_centers = t_adc[n_fid//2::n_fid]` (one ADC-center timestamp per
  echo) and marks TE via the same shared
  `plotting.plotting.nominal_te_value(echo_centers, params.ETL)` helper,
  replacing the fixed-parity `(ETL-1)//2` index. Docstring updated to
  describe the parity-aware behavior.
- [x] **150.** Resolved: `noise.py` now computes `pad_duration =
  pp.calc_duration(rg.gro) - pp.calc_duration(rg.adc)` directly, dropping
  the extra `+ sys.adc_dead_time` term (`rg.adc.dead_time` is already `0`
  by construction, so adding the real scanner dead time back on top was
  double-subtracting it). Added
  `test_noise_repetition_duration_matches_epi_readout` in
  `tests/test_trajectory_matches_schedule.py`, building a real
  `noise.seq` and confirming its first `[ADC + delay]` repetition sums
  exactly to `rg.gro`'s duration.
- [x] **151.** Resolved: `run_b0map()`'s try/except now wraps the whole
  per-sequence body (subprocess call through `save_recon_nifti`), not just
  the julia subprocess call, catching `Exception` and printing `ERROR
  [{seqname}]: ...\nSkipping...` -- matching `run_rss.py`'s exact pattern.
  Verified against the existing `tests/test_preprocessing_run_b0map.py`
  suite (all 5 tests still pass).
- [x] **159.** Resolved: added a dedicated `_read_floats` helper
  (bypassing `_r`'s generic length-1 collapse) and used it for the five
  array-valued float reads in `_read_grad`'s raster/corner-points branches
  and `_read_arbitrary`'s `time`/`magnitude`/`phase`. Verified with a real
  `write_ceq`/`read_pge` round trip (a built `ArbEPI.pge`) and a synthetic
  `n_samples=1` case confirming `_read_arbitrary` now returns a length-1
  tuple, not a bare scalar.
- [ ] **166. `plotting/compare_readout_pns.py`'s `_overlay_figure` centers its
  "gx zoom" panel ~8ms (about 7.5 echo spacings) away from the actual
  nominal-TE echo, because it mixes two different time origins.** [measured]
  `_build` (`compare_readout_pns.py:68`) reads `te_realized` from
  `scan_info.mat`'s `schedules[..., 2]`, which CLAUDE.md's ".mat file
  format" section defines as "echo time in seconds since RF excitation" --
  i.e. relative to the RF pulse, not to the start of the sequence/shot.
  `_overlay_figure` (`:81`) instead samples gradients via
  `sample_gradients_tesla_per_m(v['seq'], time_range=(0.0, p.TR))`, whose
  `t=0` is the absolute start of shot 0's block sequence -- which begins
  with a fat-sat pulse and spoiler block (`sequences/ArbEPI.py:189-190`)
  *before* the excitation RF (`:199`), not the RF itself. Line 92,
  `t_c = v['te_realized']`, uses the RF-relative value directly as an
  index into the absolute-time-sampled array, with no correction for the
  fat-sat/spoiler lead-in. Reproduced directly against this repo's shipped
  default params (`Nframes=1`, seed=0): `seq.calculate_kspace()`'s
  `t_excitation[0]` = 8.052 ms (the true absolute RF start), `te_realized`
  = 34.90 ms, so the script's `t_c` (34.90 ms) is short of the correct
  absolute reference (`t_excitation[0] + te_realized` = 42.95 ms) by
  exactly 8.052 ms -- at this config's echo spacing `D` = 1076 us, about
  7.5 echo spacings, far outside the plotted `±1.5*D` zoom window. The
  panel's x-axis label ("time relative to each variant's nominal TE echo")
  and title ("~3 echo spacings around the TE echo... slow rise on the left
  of each lobe, fast fall on the right") both therefore describe an
  earlier, unrelated echo of the train, not the nominal-TE echo they claim
  to show. Scope is narrow: only `ax_gx`'s zoom window is affected -- the
  printed comparison table, the `ax_pns` total-PNS overlay curve, and the
  separate per-variant `plot_pns_one_tr` figures all use correct absolute
  time references and are untouched by this bug. Since the fat-sat/spoiler
  timing is essentially identical between the `symmetric`/`pope` variants
  (governed by `params.fatsat`/spoiler design, not
  `ro_slew_rise`/`ro_slew_fall`/`blip_slew`), both curves are shifted by
  roughly the same amount, so the two curves stay comparable to each
  other even though neither is anchored where the plot claims. Doesn't
  touch any shipped `.seq`/`.pge` output or the `main.py` pipeline -- only
  this one-off analysis script's `compare_pns.png` gx-zoom panel (last
  run 2026-08-27 per CLAUDE.md's PNS finding history). Fix: compute the
  absolute RF start time (e.g. `seq.calculate_kspace()[2][0]`) inside
  `_build` and use `t_c = t_excitation0 + te_realized` in
  `_overlay_figure`.
- [ ] **167. `preprocessing/recon_frames.py`'s `recon_frames()` unconditionally
  calls `load_smaps()` even when `recon_fn` doesn't use sensitivity maps at
  all -- crashing RSS-only reconstruction whenever no GRE/smaps cache
  exists.** [measured; citation updated 2026-09-16 against `de3d535` --
  fresh-estimation write site shifted from `smaps.py:204` to `:424`; also
  narrowed since last verified: `load_smaps` now trusts an existing stale
  smaps cache rather than crashing when only the GRE cache is missing, so
  this still crashes only on a true first-run with neither cache present]
  `recon_frames()` (`recon_frames.py:76`) opens with
  `smaps, _smaps_degre, _emap_degre, nvcoils = load_smaps(cfg, paths,
  seq_params)`, unconditional on what `recon_fn` actually needs. But
  `run_rss.py`'s own module docstring states this driver is "root-sum-of-
  squares reconstruction (**no smaps, no BART**)", and its `_rss_recon(data,
  _smaps)` (`run_rss.py:30-31`) explicitly discards its `smaps` argument
  (underscore-prefixed, never read). `load_smaps`'s fresh-estimation branch
  (`smaps.py:424`) unconditionally opens `<datdir>/recon/<seqname>_gre.h5`
  (`with h5py.File(fn_gre, 'r') as f:`) on a true first run, which raises
  `FileNotFoundError` if that file doesn't exist. Reproduced directly: a
  minimal `ksp_epi_zf.h5`
  with no matching `<seqname>_gre.h5` present (a real scenario -- deGRE
  wasn't acquired, or its cache was cleaned up independently of the EPI
  data, since nothing ties their lifecycles together) makes
  `recon_frames(cfg, paths, seq_params, _rss_recon)` raise `FileNotFoundError:
  ... 'testseq_gre.h5' ... No such file or directory` before ever touching
  the k-space data RSS actually needs. Even when the GRE cache *is*
  present, every RSS run still pays for a full ESPIRiT sensitivity-map
  estimation (or at least a cache-validity check load) for a value it then
  throws away -- the same "confirmed expensive" cost `smaps.py`'s own
  `estimate_smaps` docstring documents elsewhere in this codebase (a
  full-resolution GRE volume "thrashed 14GB+ of memory and never completed
  in over an hour" before the `cal_size` fix). Confirmed intentional-looking,
  not a typo: `tests/test_preprocessing_recon_frames.py::test_recon_frames_
  estimates_smaps_when_no_cache` already exercises (and expects) this eager
  smaps estimation, so this is a real design gap in `recon_frames()`'s API
  (no way to opt out of smaps loading), not an accidental leftover. Fix
  direction: thread whether `recon_fn` needs smaps into `recon_frames()`
  (e.g. a `needs_smaps: bool` parameter defaulting `True`, with `run_rss.py`
  passing `False`), deriving `nvcoils` from `f['ksp_epi_zf'].shape[3]`
  instead of `load_smaps`'s return value when smaps aren't needed, skipping
  `load_smaps()` entirely on that path.
- [ ] **168. `lib/make_excitation_pulse.py`'s (and `sequences/deGRE.py`'s
  identical inline copy) post-`trap4ge` RF/slice-select resync has no guard
  against producing a negative gradient delay.** [measured, not live today]
  `make_excitation_pulse.py:42` (and `deGRE.py:82`, an intentional inline
  duplicate per that file's own docstring) computes
  `gz_ss.delay = rf.delay - gz_ss.rise_time` to re-center the RF pulse in
  `gz_ss`'s flat top after `trap4ge` rebuilds the trapezoid with raster-
  rounded rise/flat/fall times (`trap4ge` always resets `delay` to 0
  regardless of whether rounding changed anything, per `deGRE.py`'s own
  docstring paragraph explaining why this resync exists). `rf.delay` is
  fixed at RF-pulse-construction time (effectively `max(pre-trap4ge
  gz.rise_time, sys.rf_dead_time)`); if `crt` is large enough that
  `trap4ge`'s rounded-up `gz_ss.rise_time` exceeds that margin, the
  subtraction goes negative, which pypulseq's `seq.check_timing()` flags
  as a hard `NEGATIVE_DELAY` error. Reproduced directly by calling
  `make_excitation_pulse` with the repo's real default `fa`/`rf_dur`/
  `rf_tb`/`fov`/`sys` at three `crt` values: shipped default `crt=4e-6` ->
  `gz_ss.delay=88.00us` (safe, matches CLAUDE.md's "trap4ge is a proven
  no-op at crt==grad_raster_time" claim); the one alternate value CLAUDE.md
  itself names as a plausible future setting, `crt=20e-6` (reverting to
  Siemens-dual-raster compatibility) -> `gz_ss.delay=80.00us` (still safe);
  but `crt=150e-6` (roughly `rf.delay`'s own 100us margin) ->
  `gz_ss.delay=-50.00us`, and adding the resulting `gz_ss` to a block and
  calling `seq.check_timing()` confirms pypulseq reports `error_type=
  'NEGATIVE_DELAY'`. So this is a real, reproducible gap -- no
  `assert`/`max(0, ...)` clamp documents or enforces the implicit
  "trap4ge's rounding delta must stay under rf.delay's margin" assumption
  -- but it isn't reachable at the shipped `crt` and isn't reachable at the
  one alternate `crt` value this codebase's own docs contemplate either, so
  it's in the same "not live today" category as items 45/122. Flagging
  because the resync logic itself fails silently at the point of cause (a
  future change to RF pulse timing -- shorter `sys.rf_dead_time`, longer
  slice-select rise time -- could trip it with no defensive check to
  explain why `check_timing()` suddenly fails). Fix direction: either clamp
  (`gz_ss.delay = max(0.0, rf.delay - gz_ss.rise_time)`, accepting the
  pulse re-decenters slightly rather than crashing) or add an explicit
  `assert gz_ss.rise_time <= rf.delay` with a message naming the `crt`/
  `rf.delay` relationship, in both `make_excitation_pulse.py` and
  `deGRE.py`.
- [x] **169.** Closed as no longer live 2026-09-16 (against `de3d535`) --
  not a code fix, a config change. `params.py`'s default protocol was
  switched from the old 240x240x45/R=9/TE=34.9ms config to "ABCD"
  (2.4mm iso, 90x90x60, R=6, TE=30ms, commit `0b9c25f`), and under the new
  defaults a fresh `main.py --ge` build prints no `calc_te_tr_delays`
  TE-feasibility warning at all -- confirmed directly this pass (full
  build, all four sequences `OK`, no warning in the log). Re-measured the
  underlying mechanism this item describes (the golden-angle start-flip
  changing `max_blip_steps`'s worst-case step) under the new defaults: it
  still nudges the worst-case step by 1 (`max_kz_step` 9->10 with the flip
  on; `max_ky_step` unchanged at 18 either way), so the mechanism itself is
  real and unfixed -- but at the new, much smaller matrix size it no longer
  pushes `min_te` past the prescribed `TE`. Revisit (reopen under a new
  item number, don't reuse 169) if a future config change reintroduces a
  `calc_te_tr_delays` TE-feasibility warning; the original finding's
  mechanism and reproduction methodology are preserved below for
  reference. Original text follows:

  `mask2epi_radial`'s golden-angle echo-train start-flip
  logic increased the worst-case ky blip step, pushing `min_te` above the
  prescribed `TE` in the (now-superseded) shipped default config -- the
  realized TE was silently ~0.2 ms later than documented, not the "min TE
  34.86 ms" CLAUDE.md's PNS finding history cites. [measured, was live in
  the shipped default config at the time] Commit `a20da01` ("distribute radial echo-train
  starts via golden angle") added `lib/mask2epi.py`'s
  `_golden_angle_flip_start` helper, which flips which physical end of
  each shot's spoke becomes schedule index 0 ("before") based on a
  running full-circle golden-angle target, so echo-train start directions
  spread around the whole circle instead of clustering in one half (see
  `mask2epi_radial`'s own docstring for the full rationale -- this part of
  the change is a deliberate, documented improvement, not itself a bug).
  But changing which points land at the start/end of each shot's tour
  also changes the largest consecutive-sample ky/kz jump `max_blip_steps`
  measures across the whole schedule, which
  `lib/make_readout_grads.py`/`calc_te_tr_delays.py` size the readout's
  blip-turnaround geometry (and hence `gro`'s duration and
  `ReadoutGrads.echo_offset`) against. Reproduced directly against this
  repo's real seed-0 default-params schedule (`load_params()`,
  `resolve_omegas`, `mask2epi_radial` per frame): `max_blip_steps` with
  `_golden_angle_flip_start` forced to always return `False` (reproducing
  the pre-`a20da01` start-selection behavior) gives `max_ky_step=37`, vs.
  **`max_ky_step=39`** with the flip active (current code) -- `max_kz_step`
  unchanged at 8 either way. This 2-sample increase in the worst-case blip
  widens the readout's required blip-turnaround area, which increases
  `pp.calc_duration(gro)` and therefore `calc_te_tr_delays`'s `min_te =
  ... + (ETL/2 - 0.5) * pp.calc_duration(gro)` term. Confirmed end to end:
  a fresh `uv run python main.py --ge` build (full default params, seed 0)
  now prints `UserWarning: Minimum achievable TE (35.104 ms) exceeds
  prescribed TE (34.900 ms)` from `lib/calc_te_tr_delays.py:54` -- a
  warning absent from every prior baseline in this file, and from
  CLAUDE.md's own "PNS finding history" section, which cites "**min TE
  34.86 ms**" for these exact tuned defaults (`slew_derate=100,
  ro_slew_rise=100, ro_slew_fall=120, blip_slew=105`). Per CLAUDE.md's own
  documented `calc_te_tr_delays` contract ("only warns, never raises... 
  silently falls back to zero padding delay, so the sequence still builds
  with the *wrong* TE/TR baked in"), the shipped default `ArbEPI.seq`
  today silently builds with `te_delay=0` and a realized TE of
  `min_te=35.104 ms`, not the prescribed `34.9 ms` -- a ~0.2 ms/~0.6%
  timing error baked into every default-params sequence this repo
  generates, with no error or loud warning in `main.py`'s own summary
  output (the warning appears mid-build, easy to miss in a long log, and
  `main.py --ge`'s own `OK`/feasibility summary says nothing about TE
  accuracy). This also means the PNS/acoustics/TE numbers throughout
  CLAUDE.md's "PNS finding history" (79.8% peak "at min TE 34.86 ms") and
  this file's own historical entries are now stale on the TE axis, on top
  of item 111/123's already-tracked acoustics staleness and this pass's
  own "Current baseline" table's updated acoustics/PNS figures above.
  Severity: real, live, and silent -- not "not live today" like items
  45/107's sibling 122/168 -- but bounded (a fraction of a millisecond,
  well inside typical EPI timing tolerances, and PNS/feasibility are still
  `OK` per this pass's fresh `main.py --ge` build). Fix direction: either
  (a) treat this as an accepted, deliberate cost of the golden-angle
  start-spreading improvement and update `TE` in `params.py` (or
  `blip_slew`/other slew tuning) to restore margin, re-verifying PNS per
  CLAUDE.md's own "re-verify after any seed/mask/R/ETL/resolution change"
  guidance (a golden-angle start-direction change is exactly this kind of
  schedule-affecting change, even though it isn't literally a
  seed/mask/R/ETL/resolution edit), or (b) revisit whether
  `_golden_angle_flip_start` can be constrained to avoid increasing the
  worst-case blip step while still spreading start directions. Either way,
  CLAUDE.md's "PNS finding history" numbers need refreshing to match --
  out of scope to edit this pass (only `docs/review-findings.md` may be
  modified), flagged here for the next pass that touches it.
- [ ] **170. `lib/calc_te_tr_delays.py`'s `min_tr` formula omits `gy_spoil`
  from its pre-excitation (fat-sat crusher) spoiler-block duration term,
  while that block now actually plays `gx_spoil`/`gy_spoil`/`gz_spoil`
  together -- undercounts `min_tr` under anisotropic resolution.**
  [measured, not live at the shipped isotropic default, live and
  quantified under anisotropic `res`] Commit `7af04d4` ("Add gy to fat-sat
  crusher, cancel gx readout residual in spoiler") changed
  `sequences/ArbEPI.py`'s (and `EPIcal.py`'s) fat-sat crusher block to play
  all three spoiler axes together:
  ```python
  seq.add_block(
      pp.scale_grad(gx_spoil, x_scale),
      pp.scale_grad(gy_spoil, y_scale),
      pp.scale_grad(gz_spoil, z_scale),
  )
  ```
  but `lib/calc_te_tr_delays.py` was not touched by that commit, and its
  `min_tr` formula's term for that same block still accounts for only x/z:
  ```python
  # lib/calc_te_tr_delays.py:61
  min_tr = (
      pp.calc_duration(rfsat)
      + max(pp.calc_duration(gx_spoil), pp.calc_duration(gz_spoil))   # no gy_spoil
      ...
  ```
  (the function's *second*, post-readout spoiler term, `:67`, does
  correctly include `gy_spoil` -- `gy_spoil` has been a parameter to this
  function since before this diff, so only the first term's `max(...)`
  call was missed, not a plumbing gap.) Reproduced and quantified: at the
  current isotropic default (`res=[0.9,0.9,0.9]mm`) all three spoiler
  durations are equal (4.676 ms derated), so `max(gx,gz)` trivially equals
  `max(gx,gy,gz)` and the bug is inert today -- confirmed by this pass's
  `main.py --ge` build showing no `min_tr` warning. Varying only `res[1]`
  (the y/phase-encode resolution) reproduces a real, growing undercount:

  | `res_y` | `dur(gy_spoil)` | `min_tr` undercount |
  |---|---|---|
  | 0.90 mm (default) | 4.676 ms | 0.000 ms (inert) |
  | 0.70 mm | 5.872 ms | 1.196 ms (live) |
  | 0.50 mm | 8.016 ms | 3.340 ms (live) |
  | 0.30 mm | 13.028 ms | 8.352 ms (live) |

  Confirmed directly by assembling the real two-block (fat-sat + crusher)
  sequence at `res_y=0.3mm` and measuring its actual duration (17.388 ms)
  against what `min_tr`'s formula computes for that same part (9.036 ms)
  -- an 8.352 ms gap, exactly `calc_duration(gy_spoil) -
  max(calc_duration(gx_spoil), calc_duration(gz_spoil))`. Effect: since
  `min_tr` is under-reported, `tr_delay = floor((TR - min_tr)/raster) *
  raster` comes out larger than the true minimum requires, so the
  *realized* total TR runs measurably longer than intended by exactly the
  undercount -- the mirror-image failure mode of the TE bug CLAUDE.md
  already documents fixing ("saved per-echo times and the realized TE ran
  ~0.6-0.7 ms late because calc_te_tr_delays's min_te omitted the gro1
  lead-in block..."), just for TR and via this newly-introduced omission.
  `seq.check_timing()` cannot catch this -- it only validates raster
  alignment of the already-correct real blocks, not that
  `calc_te_tr_delays`'s own estimate matches them. No test exists for
  `lib/calc_te_tr_delays.py` at all (`tests/` has no
  `test_calc_te_tr_delays.py`, matching item 144's own coverage-gap
  finding), so nothing would catch this. Fix: change
  `lib/calc_te_tr_delays.py:61` to `max(pp.calc_duration(gx_spoil),
  pp.calc_duration(gy_spoil), pp.calc_duration(gz_spoil))`, and add an
  anisotropic-resolution regression case (could share test infrastructure
  with item 142's suggested anisotropic-`res` test for `make_spoilers.py`).
- [ ] **171. `ge/coppe.py`'s hop-2 SSH-failure fix (commit `3de4d58`) was
  applied to only 1 of 7 ssh/scp invocations in the file, leaving the
  identical silent-failure mode live on every other hop.** [measured code
  state; the underlying failure mode was previously verified live by this
  same commit's own investigation, not independently re-triggered here]
  The commit's own comment (`ge/coppe.py:129-134`) states, as an empirical
  finding, that `-q` "suppresses the actual auth-failure text too (e.g.
  'Host key verification failed.'), not just the progress meter" -- and
  removes `-q` from `_TRANSFER_SCRIPT`'s hop-2 `scp` (`ge/coppe.py:141-142`)
  for exactly that reason, alongside a new `_ssh_env()` helper (stripping
  `DISPLAY`/`SSH_ASKPASS`/`SSH_ASKPASS_REQUIRE`) and `BatchMode=yes`/
  `PreferredAuthentications=publickey` to force a fast, diagnosable
  non-interactive failure instead of an askpass fallback -- a real,
  structural fix, not a fragile stderr-text-classification one (no
  output-heuristic grepping was introduced anywhere in the diff). But `-q`
  is still present, unremoved, on every other ssh/scp call in the module:
  `ge/coppe.py:241` (`discover_relay_ip`'s `ssh -q user@relay`), `:264` and
  `:276` (`stage_tarball_on_relay`'s `ssh -q ... mktemp -d` and `scp -q`),
  `:293` (`cleanup_relay_staging`'s `ssh -q ... rm -rf`), and `:327-329`
  (`build_ssh_prefix`, both the outer local->epyc/goliath hop and the
  nested epyc/goliath->scanner hop -- used by every `run_remote` call:
  `query_existing_entries`, `query_run_entries`, `claim_entry_numbers`,
  `release_locks`, and the transfer trigger itself). Since `-q` suppresses
  the SSH client's own client-side auth-failure diagnostics (not remote-
  command output, which `-q` doesn't affect), an authentication failure on
  any of these remaining hops -- e.g. a rotated/expired hop-1 key, a relay
  host-key change, a bad `--user` -- would still surface only as
  `run_remote`'s generic `RuntimeError(f'remote command failed (exit
  {result.returncode})\n{detail}')` with `detail` empty or unhelpful,
  reproducing the exact "bare, undiagnosable exit 1" symptom this commit
  set out to fix, just on a different hop. Not a rare edge case: hop 1
  (`query_run_entries`, the very first network call `main()` makes) is on
  the critical path of every single invocation. `run_remote`'s docstring
  (`ge/coppe.py:348-361`) discusses only the `DISPLAY`/askpass side of this
  fix and doesn't mention that `-q` independently undermines the
  "surfaces as diagnosable error" property the commit message claims for
  the fix as a whole. Fix direction: drop `-q` from the same five
  remaining call sites (keeping stderr capture, already correct there) the
  same way it was dropped from `_TRANSFER_SCRIPT`.
- [ ] **181. `sequences/deGRE.py`'s ADC delay is derived from a pre-`trap4ge`
  gradient object, not the one actually played -- if `crt` ever diverges from
  `grad_raster_time`, the readout's first sample lands inside the gx ramp
  instead of on the flat top.** [measured] `deGRE.py:112-119`:
  ```python
  gxtmp = pp.make_trapezoid(
      'x', system=sys, amplitude=params.Nx_degre * deltak[0] / Tread, flat_time=Tread
  )
  ...
  adc = pp.make_adc(params.Nx_degre, system=sys, duration=Tread, delay=gxtmp.rise_time)
  ```
  `gxtmp` is a plain `pp.make_trapezoid(...)`, never passed through `trap4ge` --
  it exists only so `gx_pre`'s area can be derived from `-gxtmp.area / 2` and so
  `adc.delay` can be set from its `rise_time`. But the gradient actually played
  during the readout, `gx` (`:122-129`), is a *separate* object built afterward
  with a slightly different `flat_time` (`Tread + adc.dead_time`) and *is*
  passed through `trap4ge(..., crt, sys)`, which rounds rise/flat/fall times up
  to `crt` multiples -- so `gx.rise_time` only equals `gxtmp.rise_time` when
  `trap4ge` is a no-op, i.e. `crt == grad_raster_time` (today's shipped
  default, `crt = 4e-6`, per `params.py:287`'s comment). Reproduced directly:
  at `crt=4e-6`, `gxtmp.rise_time == gx.rise_time == 156 us` (inert); at
  `crt=20e-6` (the one alternate value CLAUDE.md itself names as a plausible
  future setting, for Siemens-dual-raster compatibility), `gx.rise_time`
  becomes `160 us` while `adc.delay` stays at `gxtmp`'s stale `156 us`.
  Building the real `generate_degre()` output at `crt=20e-6` and inspecting
  the first acquisition block confirms the consequence directly: the gx flat
  top starts at `0.00016 s` but the ADC starts at `0.000156 s` -- 4 us (one
  full dwell) early, so the ADC's first sample (centered at
  `adc.delay + 0.5*dwell = 0.000158 s`) lands during the ramp, not the flat
  top, corrupting kx-trajectory linearity for that sample.
  `seq.check_timing()` passes (it only validates raster alignment of each
  object in isolation, not cross-object consistency between `adc.delay` and a
  *different* gradient's realized rise time), and no test exercises
  `crt != grad_raster_time` for deGRE (`tests/test_trajectory_matches_schedule.py`'s
  only deGRE-specific cases, `test_degre_excitation_is_centered` and
  `test_degre_raises_actionable_error_below_minimum_tr`, neither vary `crt`
  nor check ADC/gradient alignment). Distinct from the two already-documented
  "same class" `trap4ge`-resync issues (the RF-pulse decentering CLAUDE.md's
  `trap4ge` paragraph describes, and item 168's `gz_ss.delay` negative-delay
  gap): `gxtmp` itself is never passed through `trap4ge`, so it's outside the
  audited "0 of 11 real call sites" set those cover -- this bug is the
  *mismatch* between an untransformed reference object and the transformed
  one actually played, not a `trap4ge` call site misbehaving. Fix: derive
  `adc.delay` from the same post-`trap4ge` gradient that's actually played
  (e.g. build/trap4ge `gx` first, or trap4ge `gxtmp` too, before reading a
  rise time for `adc.delay`), and add a regression test building
  `generate_degre` at a non-default `crt` (e.g. `20e-6`) asserting
  `adc.delay == gx.delay + gx.rise_time`.
- [ ] **182. `preprocessing/recon_frames.py`'s `use_parfor=True` path drains
  its own memory-bounding frame generator eagerly, defeating the design its
  own comment describes.** [measured] `recon_frames()` builds
  `frame_data = (f['ksp_epi_zf'][:, :, :, :, frame] for frame in
  range(nframes))` specifically, per the adjacent comment (`recon_frames.py:89-92`),
  "to bound memory" for a full-res acquisition that can otherwise exceed
  physical RAM. When `cfg.use_parfor` is `True` (`:96-98`), this generator is
  handed straight to `concurrent.futures.ProcessPoolExecutor.map()`. The
  stdlib `Executor.map()` implementation eagerly builds
  `fs = [self.submit(fn, *args) for args in zip(*iterables)]` before
  returning anything -- i.e. it fully drains the iterable up front,
  submitting every frame's task (and therefore reading every frame's HDF5
  slice into memory) regardless of worker count or how fast tasks are
  consumed. Reproduced directly two ways: (a) a standalone repro
  (`ProcessPoolExecutor(max_workers=2).map(slow_fn, generator)`, `slow_fn`
  sleeping 0.3s) shows all 10 generator items produced before `map()` even
  returns, well before either worker finishes its first task; (b) an
  instrumented real `recon_frames()` run (patching `h5py.Dataset.__getitem__`
  to timestamp reads), `use_parfor=True`, `recon_fn` sleeping 0.2s: all 8
  frame slices were read from the HDF5 file within 0.05s -- before any
  worker had finished even its first (0.2s) task -- confirming the same
  eager-drain happens in the real code path, not just the abstract repro. At
  this repo's real scale (240x240x45x18-coil complex64, ~373MB/frame per
  CLAUDE.md's own cited chunk size, up to 30 frames ~= 11GB), this means the
  `use_parfor=True` path reads essentially the entire acquisition into
  memory at once -- exactly the OOM scenario the streaming-generator design
  exists to avoid. Distinct from the already-resolved item 105 (which fixed
  re-pickling `smaps` per dispatched task, not this generator-draining
  issue). Latent in the shipped default (`PreprocessingConfig.use_parfor`
  defaults to `False`), but real and reproducible for any user who opts into
  parallel reconstruction -- a real, documented config knob, not a
  hypothetical. No test in `tests/test_preprocessing_recon_frames.py`
  exercises `use_parfor=True` at all, so nothing catches this. Fix: bound
  submission with a semaphore/rolling window (submit new work only as prior
  futures complete), or dispatch via `executor.submit()` one frame at a time
  against a bounded in-flight queue, instead of a bare `.map()` over an
  unbounded-lookahead generator.
- [ ] **183. `preprocessing/julia/b0map.jl`'s `romeo_finit` never receives
  the sensitivity maps the module docstring says replace its noisy
  coil-combine fallback -- the ROMEO-unwrapped `finit` is always built from
  the same low-SNR phase-contrast combine, regardless of whether real `smap`
  was loaded.** [verified by code reading -- no `julia` executable available
  in this environment to execute it, same constraint this repo's own test
  suite already accepts for this file] The module docstring (`b0map.jl:71-90`)
  states the `smap` feature "replaces MRIFieldmaps' own phase-contrast
  coil-combine fallback (`coil_combine(images, nothing)`, ...) with a true
  matched-filter combine (`coil_combine(images, smap)`, ...)", motivated by
  reduced noise "in this pipeline's real low-per-coil-SNR object-center
  regions." That's accurate for the main fit call, `b0map(finit, images,
  echotime; smap, mask, precon)` (`:242`), which does receive `smap`. But
  `romeo_finit(images, echotime, mask)` (`:192-198`) -- the function that
  builds the NCG solve's *initial guess* by ROMEO-unwrapping the coil-combined
  phase difference -- has no `smap` parameter at all, and its one
  `coil_combine` call (`:193`) is hardcoded `coil_combine(images, nothing)`
  unconditionally. In `main()` (`:200-242`), `smap` is fully loaded from
  `smaps_h5_path` (`:221-231`) *before* `finit = romeo_finit(images,
  echotime, mask)` is called (`:234`) -- so a real `smap` is sitting right
  there, unused by the one call that could use it. Since `main()`'s own
  docstring stresses that a bad `finit` can make the NCG solve converge to
  the wrong 2π branch entirely (not just a locally noisier optimum), this gap
  means the `smap` improvement documented in CLAUDE.md's B0 off-resonance
  section is only half-applied: the initial guess is seeded from exactly the
  noisier combine the feature was added to move away from, in precisely the
  low-SNR regions that motivated it. `preprocessing/run_b0map.py` now passes
  a real `smaps_path` whenever `load_smaps()` succeeds (the common case, not
  a corner case), so this is live in essentially every real run today, not a
  rare edge case. `tests/test_preprocessing_run_b0map.py` never exercises the
  `smaps_h5_path` argument at all (confirmed via grep -- zero occurrences),
  so nothing would catch this either way. Fix: thread `smap` into
  `romeo_finit`'s signature and call `coil_combine(images, smap)` when `smap`
  is available (mirroring what the main fit call already does), falling back
  to `coil_combine(images, nothing)` only when no `smap` was loaded.
- [ ] **184. `recon/lowrank.py`'s `img2patches`/`patches2img` silently zero
  out real image voxels whenever `stride_size > patch_size` on an axis --
  unvalidated, and untested in that regime.** [measured] `_patch_starts(n,
  patch, stride)` (`lowrank.py:21-23`) places patch starts at
  `min(i*stride, n-patch)`, guaranteeing the *last* patch reaches `n-patch`,
  but nothing constrains `stride <= patch` the way `img2patches`/
  `patches2img` already validate `stride > 0` (`:29-30`) -- so when
  `stride > patch`, consecutive patch windows can leave real gaps between
  them that no patch ever covers. Reproduced directly:
  ```python
  img = torch.arange(23, dtype=torch.float32).reshape(23,1,1,1)
  P = img2patches(img, (3,1,1), (10,1,1))
  patches2img(P, (3,1,1), (10,1,1), (23,1,1))
  # -> [0,1,2, 0,0,0,0,0,0,0, 10,11,12, 0,0,0,0,0,0,0, 20,21,22]
  ```
  Voxels 3-9 and 13-19 come back exact zero -- not an approximation, an
  actual overwrite, since `patches2img`'s `pcount` (`:57,65,68`) is exactly 0
  there and gets `.clamp_(min=1.0)`'d before the final `img / pcount`
  divide, so those voxels compute `0/1 = 0` with no warning, NaN, or error.
  At a more realistic 3D scale (`23x23x5`, patch `(3,3,3)`, stride
  `(10,10,10)`), `patchSVST(..., beta=0.0)` -- which should be a near-identity
  round-trip -- zeroes **84.7%** of all voxels. Since `reconstruct.py`'s
  `g_prox`/`reg_cost` call `patchSVST`/`img2patches` every iteration for
  every scale (`reconstruct.py:257,266`), a config that trips this would
  silently zero out most of a scale's contribution to `X_recon` on every
  iteration. Latent, not live today: every current call site
  (`reconstruct.py`'s own tests, `run_b0_recon.py`/`validate_against_mslr.py`,
  which both read `patch_sizes`/`strides` from the external Julia reference
  `.mat` already byte-validated against real MSLR output per CLAUDE.md's
  table) uses `stride <= patch` on every axis -- confirmed by grepping every
  `strides=`/`patch_sizes=` call site repo-wide. `tests/test_recon_lowrank.py`'s
  only stride-related test (`test_img2patches_rejects_nonpositive_stride`)
  checks `stride <= 0`, not `stride > patch`; every other test uses
  non-overlapping or half-overlapping strides. Fix: assert
  `all(s <= p for s, p in zip(stride_size, (psx,psy,psz)))` in
  `img2patches`/`patches2img` (mirroring the existing nonpositive-stride
  guard) and raise a clear `ValueError`, or have `_patch_starts` clamp
  `stride` to `patch`; either way, add a regression test exercising
  `stride > patch`.
- [ ] **196. `sampling/pd_sample.py`'s `crop_corner=True` post-crop can
  strip calibration-region cells at a high enough `calib_frac`, directly
  contradicting the function's own docstring claim that `calib_mask` cells
  are always fully sampled.** [measured; found 2026-09-16 against
  `de3d535`] The calibration region changed from a centered ellipse
  (`rho <= rho_calib`, always inscribed within the full-grid unit ellipse)
  to a centered *rectangle* (`8efa7dd`, `_calib_mask_rect`). `pd_sample`'s
  docstring still claims the rectangle's corners "are still forced fully
  sampled via `calib_mask` directly" even under `crop_corner`, but
  `crop_corner=True`'s post-hoc step (`mask = mask * (rho <= 1)`) uses the
  *full-grid* inscribed-ellipse mask, and when the rectangle's own
  `side_frac > 1/sqrt(2) ~= 0.707` its corners fall outside `rho <= 1` and
  get zeroed even though they're in `calib_mask`. Reproduced directly:
  `ny=nx=80, accel=1.5, calib_frac=0.9` gives `side_frac=0.775`, and 40 of
  3721 calibration cells are absent from the final mask
  (`mask[calib_mask].all()` is `False`) -- the same invariant
  `test_pd_sample_calibration_region_fully_sampled` checks, but that
  test's `calib_frac=0.2` never approaches the threshold. Not reachable at
  the shipped default (`pd_calib_frac=0.2`, `R=6` gives `side_frac~=0.18`),
  so this is "not live today" in the same class as items 45/122/168/136(a)
  -- but it's a real, freshly-introduced regression from the
  ellipse-to-rectangle change in `8efa7dd` (the old elliptical calib
  region was mathematically immune to this, since every ellipse point
  already satisfies `rho <= rho_calib <= 0.999 < 1`). Fix direction:
  either exclude `calib_mask` cells from the `crop_corner` multiply
  (`mask = calib_mask | (mask * (rho <= 1))`), or clamp `side_frac` to
  `1/sqrt(2)` when `crop_corner=True` so the rectangle's corners never
  leave the inscribed ellipse; add a regression test with a large
  `calib_frac`/low `accel` asserting `mask[calib_mask].all()` under
  `crop_corner=True`.
- [ ] **197. `lib/readout_from_params.py`'s `find_min_feasible_dwell`
  catches a bare `AssertionError`, silently masking unrelated,
  dwell-independent geometry bugs behind a misleading "no feasible dwell"
  error.** [measured; found 2026-09-16 against `de3d535`]
  `make_readout_grads` raises `AssertionError` for at least two
  structurally different reasons: (1) the dwell-dependent "triangular
  lobe" case `find_min_feasible_dwell`'s search is meant to route around,
  and (2) a dwell-*independent* blip-duration/`crt` raster mismatch
  (`assert abs(t_s / crt - round(t_s / crt)) < 1e-9`,
  `lib/make_readout_grads.py:232`). A real `max_ky_step` (223, reachable
  for masks with large blip jumps) triggers only the second assertion
  (`blip_duration (964.0 us) must be an even multiple of crt`), which does
  not depend on dwell at all -- confirmed directly: calling
  `find_min_feasible_dwell(223, 0, p)` loops through all 100 candidate
  dwells uselessly (`except AssertionError: continue` at
  `readout_from_params.py:118-127`) and then raises `RuntimeError: No
  feasible ADC dwell found ... this Nx/fov/slew/mask combination may be
  fundamentally infeasible` -- hiding the real, unrelated, easily-fixable
  root cause (a blip-geometry/`crt` mismatch) behind a message that steers
  debugging toward dwell/Nx/fov tuning instead. Fix direction: only catch
  the specific dwell-feasibility assertions (e.g. give `make_readout_grads`
  a dedicated exception type for the triangular-lobe/coverage-bump cases,
  or check the relevant condition directly before calling), and let
  unrelated assertions propagate with their real message.
- [ ] **199. `preprocessing/lowres_calib_recon.py`/`preprocessing/
  r2star_map.py` each paste an undocumented copy of `run_rss.py`'s `_ift3`,
  losing the odd-axis complex-value warning its own source carries --
  and, unlike prior copies of this bug, both new files return the
  complex, not just magnitude, image.** [measured; found 2026-09-16
  against `de3d535`, commit `42edeb3`] `run_rss.py:17-32`'s `_ift3`
  docstring explicitly warns its `fftshift`-in/`fftshift`-out pairing (not
  the canonical `ifftshift`-in) is "NOT shift-equivalent on an odd-length
  axis... safe here only because every consumer... takes a magnitude... a
  future complex-valued consumer would inherit it silently."
  `gre_diagnostics.py` correctly imports this shared function (item 91's
  fix). `lowres_calib_recon.py:83-85` and `r2star_map.py:52-54` instead
  paste an undocumented copy each. `lowres_calib_recon()`'s own module
  docstring cites a real measured example where the calibration-region
  crop is `49 x 10` (`Ny_eff=49`, and `Nx_eff = round(Ny_eff*fov_x/fov_y)
  = 49` too) -- both **odd** -- and `lowres_calib_recon()` returns the
  complex, sensitivity-weighted-combined `img` (not just a magnitude) from
  exactly this odd-sized grid, unlike every prior consumer of this
  function class (item 64's writeup). Verified numerically (`np.allclose`
  on a `7x9x5` odd-axis synthetic case) that the bad/good pairings agree
  on magnitude but diverge on the complex value, both alone and after a
  smaps-weighted coil combine -- confirming the bug is latent today only
  because `lowres_calib_recon.py`'s own `main()` happens to take
  `np.abs()` before writing NIfTI/PNG, exactly the trap the original
  docstring warned about, but that trap is now invisible to a reader of
  either new file, and a future caller reading `img` directly (not
  `np.abs(img)`) would inherit it silently. Same bug class as items
  44/64/91/108/120, now a fifth and sixth occurrence. Fix: delete both
  local copies and `from preprocessing.run_rss import _ift3` in both
  files (as `gre_diagnostics.py` already does), so the warning travels
  with the function.

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
  [measured; re-verified 2026-09-16 against `de3d535`, still open,
  unchanged. Note: item 179 below is a near-duplicate of this same claim
  (same file, same stale-`L=6` text) -- kept as separate items per this
  doc's never-reuse-a-number convention, but treat as one fix] The module docstring (`:9-11`) says "...L=6 (the current
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
  actual post-item-90 implementation in the same file.** [measured;
  citation updated 2026-09-16 against `de3d535` -- write site shifted from
  `:169-171` to `:172`, substance unchanged]
  `run_recon`'s docstring (`reconstruct.py:172`) says the `echo_times`
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
- [ ] **128. `preprocessing/recon_frames.py`'s module docstring claims a
  smaps-cache legacy-format branch is unreachable, but that branch is the
  normal path on every first pipeline run and is already exercised by this
  repo's own tests.** [measured] The docstring (`recon_frames.py:14-17`)
  says: "The smaps-cache legacy-format fallback (`recon_frames.m`'s 'cache
  file has smaps_raw/emaps but no smaps yet' branch) isn't ported either --
  `preprocess.py`, this port's only writer, always writes the full format,
  so that branch can never be reached here." Both halves are false today:
  (a) `recon_frames.py` itself (same file, `:76`) calls
  `preprocessing.smaps.load_smaps()`, which is *also* a writer -- it
  creates the cache from scratch when none is valid
  (`smaps.py:207-215`) -- so `preprocess.py` is not "this port's only
  writer"; this docstring predates `load_smaps` being factored out of
  `recon_frames.py` into `smaps.py` (per `smaps.py`'s own docstring). (b)
  `preprocess.py`'s STEP 3 cache write (`preprocess.py:343-344`) writes
  only `smaps_raw`/`emap` (plus `smaps`, written just above) -- three
  keys, never `smaps_degre`/`emap_degre` -- which `smaps.py:179`'s
  `has_degre = 'smaps_degre' in f and 'emap_degre' in f` check treats as
  part of the "full" format. Since `preprocess.py`'s STEP 3 always runs
  before `recon_frames.py` in the documented pipeline order, the cache
  `recon_frames.py`/`load_smaps` finds on essentially every first run is
  exactly this 3-key "legacy" shape -- the branch this docstring calls
  unreachable is the routine case, handled by `smaps.py:184-192`'s own
  backfill block (which the `smaps.py` docstring correctly documents as
  real). Confirmed live, not hypothetical:
  `tests/test_preprocessing_recon_frames.py`'s `_make_fixture`
  (`:25-44`) writes a cache with exactly these 3 keys and no `fn_gre`,
  so `test_recon_frames_uses_cached_smaps_and_reconstructs_all_frames`/
  `test_recon_frames_caps_at_cfg_nframes` already exercise the backfill
  path on every run, without either the fixture or the assertions calling
  that out. Distinct from item 117 (which flags `preprocess.py`'s STEP 3
  as a *duplicate* of `load_smaps`'s caching logic, not this docstring's
  incorrect claim about which branches are reachable). Fix: reword
  `recon_frames.py:14-17` to describe the real, current relationship (two
  writers, an intentionally-backfilled older cache format), or resolve it
  by fixing item 117 (making `preprocess.py` call `load_smaps` directly),
  which would make the claim true again.
- [ ] **133. The `<seqname>_gre.h5`/`smaps_<seqname>_sigpy.h5` cache paths
  are hand-built with the identical f-string independently in many
  separate files instead of being `SeqPaths` fields, and the count of
  independent call sites keeps growing.** [measured; citations updated
  2026-09-16 against `de3d535` -- `smaps.py`'s two sites shifted from
  `:234`/`:232` to `:376`/`:374` by the crop/mask/GPU/smoothing rewrite;
  `run_b0map.py`'s site is now `:76`; `preprocess.py`/`gre_diagnostics.py`
  unchanged] `<datdir>/recon/<seqname>_gre.h5`'s
  path is independently constructed via `os.path.join(cfg.datdir, 'recon',
  f'{paths.seqname}_gre.h5')` (or the equivalent with a bare `seqname`) in
  `preprocess.py:310`, `smaps.py:376`, `run_b0map.py:76`, and
  `gre_diagnostics.py:39`; `<datdir>/recon/smaps_<seqname>_sigpy.h5`'s path
  independently in `preprocess.py:324`, `smaps.py:374`, and
  `run_b0map.py:91`. `SeqPaths`
  (`preprocessing/config.py`) already centralizes every *other*
  per-sequence path (`scan_info`, `cal`, `noise`, `epi`, `recon`) for
  exactly this reason, but conspicuously omits these two. Two brand-new
  files added by commit `42edeb3` compound this further, rather than
  reusing `SeqPaths`: `preprocessing/lowres_calib_recon.py:179`
  independently re-derives the sigpy smaps path (an 8th site), and
  `preprocessing/r2star_map.py:81-82` independently re-derives *both* the
  GRE cache path (a 9th site) and the `<seqname>_b0map.h5` path (a 3rd
  site for item 188's separate cache-path pattern). Currently harmless --
  confirmed every site still uses the identical format string -- so this
  remains a latent-drift risk, not a live bug: distinct from item 130 (the
  batch-driver *skeleton* duplication) and item 117 (duplicated cache-
  *validity* logic, not path construction). A future rename of either
  cache file's naming convention would require remembering to update
  every one of these independent call sites; a missed one would silently
  break the pipeline (e.g. `smaps.py` looking for a GRE cache at a path
  `preprocess.py` no longer writes to) with no error until a downstream
  stage fails to find its input. Fix: add `gre_cache`/`smaps_cache` fields
  to `SeqPaths` (computed once in `set_seq_paths`, the same place the
  other five paths are built) and update all 9 call sites (plus item
  165's `recon/` pair, and item 188's `b0map_cache` sites) to read them
  instead of re-deriving the filename.
- [ ] **140. `preprocessing/nifti_io.py`'s module docstring caller list is
  stale on two counts: it names a module that no longer calls
  `save_recon_nifti`, and omits one that does and contradicts its
  "always the EPI grid" claim.** [measured; citations updated 2026-09-12
  against `ecb8f2f`] `nifti_io.py:2-9` names
  callers as "run_rss.py/run_cg_sense.py/run_recon_sigpy.py ...
  preprocess.py/recon_frames.py (sensitivity maps ...), and
  run_b0map.py (the field map itself ... on the EPI grid, same as every
  other NIfTI this pipeline writes ...)". [citation updated 2026-09-16
  against `de3d535` -- `smaps.py`'s two calls shifted from `:273,306` to
  `:415,448` by the crop/mask/GPU/smoothing rewrite, substance unchanged]
  A repo-wide grep of
  `save_recon_nifti(` calls shows: (a) `recon_frames.py` never calls
  `save_recon_nifti` -- that responsibility moved to `smaps.py`
  (`smaps.py:415,448`) per `smaps.py`'s own docstring ("was
  `recon_frames.py`'s private `_load_smaps` -- moved here"), so the
  docstring names the wrong module; (b) `gre_diagnostics.py:75` also
  calls `save_recon_nifti` and isn't mentioned at all -- and it passes
  `fov=sp.fov_degre` (`gre_diagnostics.py:76`, the deGRE grid), directly
  contradicting the same sentence's blanket claim that every NIfTI this
  pipeline writes is "on the EPI grid". Severity is low (documentation
  only) -- this is the same docstring item 67 already touched for a
  different sentence in the same file. Fix: replace "recon_frames.py"
  with "smaps.py" in the caller list and add `gre_diagnostics.py`
  (deGRE-grid GRE-echo images), noting it as the one caller not on the
  EPI grid.
- [ ] **141. Addendum to item 117: `preprocess.py`'s STEP 3 never writes
  `smaps_degre`/`emap_degre`, so `smaps.py`'s "legacy cache" backfill
  branch fires on every fresh full-pipeline run, not just an occasional
  older cache.** [measured; citations updated 2026-09-15 against `b701489`,
  substance and liveness confirmed unchanged] Item 117 (still open)
  already flags that
  `preprocess.py`'s STEP 3 hand-rolls a narrower copy of `smaps.py`'s
  `load_smaps()` caching logic instead of calling it directly. A
  concrete, previously-undocumented consequence of that narrowness:
  STEP 3's fresh-estimation branch (`preprocess.py:335-347`, unchanged by
  `b701489` -- that commit's gzip-compression addition lands later in the
  same function) writes only
  `smaps_raw`/`emap`/`smaps` + `Nvcoils` -- it never computes or writes
  `smaps_degre`/`emap_degre`. `smaps.py`'s canonical `load_smaps`
  (`smaps.py:427-444`, was `:286-303` -- `smaps.py` was substantially
  rewritten by `b701489`'s crop/mask/GPU/smoothing changes) always
  computes and writes both
  alongside a fresh
  estimate, and its own docstring (now `smaps.py:370-372`, was `:225-227`)
  describes the
  no-`smaps_degre` case as "an older cache written before these existed"
  that gets "backfilled in place" -- language implying an occasional,
  legacy case. Reproduced directly: writing a cache with exactly STEP 3's
  4-key shape (no `smaps_degre`/`emap_degre`) and then calling
  `smaps.load_smaps()` on it prints "Backfilling deGRE-grid smaps/emap
  into ..." and recomputes them via a second
  `process_smaps`/`resize_to_epi_grid` pass -- every time, for every
  fresh full-pipeline run today, not a rare legacy-cache case. Severity:
  low (the backfill recomputes correctly, so this is wasted duplicate
  work plus a misleadingly-scoped docstring, not a wrong result), but it
  sharpens item 117's own framing ("left to `load_smaps()`'s documented
  backfill path the first time `recon_frames.py` or `run_b0map.py` runs
  later") -- the backfill isn't an occasional fallback, it's the normal
  path on every fresh run. Fix: resolved together with item 117 --
  calling `smaps.load_smaps()` directly from STEP 3 instead of
  duplicating its cache-writing logic fixes both the Nvcoils-check drift
  item 117 already flags and this omission at once.
- [ ] **142. `lib/make_spoilers.py` doesn't share one duration across its
  x/y/z trapezoids, unlike the structurally-identical
  `lib/make_prephasers.py` (already fixed for exactly this reason -- see
  the dangling-but-real item 28, item 113).** [measured, low severity,
  not live at the shipped default; re-verified/updated 2026-09-12 against
  `ecb8f2f` -- `lib/make_spoilers.py` was reworked by commit `8448ff3`
  (the cycles/voxel redesign) to take `res`+`n_cycles_spoil` instead of
  `Nx`/`Ny`/`Nz`/`fov`/a scalar, but the underlying gap is unchanged]
  `lib/make_spoilers.py:23-46` still
  builds `gx_spoil`/`gy_spoil`/`gz_spoil` independently in a loop, each
  via its own `pp.make_trapezoid(axis, ..., area=...)` call with **no
  shared `duration=`** -- the exact pattern `make_prephasers.py` was fixed
  for (item 28/113). The rework changes *when* durations diverge only
  quantitatively, not qualitatively: `area = n_cyc / res_ax` now, and
  since every axis is currently built at the same `n_cyc =
  params.spoil_cycles_max` (see `sequences/ArbEPI.py`'s/`EPIcal.py`'s
  `make_spoilers(params.res, [params.spoil_cycles_max] * 3, ...)` call),
  the three durations still coincide exactly whenever `res` is isotropic
  -- measured now at **4.428 ms** each (up from the old commit's 2.588 ms,
  since `spoil_cycles_max=4.0` replaced the old flat `n_cycles_spoil=2`).
  Reproduced the divergence again by varying only `res` to `[0.9, 0.9,
  1.8] mm`: z-axis spoiler duration drops to **2.340 ms** while x/y stay
  at **4.428 ms** -- same "inert at isotropic default, live under
  anisotropic resolution" situation as before, just with updated absolute
  numbers. Still no `test_make_spoilers.py`. Fix: apply the same
  shared-`duration` construction `make_prephasers.py` uses, and add an
  analogous anisotropic-`res` regression test (this could share
  infrastructure with item 170's fix, which needs the same anisotropic-
  resolution scenario for a related `calc_te_tr_delays` bug).
- [x] **152.** Resolved: `gre_diagnostics.py`'s module docstring now names
  the actual keys read (`finit_hz`/`b0map_hz_degre`/`mask_degre`) and
  states the dependency on `run_b0map.py`'s post-processing having
  completed, not just `b0map.jl` itself; added an explicit `KeyError`-
  avoiding check mirroring the existing `TE_degre`-missing guard.
- [x] **153.** Resolved: CLAUDE.md's data-flow diagram's first arrow now
  reads `resolve_omegas(params)` instead of the stale
  `gen_sampling_masks(R, params)`, matching `main.py`'s actual call.
- [x] **156.** Resolved: added a `discard_duration` row to README's USER
  CONFIGURATION table, between `T1` and `ETL` (matching `load_params()`'s
  real field order).
- [x] **157.** Resolved: CLAUDE.md's `sampling/external_mask.py` paragraph
  now distinguishes the two real `Params` dataclass fields
  (`custom_mask_path`/`custom_omegas`) from the `load_params()`-local
  `custom_mask_key` variable, instead of listing all three as dataclass
  fields.
- [ ] **160. `sampling/caipi_sample.py`'s `balanced_factors` docstring (and
  `tests/test_caipi_sample.py`'s matching comment) mislabels a hypothetical
  example as "this repo's default."** [measured, low severity] Both
  `sampling/caipi_sample.py:36-37` and `tests/test_caipi_sample.py:35` say
  "At (Ny, Nz, R) = (240, 60, 4) (this repo's default `res`)...". The
  repo's actual shipped default is `(Ny, Nz, R) = (240, 45, 9)`
  (`params.py`: `N = [240, 240, 45]`, `R = 9`) -- `(240, 60, 4)` appears
  nowhere else in the codebase; it's a synthetic example chosen, per item
  147's own resolution note, specifically because it "does survive the
  restriction and still demonstrates non-square reweighting" (unlike the
  real `(240, 45, 9)`, which collapses to the less-illustrative `(3, 3)`).
  The parenthetical "(this repo's default `res`)" is presumably meant to
  say only the voxel resolution constant (0.9mm, which does inform the
  FOV-weighting math the example demonstrates) matches the shipped
  default, but as written it reads as claiming the whole `(Ny, Nz, R)`
  tuple is the shipped default, which is false and could mislead a future
  reader into thinking `Nz=45`/`R=9` isn't the real config. Fix: reword to
  something unambiguous, e.g. "(at this repo's default 0.9mm `res`, with a
  hypothetical Nz/R chosen to survive the restriction)", in both
  `caipi_sample.py` and the test file's matching comment.
- [ ] **176. `sequences/ArbEPI.py` and `sequences/EPIcal.py` both seed their
  per-shot spoiler-randomization RNG with the identical literal `0`,
  undocumented as to whether the sharing is intentional.** [verified, low
  severity, design-choice rather than clearly a bug] Commit `8448ff3`
  added `spoil_rng = np.random.default_rng(0)` at both
  `sequences/ArbEPI.py:203` and `sequences/EPIcal.py:85`. Since both are
  freshly seeded with the same constant, the two sequences draw the *same*
  sequence of `(cx, cy, cz)` cycles/voxel triples, just consumed at
  different starting points in their respective per-shot loops (ArbEPI's
  draw index corresponds to `(frame=i//Nshots, shot=i%Nshots)` starting at
  frame 0/shot 0; EPIcal's corresponds to `shot = i - Ndummyshots`
  starting at the first *dummy* shot) -- e.g. EPIcal's dummy shot 0 gets
  the exact same triple ArbEPI's frame-0/shot-0 got. This isn't flagged as
  a correctness bug -- there's no apparent reason EPIcal's spoiler draws
  need to be decorrelated from ArbEPI's for the stated purpose (breaking a
  residual coherence pathway within *one* sequence's own repeated TR
  structure, per `params.py`'s `spoil_cycles_min`/`max` comment) -- but
  it's undocumented, so a future reader could easily mistake the
  duplicated literal `0` for copy-paste residue rather than a deliberate
  choice. Fix: add a short comment beside each `spoil_rng =
  np.random.default_rng(0)` line (or in `params.py`'s
  `spoil_cycles_min`/`max` docstring) stating explicitly that sharing seed
  0 across the two generators is intentional and why.
- [x] **177.** Closed as superseded, 2026-09-15 against `b701489`: the
  code this item cited no longer exists in that form.
  `preprocessing/smaps.py`'s `crop`/mask redesign (`b701489`) removed the
  pre-resize masking step this item's `# 1. Eigenvalue support mask...`
  comment used to label (see the new comment at `smaps.py:259-266`
  explaining why that step became unnecessary) and rewrote the z-crop/
  resize step's comment (now `smaps.py:268-270`) without a leading number
  either -- so the `# 1.`/`# 2+3.` comments this item quoted are gone
  verbatim, not just shifted. See item 192 for the fresh, differently-shaped
  finding this left behind (a single orphaned `# 4.` with no `1`/`2`/`3`
  anywhere above it, which is arguably a worse inconsistency than the one
  this item originally described, not a fix).
- [ ] **178. `preprocessing/smaps.py`'s `_masked_gaussian_smooth` uses
  `scipy.ndimage.gaussian_filter`'s default `mode='reflect'` boundary
  handling, inconsistent with `preprocessing/grid_resize.py`'s documented
  `mode='nearest'` convention for the same pipeline -- confirmed to have
  no live effect, but undocumented.** [verified, very low severity;
  citation updated 2026-09-15 against `b701489` -- `_masked_gaussian_smooth`
  moved to `smaps.py:194-217` and its three unset-`mode` calls to
  `smaps.py:210,214-215` after the surrounding crop/mask redesign;
  substance unchanged] `smaps.py:210,214-215` (originally added by
  `0e4e86e`) call
  `ndimage.gaussian_filter(weight, sigma_vox)` and the matching call on
  the numerator array with no explicit `mode=`, so reflect-padding applies
  at the outermost voxels of the target grid. `grid_resize.py`'s module
  docstring (`grid_resize.py:36-39`) explicitly chose `mode='nearest'`
  over reflect/wrap for the resize step specifically because those "mix in
  wrap-around or reflected samples that make even less physical sense for
  a truncated anatomical/field-map volume" -- the same reasoning would
  apply here. Verified this has no live effect by construction, not just
  by assumption: the object essentially never touches the outer edge of
  the acquisition FOV, and where the mask is 0 at the array edge,
  `_masked_gaussian_smooth`'s `denom` clamps to 1 and `num` is 0, giving
  exactly 0 regardless of padding mode. Fix: pass `mode='nearest'`
  explicitly for consistency with `grid_resize.py`'s stated convention, or
  add a one-line comment explaining why reflect is fine here.
- [ ] **192. `preprocessing/smaps.py`'s `process_smaps` now has a single
  orphaned `# 4. Normalize` comment with no `1`/`2`/`3` anywhere above it
  -- the successor to item 177, left behind by the same crop/mask redesign
  that closed it.** [measured, very low severity] `b701489`'s rewrite of
  `process_smaps` (`smaps.py:220-342`) removed the pre-resize masking step
  item 177's `# 1. Eigenvalue support mask...` comment used to label
  (replaced by an unnumbered explanatory comment at `smaps.py:259-266`
  about why that step is no longer needed) and left the z-crop/resize
  comment (`smaps.py:268-270`, was `# 2+3. Crop z...interpolate...`)
  unnumbered too -- but `# 4. Normalize...` (`smaps.py:337`) survived
  verbatim, still carrying its old number. The module docstring
  (`smaps.py:229`) still advertises five stages ("Mask, z-crop, resize,
  smooth, and RSS-normalize"), and the two stages between resize and
  normalize -- the post-resize re-mask (`smaps.py:308-310`) and the
  Gaussian-smoothing block (`smaps.py:331-335`) -- are both unlabeled, same
  as item 177 already found. What's new: a lone "4." with nothing before
  it reads as though three steps were deleted by mistake (or that there
  are only 4 steps total), which is a more actively misleading signal than
  177's original "numbering undercounts the steps" framing. Fix: either
  drop the leftover "4." (matching the other four now-unnumbered stage
  comments, which is the simpler fix given how much this function has
  already been restructured) or renumber all five stages consistently.
- [ ] **179. `recon/sweep_time_segments.py`'s module docstring and its own
  printed sweep-table marker both still call `L=6` "the current production
  default," stale since item 82 changed the default to `L=32`.**
  [measured, low severity, self-referentially ironic; re-verified
  2026-09-16 against `de3d535`, still open, unchanged. Note: this is the
  same claim as item 112 above (same file) -- see that item's note] The module docstring
  (`recon/sweep_time_segments.py:9-11`) says "...L=6 (the current
  production default, params.py-adjacent choice in
  operators_b0.py/run_b0_recon.py)..." and the sweep report's own printed
  marker (`:148`, `marker = "  <- current default" if L == 6 else ""`)
  labels the `L==6` row as current -- but `recon/operators_b0.py:151`,
  `recon/reconstruct.py:161`, and `recon/run_b0_recon.py:69,149` all
  default `L`/`L_b0` to **32**, matching CLAUDE.md's explicit statement
  that this exact sweep script is what established L=32 as the correct
  choice over the old L=6 default. Both the docstring sentence and the
  `L == 6` marker condition were evidently never updated once that
  conclusion took effect. Severity is low (a one-off analysis script, not
  part of any pipeline or test), but the mislabeled `<- current default`
  marker appears in the script's actual printed output, not just a
  comment, so anyone re-running the sweep today gets a table flagging the
  wrong `L` as current. Fix: update both the docstring sentence and the
  `L == 6` marker condition to `L == 32` (or read the production default
  from a single named constant so this can't re-drift the next time it
  changes).
- [ ] **180. `recon/validate_against_mslr.py`'s module docstring documents
  only the three radial-dataset validation configs, omitting the three
  matching laminar-dataset configs CLAUDE.md documents as validated via
  this same script.** [measured, low severity, documentation only]
  `recon/validate_against_mslr.py:20-33`'s "Validated results (2026-08-25,
  RTX A6000, 20260822ball_radial dataset...)" paragraph lists only the
  radial L/G/G+L rows. CLAUDE.md's `recon/` section documents six configs
  from the same 2026-08-25 validation run -- the same three radial rows
  plus three matching laminar rows (`20260822ball_laminar`) -- explicitly
  stating "all six run via `recon/validate_against_mslr.py`." This isn't
  just a stale copy-paste: the script's own inline comment
  (`recon/validate_against_mslr.py:103`, "Multi-scale reg_cost...
  accumulates more floating-point noise... measured ~1-2e-4 on both real
  G+L runs (radial and laminar)") already references the laminar run's
  measured tolerance directly -- i.e. the laminar validation was
  incorporated into the script's tolerance-setting logic but never added
  to its own results-table docstring. A maintainer reading only this
  file's docstring would incorrectly believe laminar was never validated
  against real MSLR output. Fix: add the three laminar rows (or a pointer
  to CLAUDE.md's fuller six-row table) to
  `recon/validate_against_mslr.py:20-33`.
- [ ] **185. CLAUDE.md's "Plotting" paragraph undercounts
  `plotting/plot_last_run.py`'s functions -- "four" should be "five" -- and
  has been wrong since the sentence was written.** [measured, low severity]
  `CLAUDE.md`'s Plotting paragraph says "`plotting/plot_last_run.py` drives
  all four plotting functions against the most recent `output/` run." But
  `plot_last_run.py:19-23,36-53` actually calls five: `plot_sampling_mask`,
  `plot_psf`, `plot_trajectory`, `plot_one_tr`, and `plot_pns_one_tr`,
  writing `mask.png`, `psf.png`, `trajectory.png`, `one_tr.png`, and
  `PNS_one_tr.png`. Git-blaming both the CLAUDE.md sentence and
  `plot_pns_one_tr`'s definition shows they were introduced in the *same*
  commit (`f7fdf0c`, 2026-08-20) -- so this wasn't a later addition making a
  previously-true "four" go stale, the claim was already off-by-one the
  moment it was written. Distinct from item 114 (README's separate,
  already-tracked omission of `PNS_one_tr.png` from its own file list) --
  this is CLAUDE.md's function-count claim, a different sentence in a
  different file. Fix: change "four" to "five", optionally naming
  `plot_pns_one_tr` alongside `plot_one_tr` the way the paragraph already
  singles out `plot_one_tr`.
- [ ] **188. `<seqname>_b0map.h5`'s cache path is hand-built independently in
  two files -- a third, previously-untracked instance of item 133's already-
  documented pattern.** [measured] `preprocessing/run_b0map.py:76`
  (`output_path = os.path.join(cfg.datdir, 'recon', f'{seqname}_b0map.h5')`)
  and `preprocessing/gre_diagnostics.py:40` (`fn_b0map = os.path.join(
  recon_dir, f"{seqname}_b0map.h5")`) each independently construct the same
  filename pattern, the same way item 133 already documents for
  `<seqname>_gre.h5`/`smaps_<seqname>_sigpy.h5` across `preprocess.py`/
  `smaps.py`/`run_b0map.py`/`gre_diagnostics.py` -- but item 133's own
  citation list doesn't mention `_b0map.h5` at all, and a repo-wide check
  confirms this file's own numbering has never recorded it before now.
  Currently harmless (both sites use the identical `f'{seqname}_b0map.h5'`
  format string) -- the same "latent-drift risk, not a live bug" class item
  133 already describes: a future rename of the b0map cache convention that
  updates one site but not the other would silently break
  `gre_diagnostics.py`'s consumption of a file `run_b0map.py` no longer
  writes there, with no error until that downstream read fails to find its
  input. Fix: fold into item 133's own resolution -- add a `b0map_cache`
  field to `SeqPaths` alongside the `gre_cache`/`smaps_cache` fields that
  item proposes, computed once in `set_seq_paths`, and update both call
  sites above to read it instead of re-deriving the filename.
- [ ] **190. `preprocessing/preprocess.py`'s new gzip compression on
  `ksp_epi_zf` carries no in-code rationale, and the sibling module that
  documents this exact dataset's read performance now silently describes
  data from before the change.** [measured, low-medium severity]
  `preprocess.py:411-412` (`compression='gzip', compression_opts=4`, added
  by `b701489` to the `ksp_epi_zf` dataset's `create_dataset` call) carries
  no comment at all -- but the commit that added it states a concrete,
  measured justification ("a real 210GB file compresses ~117x with gzip
  level 4 (measured on a real frame chunk: 231MB -> 2MB)") that appears
  nowhere in the source tree: not in `preprocess.py` itself, not in
  CLAUDE.md's `.mat`/`.h5` file-format paragraphs (`grep -n
  "gzip\|compress" CLAUDE.md` finds only unrelated "coil compression"
  hits), and not in `recon/reconstruct.py:45-57`'s `_load_array`
  docstring, which documents this exact dataset's own read-performance
  characteristics ("~500 MB/s reading one same-sized chunk slice at a
  time") on data that predates this compression change and doesn't
  mention it's now compressed at all. This repo's own established
  convention (visible throughout CLAUDE.md and nearly every design
  decision in `smaps.py`/`grid_resize.py`) is to record measured rationale
  for exactly this kind of choice directly in code so it survives
  independent of git history -- here a real, well-justified number is
  invisible to anyone reading the source. `tests/test_recon_reconstruct.py`'s
  fixtures also write `ksp_epi_zf` uncompressed and unchunked, so
  `_load_array`'s chunk-by-chunk read path is untested against real
  gzip-compressed data. Fix: add a short comment at `preprocess.py:409-412`
  carrying the ~117x/231MB->2MB measurement (or a pointer to `b701489`),
  and a one-line caveat in `recon/reconstruct.py`'s `_load_array`
  docstring noting the dataset is now gzip-compressed and that the cited
  throughput figure predates that change.
- [ ] **191. README.md's Architecture file tree omits `ge/validate_pns.py`
  from its `ge/` subsection, even though CLAUDE.md cites it by name as
  `ge/pns.py`'s MATLAB-validation script.** [measured, low severity]
  `README.md:209-221` lists all 12 other `ge/*.py` modules with a one-line
  description each (`ge_export.py`, `ceq.py`, `blocks.py`, `seq2ceq.py`,
  `writeceq.py`, `read_pge.py`, `pns.py`, `acoustics.py`, `check.py`,
  `validate_against_matlab.py`, `coppe.py`), but `ge/validate_pns.py`
  (present in the repo -- `ls ge/*.py` lists 13 files, this is the only
  one absent from README's tree) is never mentioned anywhere in README.md
  (`grep -n validate_pns README.md` returns nothing). CLAUDE.md's PNS
  section names it in the same breath as `read_pge.py`/
  `validate_against_matlab.py`: "Both `ge/pns.py` and `ge/acoustics.py`
  match real MATLAB output to float64/float32 precision on identical
  input (`ge/validate_pns.py` + the since-removed
  `dump_pns_test.m`/`dump_acoustics_test.m`)". Severity is low
  (documentation completeness only, same class as item 185's "four should
  be five"), but it's a real, verified gap, not a duplicate (grepped this
  file for `validate_pns`/`ge/ tree` -- no hits). Fix: add a line after
  `read_pge.py` or `pns.py` in `README.md`'s `ge/` tree, e.g.
  `validate_pns.py   Validates ge/pns.py against real MATLAB pge2.pns.m
  output (not a pytest test)`.
- [ ] **201. `preprocessing/r2star_map.py`'s module docstring cites an
  in-repo consumer file that doesn't exist on this branch, without the
  "(unmerged)" qualifier every other reference to it uses.** [measured;
  found 2026-09-16 against `de3d535`, commit `42edeb3`]
  `r2star_map.py:3` says "for the generalized complex field-map correction
  in `recon/lowres_calib_recon_b0complex.py`" with no caveat. Every other
  reference to that filename in the repo (`CLAUDE.md`'s B0-correction
  section, `recon/operators_b0.py`, `recon/run_b0_recon.py`,
  `tests/test_recon_operators_b0.py`) explicitly marks it "(unmerged)"/
  "worktree-lowres-calib-recon branch" -- confirmed the file is genuinely
  absent from this checkout (`find . -name
  'lowres_calib_recon_b0complex.py'` returns nothing). Very low severity,
  documentation-only. Fix: add the same "(unmerged, exploratory branch)"
  qualifier used everywhere else.

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
  guard for item 96's real, previously-shipped PSF bug.** [measured; citation
  sharpened 2026-09-13 -- one supporting sub-claim went stale, substance
  unchanged] `tests/test_plotting.py` now exists (added for items 127/149,
  importing `plotting.plotting.nominal_te_value`) so this item's original
  "no file under `tests/` ... imports `plotting.plotting` at all" clause is
  no longer literally true. But that file only tests the standalone
  `nominal_te_value` helper -- it does not smoke-test or regression-guard
  any of the five actual plotting *functions*, so a repo-wide grep still
  confirms none of `plot_psf`, `plot_trajectory`, `plot_sampling_mask`,
  `plot_one_tr`, `plot_pns_one_tr` is referenced by any test, and this
  item's substantive finding is entirely unaddressed. This matters
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
- [x] **116.** Closed as superseded, no code change needed here. This item
  asked for a `pytest.raises(ValueError)` regression test exercising
  `sampling/ticaipi_sample.py:39-46`'s divisibility guard via its own
  cited repro, `ticaipi_sample([240, 45], 4, 0)`. Item 147's later fix to
  `balanced_factors` (restricting it to only ever return a `(Ry, Rz)` pair
  that evenly divides `(Ny, Nz)`, or raise first) removed the only path by
  which `ticaipi_sample` could reach its own guard with a non-dividing
  pair: `ticaipi_sample([240, 45], 4, 0)` no longer raises at all
  (`balanced_factors([240, 45], 4) == (4, 1)`, which now divides evenly),
  so the guard is provably unreachable through the public API today (the
  code already carries a comment acknowledging this, added alongside item
  147's fix: "kept as cheap defense-in-depth against a future regression
  in that guarantee"). The equivalent invariant this item cared about --
  that a non-dividing split gets rejected somewhere, not silently
  double-sampled -- is already covered by item 147's own
  `test_balanced_factors_raises_when_no_factor_pair_divides`. A test
  targeting `ticaipi_sample`'s own guard directly would need to
  monkeypatch `balanced_factors` to force a non-dividing pair through,
  which tests the guard's existence but not anything a real caller can
  trigger -- not worth the complexity for defense-in-depth code.
- [ ] **129. `recon/save_result.py` has zero test coverage anywhere in the
  repo, including no regression guard for the exact GPU-tensor-ordering
  bug its own docstring says previously destroyed a completed
  reconstruction.** [measured] A repo-wide grep for `save_result` under
  `tests/` finds nothing; `recon/save_result.py` is never imported by any
  test file. Its own module docstring and inline comment
  (`save_result.py:1-10,23-27`) explain a fix baked into the current code:
  `result.X_recon.detach().cpu().numpy()` and the raw-complex `.h5` write
  must both happen *before* handing data to
  `preprocessing/nifti_io.save_recon_nifti` -- "getting that boundary
  wrong once already lost a completed real reconstruction." This is
  distinct from item 115 (plotting/'s coverage gap) and from
  `save_recon_nifti` itself (which *is* tested, in
  `tests/test_preprocessing_nifti_io.py`) -- what's untested is
  `save_result()`'s own orchestration: the CPU/GPU boundary, the `.h5`
  dataset/attrs construction (`X`, `X_recon`, `omega`, `dc_costs`,
  `reg_costs`, `restarts`, `rel_changes`, plus four scalar attrs), and the
  `**result.meta, **extra_attrs` merge into the JSON sidecar (a future
  field added to both `ReconResult.meta` and a caller's `extra_attrs`
  would silently collide as a duplicate-kwarg `TypeError`, also
  untested). Nothing in the suite would catch a future edit that
  reintroduces the CUDA-tensor-into-`save_recon_nifti` ordering bug, or
  reorders the writes, or breaks the meta/extra_attrs merge. Fix: add
  `tests/test_recon_save_result.py` with a small synthetic `ReconResult`
  (gated via `pytest.importorskip("torch")` like the rest of `recon/`'s
  tests, using a CUDA tensor when available and CPU otherwise) asserting
  the `.h5`/`.json`/`.nii.gz` triplet is written correctly and that a CUDA
  `ReconResult.X_recon` doesn't crash the nifti write.
- [ ] **134. `ge/writeceq.py`'s `write_ceq` (the .pge binary writer) and
  `ge/read_pge.py`'s `read_pge` (its read-back counterpart) have zero
  pytest coverage anywhere in the repo -- including no regression guard
  for item 126's confirmed-live bug, which lives inside `write_ceq`
  itself.** [measured] A repo-wide grep (`grep -rln "write_ceq\|writeceq"
  tests/ ge/`) finds `write_ceq` referenced only in `ge/ceq.py` (the
  dataclass it consumes), `ge/ge_export.py` (its one production caller),
  `ge/read_pge.py`/`ge/validate_against_matlab.py` (round-trip
  read-back/MATLAB-comparison tooling) -- never in any file under
  `tests/`. Confirmed by reading `tests/test_ge_check.py` (the only test
  file that imports from `ge.*`) and `tests/test_seq2ceq.py` (the only
  other one) in full: neither imports `ge.writeceq`, `ge.read_pge`, or
  `ge.ge_export`, and `tests/conftest.py`'s `built_seq_dir` fixture (used
  by both) only calls `generate_arbepi`/`generate_noise`, never
  `export_to_ge`/`write_ceq`. The only things that have ever exercised
  `write_ceq` end to end are `main.py --ge` (a full manual CLI run, not
  part of the pytest suite) and `ge/validate_against_matlab.py` (needs a
  local MATLAB install and a fresh MATLAB-generated reference `.pge`,
  neither available in this environment or CI). This matters concretely,
  not just as a coverage-percentage gap: item 126 (still open) documents
  a real, confirmed-reproducible off-by-one bug inside `write_ceq` itself
  (the `NMAXBLOCKSFORGRADHEATCHECK` sliding-window block count, silently
  wrong for any segment whose block count divides 40000 evenly) that no
  test would catch today or after a fix -- the exact same "no regression
  guard for a known bug" pattern already flagged for `plotting/` (item
  115) and `recon/save_result.py` (item 129, directly above). Fix: add
  `tests/test_ge_writeceq.py` with a synthetic small `Ceq` (a handful of
  parent blocks/segments/loop rows, no need for a real `.seq` file) that
  round-trips through `write_ceq` -> `read_pge` and asserts the read-back
  fields match the input `Ceq` -- this would also directly regression-test
  item 126's fix once applied (construct a synthetic `Ceq` whose segment
  block count divides `NMAXBLOCKSFORGRADHEATCHECK` evenly, matching that
  item's own repro).
- [ ] **135. `recon/reconstruct.py`'s entire `fn_b0map` branch in
  `run_recon` -- including the item-93 `sigma1A` auto-measurement and its
  `ValueError` guard -- has zero test coverage.** [measured; citation
  updated 2026-09-16 against `de3d535` -- the branch grew from
  `:202-227` to `:213-247` after commit `1b4704a`'s r2star_map/t_ref_s
  shape-assert block was inserted, substance and test-coverage gap
  unchanged] `tests/test_recon_reconstruct.py` is the only test file
  exercising `run_recon`, and every one of its calls passes `sigma1A`
  explicitly (`sigma1A=1.0`) with no `fn_b0map` argument at all (confirmed
  by reading the file in full, and by `grep -rn
  "_load_echo_times\|_load_normalized_smaps\|fn_b0map" tests/` finding no
  hits outside `recon/reconstruct.py`/`recon/run_b0_recon.py`
  themselves). So none of the following -- all inside
  `reconstruct.py:213-247` -- are exercised by any test: the
  `b0map_hz.shape == (Nx,Ny,Nz)` assert, the `run_recon`-side call into
  `build_encoding_operator_b0` (as opposed to
  `tests/test_recon_operators_b0.py`'s standalone direct calls to that
  function), the `ValueError` raised when both `sigma1A` and `fn_b0map`
  are `None` (item 93's fix), and the auto-`sigma1A`-via-power-iteration
  branch taken when `fn_b0map` is set and `sigma1A` is `None` (also item
  93). The only thing that has ever run this branch end to end is the
  one-off production driver `run_b0_recon.py` against real, uncommitted
  acquisition data -- never in the test suite. Distinct from item 129
  (`save_result.py`, no test file at all) and item 115 (`plotting/`, no
  test file at all) in that here the *surrounding* function (`run_recon`'s
  plain, non-B0 path) is well covered
  (`test_run_recon_smoke`/`test_run_recon_recovers_signal_without_regularization`)
  -- it's specifically the B0-correction branch item 93 added real
  failure-mode/auto-estimate logic to that has no equivalent coverage, so
  a future refactor of that branch (the shape assert, the `ValueError`
  condition/message, or the auto-estimate call) could silently break any
  of the four behaviors above with nothing to catch it. Fix: add a test
  building a small synthetic `fn_ksp`/`fn_smaps`/`fn_b0map` fixture (the
  existing `test_run_recon_smoke` fixture extended with a synthetic
  `b0map_hz` dataset) and asserting `run_recon(..., fn_b0map=...,
  sigma1A=None)` both raises the documented `ValueError` when `fn_b0map`
  is also `None` and successfully auto-measures `sigma1A` and completes
  when `fn_b0map` is set.
- [x] **143.** Closed 2026-09-10: this item's own stated purpose was
  logging a correction already made in the same pass, as a guard against a
  stale cached copy of the old (wrong) claim misleading a future reader --
  not an outstanding code or doc fix. Re-checked this pass: no other file
  in the repo, including CLAUDE.md, ever made the "11 skips are
  GERecon+julia-gated" claim this item corrected (grepped for
  `GERecon`/`skip` across CLAUDE.md -- no matching claim found), so there
  is no stale copy left anywhere for a reader to be misled by. The
  "Current baseline" section above continues to carry the corrected skip
  breakdown every pass. Original text, for the historical record: This
  item's own "Current baseline" skip-count breakdown was misattributed --
  corrected in place [the originating pass], logged here so a
  cached/historical copy doesn't mislead a future reader. [measured]
  Previous "Current baseline" sections here stated the pytest suite's 11
  skips (`--extra preprocessing` synced) are "gated on the real `GERecon`
  SDK and a `julia` executable, neither available in this environment,"
  with `recon` extras (`torch`/`mirtorch`) called out as "not
  re-measured" separately -- implying the 11 are GERecon+julia-only.
  Running `uv run pytest -q -rs` (the same sync) and reading every skip
  reason directly shows this was wrong on both counts: of the 11 skips,
  **6** are `could not import 'torch'` (all six `tests/test_recon_*.py`
  files -- i.e. the `recon` extra, already inside the "11" being
  described) and **5** are `julia executable not found on PATH` (all in
  `tests/test_preprocessing_run_b0map.py`). **Zero** are GERecon-gated --
  a repo-wide grep confirms there is no test file for `raw_io.py` and no
  `pytest.importorskip("GERecon")` anywhere under `tests/`, so GERecon is
  never exercised by the suite at all, skipped or otherwise, not merely
  "unavailable in this environment." This pass also installed the
  `recon` extras fresh (`uv sync --extra recon` succeeds cleanly:
  `torch==2.13.0+cu130` CPU-only, `mirtorch==0.3.1`) and confirmed all 34
  `tests/test_recon_*.py` cases pass with them active -- see the
  "Current baseline" section above for the corrected figures. Logged as
  a numbered item, matching item 111's precedent (a stale baseline
  figure gets both a table correction and a backlog entry), since a
  reader citing the old sentence from git history would draw the wrong
  conclusion about what `uv sync --extra recon` additionally covers.
- [ ] **144. `lib/calc_te_tr_delays.py`'s documented warn-not-raise
  fallback -- a load-bearing design decision CLAUDE.md calls out
  explicitly -- has zero test coverage anywhere in the suite.**
  [measured] CLAUDE.md states: "`calc_te_tr_delays.py` only warns, never
  raises, if the prescribed `TE`/`TR` are unachievable -- it silently
  falls back to zero padding delay, so the sequence still builds with
  the *wrong* TE/TR baked in." `lib/calc_te_tr_delays.py:51-57` and
  `:70-76` are the two `warnings.warn(...)` + zero-delay-fallback sites
  this describes, confirmed still exactly matching that description by
  direct manual exercise (`TE=1ms` against default params correctly
  warns "Minimum achievable TE (32.104 ms) exceeds prescribed TE (1.000
  ms)." and falls back to `te_delay=0.0`). But a repo-wide grep for
  `pytest.warns` across the entire `tests/` directory returns zero
  matches -- no test anywhere in this suite ever asserts on a
  `warnings.warn` call, and there is no dedicated
  `tests/test_calc_te_tr_delays.py` at all; every existing
  sequence-generation test uses default or otherwise-achievable timing,
  so neither the TE nor the TR warn-and-fallback branch is ever
  exercised. This matters concretely: CLAUDE.md's very next paragraph
  tells a reader not to "hand-derive feasibility... call
  `calc_te_tr_delays` directly (or scan across candidate `ETL` values)
  to check" -- i.e. this warn-not-raise behavior is meant to be relied
  on directly by callers sweeping parameters, exactly the kind of
  documented-but-unguarded behavior items 115/116/129/134/135 already
  flag elsewhere in this codebase. A future refactor could silently flip
  this to raise, or break the zero-delay fallback, with nothing in the
  suite to catch it. Fix: add `tests/test_calc_te_tr_delays.py` with
  `pytest.warns(UserWarning, match=...)` cases for both the TE- and
  TR-unachievable branches, asserting the returned `te_delay`/`tr_delay`
  is `0.0` in each case.
- [ ] **161. `preprocessing/run_b0map.py`'s item-151 fix (widening the
  per-sequence try/except to wrap the whole batch-driver body, not just
  the julia subprocess call) has no regression test, despite the pattern
  now being testable without either Julia or GERecon.** [measured] All 5
  tests in `tests/test_preprocessing_run_b0map.py` carry a module-level
  `pytestmark = pytest.mark.skipif(shutil.which('julia') is None, ...)`,
  and none exercises a *post*-subprocess failure (e.g. `resize_to_epi_grid`
  or the `.h5`/NIfTI write raising) to confirm the batch driver catches it
  and continues to the next sequence, rather than crashing the whole batch
  -- exactly the behavior item 151 changed (previously only
  `subprocess.CalledProcessError` was caught around the julia call itself;
  now the entire per-sequence body is, `run_b0map.py:105-162`). Verified
  directly with a Julia-independent reproduction (mocking
  `shutil.which`/`subprocess.run`/`resize_to_epi_grid`/`load_smaps`, no
  real Julia or GERecon needed): `run_b0map()` prints `ERROR [seq1]:
  boom\nSkipping...` when `resize_to_epi_grid` raises, and still completes
  the batch (`Batch complete.`) rather than propagating the exception --
  so the fix is real and correct, but entirely unguarded, the same "no
  regression test for a just-fixed real behavior" pattern items 129/134/135
  already flag elsewhere in this repo. Severity: low (doesn't change
  current behavior, since the fix is already correct) -- cheap to fix: add
  a Julia-independent test to `tests/test_preprocessing_run_b0map.py` (or
  a new file outside the `julia`-skip gate) that monkeypatches
  `subprocess.run`/`resize_to_epi_grid`/`load_smaps` the way the repro
  above does and asserts the batch survives a mid-sequence failure and
  prints the expected `ERROR .../Skipping...` message.
- [ ] **162. `preprocessing/gre_diagnostics.py` has zero test coverage
  anywhere in the repo, including for the two `KeyError`-avoiding guard
  clauses item 152 just added.** [measured, low severity] A repo-wide grep
  confirms no `tests/test_preprocessing_gre_diagnostics.py` exists and no
  other test file imports `preprocessing.gre_diagnostics`. Item 152's fix
  added a `KeyError`-avoiding check for `TE_degre` (already existed,
  copied from item 78's pattern) and a brand-new one for
  `b0map_hz_degre` (`gre_diagnostics.py:82-89`) -- neither is exercised by
  any test. Same "one-off diagnostic script with no test file" gap item
  115 already documents for `plotting/`, not yet flagged for this file.
  Fix direction: a small synthetic-fixture test (mirroring
  `run_b0map.py`'s own test fixtures) asserting both `ValueError` guards
  fire on malformed inputs, and that `main()` completes and writes the
  expected PNG/NIfTI files on well-formed ones.
- [ ] **163. `recon/operators_b0.py`'s frame-shared `c_phasors`/`b_by_echo`
  tensors -- the fix for a documented real CUDA-OOM bug -- have no
  regression test for the sharing/object-identity property that fix
  depends on.** [measured; citation updated 2026-09-16 against `de3d535`
  -- shifted from `:213-221` to `:216-224`, substance unchanged]
  `operators_b0.py:216-224`'s own docstring
  explains: an earlier version built an independent `(L,*N)` `c_phasors`
  copy per frame, and at this repo's real scale that redundancy alone was
  large enough (`L=16`) to push a real reconstruction into a CUDA OOM. The
  fix (current code) constructs `c_phasors`/`b_by_echo` once in
  `build_encoding_operator_b0` and passes the *same object* into every
  frame's `GatheredSenseB0`. Verified this sharing currently holds:
  `all(f.c_phasors.data_ptr() == A.A[0].c_phasors.data_ptr() for f in
  A.A)` and the same for `b_by_echo` both come back `True` for a freshly
  built operator. But no test anywhere checks `data_ptr()`/tensor-identity
  across frames -- `test_build_encoding_operator_b0_matches_manual_per_frame_construction`
  and its siblings only check output *values* match, which would pass
  equally well if a future refactor accidentally cloned `c_phasors`/
  `b_by_echo` per frame and silently reintroduced the OOM this fix exists
  to prevent. Same "documented real bug, no test guards the fix" pattern
  as items 129/134/135/161/162. Fix: add an assertion in an existing
  `build_encoding_operator_b0` test that every frame's
  `.c_phasors`/`.b_by_echo` share `data_ptr()` with frame 0's.
- [ ] **164. `recon/operators_b0.py`'s `_check_b_weight_row_sums` --
  the detector for a real, documented signal-loss/incoherent-noise bug --
  is never tested actually firing on a bad input.** [measured; citation
  updated 2026-09-16 against `de3d535` -- shifted from `:118-141` to
  `:120-144`, substance unchanged]
  `operators_b0.py:120-144`'s docstring explains this check exists
  specifically because `nbins=20` was confirmed as the root cause of a
  real ill-conditioned-segmentation-fit failure on real reconstructions
  (see CLAUDE.md's `nbins` paragraph). The only test that touches it,
  `test_production_nbins_avoids_row_sum_warning`, asserts the warning does
  *not* fire at `nbins=128` (the production default) -- it never asserts
  the warning *does* fire at a known-bad `nbins` (e.g. 20, the exact value
  the docstring blames). Every other test in the file builds operators via
  a local `_build_b0_operator` helper that calls `mri_exp_approx` directly
  and never reaches `build_encoding_operator_b0`/`_check_b_weight_row_sums`
  at all. Confirmed the detection logic does still work today (reproduced
  the warning firing with a small `nbins=10` synthetic case at real-scale
  field-map range), but a future change that weakens or inverts the
  threshold (e.g. a `tol` sign flip, or the check silently becoming a
  no-op) would pass the entire suite undetected -- the same
  "fix exists, positive case untested" gap as item 163 just above. Fix:
  add a test building `build_encoding_operator_b0(..., nbins=20)` (or
  similarly coarse) at a scale reproducing the asymmetric in-object range
  from CLAUDE.md's `nbins` paragraph, asserting `pytest.warns` fires.
- [ ] **172. The new per-shot spoiler-cycles randomization and
  `gx_residual` net-kx cancellation (`sequences/ArbEPI.py`/`EPIcal.py`,
  commit `7af04d4`) has zero test coverage despite real, non-trivial
  arithmetic.** [measured -- verified correct by hand/script this pass,
  but unguarded] `grep -rn "spoil_cycles|gx_residual|spoil_rng" tests/`
  returns nothing. This is exactly the kind of gradient/blip-logic change
  CLAUDE.md says `tests/test_trajectory_matches_schedule.py` exists to
  guard ("this is the test to extend when changing gradient/blip logic"),
  yet that file wasn't extended here. This pass independently verified two
  key correctness properties by direct calculation against
  `Sequence.calculate_kspace()` (not just by reading the code): (a)
  `gx_residual = gx_pre.area * rg.gx_pre_scale * (1 if ETL % 2 == 0 else
  -1)` matches the true net kx position after `gx_pre + gro1 + the full
  readout train` exactly (to float precision) for both even and odd `ETL`;
  (b) `pp.scale_grad(gx_spoil, (x_scale * gx_spoil.area - gx_residual) /
  gx_spoil.area)` is algebraically self-consistent -- it delivers physical
  area `x_scale*gx_spoil.area - gx_residual`, which added to the
  pre-spoiler kx position (`gx_residual`) lands exactly at the intended
  `x_scale*gx_spoil.area`, in both `ArbEPI.py:276` and `EPIcal.py:154`.
  So there's no live bug today, but none of this is guarded by a
  regression test -- a future edit to any of these formulas (a sign flip,
  an off-by-one in the scale-factor derivation) would not be caught by the
  existing suite. Fix: add (a) a k-space-based test asserting the
  post-readout-spoiler-block kx/ky/kz position matches the intended
  `x_scale`/`y_scale`/`z_scale * area` targets (modulo item 138's known
  y/z offset) for a couple of `ETL` parities, and (b) a
  `make_spoilers.py`-level duration/area unit test per item 142.
- [ ] **173. `ge/coppe.py`'s hop-2 SSH-failure fix (commit `3de4d58` --
  the `BatchMode`/`PreferredAuthentications`/`-q` change in
  `_TRANSFER_SCRIPT`, and the new `_ssh_env()` helper) has zero test
  coverage, matching the "no regression test for a just-fixed bug" pattern
  items 129/134/135/161 already flag elsewhere in this repo.** [measured]
  `grep -rn "coppe" tests/` finds only `tests/test_coppe_assign.py`, whose
  own docstring explicitly disclaims the remote-facing functions ("thin
  wrappers around subprocess calls to a real scanner and aren't exercised
  here") and whose imports (`assign_entry_numbers, find_pge_files,
  split_reused_files, stage_entry_files`) never touch `_ssh_env`,
  `_TRANSFER_SCRIPT`, `run_remote`, or any ssh/scp-invoking function.
  Unlike the genuinely network-dependent functions that file's docstring
  excuses, both new pieces here are trivially unit-testable with zero
  network/SSH dependency: `_ssh_env()` is a pure function (assert it
  strips exactly `{DISPLAY, SSH_ASKPASS, SSH_ASKPASS_REQUIRE}` from a
  supplied env dict and leaves everything else untouched), and
  `_TRANSFER_SCRIPT` is a plain module-level string (assert it contains
  `BatchMode=yes` and `PreferredAuthentications=publickey` and that its
  `scp` invocation no longer carries a bare `-q`). Given item 171 above,
  such a test would also directly regression-guard against silently
  reintroducing `-q` on the hop-2 leg, or against a future edit forgetting
  to route a new subprocess call through `_ssh_env()`.
- [ ] **174. `preprocessing/preprocess.py`'s STEP 3 smaps branch --
  including the new `smooth_sigma_mm` threading added by `0e4e86e` -- has
  no dedicated test.** [measured, low severity] A repo-wide grep confirms
  `tests/test_preprocessing_preprocess.py` has zero references to
  `smaps`/`process_smaps`/`estimate_smaps` -- STEP 3's cache-validity
  check, its `process_smaps` call (now including
  `smooth_sigma_mm=cfg.smaps_smooth_sigma_mm`, `preprocess.py:341`), and
  its narrower write (see item 117's sharpened entry above) are entirely
  untested in isolation; they only run implicitly whenever `preprocess()`
  itself is exercised end-to-end (GERecon-gated, so effectively never in
  this suite). This means the new `smooth_sigma_mm` threading was verified
  by code reading this pass, not by any test. Fix: a light regression test
  the way `tests/test_preprocessing_recon_frames.py` covers `load_smaps`'s
  call sites, or -- better -- fold into whatever eventually resolves item
  117 (calling `load_smaps` directly from STEP 3 would make this moot).
- [ ] **175. `preprocessing/smaps.py`'s two new Gaussian-smoothing tests
  don't exercise an anisotropic target grid, the one shape of bug the
  physical-mm sigma conversion could plausibly hide.** [measured, low-
  medium severity; citation updated 2026-09-15 against `b701489` -- the
  two tests moved to `tests/test_preprocessing_smaps.py:104-142`/`:145-196`
  (was `:100-133`/`:136-187`) and the `vox_mm`/`sigma_vox` lines moved to
  `smaps.py:332-333` (was `:190-191`) after the surrounding crop/mask
  redesign; substance unchanged] Both new tests (`tests/test_preprocessing_smaps.py:104-142`,
  `:145-196`, added by `0e4e86e`) use isotropic `fov`/`n_target`
  (`(0.2,0.2,0.2)`/`(40,40,40)` and `(0.18,0.18,0.18)`/`(91,91,91)`).
  `smaps.py:332-333`'s `vox_mm = np.array(fov) / np.array(n_target) *
  1000; sigma_vox = smooth_sigma_mm / vox_mm` is only meaningfully tested
  when x/y/z voxel size is identical -- a per-axis mix-up (e.g. swapping
  `fov`/`n_target` order, or using the wrong tuple) would silently pass
  both current tests. This repo's own real acquisitions are *not*
  isotropic (e.g. CLAUDE.md's `240,240,45` grids -- z is coarser than x/y
  by ~5x), so this is exactly the untested regime. Checked the arithmetic
  by hand for a realistic case (`fov=(0.24,0.24,0.1013)`,
  `n_target=(240,240,45)` -> `vox_mm~=(1.0,1.0,2.25)` -> `sigma_vox~=
  (6.0,6.0,2.67)`, physically sensible -- larger index-space sigma on the
  finer axis to keep the same mm-scale blur) -- no live bug today, just an
  untested axis-order assumption. Fix: add one anisotropic-grid test
  (e.g. asserting the fitted/observed smoothing extent in physical units
  is similar across axes despite differing voxel size, or directly
  asserting `sigma_vox` per-axis via a stub).
- [ ] **187. `sampling/gen_sampling_masks.py`'s `'rand'` sampling method has
  zero test coverage through its actual dispatch path.** [measured] The
  `'rand'` branch (`gen_sampling_masks.py:70-72`, including the
  `rand_gaussian_sigma = np.array([Ny, Nz]) / 6` default at `:45-47`) wires
  `gen_gaussian_pdf`'s weight map into `rand_sample` -- but
  `tests/test_gen_sampling_masks.py` only ever sets `sampling_method` to
  `'caipi'`, `'ticaipi'`, `'pd'`, or a deliberately-invalid `'bogus'` (for
  the unknown-method error case); a repo-wide grep confirms no call
  anywhere (not in `tests/`, not in `main.py`) ever exercises
  `gen_sampling_masks(..., params)` with `params.sampling_method ==
  'rand'`. `rand_sample.py`/`gen_gaussian_pdf.py` each have their own
  dedicated unit tests (`tests/test_rand_sample.py`,
  `tests/test_gen_gaussian_pdf.py`), so the individual building blocks are
  covered, but the actual integration through `gen_sampling_masks` --
  including the `rand_gaussian_sigma=None` default-fallback path -- is not.
  `'rand'` isn't the shipped default (`'pd'` is), so this is latent, not
  live, but it's a real, documented, user-selectable config option running
  with no regression coverage at all, unlike its three siblings. Fix: add
  a `test_gen_sampling_masks_rand_*` case to `tests/test_gen_sampling_masks.py`
  analogous to the existing `caipi`/`ticaipi`/`pd` ones (shape, dtype,
  sample count, and that the default `rand_gaussian_sigma=None` path
  doesn't crash).
- [ ] **189. `preprocessing/smaps.py`'s new `_default_device()` GPU/CPU
  auto-selection has zero test coverage.** [measured, low severity]
  `_default_device()` (`smaps.py:31-45`, added by `b701489`) decides
  GPU-vs-CPU dispatch for every real ESPIRiT calibration call via
  `sp.config.cupy_enabled` + `cupy.cuda.runtime.getDeviceCount()`, and is
  now the implicit default (`estimate_smaps(..., device=None)`) for the
  entire sensitivity-map pipeline. `grep -n "_default_device\|device="
  tests/test_preprocessing_smaps.py` finds no reference -- neither branch
  (cupy-enabled-with-a-visible-device vs. the CPU fallback) is exercised
  by an explicit test; `estimate_smaps`'s own tests call it only with the
  default `device=None`, implicitly running whichever branch this
  environment happens to hit (almost certainly the no-cupy/CPU fallback)
  with no assertion on which path was taken. Severity is low (a short,
  straightforward function), but it's the entry point that decides GPU
  dispatch for every real ESPIRiT run in the pipeline, and a regression
  here (e.g. a typo in the `cupy.cuda.runtime.getDeviceCount()` call)
  could silently always fall back to CPU, or crash on a machine with cupy
  installed but no visible device, without any test catching it -- the
  same "no regression test for a just-added real behavior" pattern items
  161/174/175/178 already flag elsewhere in this same file/module. Fix:
  add tests that monkeypatch `sp.config.cupy_enabled`/a stub `cupy` module
  and assert `_default_device()` returns `sp.cpu_device` when cupy is
  disabled and `sp.Device(0)` when a device is reported present.
- [ ] **198. `lib/readout_from_params.py`'s new acoustic-resonance-band
  dwell-avoidance code (`find_min_feasible_dwell`/
  `_echo_spacing_in_forbidden_band`) has zero test coverage despite real
  physical-safety framing.** [measured; found 2026-09-16 against
  `de3d535`, commit `de3d535` itself (item 194's fix)] This is a
  substantial feature (+105 lines) motivated by "a genuine hardware-damage
  risk on real scanner gradient coils, not just a modeling nicety, per
  explicit user information" (CLAUDE.md's item-194 paragraph), but
  `grep -rn "find_min_feasible_dwell\|_echo_spacing_in_forbidden_band\|
  ACOUSTIC_MARGIN_US" tests/` finds no direct test -- only incidental
  mentions in `test_trajectory_matches_schedule.py`'s docstrings. No test
  exercises the acoustic-band-skipping branch itself (e.g. constructing a
  case where the geometrically-cheapest dwell lands in a forbidden band
  and asserting the search steps past it), matching this repo's own
  recurring "fix/feature landed, no regression test" pattern (items
  129/134/135/161-164/172/173/189). Fix direction: add a test with a
  synthetic `params.spec.ge_coil`/band table (or monkeypatched
  `_ESP_BANDS_US`) where the first feasible dwell's `gro` duration falls
  inside a forbidden band, asserting `find_min_feasible_dwell` returns the
  next dwell instead.
- [ ] **202. `preprocessing/lowres_calib_recon.py`/`preprocessing/
  r2star_map.py` have zero test coverage.** [measured; found 2026-09-16
  against `de3d535`, commit `42edeb3`] No
  `tests/test_preprocessing_lowres_calib_recon.py`/
  `test_preprocessing_r2star_map.py` exists, and no other test imports
  either module (confirmed by grep). Same "no regression test for
  just-added real behavior" pattern items 161/162/174/175/178/189/198
  already document elsewhere in `preprocessing/`. Given item 199 above,
  a test on an odd-sized synthetic grid asserting `lowres_calib_recon()`'s
  magnitude output is correct (and ideally regression-locking that fix)
  would be the highest-value addition.

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
  duplicate is already narrower and drifting.** [verify; citation updated
  2026-09-15 against `b701489` -- item 41's `Nvcoils` check moved to
  `smaps.py:386-392` (was `:244-250`) after the crop/mask/GPU/smoothing
  redesign; substance unchanged] `preprocess()`'s
  STEP 3 (`preprocess.py:323-353`) hand-rolls the same "check cached
  `Nvcoils` attr, load-or-estimate-and-cache" pattern
  `smaps.load_smaps()` (used by `recon_frames.py:76`) already implements
  -- but narrower: it never computes/writes `smaps_degre`/`emap_degre`
  (the deGRE-grid maps `run_b0map.py` needs), leaving that to
  `load_smaps()`'s documented backfill path the first time
  `recon_frames.py` or `run_b0map.py` runs later. Not a correctness bug
  today (the backfill path is real and tested), but it's duplicated
  cache-validity logic in two places that can already drift: item 41's
  fix made `smaps.py:386-392` compare `int(f.attrs['Nvcoils'])`
  against a freshly-read `ksp_gre.shape[-1]`, while `preprocess.py:330`
  still compares `f.attrs.get('Nvcoils') == Nvcoils` -- similar but not the
  same check, with no test pinning them to identical behavior. Since
  `paths.recon`'s GRE cache (read by `load_smaps`) is the very file STEP 2
  just wrote moments earlier, `preprocess.py` could call `load_smaps(cfg,
  paths, seq_params)` directly instead -- which would also produce a
  complete cache (with `smaps_degre`/`emap_degre`) on the very first run
  rather than deferring that to a later backfill.

  **Sharpened 2026-09-12** (`ecb8f2f`): `smaps.py`'s new
  edge-smoothing feature (`process_smaps`'s `smooth_sigma_mm` parameter,
  added by `0e4e86e`) makes this duplication concretely worse, not just
  theoretically riskier. `preprocess.py`'s STEP 3 is now a **fourth**
  call site (on top of `load_smaps`'s three) that has to be kept in sync
  with `process_smaps`'s growing signature by hand -- `preprocess.py:341`
  does correctly thread `smooth_sigma_mm=cfg.smaps_smooth_sigma_mm`
  through today (verified: all 4 production `process_smaps(` call sites
  repo-wide pass it consistently, so a fresh STEP 3 estimate's
  smoothing/masking behavior is currently functionally identical to one
  produced via `load_smaps`), but this is a demonstrated instance of
  exactly the drift risk this item already warns about -- the
  `Nvcoils`-check divergence noted above has already persisted through
  this change unnoticed, and a future `process_smaps` parameter could
  just as easily be missed in one of the 4 sites next time. `preprocess.py`'s
  STEP 3 smaps branch itself also has no dedicated test
  (`tests/test_preprocessing_preprocess.py` has zero references to
  `smaps`/`process_smaps`/`estimate_smaps`), so nothing would catch such a
  miss either -- see item 174.
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
- [ ] **130. The Stage-2 batch-driver skeleton is duplicated near-verbatim
  across five files.** [measured] `preprocessing/run_preprocessing.py`,
  `run_rss.py`, `run_cg_sense.py`, `run_recon_sigpy.py`, and
  `run_b0map.py` all share the identical outer skeleton: `print(f'Batch:
  {len(cfg.seqnames)} sequence(s) in {cfg.datdir}')`, then `for i,
  seqname in enumerate(cfg.seqnames, start=1): print(f'\n[{i}/{len(cfg.
  seqnames)}] {seqname}')`, a per-sequence `try/except Exception as e:
  print(f"ERROR [{seqname}]: {e}\nSkipping...")` (each carrying the same
  `# noqa: BLE001` comment citing "mirrors the sibling batch drivers'
  try/catch"), and a trailing `print('\nBatch complete.')`. In the three
  Stage-2 recon drivers (`run_rss.py:39-57`, `run_cg_sense.py:18-46`,
  `run_recon_sigpy.py:15-49`) the duplication runs deeper: each also
  repeats `paths = set_seq_paths(...)`, `seq_params =
  load_seq_params(paths)`, an `out_dir = os.path.join(cfg.datdir, 'recon',
  'basic'); os.makedirs(out_dir, exist_ok=True)`, a call into
  `recon_frames(cfg, paths, seq_params, recon_fn)`, and a
  `save_recon_nifti(fn_recon, img, ..., seqname=seqname,
  runtime_s=runtime_s, **sp)` call -- differing only in the `recon_fn`
  construction and the extra kwargs passed to `save_recon_nifti`.
  Confirmed by direct side-by-side comparison of `run_rss.py`/
  `run_cg_sense.py` -- the shared structure is real, unambiguous
  duplication (not superficial similarity), and the drivers' own comments
  already acknowledge they're siblings of one another. (Re-verified
  2026-09-10: `run_b0map.py`'s own try/except was widened by item 151's
  fix to wrap its entire per-sequence body, now spanning lines 105-162 --
  it still carries the identical `# noqa: BLE001`/print-message pattern
  described above, so this item's substance is unchanged, only that one
  citation's line range moved.) Fix direction: a
  shared helper (e.g. a `_run_batch(cfg, make_recon_fn, fn_recon_name,
  extra_attrs)` in a small shared module, or a decorator/context-manager
  wrapping the per-sequence try/except+prints) could factor out the outer
  skeleton across all five drivers and the inner recon-specific portion
  across the three Stage-2 drivers.
- [ ] **145. `recon/`'s `_complex_randn` test helper is duplicated
  verbatim in two test files despite an established cross-file-reuse
  precedent right next to it.** [measured, low severity]
  `tests/test_recon_operators.py:15-19`,
  `tests/test_recon_b0_correction.py:30-34`,
  `recon/sweep_time_segments.py:49-53`, and
  `recon/benchmark_b0_cost.py:48-52` each independently define the
  identical 4-line function:
  ```python
  def _complex_randn(*shape, seed):
      g = torch.Generator(device=DEVICE).manual_seed(seed)
      real = torch.randn(*shape, generator=g, device=DEVICE)
      imag = torch.randn(*shape, generator=g, device=DEVICE)
      return (real + 1j * imag).to(torch.complex64)
  ```
  Meanwhile `tests/test_recon_operators_b0.py:19` already does
  `from tests.test_recon_b0_correction import DEVICE, _complex_randn,
  _setup` -- i.e. this file set has already established cross-file
  import (not redefinition) as its own convention for this exact helper,
  making the other three copies an inconsistency with that established
  pattern, not merely incidental similarity. The two `recon/` script
  copies (`sweep_time_segments.py`, `benchmark_b0_cost.py`) each carry an
  explicit, defensible comment for why they don't import from `tests/`
  (staying a standalone analysis script with no test-suite dependency) --
  those two are fine as-is and shouldn't change. The two `tests/`-side
  copies have no such stated reason. Verified all four bodies are
  byte-identical (no behavioral risk either way) and that the full
  `tests/test_recon_*.py` suite (34 cases) passes with the real
  `torch`/`mirtorch` extras installed. Fix: have
  `tests/test_recon_operators.py` import `_complex_randn`/`DEVICE` from
  `tests/test_recon_b0_correction.py`, the same way
  `test_recon_operators_b0.py` already does, eliminating the one
  unjustified duplicate.
- [x] **154.** Resolved: dropped `ScannerSpec.psd_rf_wait`/`psd_grd_wait`
  and their per-scanner values/comments (zero readers anywhere in the
  repo, no near-term consumer found in `scanners.py`'s own comments).
- [x] **155.** Resolved: dropped `N`/`N_degre`/`res_degre`/`T1`/`duration`
  from the `Params` dataclass (and the matching `Params(...)` constructor
  kwargs) -- `load_params()` still computes them as plain local variables
  to derive `fov`/`Nx`/`Ny`/`Nz`, `fov_degre`/`Nx_degre`/etc., `fa`/
  `alpha_degre`, and `Nframes`. Verified `load_params()` still runs and
  the full test suite still passes.
- [x] **158.** Resolved: `ge/seq2ceq.py`'s pass-1 loop now uses
  `get_block_type(b).has_trid` for the TRID-presence check instead of a
  duplicated `if b.label is not None:` scan (the value-extraction loop
  itself is unchanged).
- [ ] **165. `recon/run_b0_recon.py`'s `ArbEPI_epi_zf.h5`/
  `smaps_ArbEPI_sigpy.h5` cache-path construction is duplicated verbatim in
  `recon/validate_against_mslr.py`.** [measured, low severity]
  `run_b0_recon.py:72-73` and `validate_against_mslr.py:132-133` each
  independently build `os.path.join(recon_dir, "ArbEPI_epi_zf.h5")` /
  `os.path.join(recon_dir, "smaps_ArbEPI_sigpy.h5")` from a
  locally-derived `recon_dir`, instead of sharing a helper or reading from
  `preprocessing/config.py`'s `SeqPaths`. Same flavor as item 133
  (preprocessing-side cache-path duplication across `preprocess.py`/
  `smaps.py`/`run_b0map.py`/`gre_diagnostics.py`), just on `recon/`'s own
  one-off driver scripts, not previously flagged there. Severity is low:
  both are standalone, uncommitted-real-data-dependent scripts (not part
  of any automated pipeline), and the two copies are currently
  byte-for-byte consistent -- but a future rename of either cache file's
  naming convention (the same risk item 133 already documents for its own
  seven call sites) would need remembering to update this pair too. Fix:
  fold into item 133's fix if `SeqPaths` grows `gre_cache`/`smaps_cache`
  fields, or otherwise factor the two literals into one shared constant/
  helper these two scripts both import.
- [ ] **186. `recon/run_b0_recon.py` carries its own independent copy of
  `_load_omega` instead of importing `recon/reconstruct.py`'s, inconsistent
  with the sibling helpers items 90/94 already deduplicated the same way.**
  [measured, low severity -- not a live bug, item 74 already fixed the
  correctness issue in both copies] `run_b0_recon.py:36` already imports
  `_load_array, _load_echo_times, _load_normalized_smaps, run_recon` from
  `recon.reconstruct` -- three helpers items 90 and 94 each added "next to
  `_load_omega`" specifically so both driver scripts would share one
  implementation. But `_load_omega` itself is not among the imports;
  `run_b0_recon.py:41-66` instead defines its own separate function whose
  own docstring says "Mirrors `recon/reconstruct.py`'s own `_load_omega`" --
  one returns a `torch.Tensor` already on-device (`reconstruct.py:69-103`),
  the other a `numpy.ndarray` (`run_b0_recon.py`, converted to a tensor by
  its caller at `:93`). Item 74's own resolution text (still describing this
  as fixed) explicitly frames the change as making `run_b0_recon.py`'s copy
  "mirror `reconstruct._load_omega`'s non-fallback path" -- i.e. it kept two
  parallel implementations in sync by hand, even as items 90/94 eliminated
  the equivalent duplication for the two functions sitting right next to it
  in the same file. Same duplication class item 74 itself named for this
  exact function, just never finished. Fix: have `run_b0_recon.py` import
  `_load_omega` from `recon.reconstruct` (converting to numpy at the call
  site if needed) the same way it already imports the other three helpers,
  removing its own copy -- or, if the torch/numpy return-type difference is
  deliberate, factor out one shared core (read `omegas` + fallback) with a
  thin per-caller wrapper.
- [x] **193.** Resolved: `sampling/pd_sample.py`'s `_poisson_disc_core_jit`
  drew its single initial active point uniformly over the *whole* grid,
  including the pre-filled calibration region -- a seed landing inside it
  collides on every one of its `max_attempts` tries (everything nearby is
  already "occupied"), the active list drops to zero on the first outer
  iteration, and the function returns `calib_mask` completely unchanged.
  Because `pd_sample` reuses the *same* fixed seed across every
  binary-search iteration (by design, see the module docstring's point 1),
  this silently killed genuine Poisson-disc placement for the entire call,
  not just one unlucky iteration -- `pd_sample`'s exact-count step then
  filled the whole non-calibration budget via uniform-random selection
  instead of density-tapered placement, with no error or warning. [measured,
  medium severity] Confirmed by direct reproduction: a seed landing inside a
  ~13%-area calibration region returned zero grown points every time.
  Probability of triggering scales with the calibration region's area
  fraction, which was about to grow under the same change that surfaced
  this (see `sampling/pd_sample.py`'s `calib_frac` redefinition from an
  area-matched ellipse to a per-axis fraction-of-kmax rectangle). Fixed by
  rejection-sampling the initial seed against `calib_mask` (module
  docstring point 4); verified the fix eliminates the failure across 2000
  seeds at a deliberately large (~18%) calibration-region fraction where
  the old code failed on the very first seed tried.
- [x] **194.** Resolved 2026-09-15: `lib/readout_from_params.py`'s
  `find_min_feasible_dwell` (item 146/193) only checked the "triangular
  lobe" geometric feasibility of a candidate dwell, not whether the
  resulting echo spacing lands in the scanner coil's forbidden
  acoustic-resonance band (`ge/acoustics.py`'s `_ESP_BANDS_US`, e.g.
  `xrm`'s x/y band 410-510us, z band 360-440us) -- `ge/check.py`'s real
  FFT-based acoustics check only *warns* on this (matching MATLAB's own
  non-blocking `check_grad_acoustics.m` behavior, see the GE export
  section above), but exceeding it is a genuine hardware-damage risk on
  real scanner gradient coils, not just a modeling nicety, per explicit
  user information. [measured, real risk] Reproduced building a
  fully-sampled (R=1) 5.4mm-isotropic ArbEPI variant
  (`Nx=Ny=40,Nz=27,ETL=60,Nshots=18`): the auto-selected minimal-feasible
  dwell (6us) gave a 484us echo spacing, landing squarely inside `xrm`'s
  410-510us forbidden band -- measured acoustics 0.6203, over twice the
  0.3 threshold (vs. 0.023-0.066 for every other sequence generated this
  session). Fixed by extending `find_min_feasible_dwell` to also reject
  any dwell whose `pp.calc_duration(rg.gro)` echo spacing falls within
  `ACOUSTIC_MARGIN_US` (20us) of any of `params.spec.ge_coil`'s forbidden
  bands (checked against the union of all three axes' band lists, matching
  `check_grad_acoustics.m`'s own cross-product check rather than just the
  x-axis list) -- landed on dwell=8us (echo spacing 544us, clear of the
  band with margin) for the reproducing config, dropping acoustics to
  0.0666 with PNS essentially unchanged (69.0%/63.2%, both still well
  under the 80% normal-mode line). Verified the default R=6/2.4mm config's
  own dwell selection (4us, echo spacing 680us, already clear of the band)
  is unaffected. Full test suite still passes (146 passed, 15 skipped).
  Not addressed: `ACOUSTIC_MARGIN_US` is a single flat margin applied
  uniformly, not derived from any measured sensitivity to how close is
  "close enough" -- tighten or loosen it if a future measured acoustics
  number sits uncomfortably close to 0.3 despite clearing the band by more
  than the margin, or vice versa.
- [x] **195.** Resolved 2026-09-15: `sampling/pd_sample.py`'s calibration
  region (`8efa7dd`, "redefine calibration region as a per-axis
  kmax-fraction rectangle") sized `calib_mask` as a *fixed fraction of
  k-space* (`side_frac**2 * Ny*Nz` pixels), independent of `accel`/
  `target_samples` -- a deliberate change at the time, but a real
  regression at high acceleration: `target_samples` shrinks much faster
  than the grid does as R grows, so a fixed-k-space-fraction region can
  end up holding nearly the *entire* sample budget, leaving almost no
  samples for actual incoherent variable-density coverage outside it.
  [measured, real quality impact] Reported directly by the user after
  inspecting the generated 1.6mm/0.8mm sampling-mask plots for this
  session's `ball` protocol ("we pretty much only get a low-res calib
  region with a handful of samples outside"); confirmed numerically at
  the real 0.8mm/R~94 config (`Ny=270,Nz=180`, `pd_calib_frac=0.2`): the
  fixed-kmax-fraction region held 487 of 520 target samples (93.6%),
  leaving only 33 for the entire rest of k-space. This was the deeper
  cause of that config needing `pd_calib_frac` dropped to 0.1 as a
  workaround (see item 194's neighboring conversation) -- a workaround,
  not a fix, since it only shrank the problem rather than removing its
  R-dependence.

  Fixed by `_calib_side_frac(target_samples, nx, ny, calib_frac)`: derives
  the rectangle's per-axis side fraction from `calib_frac * target_samples
  / (nx*ny)` (restoring the *original* pre-`8efa7dd` semantics --
  `calib_frac` = fraction of the R-dependent sample budget, not of
  k-space -- see `2a7ec06`, which `8efa7dd` had superseded) instead of
  treating `calib_frac` itself as the rectangle's side fraction. Keeps
  `8efa7dd`'s rectangle *shape* (and its real seed-rejection-sampling
  fix, item 193, which is independent of region shape) -- only how the
  rectangle's *size* is computed changes. Verified: at the same 0.8mm/R~94
  config, `pd_calib_frac=0.2` (the actual default, no override needed
  anymore) now gives a calibration region of 117 of 520 samples (22.5%,
  matching the requested fraction) instead of 487 (93.6%); regenerating
  that session's 0.8mm ArbEPI sequence with the fix and the default
  `pd_calib_frac=0.2` (no longer needing the 0.1 workaround) dropped peak
  PNS from 95.0% to 84.2% as a side effect (a denser non-calibration
  region gives `mask2epi_radial` shorter average consecutive-sample steps
  to work with). `tests/test_pd_sample.py`'s new
  `test_calib_side_frac_scales_with_sample_budget_not_grid` locks in the
  core invariant (side_frac, and hence realized calib pixel count, must
  shrink as accel grows at fixed `calib_frac`); full test suite (147
  passed) and this session's four resolution variants (5.4mm R=1,
  2.4mm R=6, 1.6mm R~14.5, 0.8mm R~93.5) all regenerated clean after the
  fix.
- [ ] **200. `preprocessing/lowres_calib_recon.py`'s `_load_chunked`
  duplicates `recon/reconstruct.py`'s `_load_array` chunking algorithm.**
  [measured, low severity; found 2026-09-16 against `de3d535`, commit
  `42edeb3`] Both implement identical "read chunk-by-chunk along the last
  axis when `d.chunks[-1] < d.shape[-1]`" logic to avoid the documented
  HDF5 chunk-cache pathology (`recon/reconstruct.py`'s own docstring
  describes measuring ~7 MB/s vs. ~500 MB/s for this exact fix).
  `lowres_calib_recon.py:88-101` takes an already-open `h5py.File` rather
  than a path, so a straight import isn't possible (and `preprocessing/`
  pulling in `recon/` as a dependency would be an unwanted new coupling
  between the two optional-extra packages) -- a body-only refactor into a
  small shared helper would be needed instead. Correctly implemented on
  both sides, just cross-package duplication of a fix that was
  non-trivial to discover once. Fix direction: factor the chunked-read
  loop into a tiny shared module (e.g. `preprocessing/matio.py`, imported
  by both packages) both `preprocessing/` and `recon/` can call.
- [x] **203.** Resolved 2026-09-16: `preprocessing/grid_resize.py`'s
  `resize_to_epi_grid` (used by both `smaps.py`'s `process_smaps` and
  `run_b0map.py`'s field-map resize) unconditionally raised whenever the
  *target* (EPI) z-FOV exceeded the *source* (deGRE) z-FOV, with no
  fallback -- correct in general (there's no real coil-sensitivity/field
  data to resize *from* for a z-extent deGRE never covered), but a real,
  already-acquired dataset this session hit exactly that case: deGRE's
  fixed 144mm z-FOV (72 partitions @ 2mm) vs. the 5.4mm-resolution EPI
  variant's own 145.8mm z-FOV (`Nz=27` from `round(144mm/5.4mm)`, which
  rounds *up* past the 144mm target -- flagged proactively before
  scanning ("is the deGRE FOV matching the EPI scan?"), then confirmed
  live once real data landed and `process_smaps` actually raised on it).
  Since both acquisitions are already on disk, neither retroactive fix
  (shrinking the EPI z-extent, or re-acquiring a taller deGRE) is
  possible -- the practical options were skip smaps/B0-correction for
  that one resolution entirely, or accept a zero (not fabricated) value
  at the ~0.9mm-per-side edge deGRE never measured. Explicit user
  decision: zero-pad it.

  Fixed by a new `zero_pad_z: bool` parameter on `resize_to_epi_grid`
  (default `False`, preserving the existing raise for every other
  caller): when set, resizes onto only the inner target-grid z-slices
  within the source's real coverage (computed conservatively -- ceil the
  inner start, floor the inner end, so a boundary slice straddling real/
  fake coverage is zeroed entirely rather than credited with partial real
  data) and zero-fills the rest, rather than raising. Zero, not edge-
  replication, is the correct fill: real coil sensitivity/field values
  vary fastest right at a slab's edge, the worst place to fake constancy.
  Threaded through as an opt-in parameter on `process_smaps` and
  `run_b0map` (not a new `PreprocessingConfig` field -- this is a
  per-dataset condition, not a general scan setting) so every other
  resolution's behavior is byte-identical to before. Verified on the real
  5.4mm case (`tests/test_preprocessing_grid_resize.py`'s
  `test_resize_to_epi_grid_zero_pad_z_matches_real_5p4mm_config`): exactly
  slices 0 and 26 of 27 zeroed, matching the actual cache files built for
  this dataset (`smaps_1_1x_5.4mm_sigpy.h5`, `1_1x_5.4mm_b0map.h5`) --
  both of which now exist and were consumed successfully by downstream
  RSS and B0-informed CG-SENSE reconstructions of real data, not just
  unit-tested in isolation. 2 new tests total; full suite (149 passed)
  unaffected.
