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

## Open TODOs (review findings, 2026-09-02 pass 5, against `25457b8`)

A fifth repo-wide review. The source tree has **not changed** since the
two pass-4 reviews below completed (`25457b8`, the commit this pass
starts from, only touched CLAUDE.md itself -- confirmed via `git diff
main HEAD --stat` against the last source commit, `f00e2ee`) -- so every
item recorded in the pass-4 sections is reproducibly still open exactly
as written, with no re-verification needed beyond re-running the two
baseline checks: `uv run ruff check .` still reports **31 errors** (29
`E501` + the same `F401`/`F841` pair, item 86) and `uv run pytest` still
reports **120 passed, 20 skipped**, both byte-for-byte matching the
recorded pass-4 numbers. Items 79 and 80 remain the only pass-4-or-
earlier items already resolved (by `f00e2ee`, see that pass's addendum).

Since nothing in the tree changed, this pass did not re-derive findings
already on record -- it instead ran four independent fresh-eyes reviews
over the areas that had the least dedicated attention in the B0-focused
pass-4 reviews: `ge/`, `sampling/`+`lib/`(minus the already-heavily-
reviewed mask2epi/readout-grads/trap4ge/calc_te_tr_delays files)+
`plotting/`+root config, `preprocessing/`'s non-B0 modules, and
`recon/`'s non-B0 modules plus overall doc-vs-code consistency. Six new
findings came out of that (items 96-101 below); everything each review
flagged as a candidate was independently re-verified against the actual
source in this pass before being recorded (not simply transcribed from
the sub-review). Numbering **continues at 96** for the same reason every
pass continues rather than restarts.

Same conventions: **[measured]** = reproduced by running the code (or by
arithmetic on shapes the code fixes), with the number quoted.
**[verify]** = suspicious, but needs a judgement call against the
reference implementation before acting.

### Correctness

- [ ] **96. [measured] `plot_psf` has its `fftshift`/`ifftshift` backwards,
  mis-centering the PSF by one voxel on any odd-length axis -- live at
  this repo's real dimensions.** `plotting/plotting.py:197` computes
  `psf = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(omega)))`. Every
  other centered-array convention in this codebase (`plot_sampling_mask`'s
  `ky_samp = (ys - Ny/2) * deltak_y` two lines up in the same file;
  `mask2epi.py`'s `cy, cz = Ny/2, Nz/2`) treats `omega`'s array index
  `N/2` as `k = 0` -- correctly un-centering it for a transform needs
  `ifftshift` on the *input* and `fftshift` on the *output*
  (`fftshift(ifft2(ifftshift(x)))`), the reverse of what's written here.
  `fftshift`/`ifftshift` are identical on even-length axes, so this is
  invisible on `Ny` (240, even) but not on `Nz` (45, odd, per `params.py`'s
  default `N = [240, 240, 45]`). Measured directly: for an all-ones mask
  the true PSF is a single delta at `(Ny//2, Nz//2)`; the code's formula
  places it at `(Ny//2, Nz//2 + 1)`. At the real production scale
  (`Ny,Nz = 240,45`) the code's PSF magnitude equals the correct one
  circularly shifted by exactly +1 sample along z (`np.roll(correct, 1,
  axis=1)` matches to `4.3e-18`, float noise; the same shift along the
  even y-axis does not match, confirming the bug is isolated to the odd
  axis as the shift-convention analysis predicts). Not the same bug class
  as items 44/64 (`_ift3` and friends): those swap only *one* shift, which
  item 64 proves is a harmless linear phase ramp under `np.abs()`; this
  swaps *both*, which does not cancel (as measured above -- `psf_mag`
  itself, not just the complex value, is shifted). Diagnostic-only impact
  (`output/psf.png` via `--plot`, never feeds sequence generation, GE
  export, or reconstruction) but a real, visible defect with zero test
  coverage (`grep -rn "plot_psf" tests/` -> no hits). Fix:
  `np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(omega)))`.
- [ ] **97. [measured] `Emax_n` is silently wrong for any segment whose
  gradient energy is always exactly zero -- live on this repo's own
  shipped `noise.seq` -> `noise.pge` export.** `ge/ceq.py:56-57` defaults
  `Emax_val: float = 0.0`, `Emax_n: int = 1`; `ge/seq2ceq.py:169-171`'s
  gradient-heating loop only overwrites them on strict `e_total >
  seg.Emax_val`. A segment whose combined gradient energy is genuinely
  `0.0` in every instance never satisfies that condition, so `Emax_n`
  stays at the class default `1` -- a hardcoded row number belonging to
  whichever segment occupies row 1 of the whole sequence, not necessarily
  (and, as measured, not actually) a row belonging to *this* segment.
  `ge/writeceq.py:242` writes that incoherent value verbatim into the
  `.pge` binary. Measured by running `seq2ceq()` on the real
  `output/noise.seq` (this repo's own sequence with genuinely zero
  gradient energy throughout, per CLAUDE.md's own "0.0 mT/m max grad /
  0.0 T/m/s max slew" record): segment 2's own rows start at 121
  (`seg.rows == [121]`), yet its exported `Emax_n` is `1`, which
  `ceq.loop[0, 0] == 1` confirms belongs to segment 1. Re-ran the same
  check on `ArbEPI.seq`/`EPIcal.seq`/`deGRE.seq` (all nonzero-energy
  everywhere) and every segment's `Emax_n` correctly resolves to itself --
  confirming the mechanism works whenever energy is nonzero and only
  breaks in the always-zero-energy case. Distinct from the already-
  documented "stale 1-indexed column choice" deviation in the same
  function (that one is about *which columns* feed `e_total`, not about
  this default-value fallback when `e_total` is legitimately zero
  everywhere). Practical GE-PSD-interpreter impact of a wrong `Emax_n` on
  a no-gradient segment is unverified (no way to run the real interpreter
  here), but the exported value is objectively incoherent. Fix: seed
  `Emax_n` from `seg.rows[0]` (the segment's own first row) rather than a
  cross-segment constant `1`, so the zero-energy case degrades to "this
  segment's own first instance" instead of "segment 1's".
- [ ] **98. Three of `seq2ceq`'s four block-walking loops assume the final
  segment instance is complete, and the first one to walk off the end
  raises a raw, unhelpful `KeyError` rather than an actionable error.**
  `ge/seq2ceq.py:74`'s "detect variable delay blocks" loop
  (`while n <= ceq.nMax`), `:117`'s loop-table construction (same guard),
  and `:159`'s gradient-heating loop (`while n < ceq.nMax`) all stride `n`
  forward in `nBlocksInSegment`-sized steps with no check that the last
  segment instance actually has that many blocks left. Only the
  consistency-check loop at `:138-144` guards this (`if n +
  seg.nBlocksInSegment > ceq.nMax: break`) -- the other three don't share
  its pattern. Reproduced with a minimal synthetic pypulseq sequence (two
  complete TRID-segment instances, a third truncated to just its
  TRID-label block): `seq2ceq()` crashes inside the first, unguarded loop
  at `ge/seq2ceq.py:77` with `KeyError: 6` from deep inside
  `pypulseq.Sequence.block.get_block` -- not one of this file's own
  actionable `ValueError`s used for other malformed-input cases nearby.
  Confirmed this never fires on any of the four real `output/*.seq` files
  this repo ships (their final instances are all complete) -- latent, not
  live -- but `seq2ceq(seq)` sits on `check_seq_feasibility`'s production
  path (`ge/check.py:237`, used by both `ge_export.py`'s pre-flight check
  and `main.py --ge`), so a future sequence-generator bug or a
  debugging/truncation script that ever produced a mid-TR-truncated
  sequence would surface as an opaque `KeyError` instead of a clear
  diagnostic. Fix: apply the same `if n + seg.nBlocksInSegment >
  ceq.nMax: break` (or raise with a clear message) to the other three
  loops.
- [ ] **99. [measured] `preprocessing/cg_sense.py` infers its sampling
  mask from a single coil's exact-zero values -- the same bug class items
  7/74 already fixed elsewhere in the repo, still live in this file.**
  `preprocessing/cg_sense.py:54-55` is `mask =
  np.abs(np.take(kdata_zf, 0, axis=coil_dim)) > 0` -- derived from coil
  index 0 alone. Item 74's own docstring (quoting item 7) explains exactly
  this risk generically ("no per-coil check can catch it"), and that fix
  never touched `preprocessing/`. The sibling Stage-2 driver in the same
  package does this safely: `preprocessing/recon_sigpy.py:65`'s `weights =
  (sp.rss(ksp_cf, axes=(0,)) > 0)` uses the root-sum-of-squares across
  *all* coils, so the two drivers are now inconsistent, and `cg_sense.py`
  is the weaker one. Measured failure mode: with coil 0 forced to all
  zeros, `cg_sense()` raises a numpy `RuntimeWarning: invalid value
  encountered in scalar divide` (the first CG step computes `alpha =
  rsold / ... = 0/0`) and returns an all-NaN reconstruction, silently --
  easy to miss inside `run_cg_sense.py`'s per-sequence `except Exception`
  batch loop. Caveat: `cg_sense.py` runs on `ksp_epi_zf` *after* Stage 1's
  PCA coil compression, so "coil 0" here is the dominant virtual coil (a
  weighted combination of all physical channels), not a single physical
  receive element -- a genuinely dead physical coil is unlikely to zero
  that virtual coil exactly, so the everyday trigger probability is lower
  than a naive reading suggests, but the structural defect and its
  measured catastrophic-and-silent failure mode are real. Fix: derive the
  mask from `recon_frames`'s already-available `omegas` dataset, or at
  minimum switch to an RSS-across-coils check matching `recon_sigpy.py`.

### Consistency & documentation

- [ ] **100. Item 84's README-architecture-tree claim was already fixed by
  `f00e2ee`, but neither pass-4 addendum said so -- only its
  `pyproject.toml` sub-claim is still open.** Item 84 lists six `recon/`
  modules and `gre_diagnostics.py` as missing from README's architecture
  tree. `f00e2ee` (titled "Fix B0 field map noise: switch b0map.jl's NCG
  preconditioner to :diag") touched `README.md` in the same commit and
  added exactly these: `README.md:193` lists `gre_diagnostics.py` under
  `preprocessing/`, and `:203-213` list `operators_b0.py`,
  `b0_correction.py`, `save_result.py`, `run_b0_recon.py`,
  `sweep_time_segments.py`/`benchmark_b0_cost.py` under `recon/` --
  confirmed current against the real directory listing. The pass-4
  addendum explicitly re-verified items 74/79-84/86-89 against `f00e2ee`
  and marked 79/80 resolved, but not 84, even though 84's main claim was
  fixed by that same commit. Only 84's other sub-claim -- `pyproject.toml`
  `:43-44`'s dangling "see recon/README" comment (no `recon/README*`
  exists) -- is still accurate and still open. Narrow item 84 to just
  that one line.
- [ ] **101. Two small doc gaps, worth folding into the next general docs
  pass.** (a) `preprocessing/matio.py:7-9`'s module docstring still
  quotes `schedules`' pre-echo-time shape (`h5py raw (2, 60, 20, 30)` ->
  logical `(30, 20, 60, 2)`, "`Nframes x Nshots x ETL x 2` layout"),
  stale since `8071542` ("Resize B0 field maps to the EPI grid, add
  per-echo timing, consolidate scan_info.mat") made it `x3` -- confirmed
  against `sequences/ArbEPI.py:299-301`'s 3-channel concatenation and this
  file's own ".mat file format" section. `read_mat_array`'s `.transpose()`
  is shape-agnostic, so there's no functional bug, only a stale worked
  example; update the two shape tuples and the channel count. (Not the
  same array as `lib/mask2epi.py:75`'s own, still-correct "`ETL x 2`"
  comment -- that one describes `mask2epi`'s internal array *before*
  `ArbEPI.py` appends the echo-time column.) (b) `README.md:22`'s core-
  dependency sentence ("Depends on `pypulseq` ..., numpy, scipy,
  matplotlib, hdf5storage, and numba") omits `tqdm`, which
  `pyproject.toml:19` lists as a real core (non-optional) dependency and
  which `sequences/ArbEPI.py:23`/`sequences/deGRE.py:48` both actually
  import. Add it to the list.

### Conciseness & performance

- [ ] **102. `preprocessing/epi_gridding.py`'s `reconecho()` is dead
  code.** Defined and documented at `:34-51` (including a docstring
  cross-reference from `rampsamp2cart`'s own docstring), but has zero
  callers anywhere in the repo or its tests (`grep -rn "reconecho\b"
  --include=*.py .` matches only the definition and its own docstring
  mentions; `grep -rn reconecho tests/` -> no hits). Everything that
  actually grids EPI data goes through `rampsampepi2cart`/
  `rampsamp2cart`, which reimplement the same per-echo NUFFT+DCF logic
  inline instead of calling it. Not the same dead-field class as items 33
  or 73 (those are struct fields/parameters, not a whole unused
  function). Either delete it, or wire `rampsamp2cart`'s inner loop to
  call it (it would need batching to be useful as-is).

## Open TODOs (review findings, 2026-09-02 pass 4, against `b67bcc0`)

Findings from a fourth repo-wide review, the first to cover `b67bcc0`
("Add B0 off-resonance correction to recon/, plus GRE/field-map
diagnostics") -- ~1400 lines across 8 new modules and 2 new test files
that no earlier pass saw. Numbering **continues at 74** for the same
reason every pass continues rather than restarts: source files and this
file's own prose cross-reference earlier item numbers by name, so numbers
are never reused.

Same conventions: **[measured]** = reproduced by running the code (or by
arithmetic on shapes the code fixes), with the number quoted.
**[verify]** = suspicious, but needs a judgement call against the
reference implementation before acting.

**Nothing from passes 2 or 3 has been fixed.** Items **61-73** are all
still open, as are **36-60** and **8, 13, 15, 17, 32, 33** -- spot-checked
this pass by re-confirming the exact code each names is unchanged
(`deGRE.py:237`'s `params.fov` for item 61, `deGRE.py:204`'s
`max(0, iZ - 1)` for 72a, `EPIcal.py:76`'s 1-based shot loop for 60,
`recon_frames.py:101`'s `min(cfg.Nframes, ...)` for 42, `trap4ge.py`'s
still-absent `flat_area` write for 43, `noise.py:42-43`'s zeroed
`adc_dead_time` for 13, and `noise.py`'s still-absent `set_definition`
calls for 65).

Baseline at review time (`uv sync --extra test --extra lint`, plain main
venv, no `output/` directory):

| check | pass 3 recorded | this pass |
|---|---|---|
| `uv run ruff check .` | 20 errors (all `E501`) | **31 errors** (see 86) |
| `uv run pytest` | 120 passed, 18 skipped | **120 passed, 20 skipped** (see 87) |

Only the plain-venv row was re-measured; the three other combinations in
pass 3's table shift by the same +2 skips (both new test modules are
torch-gated, not `output/`- or sigpy-gated) but were not re-run.

One thing checked and found **correct**, recorded so nobody "fixes" it:
`mri_exp_approx`'s calling convention. Read directly from
mirtorch 0.3.1's `mirtorch/linear/mri.py`, the pinned dependency: the
signature is `mri_exp_approx(b0, bins, lseg, t)`, so
`operators_b0.py:212`'s `mri_exp_approx(b0_neg, nbins, L, unique_t_ms)`
has `nbins`/`L` in the right order; the function expects `b0` in **Hz**
and `t` in **milliseconds** (it divides by 1000 internally, twice -- once
for the segment centers, once for the fit target), which is why passing
`echo_times_s * 1000` against an unscaled Hz field map is right and not a
1000x error; and the `tl` it returns is already back in **seconds**.
`Gmri`'s own defaults really are `L=6, nbins=20`, so both "matching
mirtorch's own Gmri default" claims are accurate as stated.

### Correctness

- [ ] **74. [measured] `run_b0_recon.py` re-introduces the exact-zero
  sampling-mask inference that pass-1 item 7 removed -- under the same
  function name as the fixed one, and at the cost of an extra full ~11 GB
  read.** `recon/reconstruct.py:69`'s `_load_omega` was changed by item 7
  to prefer the authoritative `omegas` dataset over deriving the mask from
  `ksp != 0`, with a docstring explaining that a real sample rounding to
  exactly `0+0j` after phase correction silently becomes "not acquired"
  and that no per-coil check can catch it. `recon/run_b0_recon.py:41`
  defines its *own* `_load_omega(fn_ksp)` whose entire body is
  `_load_array(fn_ksp, "ksp_epi_zf")[:, :, :, 0, :] != 0` -- exactly the
  derivation item 7 called a bug, on coil 0 alone (so not even the
  fallback branch's per-coil consistency assert applies). The mask it
  produces feeds `build_encoding_operator_b0`, whose per-frame `idx` then
  decides which k-space locations the whole B0-corrected reconstruction
  fits. The same fix removes a large cost: `omegas` is `(Ny,Nz,Nt)` bool
  (a few hundred KB) while this reads the full `(Nx,Ny,Nz,Nc,Nt)`
  archive, which that function's own docstring measures at ~23 s and
  concedes means "ksp_epi_zf gets loaded twice across this driver". Read
  `omegas` and the second load disappears entirely, along with the
  docstring's whole coil-0-slicing-is-slower digression. Fix: import and
  reuse `reconstruct._load_omega`, or read the dataset directly.
- [ ] **75. [measured] `build_encoding_operator_b0` materializes 2.2 GB of
  per-frame `b_weights` that hold only 60xL distinct values -- 3.3x the
  shared-`c_phasors` cost the same function was already refactored to
  avoid.** `operators_b0.py:231`'s `b = b_by_echo[pos]` is advanced
  indexing, so each of the `Nt` frames gets an independent, fully
  materialized `(K, L)` complex64 tensor. At this repo's real scale
  (`Nx,Ny,Nz = 240,240,45`, `R ~ 9` so `K = 288000`, `Nt = 30`) and the
  `L = 32` the sweep script argues for: `288000 * 32 * 8 B = 73.7 MB` per
  frame, **2.21 GB total**, every byte of it a gather from the same
  `(ETL, L) = (60, 32)` table -- 15360 distinct values, 61 KB. For
  comparison the shared `c_phasors` this function's own docstring
  describes being refactored to share across frames (after an L=16
  version "push[ed] a real reconstruction into a CUDA OOM") is
  `32 * 2592000 * 8 B = 664 MB`. So the redundancy that was fixed is now
  the *smaller* of the two. Fix: keep `pos` per frame (int64:
  `288000 * 8 B = 2.3 MB` per frame, 69 MB total, ~32x smaller; int32
  halves that again) plus the one shared `b_by_echo` table, and index
  inside `_apply`/`_apply_adjoint` -- `self.b_weights[:, il:il+1]` becomes
  `b_table[pos, il:il+1]`, the same gather, just not precomputed 30 times.
- [ ] **76. `estimate_spectral_norm` runs a fixed 30 power iterations with
  no convergence check, and the error it can make is the unsafe one.**
  `operators_b0.py:236` iterates exactly `niter` times and returns
  `A.apply(x).norm()`. Power iteration approaches `sigma1` **from below**,
  so a not-yet-converged result is an *under*-estimate; `run_recon` then
  computes `L = Nscales * sigma1A**2` (`reconstruct.py:194`) as POGM's
  Lipschitz constant, and an under-estimated `L` means a step size `1/L`
  that is too *large* -- the divergence direction, not the slow-but-safe
  one. Nothing downstream re-checks it: `run_b0_recon.py` prints the value
  next to the uncorrected reference, which catches a wild miss by eye but
  only if someone is watching. The repo already contains the right
  pattern -- `recon/solvers.py:188`'s `poweriter` takes `niter=200,
  tol=1e-6` and returns early on a converged ratio -- and it is the same
  computation, which is the disposition question item 49 raised (see 89).
  Fix: iterate to a tolerance, or keep 30 iterations and multiply by a
  small safety factor, but say which and why.
- [ ] **77. [measured] `operators_b0.py`'s `nbins` docstring blames a
  weighting that mirtorch does not do.** It describes `mri_exp_approx` as
  fitting from "an *equal-width*, magnitude-weighted histogram of
  `b0map_hz`'s *entire* range" and argues that at `nbins=20` "nearly all
  the histogram's magnitude-weighted mass falls into 1-3 bins near zero".
  mirtorch 0.3.1's `_uniform_histogram` is
  `histogram.scatter_add(0, indices, torch.ones_like(values))` -- a plain
  **voxel-count** histogram with no magnitude weighting anywhere in the
  call path (MIRT's own `mri_exp_approx` does accept a weight vector;
  mirtorch's port dropped it). The conclusion is unaffected and if
  anything stronger -- background dominates by count too, and the measured
  row-sum evidence ([0.12, 2.89] at nbins=20 vs [0.9985, 1.0022] at
  nbins=100) stands on its own -- but the stated mechanism is wrong, and
  it is the sentence someone would reason from when picking `nbins` for a
  differently-shaped field map. Fix the two "magnitude-weighted" phrases;
  note while there that the equal-width range is set by `b0.amin()`/
  `amax()` over the whole volume, which is what actually makes an
  asymmetric in-object range expensive in bins.
- [ ] **78. `gre_diagnostics.py` computes `n_echoes` generically, then
  hardcodes two echoes.** `preprocessing/gre_diagnostics.py:48` reads
  `n_echoes = ksp_echoes.shape[3]` and builds `img_echoes` over
  `range(n_echoes)`, but `:79-80` index `img_echoes[..., 1]` and
  `te_degre[1]` unconditionally, and `:83`'s ratio panel assumes exactly
  two. `preprocessing/config.py:185` defaults `n_echoes_degre` to `1` for
  a `scan_info.mat` snapshot written before the dual-echo upgrade -- which
  this file's own `preprocessing/` section describes as a deliberate,
  supported case ("a durable, non-regeneratable per-acquisition data
  record, not something to raise `KeyError` on") -- and on such a dataset
  this is a bare `IndexError` from a diagnostic script whose whole purpose
  is being run when something already looks wrong. Same line, `:46`'s
  `te_degre = f.attrs["TE_degre"]` is an unguarded `KeyError` on the same
  class of older cache, since `preprocess.py` writes that attr only
  "whenever `seq_params.TE_degre` is available". Either guard both, or
  assert `n_echoes == 2` up front with a message naming the cause.
- [ ] **79. `b0map.jl` records `l2b` in its output attributes but not
  `niter` -- the one parameter its own new docstring says `l2b` is
  meaningless without.** The commit adds both as CLI overrides and adds a
  docstring paragraph reporting that raising `l2b` alone, holding `niter`
  at 30, "measurably did *not* change the fit (tested l2b in [-6, 8], a
  16384x range ... with identical results to 4 significant figures)". The
  output block then writes `attributes(f)["l2b"] = l2b` and stops there.
  So a `<seqname>_b0map.h5` on disk records the knob that provably does
  nothing on its own and omits the one that makes it bite: two field maps
  from `l2b=4, niter=30` and `l2b=4, niter=300` are materially different
  and indistinguishable after the fact. One line, next to the `l2b` one.

### Consistency & documentation

- [ ] **80. `b0map.jl`'s new `l2b`/`niter` overrides are unreachable from
  the Python driver, so the batch pipeline cannot produce the tuned field
  map the docstring describes.** `preprocessing/run_b0map.py:67-73` builds
  the subprocess argv as `[julia, --project, script, gre_cache_path,
  output_path, str(cfg.b0map_mask_thresh)]` -- three positional args, no
  fourth or fifth -- and `PreprocessingConfig` (`config.py:60-63`) carries
  only `b0map_mask_thresh`. The regularization tuning the commit adds
  these parameters *for* (a measured ~2.7x SNR gradient toward the
  object's center, propagating under-regularized field-map noise into
  visible reconstruction artifacts) can therefore only be exercised by
  invoking `julia` by hand, outside the driver, and never reproducibly
  from a config. Fix: add `b0map_l2b`/`b0map_niter` to
  `PreprocessingConfig` and pass them through, defaulting to b0map.jl's
  own `-6.0`/`30` so existing behavior is unchanged.
- [ ] **81. Three docstrings still call time-segmented correction
  "not-yet-implemented" in the commit that implements it.**
  `recon/b0_correction.py`'s module docstring says the residual blur
  "needs full time-segmented correction (a separate, not-yet-implemented
  stage)"; `tests/test_recon_b0_correction.py`'s
  `test_realistic_regime_only_partly_corrects` docstring closes with the
  same phrase verbatim; and `b0_correction.py`'s sign-convention paragraph
  says the convention "has not yet been verified against a real
  reconstruction" while `run_b0_recon.py` exists precisely to run one.
  `recon/operators_b0.py` landed in the same commit and its own docstring
  correctly describes itself as "the actual fix that regime needs", so the
  repo now states both things. The static stage is still worth keeping
  (it is the cheap sign/scale check, and `test_l1_matches_static_
  correction` ties the two stages together), so this is a wording fix, not
  a deletion.
- [ ] **82. `L = 6` is still the default in four places after the sweep
  concluded 32.** `b67bcc0`'s own commit message says
  `sweep_time_segments.py`/`benchmark_b0_cost.py` "motivated picking L=32
  over the L=6 default (L=6 badly under-resolves this pipeline's real
  ETL=60 bandwidth-time product; L=32 is the smallest swept L keeping
  forward-model error under 1%)". Nothing in the tree reflects that:
  `operators_b0.py:144` `L: int = 6`, `reconstruct.py:132` `L_b0: int =
  6`, `run_b0_recon.py:58` `L_b0: int = 6` and `:140`
  `--L ... default=6`. Worse, three docstrings assert the sweep never
  happened -- `operators_b0.py`'s module docstring ("not swept against a
  real error bound the way Fable's staged plan recommends -- see
  CLAUDE.md's `recon/` section for that open item"), `run_b0_recon.py`'s
  ("not swept against a real min-max error bound"), and `run_recon`'s new
  parameter docs ("see operators_b0.py's module docstring for why L hasn't
  been swept") -- while `recon/sweep_time_segments.py` sits in the same
  directory doing exactly that sweep and printing "Smallest swept L with
  rel_error < 1%". Decide the default, set it in all four places, and
  replace the three docstring paragraphs with the sweep's actual numbers
  (which are not recorded anywhere in the repo -- only in the commit
  message). Note `L = 32` interacts directly with item 75: at L=32 the
  per-frame `b_weights` redundancy is 2.2 GB, which is the regime the
  earlier L=16 OOM came from.
- [ ] **83. Every "see CLAUDE.md's `recon/` section" pointer the new code
  adds is dangling, and two existing CLAUDE.md claims about `recon/` are
  now false.** Four new cross-references point at content this file does
  not contain: `operators_b0.py`'s and `run_b0_recon.py`'s "see CLAUDE.md's
  recon/ section for that open item" (there is no L-sweep open item),
  `b0map.jl`'s "see CLAUDE.md's recon/ section for the tuning this was
  chosen against" (no such tuning is recorded), and
  `benchmark_b0_cost.py`'s "the smaller Julia/Python benchmark numbers
  already documented there" for the *uncorrected* operator (that comparison
  is real and is in the `recon/` section -- this one only half-lands, since
  it also cites a real-scale table that does not exist). In the other
  direction: item 53 states "There is no `recon/operators_b0.py`
  (`recon/` holds `__init__`, `lowrank`, `operators`, `reconstruct`,
  `solvers`, `validate_against_mslr`) and the working tree is clean, so the
  uncommitted work it warns about is gone" -- the file now exists, the
  directory now holds twelve modules, and item 12's warning about the
  `grid_mode=True` alignment change shifting `smaps`/`b0map_hz` relative to
  the EPI grid is live again and has a concrete target
  (`GatheredSenseB0`'s `c_phasors`). Item 49's "run_recon still has no
  `save_result` driver" and the `recon/` section's "a `save_result` driver
  reusing `preprocessing/nifti_io.py`'s `save_recon_nifti` ... would be the
  natural next piece" are likewise both answered by `recon/save_result.py`,
  which does exactly that. Retarget the four pointers, correct 53 and 49,
  and give the `recon/` section a B0 subsection covering the sign
  convention, the ms/Hz calling convention noted in this section's
  preamble, the `nbins` finding, and whatever 82 settles on for L.
- [ ] **84. README's architecture tree has re-drifted -- seven modules
  missing, the same failure pass-1 item 22 fixed once.** `README.md:195`'s
  `recon/` block lists `operators.py`, `lowrank.py`, `solvers.py`,
  `reconstruct.py`, `validate_against_mslr.py` and stops; the directory now
  also holds `b0_correction.py`, `operators_b0.py`, `save_result.py`,
  `run_b0_recon.py`, `sweep_time_segments.py` and `benchmark_b0_cost.py`.
  The `preprocessing/` block above it omits `gre_diagnostics.py`. Since
  README's tree is the only place the repo enumerates its own modules for
  a newcomer, and item 22 already had to repair it once, consider whether
  it should list directories with a one-line purpose instead of every file.
  While there: `pyproject.toml`'s `recon` extra comment says "see
  recon/README or top-level docs for the CUDA wheel selection this machine
  needs" -- there is no `recon/README` (pre-existing, not from this commit).
- [ ] **85. `tests/test_recon_operators_b0.py`'s "realistic regime" is the
  toy grid `sweep_time_segments.py` was written to replace, and its tests
  pin the `nbins` value the code documents as broken.**
  `test_more_segments_reduces_error_in_the_realistic_regime` asserts
  `err_l16 < 0.1 * err_l8`, commented "L16 >= the 12 distinct ky times in
  this small synthetic grid, so it's essentially exact".
  `sweep_time_segments.py`'s module docstring says of that exact test:
  "That test's own 'L=16 is ~exact' finding is an artifact of its toy grid
  having only 12 distinct echo times (L>=12 trivially resolves every one
  exactly); it says nothing about whether L=6 ... is adequate at the real
  ETL=60 scale". So the test's name and docstring claim the realistic
  regime while the analysis script says it is not one -- and a real
  regression (say, L=32 silently becoming inadequate) cannot fail it.
  Separately, `_build_b0_operator` defaults `nbins=20` and every test uses
  that or 10, i.e. exactly the setting `operators_b0.py`'s docstring
  identifies as "the actual root cause ... of a real signal-loss-plus-
  incoherent-noise failure on real reconstructions", which also means the
  suite should be tripping `_check_b_weight_row_sums`' warning throughout.
  Fix: rename the test to say "toy grid", fold `sweep_time_segments.py`'s
  real-scale fixture in as the actual realistic-regime test (it is already
  written, and its docstring admits it duplicates the test helpers to avoid
  a `recon/` -> `tests/` import -- which inverts once the fixture lives in
  `tests/`), and add one case at the production `nbins=128` that asserts no
  row-sum warning fires.

### Test & tooling health

- [ ] **86. [measured] Lint debt grew 55% and is no longer all `E501`.**
  `uv run ruff check .` reports **31 errors**, up from the 20 items 15/70
  record. All 11 new ones are in files this commit added --
  `recon/benchmark_b0_cost.py` (4x `E501`, plus the repo's first-ever
  `F401` and `F841`), `recon/sweep_time_segments.py` (3x `E501`),
  `preprocessing/gre_diagnostics.py` (2x `E501`) -- and the pre-existing 20
  are unchanged, so this is purely the new code. The two non-`E501`
  findings are real dead code, not style: `benchmark_b0_cost.py:29` imports
  `GatheredSense` and never uses it (only `build_encoding_operator` is
  called), and `:86`'s `y = A.apply(x0)` is assigned but never read -- the
  adjoint on the next line is timed against the independent random `y0`
  instead, which is fine for a cost benchmark but means the variable is
  simply garbage. `ruff --fix` clears the `F401`; the rest is manual.
  Item 15's "now the only standing lint debt in the repo" no longer holds.
- [ ] **87. [measured] The entire B0 feature is invisible to the default
  test command.** `uv run pytest` in the documented main venv is now
  **120 passed, 20 skipped** (was 120/18). The two added skips are
  `tests/test_recon_b0_correction.py` and
  `tests/test_recon_operators_b0.py`, both whole-module
  `pytest.importorskip("torch")` -- so all ~1400 lines added by `b67bcc0`,
  including every sign-convention and adjoint check written specifically to
  guard them, run zero times in the environment CLAUDE.md's Commands
  section tells a contributor to use. That is the documented pattern for
  `recon/` and not wrong on its own, but the skip ledger is now 6
  torch-gated modules + 9 sigpy/nibabel-gated + 5 `output/`-gated cases
  (item 70's corrected count, re-confirmed exactly this pass), i.e. the
  majority of this repo's test surface is dark by default. Worth deciding
  once, at the level of items 46/70 rather than per-module: either CI
  installs the extras, or this file says plainly which fraction of the
  suite a plain `uv run pytest` actually exercises.
- [ ] **88. `benchmark_b0_cost.py` documents its own headline measurement
  backwards.** Its docstring promises to report "the static, L-independent
  memory cost of building the operator itself (b_weights across all frames
  + the shared c_phasors tensor)". Both named components scale linearly
  with L -- `c_phasors` is `(L, *N)` and `b_weights` is `(K, L)` per frame
  (item 75's 664 MB and 2.21 GB at L=32) -- and the script's own
  `build_mem` column measures exactly that L-dependence, printed once per
  swept L in a table whose whole point is comparing L values. Fix the
  sentence; the number it prints is the right one.

### Conciseness & performance

- [ ] **89. `GatheredSenseB0` is `GatheredSense` copied verbatim plus two
  factors, and `estimate_spectral_norm` is `poweriter` written a second
  time.** `operators_b0.py:83-107`'s `_apply`/`_apply_adjoint` reproduce
  `operators.py:49-66` line for line -- the same `reshape(Nc,-1).T`
  gather, the same `kc_full` scatter, and in particular the same
  `fftshift(fftn(ifftshift(.)))` / `ifftshift`-then-`fftshift` convention
  that `operators.py`'s module docstring spends a paragraph justifying and
  that `tests/test_recon_operators.py` verifies -- with `c_phasors[il]`
  and `b_weights[:,il]` inserted. Both copies must now stay in sync
  through any future change to that convention. `GatheredSenseB0`'s own
  docstring states the reduction ("L=1, b_weights=1, c_phasors=1 recovers
  it exactly") and
  `test_estimate_spectral_norm_matches_full_sampling_unity_case` proves it
  numerically, so the collapse is already justified and tested: make
  `GatheredSense` the `L=1` case, or have it subclass/delegate. Likewise
  `estimate_spectral_norm` (`operators_b0.py:236`) computes exactly what
  `recon/solvers.py:188`'s `poweriter` computes, minus the tolerance check
  (item 76) -- and `poweriter` is the dead function item 49 asked to
  either wire up or delete. `run_b0_recon.py` is the wiring item 49 wanted;
  the resolution is now to keep one implementation, take `poweriter`'s
  `tol`, and put it next to the operators it measures rather than inside
  the B0-specific module (it is documented as taking "any mirtorch
  LinearMap/BlockDiagonal"). Minor, same file: `:127`'s function-body
  `import warnings` is the only one in the repo -- `lib/calc_te_tr_delays.py`
  and `preprocessing/preprocess.py` both import it at module level.
- [ ] **90. [measured] `echo_times_s` is materialized 240x redundantly on
  the GPU, in both callers.** `reconstruct.py:181` and
  `run_b0_recon.py:88` both do
  `echo_times_2d.to(device).unsqueeze(0).expand(Nx, -1, -1, -1)
  .contiguous()`, turning preprocessing's `(Ny,Nz,Nt)` array into a dense
  `(Nx,Ny,Nz,Nt)` one: at real dims that is **311 MB** of float32 on the
  device, expanded from **1.30 MB** of distinct values, and
  `build_encoding_operator_b0` reads it only at each frame's sampled flat
  indices. The `.contiguous()` is what forces it -- `expand` alone is a
  free stride-0 view, and `reshape(-1)[idx]` on the expanded view would
  work directly. Cheaper still: index the `(Ny,Nz,Nt)` array with
  `idx // Nz`-style arithmetic and skip the broadcast entirely. On a
  pipeline whose docstrings repeatedly cite CUDA OOMs at this exact scale,
  311 MB for a value that varies along none of the axis it is copied
  across is worth reclaiming. (Both call sites are two lines apart in
  spirit and identical in text -- a shared helper would fix both, and is
  the same duplication as 74's `_load_omega`.)
- [ ] **91. `_ift3` now exists three times, and the newest copy inherits
  item 64's false justification on the one grid where it bites.**
  `preprocessing/gre_diagnostics.py:29` is a verbatim copy of
  `preprocessing/run_rss.py`'s `_ift3` (`fftshift(ifftn(fftshift(.)))`),
  and `preprocessing/julia/b0map.jl` implements the same convention a third
  time in Julia. Item 64 records that the `fftshift`-on-input spelling
  differs from the conventional `ifftshift` by a one-sample circular shift
  on odd-length axes, and is safe here only because every consumer takes a
  magnitude or a difference. `gre_diagnostics.py` runs on the **deGRE**
  grid, where `Nz_degre = 21` is odd -- it is magnitude-only too
  (`np.sqrt(np.sum(np.abs(...)**2))`), so still safe, but it is now the
  third place a future complex-valued consumer could pick the pattern up
  from. Import `run_rss._ift3` rather than copying it (or hoist it into a
  shared `preprocessing/` helper alongside `matio.py`), and fix item 64's
  docstring once, in one place.

### Addendum (second pass-4 review, against `f00e2ee`)

A second, independently-run pass-4 review covered the same `b67bcc0` B0
commit and reached the same conclusions on items 74/79-84/86-89 (including
the identical `mri_exp_approx` Hz-vs-milliseconds finding recorded in the
preamble above, arrived at separately by reading the same mirtorch
source). Only the items below were not already covered; all four were
re-verified against `f00e2ee`, not just against `b67bcc0`, so the code
they cite is current.

**Two items above are already resolved by `f00e2ee`** and can be closed
without further work: **79** (`b0map.jl` records `l2b` but not `niter`)
and **80** (`l2b`/`niter` unreachable from `run_b0map.py`). That commit
removed both CLI arguments outright -- `b0map.jl:237` now reads "l2b/niter
deliberately not passed -- MRIFieldmaps' own defaults", the writer block
no longer emits an `l2b` attribute, and `config.py:64-69` documents the
choice as deliberate and backed by an actual sweep (flat roughness for
`l2b` in [-6, 28] at both `niter=30` and convergence, with `precon=:diag`
identified as the real fix). Nothing is unreachable and nothing
half-recorded any more.

- [ ] **92. Four committed source files name an AI model ("Fable") as the
  authority for a design decision.** `recon/operators_b0.py:28` and
  `recon/run_b0_recon.py:18` ("not swept against a real error bound the
  way Fable's staged plan recommends"), `recon/operators_b0.py:55` ("see
  CLAUDE.md/Fable's discussion of why GatheredSense itself had to avoid
  ..."), and `tests/test_recon_b0_correction.py:137` ("the honest, weaker
  claim Fable's plan makes for this regime"). Item 82 quotes the first of
  these while making a different point (that the sweep did happen), so the
  attribution problem itself is unrecorded. This repo's own "Commit
  conventions" section directs that commits carry no AI identity and read
  as the user's own; source shipped by those commits is the same question,
  and a reader has no way to resolve "Fable's staged plan" against
  anything in the repository. All four sentences already state the real
  technical reason alongside the name -- delete the attribution, or cite
  the actual reference (Sutton/Noll/Fessler for the signal model,
  `sweep_time_segments.py` for the L bound). Fixing item 82's docstrings
  covers three of the four sites; the test file's is separate.
- [ ] **93. `run_recon(fn_b0map=...)` silently accepts a `sigma1A`
  measured for the *uncorrected* operator.** `recon/reconstruct.py:146`'s
  docstring is explicit -- "sigma1A is not re-estimated internally for the
  corrected operator -- the caller must supply one appropriate to it" --
  but nothing enforces it: `sigma1A` is an ordinary required kwarg
  (`:128`) consumed as `L = Nscales * sigma1A**2` (`:194`) with no
  reference to whether `fn_b0map` was given. `run_b0_recon.py` does the
  right thing (power-iterates via `estimate_spectral_norm` first) but is
  the only caller that does, and the failure mode is ugly: a too-small
  `sigma1A` means a too-small Lipschitz constant, so POGM takes an
  over-long step and diverges rather than failing cleanly, after however
  many minutes the run has already burned. Cheapest fix preserving the
  current contract: accept `sigma1A=None` and, when it is None, call
  `estimate_spectral_norm` on the operator just built -- which also
  resolves half of item 49's "no wired-up way to run a fresh
  reconstruction" for the B0 path. Interacts with item 76: whatever
  iteration count that item settles on becomes this path's default too.
- [ ] **94. `recon/run_b0_recon.py` re-implements `run_recon`'s smaps
  load + RSS normalization verbatim.** `:79-81` is a line-for-line copy of
  `recon/reconstruct.py:154-156`. Item 89 covers the `GatheredSenseB0` /
  `GatheredSense` duplication but not this one, and this copy is the more
  dangerous of the two: the whole purpose of `run_b0_recon`'s preamble is
  to measure `sigma1A` for the operator `run_recon` will build moments
  later, so any future change to how smaps are normalized silently
  desynchronizes the estimate from the reconstruction it is estimated for
  -- with no error, just a wrong step size. Note this is the *same* class
  of bug as item 74's omega divergence, in the same function, for the same
  reason. Factor the load+normalize into one helper both call.
- [ ] **95. `run_b0map.py`'s two `# noqa: BLE001` codes are inert, and one
  is doubly so.** `preprocessing/run_b0map.py:94` (`except Exception`) and
  `:108` (`except subprocess.CalledProcessError`) both carry `BLE001`
  suppressions. `BLE` isn't in `pyproject.toml`'s `select = ["E", "F",
  "I"]`, so neither does anything today (item 48's territory) -- and
  `:108` catches a *narrow* exception, so `BLE001` would not fire there
  even if `BLE` were enabled. The explanatory comments are worth keeping;
  the codes are cargo-culted from the five sibling batch drivers, whose
  `except Exception` form does match the rule. Fold into item 48, whose
  proposed `B` ruleset would newly flag `:108`'s suppression as unused.

## Still open: pass-3 review findings (2026-09-01, against `b8757a7`)

Findings from a third repo-wide review for correctness, consistency and
conciseness, run against the tree as merged to `main`. Numbering
**continues at 61** for the same reason pass 2 continued at 36: source
files and this file's own prose cross-reference earlier item numbers by
name (`preprocessing/grid_resize.py` and
`tests/test_preprocessing_grid_resize.py` cite "item 12",
`lib/make_prephasers.py` cites "item 28", the `trap4ge` paragraph cites
"item 17"), so numbers are never reused.

Same conventions: **[measured]** = reproduced by running the code, with
the number quoted. **[verify]** = suspicious, but needs a judgement call
against the MATLAB reference before acting.

**Nothing from pass 2 has been fixed yet.** Items **36-60** below are all
still open, as are **8, 13, 15, 17, 32, 33** carried from the 2026-08-31
pass -- re-verified this pass by reproducing the recorded baseline
exactly (20 `E501` ruff errors; 120 passed / 18 skipped) and by
re-measuring items 36, 37, 38 and 43 directly (numbers below). This pass
adds items 61-73 and supersedes the recorded numbers in items **43**,
**46** and **47** (see 70 and 73's cross-references).

Baseline at review time, all four combinations measured (the `output/`
column is item 46's gate -- `output/` is gitignored, so "no" is what a
fresh clone and any CI run see):

| venv | `output/` present | `uv run pytest` |
|---|---|---|
| `uv sync --extra test --extra lint` | no | **120 passed, 18 skipped** |
| `uv sync --extra test --extra lint` | yes | **125 passed, 13 skipped** |
| `+ --extra preprocessing` | no | **149 passed, 14 skipped** |
| `+ --extra preprocessing` | yes | **154 passed, 9 skipped** |

`uv run ruff check .` -> 20 errors, all `E501`, unchanged (item 15). The
9 residual skips in the richest configuration are 4 torch-gated
`tests/test_recon_*.py` module skips plus 5 julia-gated cases in
`tests/test_preprocessing_run_b0map.py`; nothing else is gated.

Whole-sequence feasibility, re-measured this pass on a full default-params
`main.py` build (GE_MR750, `PNSwt = [0.8, 1.0, 0.7]`, seed 0) -- all four
sequences pass `.ok`:

| sequence | peak PNS | acoustics | max grad | max slew |
|---|---|---|---|---|
| `ArbEPI.seq` | **79.8%** | **0.1484** | 50.00 mT/m | 119.0 T/m/s |
| `EPIcal.seq` | 78.1% | 0.1484 | 50.00 mT/m | 119.0 T/m/s |
| `deGRE.seq` | 77.4% | 0.2456 | 49.76 mT/m | 174.3 T/m/s |
| `noise.seq` | 0.0% | 0.0000 | 0.00 mT/m | 0.0 T/m/s |

ArbEPI's 79.8% and deGRE's 0.2456 both match what this file already
records; ArbEPI's acoustics does not (see 66).

### Correctness

- [ ] **61. [measured] `deGRE.seq` declares the EPI FOV, not its own.**
  `sequences/deGRE.py:237` writes `seq.set_definition('FOV', params.fov)`,
  but every gradient in that sequence is built from `params.fov_degre`
  (`deltak = [1 / f for f in params.fov_degre]` at `:85`, matrix
  `N_degre`). Confirmed in the written file: `output/deGRE.seq`'s
  `[DEFINITIONS]` block reads `FOV 0.216 0.216 0.0405` while the sequence
  actually encodes `0.216 0.216 0.042` -- **1.5 mm / 3.6% wrong on z**
  (x/y happen to match, since `fov_degre` tracks `fov` exactly there).
  `ArbEPI.py`/`EPIcal.py` are both correct; this is deGRE-only, and it is
  the same `fov`-vs-`fov_degre` confusion class as item 39. Fix:
  `params.fov_degre`. Note the excitation slab thickness on `:74`
  (`0.9 * params.fov[2]`) must *stay* on `params.fov` -- imaging the same
  slab as the EPI sequence is deliberate, see that module's docstring --
  so only the definition line changes.
- [ ] **62. [measured] deGRE exports the *prescribed* `TE_degre` while
  playing a different pair, and ΔTE is not preserved exactly -- correcting
  item 37's claim that it is.** `delay_te` is computed per echo,
  independently, as `ceil((te - te_min) / raster) * raster`
  (`sequences/deGRE.py:164`). The two prescribed TEs differ by
  `1 / fat_offres_freq` = 2236.90 us, which is **not** a multiple of the
  4 us gradient raster (559.225 rasters), so the two ceils land on
  different sub-raster residues and the difference does not survive.
  Measured (folding in items 36/37's corrections to `te_min`): realized
  TE = **3.120 / 5.356 ms** against prescribed **3.0369 / 5.2738 ms** --
  83.1 us and 82.2 us late -- and realized ΔTE = **2.2360 ms vs 2.23690 ms
  prescribed, 0.90 us short (0.040%)**. Two consequences. (a)
  `sequences/ArbEPI.py:338` writes `params.TE_degre` (the *prescribed*
  pair) into `scan_info.mat`, and `preprocessing/julia/b0map.jl` divides
  the echo phase difference by that ΔTE to get Hz -- so every field map
  carries a **0.040% scale error** (0.1 Hz at a 250 Hz offset). (b) Item
  37's "`dTE` is preserved exactly, so the field-map scaling and the
  `dTE = 1/fat_offres_freq` fat-cancellation argument are both unaffected"
  is not quite right: preserved to 0.04%, not exactly (residual fat phase
  0.0025 rad, still negligible). Both effects are small, but the exported
  number should be the realized one. Cleanest fix: derive `delay_te[1]`
  from `delay_te[0]` plus a raster-multiple offset, and export the
  realized pair rather than `params.TE_degre`. The doc claim in item 37
  needs correcting either way.
- [ ] **63. [measured] The ADC window overruns `Tread`, so the last
  samples of every echo are acquired while the ky/kz blip is already
  playing -- the opposite of what the code says it does.**
  `lib/make_readout_grads.py:309` comments "Delay blips to play after the
  ADC window closes" and sets `gy_blip.delay = gz_blip.delay = Tread`.
  But `Nfid = round(Tread / dwell / 4) * 4` (`:263`) rounds to the
  *nearest* multiple of 4, in either direction. At default params:
  `Tread` = 684.0 us, `Tread / dwell` = 342, `Nfid` = **344**, so the ADC
  spans 688.0 us -- a **4.0 us / 2-sample overrun** past the blip's own
  start. Magnitude is small: the unit blip accrues 2.08e-4 of its full
  area in its first 4 us (it is still on the rise ramp), so at the largest
  step this build uses (`max_ky_step` = 37) the worst-case ky error on
  those two samples is **0.0077 index units = 0.036 m^-1 against
  `deltak_y` = 4.63 m^-1, i.e. 0.17% of one k-space step**. Not a recall,
  but worth fixing because the comment asserts the opposite and the error
  grows as `dwell^2` (the overrun is bounded by `2 * dwell`), so a coarser
  dwell or a different `Nx` makes it worse silently. Note the ±kmax
  coverage loop just above is *not* also wrong: it integrates
  `_composite_segments` out to `(Nfid - 0.5) * dwell`, over the whole
  composite lobe, so it already accounts for the overrun. Fix: either
  correct the comment, or floor instead of round
  (`Nfid = floor(Tread / dwell / 4) * 4` -> 340 here), which costs a
  slightly larger flat top to keep the same coverage.
- [ ] **64. [measured] `run_rss.py`'s `_ift3` justifies its FFT-shift
  convention with a premise that is false for this repo's actual matrix
  size.** Its docstring defends `fftshift(ifftn(fftshift(.)))` (rather
  than the conventional `fftshift(ifftn(ifftshift(.)))`) as "identical for
  even-length axes, which every dimension in this pipeline is".
  `params.py` sets `N = [240, 240, 45]` and `N_degre = [108, 108, 21]` --
  **z is odd on both grids**. Measured on a length-45 axis, the two
  conventions differ by **144% in relative complex value** (and by exactly
  0 at n = 44, as claimed). The difference is a one-sample circular shift
  of the k-space input, i.e. a pure linear phase ramp in image space, so
  both current consumers happen to be immune -- `_rss_recon` takes
  `np.abs` (measured magnitude difference 1.7e-16), and
  `preprocessing/julia/b0map.jl`, which this file's preprocessing section
  says uses "the same centered-FFT convention as `run_rss.py`'s `_ift3`",
  differences two echoes on the same grid so the ramp cancels. But the
  stated *reason* is wrong, and any future complex-valued consumer of
  `_ift3` inherits the ramp silently on z. Fix the docstring to say the
  convention is magnitude-/difference-safe rather than shift-equivalent,
  or just switch the input to `ifftshift`. Related, and the third distinct
  spelling of this same idea in the repo (item 44 covers the first two):
  `ge/acoustics.py:77` uses `ifftshift` where
  `check_grad_acoustics.m` uses `fftshift` -- provably equivalent there,
  since `n1 + ZF_FAC * n1 = 8 * n1` is always even, as the comment two
  lines above already argues for a different purpose.

### Consistency & documentation

- [ ] **65. [measured] `sequences/noise.py` is the only sequence that
  never sets its `Name`/`FOV` definitions.** `ArbEPI.py:344-345`,
  `EPIcal.py:141-142` and `deGRE.py:237-238` all call
  `seq.set_definition('FOV', ...)` + `seq.set_definition('Name', seqname)`
  before `seq.write(...)`; `noise.py:69` writes straight out. Confirmed in
  the written file: `output/noise.seq`'s `[DEFINITIONS]` block has neither
  key, so `generate_noise`'s `seqname` parameter reaches only the
  filename. Nothing in `ge/` reads these definitions today, so this is
  cosmetic -- but it is a gratuitous divergence in the one sequence file
  that already diverges on `sys_seq` (item 56) and on `adc_dead_time`
  (item 13), and it is the odd one out on console output too (it alone
  prints `Validating sequence...` / `Sequence written to:` around its
  timing check). Worth folding into whichever of 13/56 gets fixed first.
- [ ] **66. [measured] The ArbEPI acoustics numbers in `ge/check.py` and
  in this file are stale by 5x since the POPE switch, and the disclaimer
  around them doesn't cover it.** Both cite `ArbEPI.seq: 0.028146 here vs
  MATLAB's 0.02814424`. Today's `ArbEPI.seq` measures **0.1484** through
  the same check -- a 5.3x change, from the asymmetric readout ramps
  (`9ae28d5`). The surrounding "historical record, not a current claim"
  disclaimer names only the `GRE.seq` -> `deGRE.seq` rename, so the
  ArbEPI figures read as current when they are not. The *window* half of
  that reproduction record does still hold exactly: measured
  0.100000 s = **25000 samples** at the 4 us raster, matching the recorded
  "25000 vs MATLAB's 25001". So re-record only the magnitude, and extend
  the disclaimer to cover the POPE change as well as the rename. (The full
  current four-sequence table is in this section's preamble above.)
- [ ] **67. `preprocessing/nifti_io.py`'s module docstring describes
  `run_b0map`'s pre-resize behavior.** It says the field map is written
  "on the deGRE grid, so its voxel size comes from `fov_degre`, not
  `fov`". `run_b0map.py:119-123` passes the *EPI*-grid `b0map_hz` with
  `fov=seq_params.fov`, and its own comment two lines up says "now on the
  EPI grid/fov like every other NIfTI this pipeline writes". `nifti_io`'s
  docstring is the stale half, and it names the exact wrong value someone
  would "restore" it to. Same class as item 50 (`preprocess.py`'s
  docstring contradicting this file), just in the other direction.
- [ ] **68. `tests/test_ge_check.py`'s two smoke tests check against a
  different scanner than the sequences were built for.**
  `test_check_seq_feasibility_runs` (`:75`) and
  `test_check_seq_feasibility_noise_has_no_gradients` (`:91`) both
  hardcode `SCANNERS['GE_UHP']`, but the `output/*.seq` files they read
  were built under `params.py`'s `GE_MR750` -- different `max_grad`
  (100 vs 50 mT/m) and a different `chronaxie`/`rheobase`/`alpha` triple,
  so a different PNS model entirely. Their assertions are only `>= 0`, so
  nothing is wrong today; but any strengthening of them would silently
  measure the wrong hardware, and the sibling PNS regression test in the
  same file (item 38's target) correctly uses `p.spec`. Use
  `load_params().spec` here too, and note this interacts with item 46 --
  these are exactly the tests that never run on a fresh checkout.
- [ ] **69. `preprocessing/smaps.py`'s `cal_size` "crop" is a zero-*pad*
  on z at this repo's real dimensions.** `estimate_smaps` resizes k-space
  to `(ncoils, cal_size, cal_size, cal_size)` = 24^3, but
  `Nz_degre = 21 < 24`, so sigpy pads z rather than cropping it. This is
  *not* a geometry bug -- k-space zero-padding interpolates while
  preserving FOV, so `process_smaps`' subsequent `fov_gre` arithmetic
  stays correct -- but two docstring claims are then wrong: "center-crop
  ksp_gre's spatial dims to this matrix size" is true only for x/y, and
  "so EspiritCalib's own internal calibration-region crop becomes a no-op"
  now means a 24-wide calibration region on z of which 3 rows are
  synthetic zeros, which is a (small) real effect on the fit rather than a
  no-op. Either clamp per axis (`min(cal_size, n)`) or state explicitly
  that z is padded and why that is acceptable.

### Test & tooling health

- [ ] **70. [measured] Item 46 undercounts the `output/`-gated tests (5
  cases, not 3), and item 47's recorded baselines don't reproduce in
  either venv -- superseded by the table in this section's preamble.**
  `test_check_seq_feasibility_runs` and `test_seq2ceq_self_consistency`
  are each `@pytest.mark.parametrize`d over `['noise.seq', 'ArbEPI.seq']`,
  so the gated set is **5 test cases across 3 test functions**, not 3
  tests -- confirmed by the exact 5-test delta between the `output/`-
  present and `output/`-absent rows of both venvs in the preamble table.
  Item 47's recorded "123 passed, 15 skipped" (plain venv) and "157 passed
  / 6 skipped" (preprocessing extras) both come from environments that no
  longer reproduce; the current four measurements are 120/18, 125/13,
  149/14 and 154/9. Re-record from the table above once item 46 is
  settled, and fix item 46's own count while you're there.

### Conciseness & performance

- [ ] **71. [measured] Item 43's `trap4ge.flat_area` staleness is gated by
  item 17, not just by having no consumer -- and the two should be fixed
  together.** Item 43 calls the wrong `flat_area` "latent today" because
  nothing reads it. There is a second, stronger reason it cannot currently
  fire: while `crt == grad_raster_time`, `trap4ge`'s rounding changes
  nothing (item 17), so `gin.area / gout.area` is *identically 1* and the
  stored `flat_area` is exactly right -- measured, ratio 1.0000000 at
  `crt = 4e-6`. Forcing the rounding to actually bite (`crt = 20e-6`, the
  value item 17 says to revert to for Siemens dual-raster support) makes
  it wrong immediately: stored `flat_area` **666.667 vs the correct
  657.895, off by 1.33%**, and `pp.scale_grad` propagates the error
  faithfully (scaled: stored 1333.33 vs correct 1315.79). So the one-line
  fix item 43 proposes is a prerequisite for item 17's revert, not
  independent cleanup -- worth saying so in both entries.
- [ ] **72. Two dead expressions and one misleading name.** (a)
  `sequences/deGRE.py:204`'s `pe2_steps[max(0, iZ - 1)]` sits inside
  `... if iZ > 0 else 0.0`, so `iZ - 1 >= 0` always and the `max` can
  never bind -- drop it (the neighbouring `pe1_steps[iY]` on `:203`
  correctly has no such guard); still open. (b) [x] fixed, incidentally, by
  the B0 field map investigation's `_load_smaps` ->
  `preprocessing.smaps.load_smaps` move (see the `preprocessing/`/`recon/`
  sections below): `preprocessing/recon_frames.py` used to bind
  `smaps, _nvcoils = _load_smaps(...)` with a leading underscore
  signalling "unused", then use it later to populate
  `seq_params_out['Nvcoils']` -- the (now 4-tuple, `smaps_degre`/
  `emap_degre` added) unpacking site renames it to `nvcoils`. Both were the
  same class as the first review's item 34.
- [ ] **73. `recon/lowrank.py`'s `pcount` buffer-reuse parameter is dead
  in production.** `patchSVST` accepts `pcount` and forwards it to
  `patches2img`, which zeroes and reuses it instead of allocating -- but
  the only production caller, `recon/reconstruct.py`'s `g_prox`
  (`:195`), never passes one, so every call allocates a fresh
  `(Nx, Ny, Nz)` float32 count array (10.4 MB at this repo's real
  240x240x45 dims, once per scale per iteration). PyTorch's caching
  allocator makes that nearly free, so this is tidiness rather than a
  measured cost -- but it is a parameter that exists solely to be passed
  and never is. Either thread a persistent buffer through from
  `run_recon`, or delete the parameter from both functions. Same
  disposition question as item 49's `poweriter`, and both live in the
  `recon/` module that item 49 already notes has no wired-up
  non-validation entry point.

## Still open: pass-2 review findings (2026-09-01, against `646d9a2`)

Findings from a second repo-wide review for correctness, consistency and
conciseness, run against the post-fix-pass tree. Numbering **continues at
36** rather than restarting, because three source files cross-reference
the first review's item numbers by name (`preprocessing/grid_resize.py`
and `tests/test_preprocessing_grid_resize.py` cite "item 12",
`lib/make_prephasers.py` cites "item 28", and this file's own
`trap4ge` paragraph cites "item 17") -- see the archived section below,
which is kept intact for exactly those references.

Same conventions as before: **[measured]** = reproduced by running the
code, with the number quoted. **[verify]** = suspicious, but needs a
judgement call against the MATLAB reference before acting.

Baseline at review time (`uv sync --extra test --extra lint`, plain main
venv, no `output/` directory present): `uv run ruff check .` -> 20 errors
(all `E501`); `uv run pytest` -> **120 passed, 18 skipped** (see 47).
Full default-params `generate_arbepi` build: **18.1 s** on 4 cores
(masks 1.4 s, `_compute_schedules` 7.6 s, assembly + write ~9 s) -- down
from the 46.8 s the first review measured, thanks to its items 29/30.

### Still open from the 2026-08-31 review

Carried forward so there is one actionable list; full detail in the
archived section below.

- [ ] **8** -- `check_grad_acoustics` axis cross-product: investigated,
  confirmed a faithful port, no code change. Left as a documented
  judgement call.
- [ ] **13** -- `sequences/noise.py:41-44` dead `adc_dead_time`
  arithmetic. Re-confirmed still live this pass: `spec.adc_dead_time` is
  `20e-6`, zeroed on `:42`, then added on `:43`, so the term is always 0
  and `pad_duration` is right only by accident.
- [ ] **15** -- `uv run ruff check .` errors. Still 20, still all
  `E501`, same files. Now the only standing lint debt in the repo.
- [ ] **17** -- `trap4ge` is a no-op at `crt == grad_raster_time`, and
  its docstring still claims the Siemens-10us/GE-4us dual-raster
  rationale. Neither the docstring correction nor the suggested
  no-op-pinning test landed.
- [ ] **32** -- `plot_trajectory(frame_idx=...)`'s whole-sequence
  `calculate_kspace()`: investigated, deliberately not fixed.
- [ ] **33** -- `ReadoutGrads.max_blip_area` is dead but kept
  deliberately (pre-POPE symmetric-ramp safety net).

### Correctness

- [ ] **36. [measured] `deGRE`'s realized TR is ~8.24 ms, not the
  prescribed 8.0 ms -- a live 3% timing error.** `sequences/deGRE.py`'s
  `tr_min` (`:170-177`) charges the spoiler block only
  `pp.calc_duration(gx_spoil)`, but that block (`:223-225`) also plays
  `pp.scale_grad(gy_pre, -y_step)` and `pp.scale_grad(gz_pre, -z_step)`,
  and a pypulseq block's duration is its longest event. Measured at
  default params: `gx_spoil` 760 us, `gy_pre`/`gz_pre` 1000 us each, so
  the real block is 1000 us and `tr_min` under-estimates by **240 us**
  per TR (`tr_min` as coded 5.372/7.608 ms vs. the real block sum
  5.612/7.848 ms). `delay_tr` is then 240 us too generous and the
  realized TR comes out **8.244 ms / 8.240 ms against a prescribed 8.0
  ms**. Three consequences, in rough order of importance: (a)
  `params.alpha_degre` is the Ernst angle for `TR_degre = 8 ms`, so the
  flip angle no longer matches the TR actually played; (b) the
  `delay_tr < 0` guard added by the first review's item 6 is evaluated
  against the under-estimated `tr_min`, so it can pass when the true
  minimum TR does not fit; (c) the deGRE scan runs 3% longer than
  planned. Fix: `max(pp.calc_duration(gx_spoil),
  pp.calc_duration(gy_pre), pp.calc_duration(gz_pre))`. Take the same
  `max` over the prephase block at `:218` while you're there -- all three
  happen to be exactly `Tpre` (1000 us) today, so it is currently a
  no-op, but it is the same latent shape and `gx_pre` carries a different
  area from the other two. This is the deGRE-inline analogue of the first
  review's item 28 `make_prephasers` finding, which was fixed only for
  the shared helper.
- [ ] **37. [measured] `deGRE`'s `te_min` still uses the RF-block-midpoint
  measure that `calc_te_tr_delays` was already fixed for -- 80 us late.**
  `sequences/deGRE.py:157` computes `max(calc_duration(rf),
  calc_duration(gz_ss)) / 2`, i.e. the RF *block*'s midpoint, which sits
  `(rf_ringdown_time - rf_dead_time) / 2` away from the physical RF
  center (260 us vs 100 us here). `lib/calc_te_tr_delays.py:44` was
  corrected to `max(...) - (rf.delay + pp.calc_rf_center(rf)[0])` for
  exactly this reason (see the POPE paragraph's "measured 'RF center' as
  the RF block's midpoint" note), but deGRE's inline copy was not.
  Measured: `te_min` as coded 2.7680 ms vs. 2.8480 ms corrected -- an
  **80.0 us** under-estimate, so both echoes land 80 us later than
  prescribed. Physically mild, because both echoes shift by the *same*
  80 us: `dTE` is preserved exactly, so the field-map scaling and the
  `dTE = 1/fat_offres_freq` fat-cancellation argument (see `TE_degre`'s
  definition in `params.py`) are both unaffected. Worth fixing anyway --
  the absolute TE is simply wrong, and the repo currently carries two
  different definitions of "time from RF to echo" in two files. (The
  corrected number above also folds in item 36's prephase-block `max`;
  the RF-center term alone is the whole 80 us.)
- [ ] **38. [measured] The 80% PNS regression test guards a *different*
  sequence than the one the repo ships.**
  `test_arbepi_default_params_peak_pns_under_normal_mode_limit`
  (`tests/test_ge_check.py:98`) builds with `Nframes=1`, so it only ever
  sees frame 0's sampling mask. Measured, seed=0, GE_MR750,
  `PNSwt = [0.8, 1.0, 0.7]`: **frame 0 = 78.92%**, **frame 10 = 79.84%**.
  This file's own headline number ("79.8% peak on the full ArbEPI build
  (GE_MR750, seed=0)", in the PNS finding history) is frame 10's -- so
  the shipped configuration sits ~0.9 percentage points closer to the
  limit than the thing the test measures. Given `blip_slew = 105` is an
  explicit ride-the-line choice with ~0.2% of margin, the guard can pass
  while the real 30-frame build is over 80%. Mechanism: `max_blip_steps`
  is taken over whatever frames are built, and the full build reaches
  `(max_ky_step, max_kz_step) = (37, 5)` while frame 0 only reaches
  `(37, 4)` -- frames 3, 6, 10, 11 and 14 carry the kz-5 step, and since
  blips are stored at unit amplitude and scaled per echo at assembly, the
  1-frame test literally never plays the largest kz blip the real
  sequence plays. Fix options, cheapest first: build the *worst* frame
  (compute `_compute_schedules` over all frames, pick
  `argmax` of the per-frame blip steps, build that one), or parametrize
  the test over 3-4 frames, or accept the ~5x cost of a
  `Nframes=30` build in the test. Whichever, re-record the quoted 79.8%
  alongside it so the doc and the guard measure the same thing.
- [ ] **39. `resize_to_epi_grid` raises on a z-FOV mismatch but silently
  mis-registers on an x/y one.** `preprocessing/grid_resize.py` checks
  `fov_src[2] < fov[2]` and raises, then crops z only -- x/y are handed
  straight to `ndimage.zoom`, which rescales the array without ever
  looking at the FOVs. So if `fov_src[:2] != fov[:2]` the smaps/B0 map
  come out geometrically wrong with no error anywhere. This is not
  hypothetical: `params.py:243` builds `N_degre = np.ceil(fov /
  res_degre - 1e-9)`, whose own comment already says "x/y get the same
  treatment for consistency even though nothing currently enforces it
  there". At today's values 216 mm / 2 mm = 108 exactly, so x/y match and
  the bug is dormant -- but any FOV that isn't an integer multiple of
  `res_degre` (change `res` or `N` on x/y and it stops being one) makes
  `fov_degre[:2] > fov[:2]` and silently shifts every sensitivity map and
  field map relative to the EPI grid. Fix: raise unless `fov_src[:2]` and
  `fov[:2]` agree to a tolerance, matching the z check's own strictness
  (or extend the crop to x/y, which is the more general fix but more
  code).
- [ ] **40. `preprocess()` leaks the output HDF5 handle when the resume
  fast-forward fails.** `preprocessing/preprocess.py:386-387` opens `mf =
  h5py.File(paths.recon, 'a')` and calls `resume_start_frame(mf,
  epi_reader, shots_per_frame)` *outside* the `try/finally` at
  `:402-426`. If the archive is short (the exact case resume exists to
  survive), the fast-forward's `epi_reader.next_frame()` raises a bare
  `StopIteration` -- bypassing the friendly `RuntimeError` at `:421` that
  explains what happened -- and leaves the checkpoint file open, on a
  path where the process is likely about to retry. Move the open + resume
  inside the `try`, or open `mf` with a `with`.
- [ ] **41. `smaps.load_smaps` doesn't validate the smaps cache's
  `Nvcoils`; `preprocess.py` does.** [reference updated: this function was
  `recon_frames._load_smaps` at review time, moved to
  `preprocessing.smaps.load_smaps` by the B0 field map investigation (see
  `preprocessing/`/`recon/` sections below) -- the bug itself is
  unaffected by the move, still open.] `preprocessing/preprocess.py:315-321`
  guards the shared `smaps_<seqname>_sigpy.h5` cache with `smaps_valid =
  f.attrs.get('Nvcoils') == Nvcoils` and re-estimates on a mismatch.
  `smaps.load_smaps` reads the same file and returns whatever is on disk.
  A cache left over from a run whose
  noise/deGRE data selected a different `Nvcoils` then reaches
  `recon_fn(data, smaps)` with mismatched coil counts -- a shape error at
  best, a silently wrong reconstruction at worst. Port the same attr
  check across (it already reads `f.attrs['Nvcoils']` two lines away).
- [ ] **42. `recon_frames`'s frame cap can be a float.**
  `PreprocessingConfig.Nframes` is typed `float` and defaults to
  `float('inf')`; `recon_frames.py` does `nframes = min(cfg.Nframes,
  nframes_avail)` and then `range(nframes)`. `min(inf, 30)` returns the
  int, so the default works, but any float override (`cfg.Nframes = 10.0`
  -- natural given the declared type) gives `TypeError: 'float' object
  cannot be interpreted as an integer`. Wrap in `int()`.
- [ ] **43. `trap4ge` overwrites `.area` but leaves `.flat_area` at its
  pre-rescale value.** `lib/trap4ge.py` rebuilds the trapezoid at a dummy
  amplitude, rescales `gout.amplitude` to preserve `gin.area`, then sets
  `gout.area = gin.area` -- but never touches `gout.flat_area`, which
  still reflects the dummy amplitude. Latent today: pypulseq recomputes
  `flat_area = amplitude * flat_time` when a block is read back
  (`Sequence/block.py:455`), and nothing in this repo reads `flat_area`
  directly. But `pp.scale_grad` *does* propagate `flat_area`
  (`scale_grad.py:33`), so a scaled trap4ge output carries a
  wrong-by-construction value that any future consumer would take at face
  value. One line: `gout.flat_area = gout.amplitude * gout.flat_time`.
- [ ] **44. Two different FFT-shift conventions on the same axis, in the
  same odd/even-phase pipeline.** `preprocessing/oephase.py`'s
  `epiphasecorrect` uses `fftshift(ifft(fftshift(.)))` (and the mirrored
  `fftshift(fft(fftshift(.)))` coming back), while
  `preprocessing/preprocess.py`'s `compute_oephase` uses
  `ifftshift(ifft(fftshift(.)))`. These are identical for even `nx` (the
  only case in production, `Nx = 240`) and diverge by one sample for odd
  `nx`. Pick one and use it in both places, or state explicitly at both
  sites that even-`Nx` is assumed -- this is the same class of
  even/odd-`fftshift` trap `_center_out` already carries an explicit
  warning about in `lib/mask2epi.py`.
- [ ] **45. [verify] `check_seq_feasibility`'s `max_slew` under-reports
  ramps shorter than ~2 gradient rasters.** `ge/check.py:228` computes
  `np.abs(np.diff(gw_tm, axis=1) / dt).max()` over samples taken at bin
  *centers* (`:163`, `t = t_start + (arange(nt) + 0.5) * dt`). For a ramp
  spanning `r >= 2*dt` the interior differences recover the true slew
  exactly, so today's gradients (POPE ramps are ~50 rasters) are
  reported correctly. But a 1-raster ramp -- which `trap4ge`'s
  `crt = 4e-6 == grad_raster_time` rounding permits for a small
  trapezoid -- reads at roughly *half* its real slew. This is the number
  `.ok` gates against `spec.max_slew`, so it is the wrong direction to be
  wrong in. Inherited from pypulseq's own `Sequence/calc_pns.py` sampling
  pattern, hence [verify] rather than a plain bug: decide whether to
  match pypulseq or to compute slew from the trapezoid parameters
  directly.

### Test & tooling health

- [ ] **46. The 3 tests that gate on `output/*.seq` silently skip on a
  fresh checkout.** `tests/test_ge_check.py:71,87` and
  `tests/test_seq2ceq.py:27` `pytest.skip` when `output/noise.seq` /
  `output/ArbEPI.seq` are missing -- which is every clone and every CI
  run, since `output/` is gitignored. The whole `seq2ceq` -> `.pge`
  path and the `check_seq_feasibility` smoke tests therefore only run for
  someone who happens to have run `main.py` first. The PNS test in the
  same file already shows the fix: build a small sequence into `tmp_path`
  inside the test (see item 38, which wants that test's build widened
  anyway).
- [ ] **47. [measured] This file's recorded pytest baseline is stale.**
  The 2026-08-31 item 14 entry records "123 passed, 15 skipped" for the
  plain main venv. Actual today: **120 passed, 18 skipped**. The 3-test
  delta is exactly item 46's `output/`-gated tests -- so the recorded
  number was taken in a working tree that had already run `main.py`, and
  is not reproducible from a clean checkout. Re-record it (and the
  "157 passed / 6 skipped with the preprocessing extras" figure, which
  has the same exposure) once item 46 is settled.
- [ ] **48. Ruff's `B` (flake8-bugbear) ruleset finds 45 issues the
  current `select = ["E", "F", "I"]` can't see; two are worth acting
  on.** `B023` at `lib/mask2epi.py:294` -- `max_excl` closes over
  `top_idx`/`top_vals`, which are rebound on every pass of the enclosing
  `for _ in range(max_passes)` loop. Currently safe (the closure is
  defined and consumed within one pass), but it is a genuine footgun in
  the hottest function in the file. `ARG001` at `ge/writeceq.py:103` --
  a `parent_by_id` parameter that is never used in the body. The rest are
  `B905` (12x `zip` without `strict=`), `B028` (3x `warnings.warn` with
  no `stacklevel` -- including both of `calc_te_tr_delays`' TE/TR
  warnings, which is exactly where a caller-side line number would help),
  `ARG005`/`ARG001` in test doubles and callbacks, and one `B011`
  `assert False` in `tests/test_gen_sampling_masks.py:47`. Consider
  adding `B` to `[tool.ruff.lint] select` after clearing the two above.
- [ ] **49. `recon/solvers.py`'s `poweriter` is dead, and its one
  natural consumer requires the value it computes.** `poweriter` has
  exactly one reference in the repo: its own `def`. Meanwhile
  `recon/reconstruct.py`'s `run_recon` takes `sigma1A` as a *required*
  keyword-only argument, and the only caller,
  `recon/validate_against_mslr.py`, reads it out of the Julia reference
  `.mat`. So there is currently no wired-up way to run a fresh (non-
  validation) reconstruction -- the helper that would supply `sigma1A`
  exists but nothing calls it. Either default `sigma1A=None` and fall
  back to `poweriter(A.apply, A.adjoint, ...)`, or delete `poweriter` and
  say in `run_recon`'s docstring where `sigma1A` is meant to come from.
  Related, and already noted in the `recon/` section below: `run_recon`
  still has no `save_result` driver.

### Consistency & documentation

- [ ] **50. `preprocessing/preprocess.py`'s module docstring contradicts
  this file.** Its first paragraph still says "this module has NOT been
  run end-to-end against real data" and "Running this module against a
  real acquisition is the user's own end-to-end verification step" --
  while the `preprocessing/` section below records exactly that run:
  `preprocess.py` -> `run_rss.py` on `wb_2.4mm` (GE_UHP) against a real
  MATLAB/BART RSS reference, 0.19% relative L2 error, Pearson
  r = 0.999997, metadata matching exactly. Rewrite the docstring to point
  at that validation instead of disclaiming it.
- [ ] **51. `ge/coppe.py` and `ge/README.md` are invisible in CLAUDE.md.**
  Zero occurrences of "coppe" in this file, even though the GE export
  section enumerates every other module under `ge/` by name and
  `coppe.py` is the largest file in the directory (582 lines, plus its
  own `ge/README.md` and `tests/test_coppe_assign.py`). README documents
  it; CLAUDE.md -- the file actually consulted when working in this repo
  -- doesn't mention it exists. Add a sentence to the GE section: what it
  does (SSH-copies a folder of `.pge` files to the scanner,
  auto-allocating `pge2` entry numbers), that it is UM-lab-internal, and
  that it is *not* part of the `main.py --ge` path.
- [ ] **52. `sampling/external_mask.py` is unreferenced and undocumented
  on both sides.** Zero occurrences in CLAUDE.md; README's architecture
  tree summarizes `sampling/` as "(caipi, ticaipi, pd, rand)". The only
  reference anywhere outside the module is its own test file, and
  `gen_sampling_masks` has no `'external'` branch -- so using an
  externally-supplied mask means bypassing the documented entry point
  entirely and calling `generate_arbepi(omegas, ...)` by hand. Decide
  which it is and say so: wire it in as a fifth `params.sampling_method`
  (it would need a path parameter on `Params`), or document it in both
  files as a deliberately manual escape hatch for collaborator-supplied
  masks.
- [ ] **53. The item-12 warning points at a file that doesn't exist.**
  That entry closes with "it shifts `smaps`/`b0map_hz` relative to the
  EPI grid, which changes `recon/operators_b0.py`'s per-voxel phase terms
  in the in-progress, uncommitted B0-correction work". There is no
  `recon/operators_b0.py` (`recon/` holds `__init__`, `lowrank`,
  `operators`, `reconstruct`, `solvers`, `validate_against_mslr`) and the
  working tree is clean, so the uncommitted work it warns about is gone.
  Retarget the caveat at whatever replaces it, or drop it.
- [ ] **54. `params.py`'s `seed` comment still contradicts the default it
  sits on.** `params.py:290-294` says "None = a fresh unseeded rng each
  run (current default behavior)" immediately above `seed = 0`. This is
  the same contradiction the first review's item 20 fixed -- but it was
  fixed in `main.py`'s copy of the comment and left in place at the
  definition site.
- [ ] **55. `params.py`'s `PNSwt` comment points the wrong direction.**
  `:274-275` says "see the 'PNS-driven slew limits' comment below"; that
  comment is at `:152-182`, above it.
- [ ] **56. `sequences/noise.py` builds its `pp.Sequence` on the derated
  system; `ArbEPI.py`/`EPIcal.py` deliberately use full-hardware
  `params.sys`.** `noise.py:41` is `sys_seq = copy.deepcopy(sys)` where
  `sys = derated_sys(params)`, while both other sequence files use
  `copy.deepcopy(params.sys)` and carry a comment explaining why (the
  POPE fall ramp deliberately runs above the derate). `noise.seq` has no
  gradients at all, so nothing observable changes -- but the three files
  now disagree about which system object goes into `pp.Sequence`, with no
  note saying the divergence is intentional. Make it match, or say why
  not.

### Conciseness & performance

- [ ] **57. [measured] Pass 3 (`_euclidean_uncross_refine`) is now ~90% of
  `mask2epi_radial` and the single largest remaining hot spot in sequence
  generation.** With the first review's items 29/30 landed, the full
  30-frame build is 18.1 s on 4 cores, of which `_compute_schedules` is
  **7.6 s** (~28 s of CPU work, parallelized). Profiling one frame
  (0.94 s total): `_euclidean_uncross_refine` 1.22 s cumulative, versus
  `_sum_2opt_refine` 0.09 s and `_bottleneck_2opt_order` 0.03 s. Inside
  it, two pure-Python inner loops dominate: `_segments_cross`/`ccw` at
  **289 428 scalar calls** (0.47 s) and `span_has_pinned` at **108 530**
  generator constructions over a <=3-element frozenset (0.15 s). Two
  cheap, low-risk wins: (a) vectorize `_count_crossings`' all-pairs
  orientation test with numpy (it is a fixed O(m^2) cross-product over an
  `(m, 2)` array -- the whole thing is four broadcasted `ccw`
  evaluations, no Python loop needed); (b) replace `span_has_pinned`'s
  `any(i <= p <= j for p in pinned)` with a precomputed sorted array plus
  `bisect`, or simply hoist `min(pinned)`/`max(pinned)` out of the loop.
  Neither changes any result -- both are pure hot-path rewrites of exact
  predicates -- so `tests/test_mask2epi.py`'s existing crossing-count
  assertions are the regression guard.
- [ ] **58. `preprocessing/preprocess.py`'s `_build_omegas` and
  `_build_echo_times` are the same loop written twice** -- same
  `schedules[frame, :, :, 0/1].ravel()` indices, same `(Ny, Nz, Nframes)`
  scatter, differing only in what they write. One function returning both
  arrays would halve the index arithmetic and make it structurally
  impossible for the two grids to fall out of alignment (which is the
  whole premise of `_build_echo_times`' "always disambiguable via the
  'omegas' mask written alongside" docstring).
- [ ] **59. `epi_gridding.rampsampepi2cart` forces complex128 output.**
  `dc = np.empty((nx, etl) + dr2.shape[2:], dtype=complex)` regardless of
  input dtype, in a pipeline that is otherwise complex64 end to end
  (`preprocess.py` writes `ksp_epi_zf` as `np.complex64`). Doubles the
  per-frame working set for no precision that survives the cast on the
  way out. Use `dr.dtype`, or `np.result_type(dco, dce)`.
- [ ] **60. `sequences/EPIcal.py`'s shot loop is 1-based where
  `ArbEPI.py`'s is 0-based.** `for shot in range(-params.Ndummyshots + 1,
  params.Nshots + 1)` versus `for shot in range(params.Nshots)`. `shot`
  is used for nothing but `is_dummy = shot < 1`, so the 1-based offset
  buys nothing and quietly re-introduces the MATLAB indexing convention
  the "Index convention" section below says this port deliberately left
  behind. `range(-Ndummyshots, Nshots)` with `is_dummy = shot < 0` reads
  the same and matches the rest of the repo.

## Archived: 2026-08-31 review findings (against `99055a5`) -- resolved

Kept in full because `preprocessing/grid_resize.py`,
`tests/test_preprocessing_grid_resize.py`, `lib/make_prephasers.py` and
this file's own `trap4ge` paragraph all cite items from it by number.
The items still open (8, 13, 15, 17, 32, 33) are carried forward at the
top of the newer list above.

Findings from a repo-wide review for correctness, consistency and
conciseness. **[measured]** = reproduced by running the code, with the
number quoted. **[verify]** = suspicious, but needs a judgement call
against the MATLAB reference before acting.

Baseline at review time: `uv run ruff check .` -> 21 errors; `uv run
pytest` -> **0 tests** in the plain main venv (see 14); with the
un-collectable modules excluded, 112 passed / 9 skipped; with the `recon`
extra installed too, 139 passed / 14 skipped.

Status as of the 2026-08-31 fix pass: items 1-7, 9-12, 14, 16, 18-31,
34-35 fixed (28 resolved sub-item-by-sub-item, see its entry). 13, 17, and
33 deliberately left as-is (each is the correct safety net for a future
change, currently a no-op or dead under the current design). 8 and 15
investigated and confirmed accurate but left as open judgement calls (no
code change). 32 investigated but not fixed -- see its entry: the
apparent fix is a previously-abandoned approach.

### Correctness (2026-08-31)

- [x] **1. `sequences/noise.py` readout doesn't match the imaging readout --
  blocks all Stage 1 preprocessing. [measured]** Fixed: `generate_noise`
  now builds `sys` via `derated_sys(params)` and its readout via
  `make_readout_grads_from_params()`, the same helper `ArbEPI.py`/
  `EPIcal.py` use, instead of `params.sys` + a direct `make_readout_grads`
  call. `noise.py`'s `Nfid` now matches ArbEPI/EPIcal's (344 at default
  params). Regression test: `test_noise_nfid_matches_arbepi` in
  `tests/test_trajectory_matches_schedule.py`.
- [x] **2. `preprocess()` resume replays the EPI archive from shot 0 --
  silently writes every frame to the wrong index.** Fixed: the resume
  logic is now `resume_start_frame()` (`preprocessing/preprocess.py`),
  which fast-forwards `epi_reader` past `start_frame * shots_per_frame`
  already-consumed shots before the main loop starts, instead of leaving
  the reader at the top of the archive. Regression tests (fake reader,
  same style as `scatter_frame`'s location-encoding test):
  `test_resume_start_frame_fast_forwards_past_completed_shots` and
  `test_resume_start_frame_no_checkpoint_does_not_advance_reader` in
  `tests/test_preprocessing_preprocess.py`.
- [x] **3. Zero blip step raises `ZeroDivisionError`, defeating the
  documented ETL=1 guardrail.** Fixed: `make_readout_grads` only scales a
  blip to unit amplitude (`1 / max_*_step`) when that axis's max step is
  nonzero; a zero step now yields a zero-area placeholder of the matched
  duration instead (every real step on that axis is also 0, so the
  placeholder's own amplitude never matters downstream). Tests:
  `tests/test_make_readout_grads.py`.
- [x] **4. `pogm_restart(mom="pogm")` is broken by an in-place `g_prox`.
  [measured]** Fixed: `recon/solvers.py`'s POGM branch now calls
  `g_prox(znew.clone(), zetanew)`, so `xnew is znew` never holds even
  though `recon/reconstruct.py`'s `g_prox` still legitimately mutates its
  argument in place. Regression test
  (`test_pogm_restart_matches_closed_form_lasso_regardless_of_prox_style`
  in `tests/test_recon_solvers.py`) covers both `fpgm`/`pogm` and both
  in-place/pure prox styles against a closed-form LASSO solution --
  confirmed it fails without the fix (`pogm` + in-place: 3.9% relative
  error vs. the closed form) and passes with it.
- [x] **5. `_count_crossings` is blind to an open path's first-vs-last
  segment. [measured]** Fixed: dropped the `if i == 0 and j == m - 2:
  continue` special case from both `_count_crossings` and the identical
  skip in pass-3 stage 2 (also fixed the same bug duplicated in
  `tests/test_mask2epi.py`'s own `_has_any_crossing` helper, which would
  otherwise have stayed blind to exactly this class of crossing in test
  verification). Regression test:
  `test_count_crossings_detects_first_vs_last_segment` in
  `tests/test_mask2epi.py`, using the finding's own 4-/5-point repro.
- [x] **6. `generate_degre` has no negative-`delay_tr` guard.** Fixed:
  mirrors the existing `delay_te` check, raising a `ValueError` naming
  `TR_degre` and the achievable minimum TR. Regression test:
  `test_degre_raises_actionable_error_below_minimum_tr` in
  `tests/test_trajectory_matches_schedule.py`.
- [x] **7. `run_recon` re-derives the sampling mask from exact zeros.**
  Fixed: new `_load_omega()` helper in `recon/reconstruct.py` reads the
  authoritative `omegas` dataset when present (broadcasting it across the
  readout axis), falling back to the `!= 0` derivation -- with its
  per-coil consistency assert, now only needed there -- for recon files
  written before `omegas` existed. Regression tests:
  `test_load_omega_prefers_omegas_dataset_over_exact_zero_inference` and
  `test_load_omega_falls_back_to_exact_zero_inference_without_omegas` in
  `tests/test_recon_reconstruct.py`.
- [ ] **8. [verify -- resolved, not a bug] `check_grad_acoustics` scores
  every gradient axis against every axis's forbidden bands.** Checked
  against `../ArbEPI/lib/check_grad_acoustics.m` directly: its own loop
  (`for lg=1:n3, for l1=1:length(bands), for l2=1:size(bands{l1},1)`,
  storing the full `lg x l1` cross product in `val{lg}{l1}{l2}`) has the
  *exact* same structure -- this is a faithful port, not an indexing bug,
  and the <0.04% MATLAB agreement is real, not coincidental. Documented
  with an explicit comment in `ge/acoustics.py` so nobody "fixes" it
  later. No code behavior changed.
- [x] **9. Pass-3 swap moves are over-restricted by `span_has_pinned`.**
  Fixed: dropped `span_has_pinned(i, j)` from stage 1's swap branch only
  (kept for reversal), with a comment explaining why a swap doesn't need
  it, and corrected the stage-2 docstring's "exact tie" framing (now that
  stage 1 can consider these swaps directly, a genuine tie is the *only*
  remaining reason stage 1 wouldn't take one -- previously the structural
  block was doing that work incidentally). `tests/test_mask2epi.py`'s
  existing crossing-count assertions (its own suggested guard) all still
  pass.
- [x] **10. `deGRE` omits the `gz_ss` resync and has no centering test --
  this was a LIVE bug, not a latent one. [measured]** Fixed: added
  `gz_ss.delay = rf.delay - gz_ss.rise_time` after `trap4ge`, matching
  `lib/make_excitation_pulse.py`. Correcting the doc along the way: this
  finding's own "currently benign only because trap4ge is a no-op"
  framing was wrong -- `trap4ge` always rebuilds the trapezoid from
  scratch via `pp.make_trapezoid` (see item 17's `_round_up_to_raster`
  no-op is about rise/flat/fall/amplitude only), which resets `.delay` to
  0 regardless of whether the rounding itself changes anything. Measured
  directly: before the fix, `deGRE.seq`'s receive-gain-calibration
  acquisition (y/z encoding both 0) had residual kz = -7.24 at readout;
  after, kz ~ 2.8e-14 (numerical zero). Regression test:
  `test_degre_excitation_is_centered` in
  `tests/test_trajectory_matches_schedule.py`.
- [x] **11. `apply_whitening`'s conjugation is right for the wrong stated
  reason -- footgun.** Fixed the comment, not the math (the math was
  already correct): a general column-to-row-vector transpose (`(M @ v)^T =
  v^T @ M^T`) never introduces a conjugate on its own, so
  `preprocessing/coils.py`'s old comment was false as written.
  `compute_whitening_matrix`'s comment and `apply_whitening`'s docstring
  now state the real reason -- `compute_whitening_matrix` computes `psi =
  conj(Psi)` (the transpose of the standard covariance convention), so its
  Cholesky factor `L` satisfies `psi = L @ L^H`, meaning `conj(L)` is the
  Cholesky factor of the standard `Psi`, and the whitener is
  `conj(inv(L)) = conj(W)` -- the conjugate comes from needing `conj(W)`
  in the first place, not from the row/column transpose. No test change
  needed (`test_whitening_decorrelates_and_normalizes` already covers the
  behavior; this was a comment-only fix).
- [x] **12. `resize_to_epi_grid` used scipy's pixel-center alignment, not
  MATLAB's edge/FOV alignment.** Fixed: `preprocessing/grid_resize.py` now
  calls `ndimage.zoom(..., grid_mode=True, mode='nearest')` instead of the
  default `grid_mode=False` -- matching the z-crop above it, which already
  assumes N voxels tile the FOV edge-to-edge (plain proportional
  `z_frac * Nz_src` indices, no pixel-center offset), so the crop and
  resize steps no longer disagree about where a voxel physically sits.
  Measured at this repo's real deGRE (108 @ 2mm) -> EPI (240 @ 0.9mm)
  x-axis scale (216mm FOV, no z-crop involved, isolating the resize step):
  a linear ramp resized under the old `grid_mode=False` disagreed with the
  analytic voxel-center positions by ~0.27mm mean / 0.63mm max error;
  `grid_mode=True` cuts that to ~0.006mm mean, with the residual error
  (still <0.4mm max) confined to the two outermost voxels on each axis --
  an unavoidable upsample-past-the-edge extrapolation artifact, not a
  remaining bug. `mode='nearest'` clamps that edge extrapolation to the
  boundary voxel's value rather than blending toward 0 (the
  `grid_mode=True` default `mode='constant'` behavior). No test anywhere
  in the repo pinned a specific interpolated value against the old
  convention -- the one real end-to-end validation on real data
  (`run_rss.py` vs. a MATLAB/BART RSS reference, see the `recon/`/
  `preprocessing/` sections below) reconstructs via root-sum-of-squares,
  which never reads `smaps` at all (`_rss_recon` ignores its `_smaps`
  argument), so it was silent on this convention either way. Regression
  test: `test_resize_to_epi_grid_matches_analytic_ramp_at_voxel_centers`
  in `tests/test_preprocessing_grid_resize.py` (plus a tolerance fix to
  the existing `test_resize_to_epi_grid_crops_z_and_upsamples_xyz`, which
  had encoded the old pixel-center convention at its own ramp's
  endpoints). Flagged since this is a geometry change, not just a bugfix:
  it shifts `smaps`/`b0map_hz` relative to the EPI grid, which changes
  `recon/operators_b0.py`'s per-voxel phase terms in the in-progress,
  uncommitted B0-correction work -- re-check any tuning done against the
  old alignment.
- [ ] **13. `sequences/noise.py:41-44` -- dead arithmetic.**
  `sys_seq.adc_dead_time` is set to `0` on `:42` then added on `:43`, so
  the term is always zero. `pad_duration` is right by accident; the code
  reads as if ADC dead time were accounted for. Drop the term, or read it
  from `sys` before the zeroing.

### Test & tooling health (2026-08-31)

- [x] **14. `uv run pytest` collects zero tests in the documented main
  venv.** Fixed: all nine `tests/test_preprocessing_*.py` modules now
  `pytest.importorskip("sigpy")`/`("nibabel")` before their sigpy/nibabel
  imports (`# noqa: E402` below the gate, matching `tests/test_recon_*.py`'s
  own pattern). `uv run pytest` in the plain main venv now collects
  cleanly (123 passed, 15 skipped, no collection errors); with the
  `preprocessing` extras installed, those 9 files' tests run for real
  (157 passed, 6 skipped for the GERecon/julia-gated ones).
- [ ] **15. [confirmed, count now 20] `uv run ruff check .` fails with
  errors** -- all still `E501`, same files as originally listed; the count
  dropped from 21 to 20 as an incidental side effect of item 1's fix
  (shortened one over-long `noise.py` line). Not fixed (left as an open
  cleanup item, not requested as a fix).
- [x] **16. The `test` extra isn't installed by `uv sync`.** Fixed via
  documentation (matching this item's own "or document" option): README
  already had `uv sync --extra test`, but CLAUDE.md's own Commands section
  -- the doc most likely to be consulted when working in this repo -- did
  not; added a note there spelling out both the missing-extra symptom (a
  bare `uv sync` + `uv run pytest` resolves `pytest` into an ephemeral env
  with none of this repo's own dependencies, ~27 `ModuleNotFoundError: No
  module named 'numpy'` errors that look like a broken repo) and the fix.

### Consistency & documentation (2026-08-31)

- [ ] **17. `trap4ge` is currently a no-op. [measured]** `params.py:150`
  sets `crt = 4e-6` and every `ScannerSpec` sets
  `grad_raster_time = 4e-6`; pypulseq already puts every trapezoid on the
  gradient raster, so the round-up can never change anything.
  Instrumented across `make_excitation_pulse`/`make_prephasers`/
  `make_spoilers`/`make_readout_grads`: **11 calls, 0 changed any
  rise/flat/fall time or amplitude.** So three claims are inaccurate:
  `lib/trap4ge.py:10-14`'s "both the Siemens (10us) and GE (4us) raster"
  (`crt` is GE-only now); this file's "every gradient ... a GE-hardware-
  timing requirement, not optional cleanup" (true in intent, vacuous in
  effect); and the `_round_up_to_raster` epsilon fix plus the
  `test_epical_trajectory_is_centered` failure it guards, neither of which
  can fire while `crt == grad_raster_time`. Keep it (it's the right net if
  `crt` returns to `20e-6`) but say so, and consider a test pinning the
  no-op property so the situation stays visible.
- [x] **18. `slew_derate` docs contradict the code, twice.** Fixed: both
  `params.py` and `lib/readout_from_params.py`'s docstrings now say
  `slew_derate` covers everything *except* the readout ramps and the
  ky/kz blips (which get their own derate via `blip_sys()`).
- [x] **19. `epi_trajectory` default documented wrong.** Fixed the doc,
  not the default: git history (`756a479`, "epi_trajectory to radial")
  shows the `'radial'` default was a deliberate change, not drift, so
  CLAUDE.md's mask2epi section now says `'radial'` too.
- [x] **20. `main.py`'s seed comment contradicts `params.py`.** Fixed:
  comment now says `seed` defaults to `0` (reproducible mask, matching
  every seed-dependent number quoted elsewhere), `None` for a fresh mask.
- [x] **21. Dangling "Open finding" cross-references and superseded
  prose.** Fixed: retargeted all three cross-references (`params.py`,
  `ge/check.py`, and this file's own self-reference) to "PNS finding
  history", corrected the stale prose around them, and documented the
  80/100% gate divergence in both `ge/check.py`'s comment and README's PNS
  paragraph (the gate split is a permanent design decision, independent
  of any one sequence's current number).
- [x] **22. README drift.** Fixed: removed the `matlab_reference/` block
  from the architecture tree, added a `recon/` block (tracked files only:
  `operators.py`/`lowrank.py`/`solvers.py`/`reconstruct.py`/
  `validate_against_mslr.py` -- the untracked B0-correction work in
  progress isn't documented yet), and replaced every
  `samp_locs.mat`/`kxoe<Nx>.mat` mention with `scan_info.mat`.
- [x] **23. `GRE.seq`/`GRE.pge` references outlive the sequence.**
  Re-measured (not just marked historical): today's `deGRE.seq` measures
  acoustics **0.2456**, *under* the 0.3 threshold (the old single-echo
  `GRE.seq` was 0.4024, over it) -- added to `ge/check.py` and CLAUDE.md
  alongside explicit notes that the byte-for-byte/blockRange MATLAB
  comparison numbers are historical, pinned to the pre-dual-echo
  `GRE.seq`/`GRE.pge`, and not re-validated against a fresh MATLAB run
  (the one-off `dump_*.m` scripts needed for that no longer exist in this
  repo -- see "matlab_reference/ removed").
- [x] **24. Stale references to removed MATLAB paths.** Fixed both
  (`scanners.py`, `sequences/ArbEPI.py`).
- [x] **25. `mask2epi_radial`'s half-split docstring doesn't match the
  code.** Fixed: docstring now describes the actual split-by-count
  behavior and states the tradeoff explicitly (guarantees the exact
  `target`/`ETL-1-target` counts the fixed TE echo needs, at the cost of
  "before"/"after" not being a literal geometric split around center).
- [x] **26. Two different `check_ge_feasibility` functions in `ge/`.**
  Deduplicated: `ge/check.py`'s inner function is now named
  `check_seq_feasibility` (it takes a `pp.Sequence` + `ScannerSpec`, not a
  path + `Params` -- the new name matches what it actually operates on).
  `ge/ge_export.py` drops the `import ... as _check_ge_feasibility` alias
  and imports `check_seq_feasibility` directly; `plotting/
  compare_readout_pns.py` and `tests/test_ge_check.py` (including the two
  test names themselves, `test_check_seq_feasibility_runs`/
  `test_check_seq_feasibility_noise_has_no_gradients`) updated to match;
  `plotting/plotting.py`'s docstring cross-reference now names both
  `ge.check.check_seq_feasibility` and `ge.ge_export.check_ge_feasibility`
  explicitly instead of the ambiguous bare name. `ge/ge_export.py`'s own
  `check_ge_feasibility` (path + `Params`, returns `(seq, report)`) is
  unchanged -- that's the real, load-bearing name collision this finding
  flagged, now resolved by the rename rather than worked around.
- [x] **27. `ge/check.py`'s default `pns_wt` is neither human nor
  phantom.** Kept the default at `(1.0, 1.0, 1.0)` deliberately -- added a
  comment on `check_seq_feasibility`'s signature explaining why: weighting
  every channel at 1.0 is the most conservative choice available, an
  upper bound that never *underweights* any channel relative to the IEC
  human table `[0.8, 1.0, 0.7]`. Production (`ge_export.py`) always passes
  `params.PNSwt` explicitly regardless, so this default only matters for
  direct callers (e.g. `tests/test_ge_check.py`) that don't specify a scan
  context.
- [x] **28. Smaller consistency items -- resolved individually.**
  - **`Ndummyshots` usage (only `EPIcal.py:77`).** Confirmed correct, not
    a bug: it drives magnetization to steady-state (dummy shots, played
    before `shot = 0`) before EPIcal's calibration-line acquisition
    begins. No code change.
  - **`T1_degre` duplicated `T1`.** Fixed: removed the `T1_degre` field
    (dataclass field, its `= 1.3` assignment, and the constructor kwarg)
    from `params.py`; `alpha_degre`'s flip-angle formula now references
    `T1` (set earlier in the same function) directly. Nothing outside
    `params.py` read `T1_degre`, so this is a pure dedup.
  - **`make_prephasers`'s independent per-axis durations.** Fixed:
    `lib/make_prephasers.py` now builds each axis's natural (shortest)
    trapezoid first, takes the max duration across all three, and rebuilds
    all three at that shared duration -- since a pypulseq block's duration
    is always its longest gradient event anyway, a shorter per-axis
    duration only hid what the real playout time was. Verified this was
    live (not just theoretical) with a non-isotropic FOV
    (`fov=(0.216, 0.216, 0.09)`, real Nx/Ny/Nz): before the fix, z's
    duration was 0.488 ms vs. x/y's 0.776 ms; after, all three are 0.776
    ms, with each axis's target area unchanged. Regression test:
    `test_make_prephasers_shares_one_duration_across_axes` in
    `tests/test_make_prephasers.py` (confirmed it fails without the fix).
  - **`gen_sampling_masks`'s caipi branch is deterministic.** Confirmed
    accurate, left as-is: "spatiotemporally static" is a valid subset of
    "spatiotemporally incoherent" (the module's own framing), not a
    contradiction -- no code change.
  - **`ge/acoustics.py`'s dead `MAXFREQ`/unreachable odd-length branch.**
    Investigated against `../myFuncs/check_grad_acoustics.m` (the MATLAB
    original) directly, not assumed: both `maxfreq`/`MAXFREQ` (used only
    inside MATLAB's plotting branch, which this port deliberately doesn't
    carry over -- already documented at its definition) and the
    `if mod(n1+zf1,2), zf1=zf1+1; end` odd-length guard are present
    *verbatim* in the MATLAB source too, with the same hardcoded
    `zf_fac = 7`, so `n1 + zf1 == 8*n1` is provably even in both
    languages -- this isn't a Python-port artifact or omission, the guard
    has always been dead code. Removed the unreachable branch in
    `ge/acoustics.py` (a no-op deletion, not a behavior change) with a
    comment citing this; left `MAXFREQ` as-is since its own comment
    already documents why it's unused here.

### Conciseness & performance (2026-08-31)

Measured on a full default-params `generate_arbepi` build (30 frames x 20
shots, 41 400 blocks): **46.8 s total**; items 29-30 are 82% of it.

- [x] **29. Whole-sequence `calculate_kspace()` to extract 2 echoes --
  16.7 s of 46.8 s. [measured]** Fixed exactly as suggested: a throwaway
  single-shot `pp.Sequence` (gx_pre + gro1 + two echo blocks, no y/z
  channels at all since they don't affect kx) replaces the full-sequence
  call. Verified matching the old full-sequence result to ~1e-11 (float
  noise) before switching; `test_arbepi_kxoe_matches_epical` still passes.
- [x] **30. `_compute_schedules` is serial over independent frames --
  21.7 s of 46.8 s. [measured]** Fixed via `ProcessPoolExecutor`, with a
  plain serial fallback for `Nframes <= 1` (avoids pool startup overhead
  on small/test builds). Verified parallel results match serial exactly
  on the real seed=0 mask; measured 21.7s -> 1.0s (64 cores).
- [x] **31. `main.py --ge` runs every feasibility check twice.** Fixed:
  `check_ge_feasibility` (ge_export) now returns `(seq, report)`;
  `export_to_ge` accepts optional `seq=`/`report=` to skip its own
  re-read/re-check when given. `main.py` passes both through from the
  pre-flight loop (still fails before writing any `.pge`, unchanged).
- [x] **32. Investigated, NOT fixed -- see reasoning.**
  `plot_trajectory(frame_idx=...)` also does a whole-sequence
  `calculate_kspace()`. Item 29 already eliminated the actual duplicate
  computation this finding worried about (ArbEPI.py's own full-sequence
  call in a `--plot --ge` run) -- `plot_trajectory` itself is called only
  once per `plot_last_run`, so there's no redundant *repeated* call left
  to cache away, and `plot_one_tr`/`plot_pns_one_tr` never call
  `calculate_kspace()` at all (nothing to share a cache with).
  `calculate_kspace()` itself is monolithic (`k_traj_adc` is just an
  index-slice of the fully-computed `k_traj`, confirmed by reading
  pypulseq's source -- no way to compute one without the other through
  the public API). The one remaining angle -- extracting a
  sub-`Sequence` for just the target frame -- is exactly the approach
  `plotting/plotting.py`'s own module docstring documents as tried and
  **abandoned** (a real, unexplained ~16-22 m^-1 kx offset). A *prefix*
  truncation (frames `[0..frame_idx]`, not an isolated middle frame)
  looked promising since `frame_idx` defaults to 0, but reconstructing
  blocks via `get_block()`/`add_block()` round-tripping is the same class
  of fragile operation that produced the original bug, and validating it
  thoroughly enough to trust was out of scope here. Left as an accepted,
  documented residual cost rather than risk a diagnostic-only plot
  silently showing wrong trajectories.
- [ ] **33. `ReadoutGrads.max_blip_area` is dead.** Confirmed: set in both
  branches, returned in the dataclass, never read anywhere else in the
  repo. Kept deliberately (no code change) rather than deleted: a comment
  above the field in `lib/make_readout_grads.py` now explains it's dead
  only under the current POPE (asymmetric-ramp) geometry, which sizes the
  flat top from `2 * max(a1, a_d)` instead -- it was load-bearing pre-POPE
  (`S = Nx*deltak + max_blip_area`, the symmetric-ramp case where
  `a1 == a_d == max_blip_area/2`) and stays in case a future change
  reverts to a symmetric-slew readout where it's needed again.
- [x] **34. `sequences/EPIcal.py:44-45` -- needless intermediate.** Fixed:
  collapsed `scan_info = hdf5storage.loadmat(...)` + `schedules =
  scan_info['schedules']` into one chained
  `schedules = hdf5storage.loadmat(...)['schedules']` (line still under
  100 chars, no new lint violation). `tests/
  test_trajectory_matches_schedule.py` still passes.
- [x] **35. `_bottleneck_2opt_order` evaluates no-op reversals.** Fixed:
  the reversal loop now starts at `j = i + 1`, not `j = i` (a
  single-element "reversal" can never change anything). Existing
  `tests/test_mask2epi.py` suite (37 tests) still passes.

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
  b0map_hz, echo_times_s, L, nbins)` solves the segmentation fit once
  against this pipeline's real per-frame-invariant ETL distinct echo
  times (not once per frame -- an earlier version did, which also OOM'd a
  real reconstruction by storing an independent per-frame copy of the
  shared `(L,*N)` `c_phasors` tensor). `estimate_spectral_norm` (power
  iteration) is required alongside it -- the B0-corrected operator's
  spectral norm has no closed form the way the plain SENSE operator's does.

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
as the production value (the smallest swept `L` clearing that 1% bar),
overriding `operators_b0.py`'s own `L=6` default at the call site.

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

See `README.md` for the getting-started walkthrough and the full
`Getting started` / `GE export` usage examples.
