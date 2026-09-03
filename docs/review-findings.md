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

- [ ] **8. [verify -- judgement call, not currently a bug]
  `check_grad_acoustics` scores every gradient axis against every axis's
  forbidden bands.** Checked against `../ArbEPI/lib/check_grad_acoustics.m`
  directly: its own loop (`for lg=1:n3, for l1=1:length(bands), for
  l2=1:size(bands{l1},1)`, storing the full `lg x l1` cross product) has
  the *exact* same structure -- this is a faithful port, not an indexing
  bug. Documented with a comment in `ge/acoustics.py` so nobody "fixes" it
  later. Kept on this list only as a flagged judgement call, not an action
  item.
- [ ] **13. `sequences/noise.py:41-44` -- dead `adc_dead_time`
  arithmetic.** `sys_seq.adc_dead_time` is set to `0` then immediately
  added back, so the term is always zero and `pad_duration` is right only
  by accident (it reads as if ADC dead time were accounted for). Drop the
  term, or read it from `sys` before the zeroing.
- [ ] **36. [measured] `deGRE`'s realized TR is ~8.24 ms, not the
  prescribed 8.0 ms -- a live 3% timing error.** `sequences/deGRE.py`'s
  `tr_min` charges the spoiler block only `pp.calc_duration(gx_spoil)`,
  but that block also plays `pp.scale_grad(gy_pre, -y_step)` and
  `pp.scale_grad(gz_pre, -z_step)`, and a pypulseq block's duration is its
  longest event. Measured at default params: `gx_spoil` 760 us,
  `gy_pre`/`gz_pre` 1000 us each, so the real block is 1000 us and
  `tr_min` under-estimates by **240 us** per TR. Consequences: (a)
  `params.alpha_degre` (the Ernst angle) no longer matches the TR actually
  played; (b) the `delay_tr < 0` guard is evaluated against the
  under-estimated minimum, so it can pass when the true minimum doesn't
  fit; (c) the scan runs 3% longer than planned. Fix:
  `max(pp.calc_duration(gx_spoil), pp.calc_duration(gy_pre),
  pp.calc_duration(gz_pre))`. Take the same `max` over the prephase block
  while there (currently a no-op since all three happen to be `Tpre`
  today, but the same latent shape).
- [ ] **37. [measured] `deGRE`'s `te_min` still uses the RF-block-midpoint
  measure that `calc_te_tr_delays` was already fixed for -- 80 us late.**
  `sequences/deGRE.py:157` computes `max(calc_duration(rf),
  calc_duration(gz_ss)) / 2`, the RF *block*'s midpoint, not the physical
  RF center `lib/calc_te_tr_delays.py` now uses (`rf.delay +
  pp.calc_rf_center(rf)[0]`). Measured: `te_min` as coded 2.7680 ms vs.
  2.8480 ms corrected -- 80.0 us under-estimate on both echoes. Since both
  echoes shift by the same 80 us, `dTE` is preserved to within ~0.9 us
  (see item 62), so the fat-cancellation and field-map-scaling arguments
  are unaffected -- but the absolute TE is simply wrong, and the repo now
  carries two different "time from RF to echo" definitions in two files.
- [ ] **38. [measured] The 80% PNS regression test guards a *different*
  sequence than the one the repo ships.**
  `test_arbepi_default_params_peak_pns_under_normal_mode_limit`
  (`tests/test_ge_check.py:98`) builds with `Nframes=1`, seeing only frame
  0's mask. Measured seed=0, GE_MR750: frame 0 = 78.92%, frame 10 (the
  full build's peak) = 79.84% -- the shipped config sits ~0.9 points
  closer to the 80% limit than the guard measures, because frame 0 never
  plays the largest kz blip (steps 3, 6, 10, 11, 14 do). Given
  `blip_slew=105` leaves only ~0.2% margin, the guard can pass while the
  real 30-frame build exceeds 80%. Fix options: build the worst frame
  (pick `argmax` of per-frame blip steps), parametrize over several
  frames, or accept the ~5x cost of a full `Nframes=30` build in the test.
- [ ] **39. `resize_to_epi_grid` raises on a z-FOV mismatch but silently
  mis-registers on an x/y one.** `preprocessing/grid_resize.py` checks
  `fov_src[2] < fov[2]` and raises, then crops z only -- x/y go straight to
  `ndimage.zoom` with no FOV check at all, so smaps/B0 map come out
  geometrically wrong with no error if `fov_src[:2] != fov[:2]`. Dormant
  today only because `N_degre = ceil(fov / res_degre - 1e-9)` happens to
  make x/y match exactly at current values (216mm / 2mm = 108). Fix:
  raise unless `fov_src[:2]` and `fov[:2]` agree to a tolerance, matching
  the z check's strictness.
- [ ] **40. `preprocess()` leaks the output HDF5 handle when the resume
  fast-forward fails.** `preprocessing/preprocess.py:386-387` opens `mf =
  h5py.File(paths.recon, 'a')` and calls `resume_start_frame(...)`
  *outside* the surrounding `try/finally`. On a short archive (exactly the
  case resume exists to survive), `epi_reader.next_frame()` raises a bare
  `StopIteration` -- bypassing the friendly `RuntimeError` that would
  otherwise explain what happened -- and leaves the checkpoint file open on
  a path likely about to retry. Move the open + resume inside the `try`,
  or open `mf` with a `with`.
- [ ] **41. `smaps.load_smaps` doesn't validate the smaps cache's
  `Nvcoils`; `preprocess.py` does.** `preprocessing/preprocess.py:315-321`
  guards the shared `smaps_<seqname>_sigpy.h5` cache with `smaps_valid =
  f.attrs.get('Nvcoils') == Nvcoils` and re-estimates on mismatch;
  `smaps.load_smaps` reads the same file with no such check, so a stale
  cache from a run with a different `Nvcoils` reaches `recon_fn(data,
  smaps)` mismatched -- a shape error at best, a silently wrong
  reconstruction at worst. Port the same attr check across.
- [ ] **42. `recon_frames`'s frame cap can be a float.**
  `PreprocessingConfig.Nframes` is typed `float`, default `float('inf')`;
  `recon_frames.py` does `nframes = min(cfg.Nframes, nframes_avail)` then
  `range(nframes)`. `min(inf, 30)` returns the int so the default works,
  but any float override (`cfg.Nframes = 10.0`, natural given the declared
  type) raises `TypeError: 'float' object cannot be interpreted as an
  integer`. Wrap in `int()`.
- [ ] **43. `trap4ge` overwrites `.area` but leaves `.flat_area` at its
  pre-rescale value.** `lib/trap4ge.py` rebuilds the trapezoid at a dummy
  amplitude, rescales `gout.amplitude` to preserve `gin.area`, sets
  `gout.area = gin.area`, but never touches `gout.flat_area`. Currently a
  no-op in practice (see item 17: `crt == grad_raster_time` means the
  rescale never actually changes anything, so `flat_area` happens to stay
  correct too -- measured ratio 1.0000000). Item 71 shows the two items
  are coupled: forcing the rounding to actually bite (`crt = 20e-6`)
  reproduces the `flat_area` error immediately (666.667 vs. correct
  657.895, 1.33% off), so this fix is a prerequisite for ever reverting
  item 17's `crt` back to `20e-6`. One line:
  `gout.flat_area = gout.amplitude * gout.flat_time`.
- [ ] **44. Two different FFT-shift conventions on the same axis, in the
  same odd/even-phase pipeline.** `preprocessing/oephase.py`'s
  `epiphasecorrect` uses `fftshift(ifft(fftshift(.)))` (mirrored coming
  back), while `preprocessing/preprocess.py`'s `compute_oephase` uses
  `ifftshift(ifft(fftshift(.)))`. Identical for even `nx` (the only case
  in production, `Nx=240`), diverge by one sample for odd `nx`. Pick one
  convention for both, or state explicitly at both sites that even-`Nx` is
  assumed -- same trap class `_center_out` already warns about in
  `lib/mask2epi.py`. (Related to items 64/91's third and fourth spellings
  of the same convention question elsewhere in the repo.)
- [ ] **45. [verify] `check_seq_feasibility`'s `max_slew` under-reports
  ramps shorter than ~2 gradient rasters.** `ge/check.py:228` computes
  `np.abs(np.diff(gw_tm, axis=1) / dt).max()` over bin-center samples. For
  a ramp `>= 2*dt` the interior differences recover the true slew exactly
  (today's POPE ramps are ~50 rasters, so this is currently accurate), but
  a 1-raster ramp -- which `trap4ge`'s no-op rounding (item 17) permits for
  a small trapezoid -- would read at roughly half its real slew, in the
  wrong direction for a hard `.ok` gate. Inherited from pypulseq's own
  `Sequence/calc_pns.py` sampling pattern (hence [verify], not a plain
  bug): decide whether to match pypulseq or compute slew from the
  trapezoid parameters directly.
- [ ] **61. [measured] `deGRE.seq` declares the EPI FOV, not its own.**
  `sequences/deGRE.py:237` writes `seq.set_definition('FOV',
  params.fov)`, but every gradient in that sequence is built from
  `params.fov_degre`. Confirmed in the written file: `output/deGRE.seq`'s
  `[DEFINITIONS]` reads `FOV 0.216 0.216 0.0405` while the sequence
  actually encodes `0.216 0.216 0.042` -- 1.5 mm / 3.6% wrong on z (x/y
  happen to match since `fov_degre` tracks `fov` there). Fix:
  `params.fov_degre`. The excitation slab thickness (`0.9 * params.fov[2]`)
  must stay on `params.fov` -- imaging the same slab as the EPI sequence is
  deliberate -- so only the definition line changes.
- [ ] **62. [measured] deGRE exports the *prescribed* `TE_degre` while
  playing a slightly different pair; ΔTE is not preserved exactly.**
  `delay_te` is computed per echo independently as `ceil((te - te_min) /
  raster) * raster`; the two prescribed TEs differ by `1 /
  fat_offres_freq` = 2236.90 us, not a multiple of the 4 us raster, so the
  two ceils land on different sub-raster residues. Measured (folding in
  items 36/37's corrections): realized TE = 3.120 / 5.356 ms vs.
  prescribed 3.0369 / 5.2738 ms (83.1 us / 82.2 us late), realized ΔTE =
  2.2360 ms vs. 2.23690 ms prescribed (0.90 us short, 0.040%). Two
  consequences: (a) `ArbEPI.py:338` writes the *prescribed* `TE_degre`
  into `scan_info.mat`, and `b0map.jl` divides the echo phase difference
  by that ΔTE to get Hz -- so every field map carries a 0.040% scale
  error; (b) residual fat phase after "cancellation" is 0.0025 rad, not
  exactly zero. Both small, but the exported number should be the
  realized one. Fix: derive `delay_te[1]` from `delay_te[0]` plus a
  raster-multiple offset, and export the realized pair.
- [ ] **63. [measured] The ADC window overruns `Tread` by 2 samples, so
  the last samples of every echo are acquired while the ky/kz blip is
  already playing -- the opposite of what the code's comment says.**
  `lib/make_readout_grads.py:309` comments "Delay blips to play after the
  ADC window closes" and sets blip delays to `Tread`, but `Nfid =
  round(Tread / dwell / 4) * 4` rounds to the *nearest* multiple of 4. At
  default params: `Tread/dwell = 342`, `Nfid = 344`, so the ADC spans
  688.0 us -- a 4.0 us / 2-sample overrun past the blip's start. Worst-case
  ky error on those two samples: 0.17% of one k-space step (small, but the
  comment asserts the opposite, and the error grows as `dwell^2`). Note
  the ±kmax coverage loop is *not* also wrong -- it already integrates out
  to `(Nfid - 0.5) * dwell`, accounting for the overrun. Fix: correct the
  comment, or floor instead of round (`Nfid = floor(...) * 4` -> 340),
  which costs a slightly larger flat top for the same coverage.
- [ ] **64. [measured] `run_rss.py`'s `_ift3` justifies its FFT-shift
  convention with a premise that's false for this repo's actual matrix
  size.** Its docstring defends `fftshift(ifftn(fftshift(.)))` as
  "identical for even-length axes, which every dimension in this pipeline
  is" -- but `params.py` sets `N = [240, 240, 45]` and `N_degre = [108,
  108, 21]`, z odd on both. Measured on a length-45 axis: the two
  conventions differ by 144% in relative complex value (0 at n=44, as
  claimed). The difference is a one-sample circular shift of the k-space
  input -- a pure linear phase ramp in image space -- so both current
  consumers are immune (`_rss_recon` takes `np.abs`; `b0map.jl` differences
  two echoes on the same grid, cancelling the ramp), but the stated
  *reason* is wrong and any future complex-valued consumer inherits the
  ramp silently on z. Fix the docstring to say magnitude-/difference-safe
  rather than shift-equivalent, or switch the input to `ifftshift`. Same
  convention question as items 44 and 91 (three total spellings in the
  repo); also related, `ge/acoustics.py:77` uses `ifftshift` where the
  MATLAB original uses `fftshift` -- provably equivalent there since
  `n1 + ZF_FAC*n1` is always even.
- [ ] **74. [measured] `run_b0_recon.py` re-introduces the exact-zero
  sampling-mask inference that item 7 removed elsewhere -- under the same
  function name, at the cost of an extra ~11 GB read.**
  `recon/reconstruct.py:69`'s `_load_omega` was fixed to prefer the
  authoritative `omegas` dataset over `ksp != 0`, with a docstring
  explaining a real sample rounding to exactly `0+0j` silently becomes
  "not acquired." `recon/run_b0_recon.py:41` defines its *own*
  `_load_omega(fn_ksp)` whose entire body is `_load_array(fn_ksp,
  "ksp_epi_zf")[:, :, :, 0, :] != 0` -- exactly the bug item 7 removed, on
  coil 0 alone, feeding `build_encoding_operator_b0`'s per-frame sampling
  mask. Also costs real time: `omegas` is a few hundred KB while this
  reads the full `(Nx,Ny,Nz,Nc,Nt)` archive a second time (~23 s per that
  function's own docstring). Fix: import and reuse
  `reconstruct._load_omega`, or read `omegas` directly.
- [ ] **75. [measured] `build_encoding_operator_b0` materializes 2.2 GB of
  per-frame `b_weights` that hold only 60xL distinct values -- 3.3x the
  shared-`c_phasors` cost the same function was already refactored to
  avoid.** `operators_b0.py:231`'s `b = b_by_echo[pos]` is advanced
  indexing, so each of `Nt` frames gets an independently materialized
  `(K, L)` complex64 tensor. At real scale (`K=288000`, `Nt=30`, `L=32`):
  73.7 MB/frame, 2.21 GB total, every byte a gather from the same `(ETL,
  L)=(60,32)` table (61 KB of actual distinct values). For comparison the
  shared `c_phasors` this function was already refactored to share across
  frames is 664 MB -- so the "fixed" redundancy is now the smaller of the
  two. Fix: keep `pos` per frame (2.3 MB/frame int64, 69 MB total) plus
  the one shared `b_by_echo` table, and index inside
  `_apply`/`_apply_adjoint` instead of precomputing 30 times.
- [ ] **76. `estimate_spectral_norm` runs a fixed 30 power iterations with
  no convergence check, and the error it can make is the unsafe one.**
  `operators_b0.py:236` iterates exactly `niter` times. Power iteration
  approaches `sigma1` *from below*, so an under-converged result
  under-estimates POGM's Lipschitz constant `L = Nscales * sigma1A**2`
  (`reconstruct.py:194`), giving a step size that's too *large* -- the
  divergence direction. Nothing downstream re-checks it (`run_b0_recon.py`
  only prints the value next to the uncorrected reference). The repo
  already has the right pattern: `recon/solvers.py:188`'s `poweriter`
  takes `niter=200, tol=1e-6` with early return on convergence, for the
  same computation (see item 89's disposition question). Fix: iterate to
  a tolerance, or keep 30 iterations with a documented safety factor.
- [ ] **77. [measured] `operators_b0.py`'s `nbins` docstring blames a
  weighting mirtorch does not do.** It describes `mri_exp_approx` as
  fitting from a "magnitude-weighted histogram" -- but mirtorch 0.3.1's
  `_uniform_histogram` (`histogram.scatter_add(0, indices,
  torch.ones_like(values))`) is a plain voxel-count histogram, no
  magnitude weighting anywhere in the call path. The conclusion is
  unaffected (background dominates by count too; the row-sum evidence
  stands on its own), but the stated mechanism is wrong -- and it's the
  sentence someone would reason from when picking `nbins` for a
  differently-shaped field map. Fix the two "magnitude-weighted" phrases;
  note the equal-width range is set by `b0.amin()`/`amax()` over the whole
  volume, which is what actually makes an asymmetric in-object range
  expensive in bins.
- [ ] **78. `gre_diagnostics.py` computes `n_echoes` generically, then
  hardcodes two echoes.** `preprocessing/gre_diagnostics.py:48` reads
  `n_echoes = ksp_echoes.shape[3]` generically, but `:79-80` index
  `img_echoes[..., 1]`/`te_degre[1]` unconditionally and `:83`'s ratio
  panel assumes exactly two. `config.py:185` defaults `n_echoes_degre` to
  `1` for a pre-dual-echo `scan_info.mat` snapshot -- a deliberately
  supported case per this repo's own convention -- so on such a dataset
  this is a bare `IndexError` in a diagnostic script whose whole purpose
  is being run when something already looks wrong. `:46`'s `te_degre =
  f.attrs["TE_degre"]` is similarly an unguarded `KeyError` on the same
  class of older cache. Either guard both, or assert `n_echoes == 2` up
  front with a message naming the cause.
- [ ] **93. `run_recon(fn_b0map=...)` silently accepts a `sigma1A`
  measured for the *uncorrected* operator.** `recon/reconstruct.py:146`'s
  own docstring says "sigma1A is not re-estimated internally for the
  corrected operator -- the caller must supply one appropriate to it", but
  nothing enforces it: `sigma1A` is an ordinary required kwarg consumed as
  `L = Nscales * sigma1A**2` with no reference to `fn_b0map`.
  `run_b0_recon.py` does the right thing (power-iterates first) but is the
  only caller that does; a too-small `sigma1A` means a too-small
  Lipschitz constant, so POGM diverges rather than failing cleanly, after
  however long the run has already burned. Fix: accept `sigma1A=None` and
  call `estimate_spectral_norm` on the built operator when it's `None`.
  Interacts with item 76 (whatever iteration count it settles on becomes
  this path's default too).
- [ ] **96. [measured] `plot_psf` has its `fftshift`/`ifftshift` backwards,
  mis-centering the PSF by one voxel on any odd-length axis -- live at
  this repo's real dimensions.** `plotting/plotting.py:197` computes
  `psf = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(omega)))`, the
  reverse of the correct `fftshift(ifft2(ifftshift(x)))` for un-centering
  an array whose index `N/2` represents `k=0` (the convention every other
  centered array in this codebase uses). Invisible on `Ny` (240, even),
  live on `Nz` (45, odd, per `params.py`'s default `N`). Measured: for an
  all-ones mask the true PSF is a delta at `(Ny//2, Nz//2)`; the code's
  formula places it at `(Ny//2, Nz//2 + 1)` -- at real scale the code's
  PSF magnitude equals the correct one circularly shifted by exactly +1
  sample along z (`np.roll` matches to float noise). Not the same bug
  class as items 44/64/91 (those swap only one shift, cancelling under
  `np.abs()`); this swaps both, which does not cancel. Diagnostic-only
  impact (`output/psf.png` via `--plot`, zero test coverage). Fix:
  `np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(omega)))`.
- [ ] **97. [measured] `Emax_n` is silently wrong for any segment whose
  gradient energy is always exactly zero -- live on this repo's own
  shipped `noise.seq` -> `noise.pge` export.** `ge/ceq.py:56-57` defaults
  `Emax_val: float = 0.0`, `Emax_n: int = 1`; `ge/seq2ceq.py:169-171`'s
  gradient-heating loop only overwrites them on strict `e_total >
  seg.Emax_val`, so a segment with genuinely `0.0` energy everywhere never
  satisfies that condition and `Emax_n` stays at the hardcoded class
  default `1` -- a row number belonging to whichever segment occupies row
  1, not this segment. `ge/writeceq.py:242` writes that incoherent value
  verbatim into the `.pge` binary. Measured on `output/noise.seq`:
  segment 2's own rows start at 121 (`seg.rows == [121]`), yet its
  exported `Emax_n` is `1`, which belongs to segment 1. Re-ran on
  `ArbEPI.seq`/`EPIcal.seq`/`deGRE.seq` (nonzero energy everywhere) and
  every segment's `Emax_n` correctly resolves to itself. Distinct from
  the already-documented stale-column-choice deviation in the same
  function (that's about *which columns* feed `e_total`, not this
  default-value fallback). Fix: seed `Emax_n` from `seg.rows[0]` rather
  than the cross-segment constant `1`.
- [ ] **98. Three of `seq2ceq`'s four block-walking loops assume the
  final segment instance is complete, and the first to walk off the end
  raises a raw `KeyError` rather than an actionable error.**
  `ge/seq2ceq.py:74`'s variable-delay-block-detection loop, `:117`'s
  loop-table construction, and `:159`'s gradient-heating loop all stride
  `n` forward in `nBlocksInSegment`-sized steps with no check the last
  instance has that many blocks left. Only the consistency-check loop at
  `:138-144` guards this (`if n + seg.nBlocksInSegment > ceq.nMax:
  break`). Reproduced with a synthetic sequence (two complete
  TRID-segment instances, a third truncated to just its TRID-label
  block): crashes at `ge/seq2ceq.py:77` with `KeyError: 6` from inside
  pypulseq's `get_block`, not one of this file's own actionable
  `ValueError`s. Never fires on any real shipped `output/*.seq` (their
  final instances are complete), but `seq2ceq(seq)` sits on
  `check_seq_feasibility`'s production path, so a future generator bug or
  truncation script would surface an opaque `KeyError`. Fix: apply the
  same guard to the other three loops.
- [ ] **99. [measured] `preprocessing/cg_sense.py` infers its sampling
  mask from a single coil's exact-zero values -- the same bug class items
  7/74 already fixed elsewhere, still live in this file.**
  `preprocessing/cg_sense.py:54-55` is `mask = np.abs(np.take(kdata_zf, 0,
  axis=coil_dim)) > 0`, derived from coil index 0 alone. The sibling
  Stage-2 driver does this safely: `recon_sigpy.py:65`'s `weights =
  (sp.rss(ksp_cf, axes=(0,)) > 0)` uses RSS across *all* coils, so the two
  drivers are now inconsistent. Measured failure mode: with coil 0 forced
  to all zeros, `cg_sense()` raises `RuntimeWarning: invalid value
  encountered in scalar divide` (first CG step: `alpha = rsold/... =
  0/0`) and returns an all-NaN reconstruction, silently -- easy to miss
  inside `run_cg_sense.py`'s per-sequence `except Exception` batch loop.
  Caveat: `cg_sense.py` runs on `ksp_epi_zf` *after* Stage 1's PCA coil
  compression, so "coil 0" is the dominant virtual coil, lowering the
  everyday trigger probability -- but the structural defect and its
  catastrophic-and-silent failure mode are real. Fix: derive the mask
  from `recon_frames`'s already-available `omegas` dataset, or at minimum
  switch to an RSS-across-coils check matching `recon_sigpy.py`.

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
- [ ] **56. `sequences/noise.py` builds its `pp.Sequence` on the derated
  system; `ArbEPI.py`/`EPIcal.py` deliberately use full-hardware
  `params.sys`.** `noise.py:41` is `sys_seq = copy.deepcopy(sys)` where
  `sys = derated_sys(params)`, while the other two use
  `copy.deepcopy(params.sys)` with a comment explaining why (the POPE
  fall ramp deliberately runs above the derate). `noise.seq` has no
  gradients so nothing observable changes, but the three files now
  disagree about which system object goes into `pp.Sequence` with no note
  that the divergence is intentional. Make it match, or say why not.
- [ ] **65. [measured] `sequences/noise.py` is the only sequence that
  never sets its `Name`/`FOV` definitions.** `ArbEPI.py`/`EPIcal.py`/
  `deGRE.py` all call `seq.set_definition('FOV', ...)` +
  `seq.set_definition('Name', seqname)` before `seq.write(...)`;
  `noise.py:69` writes straight out. Confirmed in the written file:
  `output/noise.seq`'s `[DEFINITIONS]` block has neither key.
  Nothing in `ge/` reads these today, so cosmetic -- but it's a
  gratuitous divergence in the one sequence file that already diverges on
  `sys_seq` (item 56) and `adc_dead_time` (item 13). Worth folding into
  whichever of 13/56 gets fixed first.
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
- [ ] **88. `benchmark_b0_cost.py` documents its own headline measurement
  backwards.** Its docstring promises "the static, L-independent memory
  cost of building the operator itself" -- but both named components
  (`c_phasors`, `b_weights`) scale *linearly* with L (item 75's 664 MB and
  2.21 GB at L=32), and the script's own `build_mem` column measures
  exactly that L-dependence across a table whose whole point is comparing
  L values. Fix the sentence; the number it prints is the right one.

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
- [ ] **59. `epi_gridding.rampsampepi2cart` forces complex128 output.**
  `dc = np.empty((nx, etl) + dr2.shape[2:], dtype=complex)` regardless of
  input dtype, in a pipeline that's otherwise complex64 end to end
  (`preprocess.py` writes `ksp_epi_zf` as `np.complex64`). Doubles the
  per-frame working set for no precision that survives the cast on the
  way out. Use `dr.dtype`, or `np.result_type(dco, dce)`.
- [ ] **60. `sequences/EPIcal.py`'s shot loop is 1-based where
  `ArbEPI.py`'s is 0-based.** `for shot in range(-params.Ndummyshots + 1,
  params.Nshots + 1)` vs. `for shot in range(params.Nshots)`. `shot` is
  used for nothing but `is_dummy = shot < 1`, so the 1-based offset buys
  nothing and re-introduces the MATLAB indexing convention CLAUDE.md's
  "Index convention" section says this port deliberately left behind.
  `range(-Ndummyshots, Nshots)` with `is_dummy = shot < 0` reads the same
  and matches the rest of the repo.
- [ ] **71. [measured] Item 43's `trap4ge.flat_area` staleness is gated
  by item 17, not just by having no consumer -- fix the two together.**
  While `crt == grad_raster_time`, `trap4ge`'s rounding changes nothing
  (item 17), so `gin.area / gout.area` is identically 1 and the stored
  `flat_area` is exactly right by coincidence (measured ratio 1.0000000
  at `crt=4e-6`). Forcing the rounding to bite (`crt=20e-6`, the value
  item 17 says to revert to for Siemens dual-raster support) makes it
  wrong immediately (stored 666.667 vs. correct 657.895, 1.33% off), and
  `pp.scale_grad` propagates the error faithfully. So item 43's one-line
  fix is a prerequisite for item 17's revert, not independent cleanup.
- [ ] **72(a). Dead expression in `deGRE.py`.**
  `sequences/deGRE.py:204`'s `pe2_steps[max(0, iZ - 1)]` sits inside
  `... if iZ > 0 else 0.0`, so `iZ - 1 >= 0` always and the `max` can
  never bind -- drop it (the neighbouring `pe1_steps[iY]` correctly has no
  such guard). (Part (b) of the original item -- the `_load_smaps`
  leading-underscore-then-used naming issue -- was already fixed
  incidentally by the `preprocessing/smaps.py` rewiring; not carried
  forward here.)
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
  `GatheredSense` the L=1 case, or subclass/delegate. Likewise
  `estimate_spectral_norm` computes exactly what `recon/solvers.py:188`'s
  `poweriter` computes, minus the tolerance check (item 76) -- keep one
  implementation, take `poweriter`'s `tol`, put it next to the operators
  it measures. Minor, same file: `:127`'s function-body `import warnings`
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
