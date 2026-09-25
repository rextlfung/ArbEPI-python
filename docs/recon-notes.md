# recon/ design notes (raw)

The module docstrings of the recon/ files that were removed in the recon restructure
(operators, mslr, run_recon, L1-wavelet_TV_B0_SENSE, lowres_calib, analysis,
hdf5_chunked_io, run_rss), kept verbatim as raw material for proper documentation.
File and function names inside refer to the old layout; see CLAUDE.md's recon/ section
for the current one. Function docstrings moved with their functions and are not repeated here.

## `recon/operators.py`

```text
Per-frame Cartesian SENSE encoding operator, block-diagonal-stacked over
time, gathered to sampled k-space locations only -- (K, Nc) per frame instead
of a dense (Nc, Nx, Ny, Nz) masked grid.

Built on mirtorch.linear.linearmaps.LinearMap + mirtorch.linear.basics.
BlockDiagonal for the operator-composition framework, in place of
../mslr-recon's MIRT.Asense / LinearMapsAA.block_diag, but implementing
_apply/_apply_adjoint directly here (rather than using mirtorch's own
mirtorch.linear.mri.Sense) for the same reason mslr-recon's src/sense_gpu.jl
gathers to sampled locations rather than keeping mirtorch.linear.mri.Sense's
dense masked-grid output: on this repo's real ball-phantom data (240x240x45,
18 coils, 30 frames, R~9), the dense representation is ~11GB per k-space-
shaped tensor and OOMs a 49GB GPU on the very first gradient evaluation --
gathering to the K sampled locations (K = Nx*Ny*Nz/R) cuts every k-space-
shaped tensor by the acceleration factor R, matching Julia's memory
footprint (mirt_mod.jl/sense_gpu.jl were written under the same 48GB budget).

The forward/adjoint math is otherwise identical to mirtorch's Sense with
norm='ortho' (see the module docstring this file used to carry, still
accurate): forward = fftshift(fftn(ifftshift(smaps .* x), norm='ortho')),
adjoint = fftshift(ifftn(ifftshift(.), norm='ortho')) summed over coils with
conj(smaps) -- the exact adjoint of the forward map for any grid size
(fftshift/ifftshift are permutation matrices, P^T = P^-1; ortho-normalized
fftn/ifftn are mutually adjoint), verified against mirtorch's own Sense by
adjoint self-consistency in tests/test_recon_operators.py.


Formerly recon/operators_b0.py
------------------------------
Time-segmented B0 off-resonance correction for recon/'s Cartesian
encoding operator -- the fuller, min-max-style stage of a staged plan for
adding B0 correction to GatheredSense (recon/operators.py).
demodulate_smaps (below) implements a cheaper static single-segment stage
first; see its section below (and CLAUDE.md's recon/ section) for why static
alone measured far too weak at this repo's real echo-train length /
field-map range (~5% error reduction, vs ~98% in an idealized small-
excursion regime) to be sufficient on its own -- this module is the actual
fix that regime needs.

Reuses mirtorch.linear.mri.mri_exp_approx (the same B0-segmentation
coefficient fit mirtorch's own non-Cartesian Gmri/GmriGram operators use,
see examples/demo_mri.ipynb's "Non-Cartesian MRI with B0 correction"
section) rather than reimplementing time-segmentation from scratch --
that function is trajectory-agnostic (a per-time-value least-squares fit
against a field-map-frequency histogram), so it slots directly into this
repo's Cartesian-with-blips/FFT-based GatheredSense in place of mirtorch's
own NUFFT-based Gmri/GmriGram, without needing torchkbnufft at all.

Sign convention: mri_exp_approx fits exp(-2j*pi*b0*t) (see its own
docstring); passing it -b0map_hz (matching mirtorch's own demo notebook's
`Gmri(..., zmap=-b0, ...)` call) composes to the physically-correct
exp(+2j*pi*b0map_hz*t) demodulation -- see the static-stage section
below for the full derivation and the reference (Sutton, Noll, Fessler,
IEEE TMI 2003) this is cross-checked against.

L (segment count) defaults to 32 here, not mirtorch's own Gmri default of
6 -- recon/analysis.py's real-scale sweep (real ETL=60 field-
map range/echo spacing) found a sharp, Nyquist-like phase transition
around L=27-32, matching this repo's real bandwidth-time product
(BT = field-map range * echo-train duration ~= 27); L=6 gives only ~35%
forward-model error reduction (barely better than no correction), while
L=32 is the smallest swept value that gets relative forward-model error
under 1%. See that script and CLAUDE.md's recon/ section (B0 subsection)
for the full sweep and the cost-vs-L tradeoff (recon/analysis.py).

Static single-segment stage (demodulate_smaps; formerly its own module,
recon/b0_correction.py) -- the first, cheapest stage of a staged plan for adding
B0 correction to GatheredSense (recon/operators.py): a single per-voxel
conjugate-phase demodulation baked into the coil sensitivity maps before the
encoding operator is built, at zero added per-iteration cost. This corrects
the dominant geometric-shift component of EPI off-resonance distortion; it
does not correct the residual blur/ghosting from differential phase accrual
across the echo train -- that needs the full time-segmented correction
stage (GatheredSenseB0, below), and this static stage was
deliberately implemented first: cheap enough to validate the field map's
sign/scale conventions in isolation before building the more expensive
machinery on top of them.

Sign convention: the forward signal model (Sutton, Noll, Fessler, "Fast,
iterative image reconstruction for MRI in the presence of field
inhomogeneities," IEEE TMI 2003, DOI 10.1109/TSP.2005.853152 -- the same
reference mirtorch.linear.mri.Gmri's own docstring cites) is

    s(t) = integral m(r) exp(i 2*pi*Delta_f(r)*t) exp(-i 2*pi*k(t).r) dr

i.e. the forward operator needs a *positive* exp(+i 2*pi*f(r)*t) phase
multiplied into the image before the spatial-encoding FFT, to reproduce the
extra phase off-resonance actually adds to the acquired signal. Cross-
checked against mirtorch's own Gmri, not just re-derived from the paper:
its demo notebook (examples/demo_mri.ipynb, "Non-Cartesian MRI with B0
correction") passes `zmap=-b0` into Gmri, whose internal mri_exp_approx
builds the per-segment demodulation as exp(-i 2*pi*zmap(r)*t_l) -- compose
the two and that's exp(-i 2*pi*(-b0)*t) = exp(+i 2*pi*b0(r)*t), the same
positive sign used here (mirtorch's `zmap=-b0` negation exists purely to
cancel mri_exp_approx's own internal negative sign, not to flip the
physical convention).

This assumes preprocessing/run_b0map.py's b0map_hz (from MRIFieldmaps.jl,
Lin & Fessler -- the same Fessler lineage as the TSP reference above, so a
priori likely to share this same sign convention by construction) already
follows it. Verified against a real reconstruction, not just assumed:
recon/run_recon.py's real runs (see CLAUDE.md's recon/ B0 subsection)
show the field map reducing, not worsening, distortion, and the
correction's expected geometric-sharpening effect is preserved after the
b0map.jl preconditioner fix (precon=:diag) that separately addressed
field-map-noise-induced speckle. Flip the sign of b0map_hz at the call
site (`demodulate_smaps(smaps, -b0map_hz, te_s)`) if a future dataset's
comparison ever shows this backwards.
```

## `recon/mslr.py`

```text
Multi-scale Locally Low-Rank (MSLR) fMRI reconstruction via decomposition.
Port of ../mslr-recon/scripts/reconstruct.jl (Ong & Lustig 2016), built on
mirtorch instead of MIRT.jl/LinearMapsAA -- see recon/operators.py and
recon/mslr.py for the individual pieces.

    X_final = X[...,0] + X[...,1] + ... + X[...,Nscales-1]

Each component X[...,k] is independently constrained to be locally low-rank
at its own patch scale (recon/mslr.py's patchSVST); data consistency is
enforced on the sum. lambda_k set by the Ong & Lustig (2016) closed-form
formula (see _reg_weights below) -- no tuning needed beyond lambda_global.

save_result (below; formerly its own module, recon/save_result.py) is what
actually persists a ReconResult -- run_recon() itself only returns one in
memory. It writes two files per result: `<fn_base>.nii.gz` + `.json`
(magnitude image + metadata sidecar, via preprocessing/nifti_io.py's
save_recon_nifti -- same format/convention every other reconstructed image
in this pipeline uses) and `<fn_base>.h5` (full-precision complex
X_recon/X plus the solver convergence trace, plain numpy-order h5py -- no
MATLAB consumer, matching this repo's own .h5-not-.mat convention for
internal artifacts).


Formerly recon/solvers.py
-------------------------
Proximal gradient method with momentum (PGM/FPGM/POGM) and gradient
restart. Port of ../mslr-recon/src/mirt_mod.jl's `pogm_restart`, itself a
modified port of MIRT.pogm_restart (Kim & Fessler, 2017/2018) adding
GPU-memory-safe scalar typing, in-place buffer reuse, and early stopping via
`conv_tol`.

The scalar-typing and buffer-aliasing machinery in the Julia version exists
specifically to avoid Float64 promotion of CuArray{ComplexF32} and to fit a
48GB-VRAM budget under Julia's broadcast-allocates-a-new-array semantics
(see mirt_mod.jl's module docstring, points 1-5). None of that applies here:
Python floats multiplied against a complex64 tensor stay complex64 (PyTorch's
weak-scalar type promotion), and PyTorch's caching allocator reuses freed
blocks without needing manual aliasing -- so this port keeps the exact
momentum/restart math (points 6-7 of that docstring: early stopping and the
extended `fun` callback) but drops the Julia-GPU-specific mechanics.


Formerly recon/lowrank.py
-------------------------
Patch extraction/recombination and singular-value soft-thresholding (SVST)
for locally-low-rank (LLR) regularization of a 4-D image time series
(Nx, Ny, Nz, Nt). Port of ../mslr-recon/src/recon.jl (Ong & Lustig 2016).

Unlike the Julia original (which loops over patches with @threads on CPU or
sequential CUSOLVER calls on GPU, to avoid a huge intermediate tensor and
work around a GPU-only cuSOLVER NaN bug), this port batches every patch into
one tensor and calls a single batched `torch.linalg.svd` -- PyTorch already
parallelizes a batched SVD internally (cuSOLVER batched routines on GPU,
multi-threaded LAPACK on CPU), so there is no need to loop by hand. The one
piece of Julia's SVST that IS still needed here is the exact-zero shortcut
below (see SVST docstring) -- it's a correctness safeguard, not a Julia-GPU
memory optimization, so it survives the port.
```

## `recon/run_recon.py`

```text
Real-data reconstruction drivers on this repo's own zero-filled k-space
(.venv-recon), all sharing the B0-corrected (or plain) GatheredSense operator
and differing in the solver. Subcommands, each with the flags of the
standalone script it replaced:

    mslr-ref    MSLR G+L with B0 (and optional R2*) correction, parameters read from a ../mslr-recon reference .mat
    mslr-local  MSLR with a single local-low-rank scale, B0-corrected by default
    cg          unregularized CG-SENSE, B0-corrected by default

    .venv-recon/bin/python -m recon.run_recon {mslr-ref,mslr-local,cg} <datdir> <name|seqname> ...

The sections below are the original module docstrings, kept verbatim.

Formerly recon/run_b0_recon.py
------------------------------
One-off driver: reconstruct real acquisitions with full time-segmented
B0 correction (recon/operators.py), replicating the existing G+L
(multi-scale) config already validated against ../mslr-recon
(recon/analysis.py) for the *uncorrected* case, and saving
results under <datdir>/recon/mslr_b0/G+L_L<L_b0>/ (recon/mslr.py) --
one directory per L (matches the convention already on disk from the
L=6/10/16 runs made during the sweep below).

--r2star additionally layers T2*/T1 amplitude-decay correction on top of
the B0 phase correction (recon/operators.py's r2star_map/t_ref_s
generalization -- see that function's docstring for the ψ(r) = i*2*pi*Δf(r)
- R2*(r) complex-field derivation and its sign-convention divergence from
the worktree-lowres-calib-recon branch's adjoint-only calib script). R2* is
estimated from the same dual-echo deGRE data already used for the B0 map
(preprocessing/r2star_map.py's two-point log-ratio) and referenced to the
nominal-TE echo's acquisition time (scan_info.mat's
schedules[0,0,(ETL-1)//2,2] -- the same value
recon/lowres_calib.py reads, read the same way here rather
than re-derived). Output moves to <datdir>/recon/mslr_b0complex/G+L_L<L_b0>/
so a --r2star run never collides with a plain B0-only run at the same L.

sigma1A is not reused from the uncorrected reference: the B0-corrected
operator's spectral norm has no known closed form (mri_exp_approx's B
weights are a least-squares fit, not guaranteed unit-norm/orthogonal -- see
operators.py), so it's measured here via power iteration
(estimate_spectral_norm) before the real reconstruction runs, on the actual
per-dataset smaps/omega/b0map/echo_times rather than assumed.

L (segment count) defaults to 32 -- see operators.py's module docstring
for the real-scale sweep (recon/analysis.py) that settled it;
--L still lets it be overridden per run without a code change.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.run_recon mslr-ref <datdir> <name>
e.g.
    .venv-recon/bin/python -m recon.run_recon mslr-ref \
        /StorageRAID/rexfung/20260822ball_laminar laminar


Formerly recon/run_mslr_local.py
--------------------------------
One-off driver: MSLR reconstruction with a single local-low-rank scale,
for this repo's own <seqname>_epi_zf.h5 / smaps_<seqname>_sigpy.h5 naming
convention (unlike run_recon.py / analysis.py, both
hardcoded to the ArbEPI_epi_zf.h5 name and, for the latter, a Julia
reference .mat that doesn't exist for a dataset that was never run through
../mslr-recon).

Motivation: recon/lowres_calib.py's fully-sampled-calibration-region
reconstruction is a fast diagnostic that uses only a few hundred of a
dataset's many thousand k-space samples -- deliberately, for speed, per its
own module docstring -- so its thermal-noise floor is far higher than the
full accelerated reconstruction's. Comparing recon/lowres_temporal_
stability.py's output on this driver's full, high-SNR reconstruction
against the calibration-region-only numbers already measured tests whether
that low-res diagnostic's own noise floor, not a real B0/T2*/spoiling
artifact, was inflating the fluctuation/drift figures measured so far.

patch_sizes/strides default to a single local scale ([(6,6,6)], [(3,3,3)]),
not the G+L multi-scale config recon/analysis.py's Julia
reference uses -- an explicit choice for this investigation (local-only
avoids the giant whole-volume SVD's added runtime/memory at the higher
accelerations here, R up to ~93.5, and isolates locally-low-rank spatial
regularization from a global-rank prior). No Julia reference exists for
this patch config on this dataset, so niters/conv_tol/mom fall back to
run_recon's own defaults rather than being read from one.

B0 correction defaults to on (`--no-b0` reverts to the original plain
operator) -- an uncorrected reconstruction shows real geometric distortion
in the phase-encode direction (classic uncorrected-off-resonance EPI
warping), confirmed by comparing this driver's plain output against
run_recon.py's B0-corrected one on the same data: the latter's object is
round, the former's is not. sigma1A has no closed form for either operator
(see operators.py's estimate_spectral_norm docstring) -- this driver
measures it once via power iteration before the real reconstruction runs,
the same pattern run_recon.py uses.

lambda_global (the Ong & Lustig regularization scale multiplying
_reg_weights' closed-form lambda_k) defaults to this dataset's own
acceleration factor R, not run_recon's flat default of 1.0 -- an explicit
choice (2026-09-18) after comparing reconstructions across this repo's
4-resolution/4-R sweep: _reg_weights' formula has no data-amplitude
normalization, so a fixed lambda_global applies disproportionately more
shrinkage to a lower-signal-amplitude (higher-resolution/higher-R, smaller
voxel volume) acquisition than a higher-amplitude one -- scaling by R
compensates for that so different resolutions in the same sweep get
comparably-effective regularization, rather than reusing the R~9
production-scale value analysis.py's own reference happened
to be tuned at. Override with --lambda-global to pin a specific value
instead (e.g. for reproducing a fixed-lambda comparison).

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.run_recon mslr-local <datdir> <seqname> [--device cuda]
e.g.
    .venv-recon/bin/python -m recon.run_recon mslr-local \
        /StorageRAID/rexfung/20260915ball 1_1x_5.4mm


Formerly recon/cg_sense_b0.py
-----------------------------
B0-informed CG-SENSE: unregularized conjugate-gradient SENSE
reconstruction using the time-segmented B0-corrected encoding operator
(recon/operators.py's GatheredSenseB0/build_encoding_operator_b0),
solved via literal conjugate gradient (Pruessmann et al.) rather than
recon/mslr.py's multi-scale-low-rank POGM solve.

Ports preprocessing/cg_sense.py's exact CG algorithm (CG on the normal
equations E^H E x = E^H y) onto mirtorch's LinearMap interface
(A.apply/A.adjoint) instead of that module's explicit FFT+mask+smaps
closures, so it works with any encoding operator sharing that contract --
here, the B0-corrected one -- not just the plain uncorrected SENSE operator
preprocessing/cg_sense.py was written for.

Unlike recon/mslr.py's run_recon (POGM, needs sigma1A -- the
operator's spectral norm -- for its step size), CG is self-scaling and
needs no such estimate: skip recon/operators.py's estimate_spectral_norm
entirely here.

Reconstructs the whole (Nx,Ny,Nz,Nt) volume in one CG run rather than
looping per frame: build_encoding_operator_b0 returns a BlockDiagonal
operator (independent per-frame GatheredSenseB0 blocks, see
recon/operators.py's build_encoding_operator docstring), so A^H A has no
cross-frame coupling -- solving jointly over the stacked tensor is
mathematically identical to solving each frame's CG independently, just
one Python-level loop instead of Nt, and matches recon/mslr.py's own
convention of treating the whole (Nx,Ny,Nz,Nt) tensor as one state array.

`num_iter` defaults to 150, not 20 -- the original default converged well at
R=1 (residual 8.6e-4 by iteration 20) but left real undersampled data
nowhere near converged: R=6 residual was still 3.3e-3 at iteration 20 (CG's
per-iteration residual decay rate is measurably slower at higher R, exactly
as SENSE-CG conditioning theory predicts -- Pruessmann et al. 2001).
Verified directly on 2_6x_2.4mm real data (2026-09-18): 150 iterations
brings the residual to 9.95e-4 (vs 3.3e-3 at 20) and the recovered image's
median/peak intensity rise substantially (median +24%, peak +7x) -- the
missing signal at 20 iterations was under-convergence, not a model defect
(see the same day's operator round-trip investigation). 150 is a compromise
across this repo's R=1-93.5 sweep: cheap for R=1 (converges in ~20 anyway)
while getting R=6 most of the way to full convergence; higher-R acquisitions
in the same sweep may still need more -- check the saved `residuals` array
and raise --num-iter if the tail hasn't flattened.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.run_recon cg <datdir> <seqname>
```

## `recon/L1-wavelet_TV_B0_SENSE.py`

```text
L1-wavelet + TV regularized, B0-informed SENSE reconstruction (.venv-recon; needs
both torch/mirtorch for the B0 operator and sigpy for the solver). This is the
only thing this module does: the time-segmented B0-corrected encoding operator
(recon/operators.py's GatheredSenseB0) is bridged into sigpy
(TorchLinopBridge), and each frame is solved with combined L1-wavelet + TV
regularization via PrimalDualHybridGradient (wavelet_tv_recon_b0), driven per
sequence by main_run.

    .venv-recon/bin/python -m recon.L1-wavelet_TV_B0_SENSE <datdir> <seqname> \
        [--lamb-l1 0.005] [--lamb-tv 0.005] [--num-iter 100] [--frames 0,1,2]

(The file name has a hyphen, so it cannot be used in an `import` statement;
`python -m` and importlib.import_module("recon.L1-wavelet_TV_B0_SENSE") both
work.)

This file replaces recon/sigpy_b0.py and recon/sigpy_recon.py. The latter's
other contents -- the plain (uncorrected) `wavelet_tv_recon`, its shared
per-frame loop `recon_frames`, and the `rss`/`cg-sense`/`l1-tv` batch drivers --
were removed; they live in git history. Its solver-rationale docstring (the O(1)
data rescaling, measured on real data) is kept below, since wavelet_tv_recon_b0
relies on the same reasoning; the other sections' mentions of `sigpy_recon.py`'s
plain-operator solver refer to that removed code. The sections below are the
original module docstrings, kept verbatim.

Formerly recon/sigpy_b0/sigpy_torch_bridge.py
---------------------------------------------
Bridges one mirtorch LinearMap (recon/operators.py, recon/operators.py)
into a pair of sigpy.linop.Linop objects, so sigpy's already-tested
regularization/solver machinery (recon/sigpy_recon.py's Wavelet +
FiniteDifference + prox.Stack + PrimalDualHybridGradient pattern) can be
reused unchanged with this repo's own B0-corrected encoding operator in
place of sigpy.mri.linop.Sense -- rather than reimplementing PDHG, wavelet
transforms, or TV proximal operators from scratch in torch.

Only sigpy (pure Python + numpy) needs to exist in .venv-recon for this --
not cupy -- so each Linop.apply()/.adjoint() call round-trips its argument
through host memory: numpy -> torch (GPU) -> mirtorch forward/adjoint ->
numpy. This costs one small host<->device copy per call (a few MB for this
repo's per-frame image/k-space sizes), negligible next to the actual
FFT + smaps + (for the B0-corrected operator) L-segment loop compute it
wraps -- not the bottleneck, so not worth adding a cupy dependency to
avoid it.

Before reusing sigpy regularization/solver defaults (e.g.
sigpy_recon.py's lamb_l1/lamb_tv) with a torch_op bridged through here,
call recon/operators.py's check_operator_unitary(torch_op, x0) once --
those defaults were tuned against sigpy.mri.linop.Sense, which is
genuinely unitary; a non-unitary torch_op (any B0-corrected
GatheredSenseB0, see that function's docstring) needs its own retuning,
not a transplanted lambda. This bridge class itself does not call it
automatically (a per-frame batch driver would otherwise pay a full power
iteration once per frame) -- callers building a new operator/driver
combination should check once, e.g. against frame 0's operator, before
looping.


Formerly recon/sigpy_b0/recon_sigpy_b0.py
-----------------------------------------
Combined L1-wavelet + total-variation regularized SENSE reconstruction,
B0-corrected -- sigpy_recon.py's exact regularization/solver pattern
(Wavelet + FiniteDifference stacked, prox.Stack of two L1Regs, solved via
PrimalDualHybridGradient), with sigpy_recon.py's plain sigpy.mri.linop.Sense
replaced by recon/operators.py's time-segmented GatheredSenseB0, bridged
into sigpy via recon/L1-wavelet_TV_B0_SENSE.py.

Motivation: run_recon.py's unregularized CG-SENSE shows real semi-
convergence at this repo's undersampling factors (R=6+) -- the well-posed
part of the image converges within ~20-30 iterations, but unconstrained
high-spatial-frequency content (a sharp object edge, worst-conditioned at
higher R) grows essentially without bound the longer CG runs, with no
natural stopping point (verified 2026-09-18: tracked individual edge
voxels through 150 CG iterations on real 2_6x_2.4mm data -- they grow
~50x from iteration 10 to 150 while the object interior is flat by
iteration ~30). A proximal-regularized solve constrains exactly that
unconstrained high-frequency content via the TV/wavelet sparsity prior,
rather than trading under-recovery for unbounded edge amplification.

Bridge validated (2026-09-18) on real 1_1x_5.4mm smaps/B0-map data before
any real-data use here: TorchLinopBridge's forward/adjoint satisfy
<Ax,y> == <x,A^Hy> to 1e-6 relative precision, and a synthetic ground-truth
recovery (complex phantom -> A -> PDHG+wavelet+TV, lambda shrinking toward
0) converges toward the true image as max_iter increases (0.141 -> 0.109 ->
0.094 -> 0.089 relative L2 error at 50/150/400/800 iterations) -- slower
than sigpy_recon.py's own plain-operator verification (which reaches ~0
by ~100 iterations), consistent with GatheredSenseB0's L=32-segment loop
making the operator more expensive/less well-conditioned per iteration
than the plain FFT+smaps Sense operator, not a bridge defect (adjoint
self-consistency alone already rules out a wiring bug).

y here is the GATHERED (K,Nc) k-space at this frame's sampled locations
(recon/operators.py's convention throughout this repo), not a dense
zero-filled [Nx,Ny,Nz,Nc] array -- unlike sigpy_recon.py's wavelet_tv_recon,
which takes the dense array and infers its own sampling mask from where it's
exactly zero. A_sigpy's ishape (image domain) is what G/proxg are built
against; its oshape must match y's shape exactly.


Formerly recon/sigpy_b0/run_recon_sigpy_b0.py
---------------------------------------------
Stage 2 batch driver: B0-corrected combined L1-wavelet + TV regularized
reconstruction (recon/L1-wavelet_TV_B0_SENSE.py), for this repo's own
<seqname>_epi_zf.h5 / smaps_<seqname>_sigpy.h5 naming convention -- the
regularized-recovery counterpart to recon/run_recon.py (unregularized) and
recon/run_recon.py (nuclear-norm regularized), all three sharing the
same B0-corrected GatheredSenseB0 encoding operator but differing in how
(or whether) they constrain the ill-posed part of the R>1 problem.

Unlike sigpy_recon.py's other Stage-2 drivers (sigpy_recon.py, sigpy_recon.py,
sigpy_recon.py -- all .venv-preprocessing, sigpy-only), this needs
torch/mirtorch for GatheredSenseB0 -- runs in .venv-recon (sigpy installed
there specifically for this bridge; see recon/L1-wavelet_TV_B0_SENSE.py's
module docstring for why cupy was not also added). Reimplements the
per-frame batch loop directly (rather than importing recon_frames.recon_frames)
since that loop's whole job -- calling recon_fn(data, smaps) per frame -- has
to change shape anyway: this driver's per-frame k-space is the *gathered*
(K,Nc) representation GatheredSenseB0 expects, not sigpy_recon.py's dense
zero-filled [Nx,Ny,Nz,Nc] per frame.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.L1-wavelet_TV_B0_SENSE <datdir> <seqname> \
        [--lamb-l1 0.005] [--lamb-tv 0.005] [--num-iter 100] [--frames 0,1,2]


Formerly recon/basic/recon_sigpy.py (docstring only; its solver code was removed)
-----------------------------------
Combined L1-wavelet + total-variation regularized SENSE reconstruction,
replacing BART's `pics -R W:7:0:lamb_l1 -R T:7:0:lamb_tv -i N -S` (run_bart.m)
with sigpy (see CLAUDE.md for why BART was dropped in favor of sigpy).

Both regularizers are combined via sigpy's standard multi-regularizer
pattern for sigpy.app.LinearLeastSquares: a stacked operator
`G = Vstack([Wavelet, FiniteDifference])` and a block-separable proximal
operator `prox.Stack([L1Reg(...,lamb_l1), L1Reg(...,lamb_tv)])` -- the same
structure sigpy.mri.app.L1WaveletRecon / TotalVariationRecon each use
individually (see their source in sigpy), just combined here rather than
applied one at a time. Solved via PrimalDualHybridGradient, the standard
solver for f(x) + g(Gx) with f smooth and g nonsmooth-but-prox-friendly on
a transformed domain.

Verified against a synthetic SENSE forward model: with fully-sampled
synthetic k-space and lamb_l1=lamb_tv shrinking to 0, the reconstruction
converges to the true image (relative error 6.5e-3 -> 7e-4 -> ~0 as lamda
goes 1e-2 -> 1e-3 -> 1e-5), confirming the Vstack/Stack/PDHG composition is
solving the intended problem rather than something subtly mis-wired.

`y` is rescaled to O(1) before solving (and the result rescaled back) --
this port's replacement for BART's `-S` flag, added after real project
data (wb_2.4mm ball phantom) surfaced the consequence of not having one:
lamb_l1/lamb_tv are tuned for O(1)-scaled data, so without this step
they're negligible against raw scanner-unit k-space (|y| ~ 1e4-1e5),
silently degrading "L1+TV" to an unregularized least-squares SENSE solve.
At this acquisition's undersampling (R=6) that's ill-posed, and a lambda
sweep at the true (unscaled) magnitude confirmed it manifests specifically
as spurious signal loss in a uniform phantom's center -- center/shell
signal ratio was flat and wrong (~0.64, vs RSS's own 0.76) from lamb=0 up
through lamb=80, only correcting once lamb reached ~1000, i.e. roughly
the scale this normalization now reaches automatically at lamb=0.005.
sigpy's power-iteration step-size calibration (max_power_iter) still
serves its own separate auto-scaling purpose (for A/G's operator norms,
not the data/regularizer scale) and remains in place alongside this.
```

## `recon/lowres_calib.py`

```text
Low-resolution reconstruction of the fully sampled k-space calibration region,
with optional B0-informed correction, and its temporal-stability check.
Subcommands:

    calib      reconstruct the calibration region (plain by default)
    stability  temporal-stability analysis of `calib`'s output

`calib` is a plain IFFT + smaps-weighted coil combine unless a B0 field map is
provided: `--b0` (uses <datdir>/recon/<seqname>_b0map.h5) or `--b0map PATH`
switch to the time-segmented, adjoint-only B0-corrected reconstruction, and
`--r2star` (with either) additionally corrects T2* amplitude decay via the
complex-field variant. torch/mirtorch are imported lazily, only on the B0
paths, so the plain path runs in .venv-preprocessing (no torch) while the B0
paths need .venv-recon:

    .venv-preprocessing/bin/python -m recon.lowres_calib calib <datdir> [seqname ...]
    .venv-recon/bin/python -m recon.lowres_calib calib <datdir> [seqname] --b0 [--r2star] [--L 32] [--nbins 128] [--device cuda]
    .venv-preprocessing/bin/python -m recon.lowres_calib stability <datdir> [<datdir> ...] [--variant {,b0,b0complex}]

The sections below are the original module docstrings, kept verbatim (the two
B0 sections' `python -m recon.lowres_calib {b0,b0complex}` usage is now
`calib --b0` / `calib --b0 --r2star`; their note that compute_calib_mask/
native_calib_grid are duplicated no longer applies -- there is one copy).

Formerly recon/lowres_calib/lowres_calib_recon.py
-------------------------------------------------
Quick low-res sanity-check reconstruction from the fully-sampled k-space
calibration region, for a fast look at a dataset without running the full
iterative Stage-2 pipeline (sigpy_recon.py / recon/).

Every frame's (ky, kz) sampling mask (see sampling/pd_sample.py's
`calib_frac`) always includes a small, fully-sampled centered region --
the calibration region ESPIRiT itself is calibrated from (via the deGRE
scan, not this one). That same guarantee holds for the *EPI* acquisition's
own per-frame mask: `omegas[..., t]` is `calib_mask | (extra incoherent
samples)` for every frame `t`, so `np.all(omegas, axis=-1)` (the
intersection across all frames) recovers exactly that calibration region --
no need to know `params.pd_calib_frac`/`R` ahead of time, or assume they
match the current defaults for a dataset acquired under different settings,
and no dependence on the region's shape either (a plain set intersection,
which works the same whether `calib_mask` is an ellipse or a rectangle).
Verified empirically on both `20260822ball_*` datasets: 362/10800 (ky, kz)
locations, a centered ellipse, identical between the radial/laminar
variants (they share the same underlying (ky, kz) mask, only the per-frame
EPI shot ordering differs) -- both acquired under `pd_sample.py`'s original
area-matched-ellipse `calib_frac` semantics (fraction of the sample
budget); a since-reverted intermediate version briefly made calib_frac a
fixed fraction of k-space instead (independent of R), but that let the
calibration region consume nearly the entire sample budget at high
acceleration, so `_calib_side_frac` restored the fraction-of-sample-budget
sizing -- now realized as a rectangle rather than the original ellipse
(see that module's docstring and docs/review-findings.md item 195). This
paragraph's specific numbers are a historical record of those two
(ellipse-shaped) datasets, not a current claim about the shape a fresh
acquisition's calibration region will have.

Since that region is exactly, not approximately, fully sampled, no
iterative reconstruction is needed: masking `ksp_epi_zf` down to it,
inverse-FFTing, and combining coils with the existing ESPIRiT sensitivity
maps (`recon/smaps_<seqname>_sigpy.h5`, already normalized so
`sum_c |s_c|^2 <= 1`, see smaps.py's process_smaps) is the correct linear
estimate directly -- `img = sum_c conj(s_c) * ifft(ksp_c)`, no
regularization or iteration.

**Reconstructs at native resolution, not zero-padded to the full (Nx, Ny,
Nz) acquisition grid.** Standard Cartesian MRI relation: resolution =
FOV/N (Delta_k = 1/FOV, and N samples span a k-space extent of N*Delta_k =
N/FOV). The calibration region's (ky, kz) bounding box -- 49 x 10 samples
on both `20260822ball_*` datasets, out of the full 240 x 45 -- caps the
achievable in-plane resolution at FOV_y/49 = 4.41 mm and FOV_z/10 = 4.05 mm,
far coarser than the full acquisition's 0.9 mm. Zero-padding that region up
to the full (Nx, Ny, Nz) grid before IFFT (an earlier version of this
script did exactly that) is pure sinc interpolation -- it doesn't add any
real information, and it makes neighboring voxels highly correlated by
construction (heavily oversampled relative to the true resolution), which
inflates variance-based diagnostics run on the result (e.g. an SVD/PCA
decomposition's "fraction of variance in the top component" -- see
preprocessing/lowres_calib_gain_drift_check.py). Reconstructing directly at
the native grid size gives the same true image content without the
redundant interpolation. `kx` (the readout direction, fully sampled on
every echo, not calibration-limited) is *also* cropped to match, to the
same effective sample count as `ky` -- `Nx_eff = round(Ny_eff * FOV_x /
FOV_y)`, which on these two datasets (FOV_x == FOV_y) works out to exactly
49, matching `Ny_eff` -- an explicit choice to keep the two in-plane axes
at matched resolution rather than leaving `kx` at full resolution while
`ky`/`kz` are calibration-limited.

Same centered-IFFT convention as preprocessing/gre_diagnostics.py's _ift3,
and the same FOV-preserving resize (`grid_resize.resize_to_epi_grid`) the
rest of this pipeline already uses to move smaps between grids of the same
FOV at different resolutions.

Usage (from repo root, .venv-preprocessing -- this module needs sigpy/h5py/
matplotlib, not torch/mirtorch, despite living under recon/ alongside the
torch-based MSLR pipeline; see CLAUDE.md's recon/ section for the venv
split rationale):
    .venv-preprocessing/bin/python -m recon.lowres_calib calib <datdir> [seqname]


Formerly recon/lowres_calib/lowres_temporal_stability.py
--------------------------------------------------------
Temporal stability analysis of lowres_calib.py's output.

That reconstruction masks every frame down to the same fixed, fully-sampled
(ky, kz) calibration region before IFFT + smaps combine (see its module
docstring) -- so unlike a full-resolution reconstruction, there is no
frame-varying undersampling mask/trajectory in the signal path here at all.
Any temporal variation this script measures is therefore attributable to
the object/system itself (thermal noise, scanner drift, motion) and *not*
to which (ky, kz) locations a given frame happened to sample -- the
standard NEMA/fBIRN-style phantom stability decomposition (percent
fluctuation + linear drift from a per-frame ROI-mean signal curve, plus a
per-voxel tSNR map) is used to quantify that directly.

variant='' reads recon/lowres_calib.py's plain output;
variant='b0' reads recon/lowres_calib.py's B0-corrected output --
see that module's docstring for why per-frame B0-induced phase, not just
system drift/noise, is expected to show up here as apparent instability
for a static object.

Usage (from repo root, .venv-preprocessing -- matplotlib/nibabel, not
torch/mirtorch, despite comparing recon/lowres_calib.py's
.venv-recon-only output; see CLAUDE.md's recon/ section "not a
single-venv package" note):
    .venv-preprocessing/bin/python -m recon.lowres_calib stability <datdir> [--seqname ArbEPI] [--variant b0]


Formerly recon/lowres_calib/lowres_calib_recon_b0.py
----------------------------------------------------
B0-corrected variant of recon/lowres_calib.py: same fully-sampled
(ky, kz) calibration region, same "no iteration, no regularization"
philosophy (`img = sum_c conj(smap_c) * ifft(ksp_c)`), but through
recon/operators.py's GatheredSenseB0 adjoint instead of a plain 3D
IFFT, so time-segmented conjugate-phase (Sutton/Noll/Fessler) B0
demodulation is included.

Motivation: even though every frame samples the exact same (ky, kz)
calibration locations (see lowres_calib.py's module docstring), the
*order* in which a given frame's shots visit them differs, so a given
(ky, kz) location is acquired at a different echo time (time since RF
excitation) in different frames -- confirmed against this repo's own
`echo_times` array: per-echo timing is frame-invariant as a *set*
(sequences/ArbEPI.py), but which (ky,kz) location maps to which echo
index varies per frame, exactly like the full acquisition. Off-resonance
phase accrues with that time, so the same k-space location carries a
different B0-induced phase from frame to frame -- for a genuinely static
object (a phantom), that is a source of *apparent* temporal instability
that has nothing to do with real signal change.

Does NOT reuse recon.operators.build_encoding_operator_b0 directly:
that function's shared time-segmentation fit is built from frame 0's
distinct echo times alone, on the (correct, for its own use case)
assumption that a frame's full ETL-worth of samples covers every echo
time the fit could ever need. That doesn't hold for the small calibration
region alone -- a given frame's calibration-region samples can miss some
of the echo times other frames' calibration samples use -- so
`_build_calib_operator_b0` below takes the union of echo times across
every frame's calibration-region samples instead of frame 0's alone.
Otherwise mirrors build_encoding_operator_b0's current construction
exactly, including GatheredSenseB0's (smaps, samp, pos, b_by_echo,
c_phasors) contract (b_by_echo shared across frames as one (n_unique_t,L)
table, each frame supplies only its own row-index vector `pos` -- not a
per-frame pre-gathered (K,L) copy).

L defaults to 32 here (not operators.py's own L=6 default) -- see
CLAUDE.md's recon/ "B0 off-resonance correction" section: a real-scale
sweep (recon/analysis.py) found L=6 badly under-resolves this
pipeline's real ETL=60 bandwidth-time product, while L=32 is the smallest
value that gets relative forward-model error under 1%.

Reconstructs at native (resolution-matched) grid size, not zero-padded to
the full (Nx,Ny,Nz) acquisition grid -- see lowres_calib.py's module
docstring for the FOV/N resolution derivation. compute_calib_mask/
native_calib_grid are duplicated from there rather than imported, to keep
this .venv-recon-only module off that file's module-level matplotlib
import (not part of the `recon` pyproject extra -- see CLAUDE.md's recon/
section "not a single-venv package" note).

Sign convention: this reconstruction only ever calls `.adjoint()`, never
`.apply()`. For the phase-only field this module builds (no R2* term),
that needs no special handling -- `c_phasors` here is a pure rotation
(|exp(i*theta)| = 1), so its conjugate *is* its own multiplicative
inverse, and GatheredSenseB0._apply_adjoint's existing `.conj()` already
does the right thing. (A complex field generalizing this to also correct
T2*/T1 decay, as recon/operators.py's r2star_map parameter supports for
the bidirectional/iterative path, needs a different, deliberately-flipped
sign for an adjoint-only reconstruction -- see
build_encoding_operator_b0's own docstring for why -- and isn't
implemented here.)

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.lowres_calib calib --b0 <datdir>         [--seqname ArbEPI] [--device cuda]


Formerly recon/lowres_calib/lowres_calib_recon_b0complex.py
-----------------------------------------------------------
Generalized-complex-field-map variant of recon/lowres_calib.py:
same fully-sampled calibration region, same adjoint-only philosophy, but
the time-segmented correction now accounts for a COMPLEX field combining
off-resonance and T2* decay, generalizing the real-only Δf(r) (Hz) that
recon/operators.py's mri_exp_approx wraps (mirtorch's own source raises
TypeError on a complex b0 input, so this generalization can't be done by
just passing it a complex array -- it needs its own spatial-basis
construction, below).

The *physical* forward-model exponent is psi(r) = i*2*pi*Δf(r) - R2*(r)
(signal decays as time since excitation increases). The array this module
actually builds and feeds to GatheredSenseB0, psi_recon = i*2*pi*Δf(r) +
R2*(r) (plus sign), is deliberately NOT that -- see
_build_calib_operator_b0_complex's inline comment for why: this
reconstruction only ever calls .adjoint(), never .apply(), and
GatheredSenseB0._apply_adjoint always conjugates c_phasors. Conjugating a
real quantity is a no-op, so building c_phasors from the physical -R2*
would make the adjoint apply the same decay a second time instead of
undoing it (confirmed empirically in the exploratory branch this was
ported from -- a version with the physical sign measured tSNR getting
monotonically *worse* through uncorrected -> phase-only -> this
complex-field version, the opposite of the expected direction). The true
multiplicative inverse of exp(psi*t) is exp(-psi*t), which only equals
exp(conj(psi)*t) when Re(psi)=0 (pure rotation, the phase-only case
lowres_calib.py already validated) -- psi_recon is chosen so
that conjugating it reproduces that true inverse instead.

Does NOT reuse recon.operators.build_encoding_operator_b0 directly, for
the same reason lowres_calib.py doesn't -- see that module's
docstring. Also does NOT copy the worktree exploratory branch's
GatheredSenseB0 call verbatim: that branch predates the current
(smaps, samp, pos, b_by_echo, c_phasors) contract (docs/review-findings.md
item 75) and passed a pre-gathered per-frame `b` instead -- adapted here to
build `pos` (a row-index vector into the shared `b_by_echo` table) exactly
like lowres_calib.py's `_build_calib_operator_b0` does.

Motivation (see CLAUDE.md's recon/ "B0 off-resonance correction" section
for the full derivation): lowres_calib.py's phase-only correction
demodulates off-resonance but leaves T2*/T1 amplitude decay uncorrected --
a given (ky,kz) calibration location is acquired at a different echo
index, hence a different amount of decay, in different frames, exactly
the same TE-scrambling mechanism that motivates the phase correction.
Generalizing Δf(r) to psi(r) corrects both simultaneously with the same
L-segment machinery.

Reference time = TE_nominal, not t=0 (excitation): both the magnitude
(R2*) and phase (Δf) corrections are the real and imaginary parts of the
SAME complex exponent exp(psi(r)*t), so they share one time reference by
construction -- shifting the per-sample times fed into the segmentation
fit by -TE_nominal makes the reconstruction target "the image as it would
appear at the prescribed TE" (the standard GRE/EPI T2*-weighted
convention) rather than "the undecayed image at the moment of excitation,"
which is neither standard nor numerically favorable (voxels with short
T2* would need very large amplification factors relative to t=0's much
earlier lead-in interval). TE_nominal is read directly from scan_info.mat's
schedules[...,2] at echo index (ETL-1)//2 -- the nominal-TE echo,
frame/shot-invariant by construction (see CLAUDE.md's mask2epi_radial
paragraph) -- not re-derived from the calibration region's own (possibly
incomplete) echo-time coverage.

Segmentation strategy: reuses mri_exp_approx's existing, already-tuned
(L=32) temporal interpolation weights (b_by_echo) and segment-placement
times (tl) UNCHANGED -- computed from Δf(r) alone, exactly as the
phase-only version does, just against TE-shifted times. Only the SPATIAL
basis functions are regeneralized: instead of mri_exp_approx's own
phase-only exp(i*2*pi*Δf(r)*tl[l]), this builds exp(psi(r)*tl[l]) directly
from the full, continuous per-voxel complex field (not the histogram-
binned reference values mri_exp_approx uses internally only to fit
b_by_echo). This is deliberately not a from-scratch joint (Δf, R2*)
segmentation fit: Δf's bandwidth-time product is what drove L=32 (see
CLAUDE.md's sweep finding, ~27 at this pipeline's real ETL=60/72ms scale);
R2*'s own decay-time product (R2*_typical * echo-train duration) is
printed by this script's main() specifically to check that it's small by
comparison, which is what justifies reusing Δf-only-tuned interpolation
weights for R2* too rather than re-deriving a joint fit.

R2*(r) comes from preprocessing/r2star_map.py's two-point estimate on the
same dual-echo deGRE data already used for Δf(r) -- see that module's
docstring.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.lowres_calib calib --b0 --r2star <datdir>         [--seqname ArbEPI] [--device cuda]
```

## `recon/analysis.py`

```text
One-off analysis and validation scripts (.venv-recon), not part of the production
path. Subcommands:

    sweep      time-segmentation count L accuracy sweep (produced the L=32 choice)
    benchmark  forward+adjoint cost vs L
    validate   field-by-field comparison against real ../mslr-recon (Julia) output

    .venv-recon/bin/python -m recon.analysis {sweep,benchmark,validate} ...

The sections below are the original module docstrings, kept verbatim.

Formerly recon/analysis/sweep_time_segments.py
----------------------------------------------
One-off analysis: sweep the time-segmentation count L (recon/operators.py)
against a synthetic ground truth scaled to this repo's REAL echo-train length
(ETL=60, ~1.2ms/echo -> ~72ms readout window) and REAL field-map range
(-300 to +70 Hz -- both numbers from operators.py's own module docstring)
-- not the 12-distinct-echo-time toy grid
tests/test_recon_operators_b0.py's test_more_segments_reduces_error_in_the_
realistic_regime uses. That test's own "L=16 is ~exact" finding is an
artifact of its toy grid having only 12 distinct echo times (L>=12 trivially
resolves every one exactly); it says nothing about whether L=6 (the current
production default, params.py-adjacent choice in operators.py/
run_recon.py) is adequate at the real ETL=60 scale, where there are up to
60 distinct echo times spanning a much larger bandwidth-time product
(BT = Delta_f_range * T_readout ~= 370 Hz * 0.072 s ~= 27).

Ground-truth construction mirrors tests/test_recon_b0_correction.py's
_brute_force_time_varying_ksp (brute-force, genuinely time-varying, one
dense FFT per echo/ky -- not a shortcut that could hide a segmentation
error) and tests/test_recon_operators_b0.py's mri_exp_approx-based operator
construction, reimplemented here (not imported from tests/) so this stays a
standalone recon/ analysis script, not a test-suite dependency. Nx/Nz/Nc are
kept small for speed -- this is a synthetic forward-model sweep, not a real
reconstruction -- only Ny=ETL and the field-map range/echo spacing need to
match real values, since the per-echo off-resonance phase model here only
depends on t_per_ky (one time per ky row, matching sequences/ArbEPI.py's
echo_times) and b0map_hz's spatial values, not on Nx/Nz/Nc.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.analysis sweep


Formerly recon/analysis/benchmark_b0_cost.py
--------------------------------------------
One-off analysis: measure the actual compute-time and GPU-memory cost of
recon/operators.py's GatheredSenseB0/build_encoding_operator_b0 as a
function of the time-segmentation count L, at this repo's REAL acquisition
scale (Nx,Ny,Nz,Nc,Nt = 240,240,45,18,30, R~9 -- see CLAUDE.md's recon/
section) -- not extrapolated from the smaller Julia/Python benchmark numbers
already documented there, which were measured for the *uncorrected*
GatheredSense operator only.

Uses synthetic (random) smaps/omega/b0map_hz/echo_times -- cost here depends
only on tensor shapes (FFT size, sample count K, L), not on real scan
content, matching recon/analysis.py's own reasoning for using
synthetic data.

Reports, for each swept L:
  - wall-clock time for one full forward (A.apply) + adjoint (A.adjoint)
    call over all 30 frames (one POGM gradient-step-equivalent)
  - peak GPU memory during that call (torch.cuda.max_memory_allocated)
  - the memory cost of building the operator itself -- dominated by the
    shared `(L,*N)` c_phasors tensor, which scales *linearly* with L (664
    MB at L=32); the per-frame `pos` index arrays (int64, one per frame)
    are the one genuinely L-independent piece, and small (69 MB total at
    this repo's real 30-frame/288000-sample-per-frame scale -- see
    recon/operators.py's GatheredSenseB0 docstring and
    docs/review-findings.md item 75, which this build_mem column measures
    the fix for)

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.analysis benchmark


Formerly recon/analysis/validate_against_mslr.py
------------------------------------------------
Validate recon/mslr.py against real ../mslr-recon (Julia/MIRT.jl)
output, field by field. Not a pytest test -- like seq2ge/validate_against_
matlab.py, this depends on real reference output that isn't committed to
this repo (machine-specific acquisition data + a completed mslr-recon run).

Usage (from repo root, using the .venv-recon environment):
    .venv-recon/bin/python -m recon.analysis validate <julia_reconstruct.mat>

The reference .mat is produced by ../mslr-recon's scripts/reconstruct.jl
(e.g. via experiments/20260822ball.jl -- run_recon's own `matwrite` call).
Every reconstruction parameter (fn_ksp/fn_smaps by convention -- see below,
patch_sizes, strides, sigma1A, lambda_global, conv_tol, niters, mom) is read
directly from the reference file rather than re-specified, so this always
replicates exactly what the reference run used, and fn_ksp/fn_smaps are
derived from the reference .mat's own directory layout
(<recon_dir>/mslr/<subdir>/<name>.mat -> <recon_dir>/ArbEPI_epi_zf.h5 +
<recon_dir>/smaps_ArbEPI_sigpy.h5 -- matching experiments/20260822ball.jl's
own `datasets` table), unless overridden with --ksp/--smaps.

Validated results (2026-08-25, RTX A6000, 20260822ball_radial dataset,
Nx,Ny,Nz,Nvc,Nt=240,240,45,18,30, R~9):

  config  iters  dc reldiff  reg reldiff  X_recon reldiff  Pearson r     py/julia s
  L       55     5.9e-7      3.0e-6       1.6e-5           0.9999999998  309/405
  G       56     1.6e-6      8.1e-6       3.8e-5           0.9999999989  96/134
  G+L     101    1.6e-6      2.1e-4       2.1e-5           0.9999999997  597/785

All three configs converge to the same iteration count as the Julia run
(confirming pogm_restart's early-stopping logic matches exactly) and match
to float32 summation-order noise -- the same class of ~1-ULP difference this
repo's seq2ge/ port already documents against real MATLAB output. Python
also runs consistently faster despite a simpler (fully-batched, not
hand-tuned) SVD/FFT implementation.
```

## `recon/hdf5_chunked_io.py`

```text
Chunk-aware reads of this pipeline's own plain-numpy-order .h5 files
(ksp_epi_zf and friends -- chunked one frame per chunk along the last
axis; see preprocessing/preprocess.py's writer). No torch/mirtorch import
here, deliberately: recon/mslr.py's `_load_array` and
recon/lowres_calib.py's (former) `_load_chunked` used to be two
independent copies of the same chunk-by-chunk-along-the-last-axis loop
(docs/review-findings.md item 200) specifically because
lowres_calib.py runs in .venv-preprocessing (no torch) and importing
recon/mslr.py would pull in recon/operators.py's mirtorch/torch
dependency for no reason -- this module has neither, so both venvs can
share it.

`read_frames_cropped`'s `spatial_slices` parameter is the fix for
docs/review-findings.md item 204: one HDF5 chunk is one frame's *entire*
spatial+coil extent (Nx,Ny,Nz,Nc), so a calibration-region-only consumer
still has to decompress every frame's full chunk once -- chunk boundaries
can't be worked around -- but it does NOT have to hold every frame's full
decompressed volume in memory *simultaneously*. Cropping each frame down
to `spatial_slices` immediately after decompressing it, before moving to
the next frame, bounds peak memory to one frame's full volume instead of
every frame's: on this pipeline's real 0.8mm/R~93.5 dataset (Nx,Ny,Nz,Nc,
Nt = 270,270,180,32,60), that's the difference between ~201GB (the whole
array, previously required just to read out a few hundred calibration-
region k-space samples) and ~3.4GB (one frame) -- see that item for the
measurement.
```

## `recon/run_rss.py`

```text
Root-sum-of-squares reconstruction: no smaps, no regularization, no B0
correction -- the plain baseline the other three drivers (recon/lowres_calib.py,
recon/L1-wavelet_TV_B0_SENSE.py, recon/run_recon.py) are compared against.

Revived (2026-09-22) after the recon/ consolidation removed the old
recon/run_rss.py + recon/recon_frames.py pair (they relied on MATLAB-literal
toppe.utils.ift3.m's fftshift-both-sides IFFT convention, since superseded
here by recon/operators.py's own verified ortho-normalized
fftshift(ifftn(ifftshift(.), norm='ortho')) adjoint -- the same convention
every other reconstruction in this repo now uses, so RSS is a fair,
consistent baseline rather than reproducing the old MATLAB port's own
odd-axis quirk (see the removed module's docstring, still in git history,
for that quirk's detail).

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.run_rss <datdir> <seqname>
```
