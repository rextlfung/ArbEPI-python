"""Per-frame Cartesian SENSE encoding operator, block-diagonal-stacked over
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
"""

import math
import warnings

import torch
from mirtorch.linear import BlockDiagonal
from mirtorch.linear.linearmaps import LinearMap
from mirtorch.linear.mri import mri_exp_approx

from recon.solvers import poweriter


class GatheredSense(LinearMap):
    """smaps: (Nc, *N) complex64. samp: (*N,) bool. Maps (*N,) <-> (K, Nc)."""

    def __init__(self, smaps: torch.Tensor, samp: torch.Tensor):
        N = tuple(smaps.shape[1:])
        Nc = smaps.shape[0]
        idx = torch.nonzero(samp.reshape(-1), as_tuple=False).squeeze(-1)
        super().__init__(N, (idx.numel(), Nc))
        self.smaps = smaps
        self.idx = idx
        self.N = N
        self.Nc = Nc
        self.dims = tuple(range(1, len(N) + 1))  # spatial dims of a (Nc,*N) tensor

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        xc = x * self.smaps  # (Nc,*N)
        kc = torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(xc, dim=self.dims), dim=self.dims, norm="ortho"),
            dim=self.dims,
        )
        kc_flat = kc.reshape(self.Nc, -1).T  # (prod(N), Nc), spatial C-order flatten
        return kc_flat[self.idx, :]

    def _apply_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        kc_full = torch.zeros(math.prod(self.N), self.Nc, dtype=y.dtype, device=y.device)
        kc_full[self.idx, :] = y
        kc_full = kc_full.T.reshape(self.Nc, *self.N)
        kc_shifted = torch.fft.ifftshift(kc_full, dim=self.dims)
        xc = torch.fft.fftshift(
            torch.fft.ifftn(kc_shifted, dim=self.dims, norm="ortho"), dim=self.dims
        )
        return (xc * self.smaps.conj()).sum(dim=0)


def build_encoding_operator(smaps: torch.Tensor, omega: torch.Tensor) -> BlockDiagonal:
    """smaps: (Nc, Nx, Ny, Nz) complex64. omega: (Nx, Ny, Nz, Nt) bool, same
    sample count K per frame (asserted by the caller). Returns an operator
    (Nx, Ny, Nz, Nt) -> (K, Nc, Nt); `.A[it].idx` gives frame it's flat
    spatial sample indices, for gathering a matching k-space target array."""
    Nt = omega.shape[-1]
    frames = [GatheredSense(smaps, omega[..., it]) for it in range(Nt)]
    return BlockDiagonal(frames)


def gather_ksp(ksp0: torch.Tensor, A: BlockDiagonal) -> torch.Tensor:
    """ksp0: (Nx,Ny,Nz,Nc,Nt) dense zero-filled k-space (this repo's own
    preprocessing/ output layout). Returns (K,Nc,Nt), gathered with each
    frame's own operator so it lines up exactly with A.apply's output."""
    Nt = ksp0.shape[-1]
    Nc = ksp0.shape[3]
    K = A.A[0].idx.numel()
    out = torch.empty(K, Nc, Nt, dtype=ksp0.dtype, device=ksp0.device)
    for it in range(Nt):
        flat = ksp0[..., it].reshape(-1, Nc)  # spatial C-order flatten, matches GatheredSense
        out[:, :, it] = flat[A.A[it].idx, :]
    return out

def demodulate_smaps(smaps: torch.Tensor, b0map_hz: torch.Tensor, te_s: float) -> torch.Tensor:
    """smaps: (Nc, *N) complex. b0map_hz: (*N,) real, Hz -- same spatial grid
    as smaps (see preprocessing/run_b0map.py's EPI-grid resize, which makes
    this true for the first time -- b0map_hz used to live on a different,
    coarser deGRE grid). te_s: effective echo time (s) this single-segment
    correction is centered at.

    A single scalar te_s suffices for every frame: this repo's per-echo
    acquisition timing is identical across every shot/frame (see
    sequences/ArbEPI.py's echo_times computation and its uniformity check in
    tests/test_trajectory_matches_schedule.py's
    test_arbepi_schedule_echo_times), so there is exactly one meaningful
    "effective TE" for this whole acquisition, not a per-frame one -- pass
    the prescribed TE (params.TE), or equivalently the mean of scan_info.mat
    schedules[...,2]/preprocessing's echo_times, which are the same value
    here by symmetry (see that same test).

    Returns smaps pre-multiplied by exp(+i*2*pi*b0map_hz*te_s) -- feed this
    into recon/operators.py's build_encoding_operator in place of the raw
    smaps. No other code change is needed: this correction is static
    (time-invariant across frames), so it needs no per-frame handling and
    no change to GatheredSense/build_encoding_operator/reconstruct.py at
    all -- it's entirely absorbed into the smaps array those already take.
    """
    angle = (2 * math.pi * te_s) * b0map_hz.to(torch.float32)
    phasor = torch.exp(1j * angle).to(smaps.dtype)
    return smaps * phasor.unsqueeze(0)


class GatheredSenseB0(GatheredSense):
    """GatheredSense (recon/operators.py) plus time-segmented B0 correction.
    Subclasses GatheredSense and delegates to its _apply/_apply_adjoint
    for the per-segment FFT/gather and scatter/IFFT/coil-combine (the
    exact same math, once per segment, with c_phasors[l] pre-multiplied
    into x on the way in) rather than a second copy of that code -- see
    docs/review-findings.md item 89.

    smaps: (Nc,*N) complex64. samp: (*N,) bool -- K True entries.
    pos: (K,) int64 -- for each sampled location, its row index into
        b_by_echo. c_phasors/b_by_echo are looked up by index inside
        _apply/_apply_adjoint rather than gathered into a (K,L) tensor up
        front -- b_by_echo is typically shared across many GatheredSenseB0
        instances (see build_encoding_operator_b0, where every frame's
        `pos` indexes the same frame-invariant table), and materializing
        each instance's own (K,L) gather duplicates that shared table's
        data L/(rows of b_by_echo) times over (measured 3.3x the shared
        c_phasors cost at this repo's real L=32 scale -- see
        docs/review-findings.md item 75). Pass `pos = torch.arange(K)` and
        `b_by_echo` already shaped (K,L) to recover the old one-tensor-per-
        instance behavior exactly (the gather is then the identity).
    b_by_echo: (n_rows,L) complex -- per-segment time-interpolation
        coefficient table, indexed by `pos`.
    c_phasors: (L,*N) complex -- per-segment, per-voxel demodulation
        phasors.
    b_by_echo/c_phasors both from mri_exp_approx (see
    build_encoding_operator_b0).

    Forward: y = sum_l b_by_echo[pos,l] * gather(FFT(smaps * c_phasors[l] * x)).
    A straightforward generalization of GatheredSense._apply/_apply_adjoint
    (L=1, b_by_echo=ones(K,1), pos=arange(K), c_phasors=1 recovers it
    exactly) -- looped and accumulated per segment rather than batched over
    L, so peak memory stays at the single-segment footprint regardless of L
    (see CLAUDE.md's recon/ section for why GatheredSense itself had to
    avoid a dense per-k-space-shaped-tensor representation in the first
    place -- the same memory pressure applies here, one level up).
    """

    def __init__(
        self, smaps: torch.Tensor, samp: torch.Tensor,
        pos: torch.Tensor, b_by_echo: torch.Tensor, c_phasors: torch.Tensor,
    ):
        super().__init__(smaps, samp)  # sets smaps/idx/N/Nc/dims, LinearMap size (K,Nc)
        assert pos.shape[0] == self.idx.numel(), (
            f"pos has {pos.shape[0]} entries, expected {self.idx.numel()} sampled locations"
        )
        assert c_phasors.shape == (b_by_echo.shape[1],) + self.N, (
            f"c_phasors shape {tuple(c_phasors.shape)} != (L,*N) = {(b_by_echo.shape[1],) + self.N}"
        )
        self.L = b_by_echo.shape[1]
        self.pos = pos
        self.b_by_echo = b_by_echo
        self.c_phasors = c_phasors

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(self.size_out, dtype=self.smaps.dtype, device=x.device)
        for il in range(self.L):
            kc_gathered = super()._apply(x * self.c_phasors[il])  # (K,Nc)
            b_il = self.b_by_echo[self.pos, il : il + 1]  # (K,1), gathered lazily
            y = y + b_il * kc_gathered
        return y

    def _apply_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(self.N, dtype=self.smaps.dtype, device=y.device)
        for il in range(self.L):
            b_il = self.b_by_echo[self.pos, il : il + 1].conj()  # (K,1)
            xc = super()._apply_adjoint(y * b_il)  # (Nc,*N)-summed scatter/IFFT/coil-combine
            x = x + self.c_phasors[il].conj() * xc
        return x


def _check_b_weight_row_sums(b: torch.Tensor, frame_idx: int | str, tol: float = 0.1) -> None:
    """sum_l b_weights[i,l] is the fit evaluated at frequency 0 (every basis
    term exp(-i*0*tl_l)=1), so it should sit close to 1.0 for every sample
    when mri_exp_approx's underlying histogram fit is well-conditioned. A
    coarse `nbins` relative to how wide/asymmetric b0map_hz's range is (a
    real bug hit once already -- see git history / CLAUDE.md's recon/
    section) makes the fit ill-conditioned instead, producing erratic
    per-sample row sums that manifest as per-k-space-sample gain errors:
    signal loss where a sample's row sum is << 1, spatially incoherent
    noise from the resulting inconsistency across samples. This is a
    necessary-but-not-sufficient check (mirroring MIRT's own
    mri_exp_approx_test self-test, ../mirt/mri/mri_exp_approx.m's wrms
    computation, more precisely) -- cheap enough to run unconditionally
    rather than only when debugging."""
    row_sums = b.sum(dim=1).abs()
    lo, hi = row_sums.min().item(), row_sums.max().item()
    if lo < 1 - tol or hi > 1 + tol:
        warnings.warn(
            f"build_encoding_operator_b0: frame {frame_idx}'s b_weights row sums "
            f"range [{lo:.4f}, {hi:.4f}] (want close to 1.0) -- the segmentation fit "
            f"looks ill-conditioned (nbins too coarse for b0map_hz's dynamic range is "
            f"the known cause; see CLAUDE.md's recon/ section). Reconstructing with "
            f"this operator is likely to show signal loss and/or incoherent noise.",
            stacklevel=2,
        )


def build_encoding_operator_b0(
    smaps: torch.Tensor,
    omega: torch.Tensor,
    b0map_hz: torch.Tensor,
    echo_times_yz: torch.Tensor,
    L: int = 32,
    nbins: int = 128,
    r2star_map: torch.Tensor | None = None,
    t_ref_s: float = 0.0,
) -> BlockDiagonal:
    """smaps: (Nc,Nx,Ny,Nz) complex64. omega: (Nx,Ny,Nz,Nt) bool, same
    sample count K per frame (asserted by the caller, matching
    build_encoding_operator's own contract). b0map_hz: (Nx,Ny,Nz) real, Hz
    -- same EPI grid as smaps, which preprocessing/run_b0map.py's grid
    resize now guarantees (b0map_hz used to live on a different, coarser
    deGRE grid; see grid_resize.py). echo_times_yz: (Ny,Nz,Nt) real,
    seconds since RF excitation at each sampled location -- preprocessing's
    own native shape (kx doesn't affect echo time; see
    preprocessing/preprocess.py's _build_echo_times), read directly rather
    than broadcast to a dense (Nx,Ny,Nz,Nt) tensor first (311 MB from 1.3
    MB of distinct values at this repo's real scale -- see
    docs/review-findings.md item 90): a flat sample index into the
    (Nx,Ny,Nz) mask maps onto this array's flat (Ny,Nz) index via `idx %
    (Ny*Nz)`, since torch's C-order flatten gives idx = ix*Ny*Nz + iy*Nz +
    iz and the echo time is constant across ix.

    nbins: mri_exp_approx builds its segmentation fit from an *equal-width*,
    plain voxel-count histogram of b0map_hz's *entire* range (background
    included, unmasked; mirtorch 0.3.1's `_uniform_histogram` is
    `histogram.scatter_add(0, indices, torch.ones_like(values))` -- no
    magnitude weighting anywhere in the call path, unlike MIRT's original
    `mri_exp_approx`, whose weight-vector argument mirtorch's port
    dropped) -- mirtorch's own Gmri default (20) turned out to be roughly
    half MIRT.jl's own recommended default (nhist=40, see
    ../mirt/mri/mri_exp_approx.m) even before accounting for anything
    dataset-specific, and was measured to badly under-resolve this
    pipeline's real field maps: ~half the volume is near-zero background
    and the in-object range is wide and asymmetric (this repo's real
    acquisitions span roughly -300 to +70 Hz, not symmetric around 0), so
    at nbins=20 nearly all the histogram's *voxel-count* mass falls into
    1-3 bins near zero (background dominates by sheer count, not
    magnitude) and the (bins x L) least-squares fit becomes severely
    ill-conditioned everywhere else -- confirmed as the actual root cause
    (not L) of a real signal-loss-plus-incoherent-noise failure on real
    reconstructions: per-sample b_weights row sums (see
    _check_b_weight_row_sums) ranged [0.12, 2.89] at nbins=20 vs
    [0.9985, 1.0022] at nbins=100 on the same real data. The equal-width
    range itself is set by `b0.amin()`/`amax()` over the whole volume,
    which is what actually makes an asymmetric in-object range expensive
    in bins. 128 is a round number comfortably past that threshold; raise
    it further before lowering it.

    Returns an operator (Nx,Ny,Nz,Nt) -> (K,Nc,Nt), the same contract as
    build_encoding_operator (including `.A[it].idx`) -- a drop-in
    replacement wherever recon/mslr.py calls it, so long as the
    caller also re-estimates sigma1A for POGM's step size (the B0-corrected
    operator's spectral norm is not guaranteed to match the uncorrected
    one's, unlike everything else about this operator's contract).

    Per-echo acquisition time is frame-invariant here: every frame samples
    the exact same set of ETL distinct echo times, just at different (ky,kz)
    locations (sequences/ArbEPI.py's echo_times depends only on echo index,
    not which (ky,kz) that echo happens to encode -- confirmed to ~1e-15
    precision by tests/test_trajectory_matches_schedule.py's
    test_arbepi_schedule_echo_times). That means mri_exp_approx's
    frequency-domain fit -- both c_phasors (which depends only on b0map_hz
    and the segment-center times tl, themselves only a function of
    min/max sample time) and the (nbins,L) pinv fit itself -- only needs
    solving *once*, against the ETL unique times, not once per frame
    against a 288000-sample-redundant target. An earlier version of this
    function called mri_exp_approx fresh per frame instead: besides wasting
    compute (re-solving an already-ill-conditioned-at-low-nbins fit 30x
    over), it stored an independent (L,*N) c_phasors copy per frame -- at
    this repo's real 240x240x45/18-coil/30-frame scale, L=16 alone made
    that redundancy large enough to push a real reconstruction into a CUDA
    OOM during the low-rank prox step (measured; see git history). Sharing
    one c_phasors tensor across every frame's GatheredSenseB0 (safe --
    _apply/_apply_adjoint only ever read it) fixes both.

    r2star_map: optional (Nx,Ny,Nz) real, 1/s, same EPI grid as b0map_hz --
    generalizes the real off-resonance field Δf(r) to a complex field
    ψ(r) = i*2*pi*Δf(r) - R2*(r), so the same segmented-exponential
    machinery corrects T2*/T1 amplitude decay alongside phase (motivation:
    a given (ky,kz) location is acquired at a different echo index, hence
    a different amount of decay, in different frames -- see
    preprocessing/r2star_map.py and CLAUDE.md's recon/ "B0 off-resonance
    correction" section). None (default) reproduces the original
    phase-only operator bit-for-bit: c_phasors then comes straight from
    mri_exp_approx's own spatial output, exactly as before this parameter
    existed, and t_ref_s is ignored.

    IMPORTANT, and NOT the sign convention used by
    recon/lowres_calib.py on the (unmerged)
    worktree-lowres-calib-recon branch: this operator is bidirectional
    (recon/mslr.py's run_recon calls both .apply() and .adjoint()
    through POGM, and estimate_spectral_norm's power iteration needs both
    too), so c_phasors here is built from the PHYSICAL forward exponent
    exp(psi(r)*t) with psi(r) = i*2*pi*Δf(r) - R2*(r) -- decaying, not
    growing, as t increases -- and GatheredSenseB0._apply_adjoint's
    existing `c_phasors[l].conj()` is left untouched. That conjugate
    already *is* the true mathematical adjoint of a per-voxel diagonal
    scaling for ANY complex c_phasors, decaying or not (the adjoint of a
    diagonal matrix is its conjugate, full stop -- no "does conjugating
    undo the decay" question ever enters into whether `_apply_adjoint` is
    correct). The branch's calib-region script instead flips the sign
    (its psi_recon = i*2*pi*Δf + R2*) because it only ever calls
    `.adjoint()`, never `.apply()`, on a non-iterative, adjoint-only
    reconstruction, and wants matched-filter-style decay *compensation*
    (an approximate deconvolution) rather than a faithful forward model.
    Porting that flipped sign into *this* bidirectional operator would
    make `.apply()` model signal growth (exp(+R2*(r)*t), unbounded as t
    grows) instead of decay: wrong physically, and it would inflate
    `estimate_spectral_norm`'s power-iteration estimate and collapse
    POGM's step size. Do not "fix" this to match the branch --
    tests/test_recon_operators_b0.py's
    `test_r2star_generalization_adjoint_is_self_consistent` locks the
    distinction in via an adjoint dot-product check, which the branch's
    convention fails and this one passes.

    t_ref_s: reference time (seconds since RF excitation) the R2* decay is
    measured relative to -- ignored when r2star_map is None. Only matters
    once -R2*(r)*t enters the exponent: with t measured from excitation
    (t_ref_s=0), short-T2* voxels get heavily down-weighted relative to
    long-T2* ones in the segmentation fit (needlessly ill-conditioned),
    and the reconstructed image would mean "the undecayed image at the
    moment of excitation" rather than the standard T2*-weighted
    image-at-TE convention. Pass the nominal-TE echo's acquisition time
    (scan_info.mat's schedules[0,0,(ETL-1)//2,2] -- see
    preprocessing/r2star_map.py's caller for how to read it) so decay
    factors straddle 1 and the reconstruction target is "the image as it
    would appear at the prescribed TE" -- matches
    recon/lowres_calib.py's own TE-referencing for the
    same reason (unlike the sign, this part of its design *is* reused
    as-is). The phase-only path (r2star_map=None) needs no such shift: a
    global time-reference change to Δf alone only rescales the image by a
    per-voxel phase, and shifting it would perturb the exact bit-for-bit
    match with the pre-R2* operator this function preserves whenever
    r2star_map is None.
    """
    Nt = omega.shape[-1]
    N = tuple(smaps.shape[1:])
    Ny, Nz = N[1], N[2]
    n_yz = Ny * Nz
    echo_times_flat = echo_times_yz.reshape(n_yz, Nt)  # (Ny*Nz,Nt), no Nx broadcast
    # R2* path references times to TE_nominal (see docstring); phase-only
    # path is unshifted so this reduces to the exact pre-R2* operator.
    fit_times_flat = echo_times_flat if r2star_map is None else echo_times_flat - t_ref_s
    b0_neg = (-b0map_hz).to(torch.float32)  # see module docstring's sign-convention note

    samp0 = omega[..., 0]
    idx0 = torch.nonzero(samp0.reshape(-1), as_tuple=False).squeeze(-1)
    t0_ms = (fit_times_flat[idx0 % n_yz, 0] * 1000).to(torch.float32)
    unique_t_ms = torch.unique(t0_ms, sorted=True)
    b_by_echo, c, tl = mri_exp_approx(b0_neg, nbins, L, unique_t_ms)
    _check_b_weight_row_sums(b_by_echo, "shared (frame 0's distinct echo times)")
    b_by_echo = b_by_echo.to(smaps.dtype)  # (n_unique_t,L), shared across every frame below

    if r2star_map is None:
        # (Nvox,L)->(L,Nvox)->(L,*N)
        c_phasors = c.transpose(0, 1).reshape((L,) + N).to(smaps.dtype)
    else:
        assert tuple(r2star_map.shape) == N, (
            f"r2star_map shape {tuple(r2star_map.shape)} != smaps grid {N}"
        )
        r2_hz = r2star_map.to(torch.float32)
        t_span_s = float((unique_t_ms.max() - unique_t_ms.min()).item()) / 1000
        bt_df = float(b0map_hz.max() - b0map_hz.min()) * t_span_s
        bt_r2 = float(r2_hz.max()) * t_span_s
        print(
            f"  build_encoding_operator_b0: R2* decay-time product = {bt_r2:.4f} vs "
            f"Δf's bandwidth-time product = {bt_df:.2f} (should be much smaller -- this "
            "is what justifies reusing Δf-only-tuned b_by_echo/tl for the complex field; "
            "see module docstring)."
        )
        # Physical forward exponent (decaying, not the branch script's
        # flipped-sign reconstruction convenience) -- see docstring.
        psi = 1j * 2 * math.pi * b0map_hz.to(torch.complex64) - r2_hz.to(torch.complex64)
        tl_c = tl.to(torch.complex64)  # tl already in seconds (mri_exp_approx divides by 1000)
        c_phasors = torch.exp(
            tl_c.reshape((L,) + (1,) * len(N)) * psi[None, ...]
        ).to(smaps.dtype)

    frames = []
    for it in range(Nt):
        samp = omega[..., it]
        idx = torch.nonzero(samp.reshape(-1), as_tuple=False).squeeze(-1)
        t_ms = (fit_times_flat[idx % n_yz, it] * 1000).to(torch.float32)
        # Map each sample's time to its row in b_by_echo, rather than
        # assuming a fixed order -- an explicit exact-match check, so a
        # dataset that ever breaks the frame-invariant-timing assumption
        # above fails loudly instead of silently mis-assigning weights.
        pos = torch.searchsorted(unique_t_ms, t_ms).clamp(max=unique_t_ms.numel() - 1)
        assert torch.allclose(unique_t_ms[pos], t_ms, atol=1e-4), (
            f"build_encoding_operator_b0: frame {it}'s sample echo times aren't "
            "a subset of frame 0's distinct times -- the frame-invariant-timing "
            "assumption this function relies on doesn't hold for this dataset."
        )
        frames.append(GatheredSenseB0(smaps, samp, pos, b_by_echo, c_phasors))
    return BlockDiagonal(frames)


def estimate_spectral_norm(A, x0: torch.Tensor, niter: int = 200, tol: float = 1e-6) -> float:
    """Power iteration estimate of sigma1(A) -- delegates to
    recon/solvers.py's poweriter (same computation, applied to A's own
    forward/adjoint) rather than a second copy of the loop: unlike the
    plain (unweighted) SENSE operator, a time-segmented
    GatheredSenseB0/BlockDiagonal's spectral norm has no known closed form
    (mri_exp_approx's B weights are a least-squares fit, not guaranteed
    unit-norm/orthogonal), so it needs to be measured before trusting it as
    POGM's Lipschitz-constant basis (`L = Nscales * sigma1A**2` in
    recon/mslr.py's run_recon) -- reusing the uncorrected operator's
    own sigma1A here would be a guess, not a measurement.

    Power iteration converges to sigma1 *from below*, so an under-converged
    estimate under-estimates the Lipschitz constant and drives POGM's step
    size the unsafe direction (too large, i.e. divergent) -- poweriter's
    tol-based early stop only returns once the ratio has stabilized,
    instead of trusting a fixed iteration count to have been enough.

    A: any mirtorch LinearMap/BlockDiagonal (.apply/.adjoint). x0: any
    nonzero starting tensor matching A's size_in (dtype/device included)."""
    return poweriter(A.apply, A.adjoint, x0, niter=niter, tol=tol)


def check_operator_unitary(
    A, x0: torch.Tensor, name: str = 'A', tol: float = 0.05, niter: int = 200,
    poweriter_tol: float = 1e-6,
) -> float:
    """Measures sigma1(A) (via estimate_spectral_norm) and warns if it's not
    close to 1.0 -- added 2026-09-18 after a real debugging session where
    exactly this gap (GatheredSenseB0's sigma1 = 1.29, not ~1.0 like the
    plain GatheredSense's 0.9998) went undiagnosed for a while: sigpy's
    sigpy_recon.py-style L1-wavelet+TV regularization (lamb_l1/lamb_tv,
    tuned against a genuinely unitary sigpy.mri.linop.Sense) silently needed
    a ~100x larger lambda once the same pattern was reused with
    recon/sigpy_b0.py wrapping a non-unitary operator -- a fixed
    lambda's *effective* regularization strength (relative to the data
    term) shifts with the operator's own norm, both because sigpy's PDHG
    step-size calibration (tau ~ 1/||A||^2 with sigma held fixed, see
    sigpy/app.py's LinearLeastSquares._get_PrimalDualHybridGradient) slows
    convergence of both the data and regularization terms together at a
    fixed iteration budget, and because POGM's own Lipschitz constant
    (recon/mslr.py's run_recon: `L = Nscales * sigma1A**2`) is
    directly sigma1(A)-dependent.

    Call this once when building a *new* encoding operator (or composing
    an existing one into a new solver/regularization scheme) rather than
    assuming unitarity -- a plain Cartesian-FFT + RSS-normalized-smaps
    operator (GatheredSense) genuinely is unitary (sigma1 ~= 1.0, tight
    tolerance -- see tests/test_recon_operators.py) and should warn if it
    ever isn't (a real regression). A time-segmented B0-corrected operator
    (GatheredSenseB0) is *not* guaranteed unitary by construction --
    mri_exp_approx's segmentation weights are a least-squares fit, not an
    orthogonal/unit-norm decomposition (see estimate_spectral_norm's own
    docstring) -- so a warning there is expected, not necessarily a bug;
    it's a reminder to re-tune (or explicitly account for) lambda/step-size
    choices made under a unitary-operator assumption, not an error to
    silence.

    Returns the measured sigma1(A) either way, so callers can reuse it
    (e.g. as POGM's sigma1A) instead of measuring twice.
    """
    sigma1 = estimate_spectral_norm(A, x0, niter=niter, tol=poweriter_tol)
    if abs(sigma1 - 1.0) > tol:
        warnings.warn(
            f"check_operator_unitary: {name}'s sigma1(A) = {sigma1:.4f}, not close to 1.0 "
            f"(tol={tol}) -- this operator is not unitary. Any regularization weight "
            "(lambda_l1/lambda_tv/lambda_global/...) or step size tuned assuming a unitary "
            "operator (e.g. sigpy_recon.py's lamb_l1=lamb_tv=0.005 default, tuned against a "
            "genuinely unitary sigpy.mri.linop.Sense) will not transfer directly -- expect to "
            "re-tune, or explicitly normalize the operator/data by sigma1(A) first.",
            stacklevel=2,
        )
    return sigma1
