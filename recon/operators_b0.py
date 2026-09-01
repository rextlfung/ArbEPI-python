"""Time-segmented B0 off-resonance correction for recon/'s Cartesian
encoding operator -- the fuller, min-max-style stage of a staged plan for
adding B0 correction to GatheredSense (recon/operators.py).
recon/b0_correction.py implements a cheaper static single-segment stage
first; see its docstring (and CLAUDE.md's recon/ section) for why static
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
exp(+2j*pi*b0map_hz*t) demodulation -- see recon/b0_correction.py's
docstring for the full derivation and the reference (Sutton, Noll, Fessler,
IEEE TMI 2003) this is cross-checked against.

L (segment count) here is a fixed engineering default (matching mirtorch's
own Gmri default), not swept against a real error bound the way Fable's
staged plan recommends -- see CLAUDE.md's recon/ section for that open item.
"""

import math

import torch
from mirtorch.linear import BlockDiagonal
from mirtorch.linear.linearmaps import LinearMap
from mirtorch.linear.mri import mri_exp_approx


class GatheredSenseB0(LinearMap):
    """GatheredSense (recon/operators.py) plus time-segmented B0 correction.

    smaps: (Nc,*N) complex64. samp: (*N,) bool -- K True entries.
    b_weights: (K,L) complex -- per-sampled-k-space-location time-
        interpolation coefficients.
    c_phasors: (L,*N) complex -- per-segment, per-voxel demodulation
        phasors.
    Both from mri_exp_approx (see build_encoding_operator_b0).

    Forward: y = sum_l b_weights[:,l] * gather(FFT(smaps * c_phasors[l] * x)).
    A straightforward generalization of GatheredSense._apply/_apply_adjoint
    (L=1, b_weights=1, c_phasors=1 recovers it exactly) -- looped and
    accumulated per segment rather than batched over L, so peak memory
    stays at the single-segment footprint regardless of L (see
    CLAUDE.md/Fable's discussion of why GatheredSense itself had to avoid
    a dense per-k-space-shaped-tensor representation in the first place --
    the same memory pressure applies here, one level up).
    """

    def __init__(
        self, smaps: torch.Tensor, samp: torch.Tensor,
        b_weights: torch.Tensor, c_phasors: torch.Tensor,
    ):
        N = tuple(smaps.shape[1:])
        Nc = smaps.shape[0]
        idx = torch.nonzero(samp.reshape(-1), as_tuple=False).squeeze(-1)
        assert b_weights.shape[0] == idx.numel(), (
            f"b_weights has {b_weights.shape[0]} rows, expected {idx.numel()} sampled locations"
        )
        assert c_phasors.shape == (b_weights.shape[1],) + N, (
            f"c_phasors shape {tuple(c_phasors.shape)} != (L,*N) = {(b_weights.shape[1],) + N}"
        )
        super().__init__(N, (idx.numel(), Nc))
        self.smaps = smaps
        self.idx = idx
        self.N = N
        self.Nc = Nc
        self.dims = tuple(range(1, len(N) + 1))
        self.L = b_weights.shape[1]
        self.b_weights = b_weights
        self.c_phasors = c_phasors

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(self.size_out, dtype=self.smaps.dtype, device=x.device)
        for il in range(self.L):
            xc = (x * self.c_phasors[il]) * self.smaps  # (Nc,*N)
            kc = torch.fft.fftshift(
                torch.fft.fftn(torch.fft.ifftshift(xc, dim=self.dims), dim=self.dims, norm="ortho"),
                dim=self.dims,
            )
            kc_flat = kc.reshape(self.Nc, -1).T  # (prod(N),Nc), spatial C-order flatten
            y = y + self.b_weights[:, il : il + 1] * kc_flat[self.idx, :]
        return y

    def _apply_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(self.N, dtype=self.smaps.dtype, device=y.device)
        for il in range(self.L):
            yw = y * self.b_weights[:, il : il + 1].conj()  # (K,Nc)
            kc_full = torch.zeros(math.prod(self.N), self.Nc, dtype=y.dtype, device=y.device)
            kc_full[self.idx, :] = yw
            kc_full = kc_full.T.reshape(self.Nc, *self.N)
            kc_shifted = torch.fft.ifftshift(kc_full, dim=self.dims)
            xc = torch.fft.fftshift(
                torch.fft.ifftn(kc_shifted, dim=self.dims, norm="ortho"), dim=self.dims
            )
            x = x + self.c_phasors[il].conj() * (xc * self.smaps.conj()).sum(dim=0)
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
        import warnings

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
    echo_times_s: torch.Tensor,
    L: int = 6,
    nbins: int = 128,
) -> BlockDiagonal:
    """smaps: (Nc,Nx,Ny,Nz) complex64. omega: (Nx,Ny,Nz,Nt) bool, same
    sample count K per frame (asserted by the caller, matching
    build_encoding_operator's own contract). b0map_hz: (Nx,Ny,Nz) real, Hz
    -- same EPI grid as smaps, which preprocessing/run_b0map.py's grid
    resize now guarantees (b0map_hz used to live on a different, coarser
    deGRE grid; see grid_resize.py). echo_times_s: (Nx,Ny,Nz,Nt) real,
    seconds since RF excitation at each sampled location -- broadcast
    across Nx from preprocessing's (Ny,Nz,Nt) echo_times array (kx doesn't
    affect echo time; see preprocessing/preprocess.py's _build_echo_times),
    0 at unsampled locations (unused there, never gathered).

    nbins: mri_exp_approx builds its segmentation fit from an *equal-width*,
    magnitude-weighted histogram of b0map_hz's *entire* range (background
    included, unmasked) -- mirtorch's own Gmri default (20) turned out to be
    roughly half MIRT.jl's own recommended default (nhist=40, see
    ../mirt/mri/mri_exp_approx.m) even before accounting for anything
    dataset-specific, and was measured to badly under-resolve this
    pipeline's real field maps: ~half the volume is near-zero background
    and the in-object range is wide and asymmetric (this repo's real
    acquisitions span roughly -300 to +70 Hz, not symmetric around 0), so
    at nbins=20 nearly all the histogram's magnitude-weighted mass falls
    into 1-3 bins near zero and the (bins x L) least-squares fit becomes
    severely ill-conditioned everywhere else -- confirmed as the actual
    root cause (not L) of a real signal-loss-plus-incoherent-noise failure
    on real reconstructions: per-sample b_weights row sums (see
    _check_b_weight_row_sums) ranged [0.12, 2.89] at nbins=20 vs
    [0.9985, 1.0022] at nbins=100 on the same real data. 128 is a round
    number comfortably past that threshold; raise it further before
    lowering it.

    Returns an operator (Nx,Ny,Nz,Nt) -> (K,Nc,Nt), the same contract as
    build_encoding_operator (including `.A[it].idx`) -- a drop-in
    replacement wherever recon/reconstruct.py calls it, so long as the
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
    """
    Nt = omega.shape[-1]
    N = tuple(smaps.shape[1:])
    b0_neg = (-b0map_hz).to(torch.float32)  # see module docstring's sign-convention note

    samp0 = omega[..., 0]
    idx0 = torch.nonzero(samp0.reshape(-1), as_tuple=False).squeeze(-1)
    t0_ms = (echo_times_s[..., 0].reshape(-1)[idx0] * 1000).to(torch.float32)
    unique_t_ms = torch.unique(t0_ms, sorted=True)
    b_by_echo, c, _tl = mri_exp_approx(b0_neg, nbins, L, unique_t_ms)
    _check_b_weight_row_sums(b_by_echo, "shared (frame 0's distinct echo times)")
    c_phasors = c.transpose(0, 1).reshape((L,) + N).to(smaps.dtype)  # (Nvox,L)->(L,Nvox)->(L,*N)

    frames = []
    for it in range(Nt):
        samp = omega[..., it]
        idx = torch.nonzero(samp.reshape(-1), as_tuple=False).squeeze(-1)
        t_ms = (echo_times_s[..., it].reshape(-1)[idx] * 1000).to(torch.float32)
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
        b = b_by_echo[pos]
        frames.append(GatheredSenseB0(smaps, samp, b.to(smaps.dtype), c_phasors))
    return BlockDiagonal(frames)


def estimate_spectral_norm(A, x0: torch.Tensor, niter: int = 30) -> float:
    """Power iteration estimate of sigma1(A) -- same pattern
    tests/test_recon_operators.py's test_spectral_norm_is_near_unity_for_
    normalized_smaps_full_sampling uses inline for GatheredSense, factored
    out here for reuse: unlike the plain (unweighted) SENSE operator, a
    time-segmented GatheredSenseB0/BlockDiagonal's spectral norm has no
    known closed form (mri_exp_approx's B weights are a least-squares fit,
    not guaranteed unit-norm/orthogonal), so it needs to be measured before
    trusting it as POGM's Lipschitz-constant basis (`L = Nscales *
    sigma1A**2` in recon/reconstruct.py's run_recon) -- reusing the
    uncorrected operator's own sigma1A here would be a guess, not a
    measurement.

    A: any mirtorch LinearMap/BlockDiagonal (.apply/.adjoint). x0: any
    nonzero starting tensor matching A's size_in (dtype/device included)."""
    x = x0 / x0.norm()
    for _ in range(niter):
        x = A.adjoint(A.apply(x))
        x = x / x.norm()
    return A.apply(x).norm().item()
