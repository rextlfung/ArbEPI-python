"""MRI encoding operators: image -> sampled multi-coil k-space.

    SENSE            smaps -> FFT -> sample
    SENSE_B0         B0 phase accrual -> smaps -> FFT -> sample (time-segmented)
    SENSE_B0_R2star  B0 phase accrual + R2* magnitude decay -> smaps -> FFT -> sample

Each is one frame's operator; build_sense/build_sense_b0/build_sense_b0_r2star
stack one per frame into a mirtorch BlockDiagonal mapping (Nx,Ny,Nz,Nt) ->
(K,Nc,Nt), where K is the number of sampled k-space locations per frame. The
output holds only those K samples (not a dense zero-filled grid), which is what
keeps memory manageable at this repo's real data sizes.

Sign convention (Sutton, Noll & Fessler, IEEE TMI 2003): the forward model is
s(t) = sum_r m(r) exp(+i 2 pi df(r) t) exp(-i 2 pi k(t).r). mirtorch's
mri_exp_approx fits exp(-i 2 pi b t), so it is passed -b0map_hz (the same
negation mirtorch's own Gmri demo uses, `zmap=-b0`).
"""

import math
import warnings

import torch
from mirtorch.linear import BlockDiagonal
from mirtorch.linear.linearmaps import LinearMap
from mirtorch.linear.mri import mri_exp_approx


class SENSE(LinearMap):
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
        # Scatter into a (Nc,prod(N))-laid-out buffer directly (along dim=1,
        # not dim=0) so the follow-up .reshape(Nc,*N) is a free view -- the
        # (prod(N),Nc)-then-.T.reshape() ordering this used to use forces a
        # full (Nc,*N)-sized copy (reshape can't return a view of a
        # transposed/non-contiguous tensor), a real, measured contributor to
        # a CUDA OOM at this repo's largest dataset scale (4_93.5x_0.8mm,
        # (270,270,180) grid, Nc=32 -- each such copy is ~3.13 GiB; see
        # CLAUDE.md's recon/ B0 subsection).
        kc_full = torch.zeros(self.Nc, math.prod(self.N), dtype=y.dtype, device=y.device)
        kc_full[:, self.idx] = y.T
        kc_full = kc_full.reshape(self.Nc, *self.N)
        kc_shifted = torch.fft.ifftshift(kc_full, dim=self.dims)
        xc = torch.fft.fftshift(
            torch.fft.ifftn(kc_shifted, dim=self.dims, norm="ortho"), dim=self.dims
        )
        return (xc * self.smaps.conj()).sum(dim=0)


def build_sense(smaps: torch.Tensor, omega: torch.Tensor) -> BlockDiagonal:
    """smaps: (Nc, Nx, Ny, Nz) complex64. omega: (Nx, Ny, Nz, Nt) bool, same
    sample count K per frame (asserted by the caller). Returns an operator
    (Nx, Ny, Nz, Nt) -> (K, Nc, Nt); `.A[it].idx` gives frame it's flat
    spatial sample indices, for gathering a matching k-space target array."""
    Nt = omega.shape[-1]
    frames = [SENSE(smaps, omega[..., it]) for it in range(Nt)]
    return BlockDiagonal(frames)


class SENSE_B0(SENSE):
    """SENSE (recon/operators.py) plus time-segmented B0 correction.
    Subclasses SENSE and delegates to its _apply/_apply_adjoint
    for the per-segment FFT/gather and scatter/IFFT/coil-combine (the
    exact same math, once per segment, with c_phasors[l] pre-multiplied
    into x on the way in) rather than a second copy of that code -- see
    docs/review-findings.md item 89.

    smaps: (Nc,*N) complex64. samp: (*N,) bool -- K True entries.
    pos: (K,) int64 -- for each sampled location, its row index into
        b_by_echo. c_phasors/b_by_echo are looked up by index inside
        _apply/_apply_adjoint rather than gathered into a (K,L) tensor up
        front -- b_by_echo is typically shared across many SENSE_B0
        instances (see build_sense_b0, where every frame's
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
    build_sense_b0).

    Forward: y = sum_l b_by_echo[pos,l] * gather(FFT(smaps * c_phasors[l] * x)).
    A straightforward generalization of SENSE._apply/_apply_adjoint
    (L=1, b_by_echo=ones(K,1), pos=arange(K), c_phasors=1 recovers it
    exactly) -- looped and accumulated per segment rather than batched over
    L, so peak memory stays at the single-segment footprint regardless of L
    (see CLAUDE.md's recon/ section for why SENSE itself had to
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


class SENSE_B0_R2star(SENSE_B0):
    """SENSE_B0 with magnitude decay as well as phase accrual.

    The real off-resonance field df(r) becomes a complex field
    psi(r) = i*2*pi*df(r) - R2*(r), and each segment's phasor is
    exp(psi(r) * t_l) instead of exp(i*2*pi*df(r) * t_l). Times t_l are
    measured from t_ref (the nominal TE), so decay factors straddle 1 and the
    image is "the image at TE" rather than "the undecayed image at excitation".

    Sign: this is the physical forward model (decaying as t grows). SENSE_B0's
    adjoint conjugates the phasors, which is the exact adjoint for any complex
    phasor, so no sign trick is needed. (An adjoint-only reconstruction can
    flip the R2* sign to compensate decay; doing that here would make .apply()
    model growth and break the adjoint test in tests/test_recon_operators.py.)

    The forward/adjoint code is SENSE_B0's; what differs is the phasors, built
    by segment_phasors. They're computed once and shared by every frame's
    operator (see build_sense_b0_r2star), since an (L,*N) copy per frame
    doesn't fit in GPU memory at real data sizes.
    """

    @staticmethod
    def segment_phasors(
        b0map_hz: torch.Tensor, r2star_map: torch.Tensor, tl: torch.Tensor, dtype: torch.dtype,
    ) -> torch.Tensor:
        """(L,*N) phasors exp(psi(r) * tl[l]), psi = i*2*pi*b0map_hz - r2star_map.

        b0map_hz (Hz), r2star_map (1/s): (*N,) on the EPI grid. tl: (L,) segment
        times in seconds relative to t_ref, from mri_exp_approx."""
        N = tuple(b0map_hz.shape)
        assert tuple(r2star_map.shape) == N, f"r2star_map shape {tuple(r2star_map.shape)} != {N}"
        r2_hz = r2star_map.to(torch.float32)
        # Reusing the df-only segmentation fit (b_by_echo, tl) for the complex
        # field is fine as long as R2*'s decay-time product is much smaller than
        # df's bandwidth-time product.
        t_span_s = float((tl.max() - tl.min()).item())
        bt_df = float(b0map_hz.max() - b0map_hz.min()) * t_span_s
        bt_r2 = float(r2_hz.max()) * t_span_s
        print(f"  SENSE_B0_R2star: R2* decay-time product = {bt_r2:.4f} vs "
              f"B0 bandwidth-time product = {bt_df:.2f} (should be much smaller)")
        psi = 1j * 2 * math.pi * b0map_hz.to(torch.complex64) - r2_hz.to(torch.complex64)
        tl_c = tl.to(torch.complex64).reshape((-1,) + (1,) * len(N))
        return torch.exp(tl_c * psi[None, ...]).to(dtype)


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
            f"build_sense_b0: frame {frame_idx}'s b_weights row sums "
            f"range [{lo:.4f}, {hi:.4f}] (want close to 1.0) -- the segmentation fit "
            f"looks ill-conditioned (nbins too coarse for b0map_hz's dynamic range is "
            f"the known cause; see CLAUDE.md's recon/ section). Reconstructing with "
            f"this operator is likely to show signal loss and/or incoherent noise.",
            stacklevel=2,
        )


def _segment_fit(
    omega: torch.Tensor,
    b0map_hz: torch.Tensor,
    echo_times_yz: torch.Tensor,
    L: int,
    nbins: int,
    t_ref_s: float = 0.0,
):
    """Shared by build_sense_b0/build_sense_b0_r2star: fit mri_exp_approx's
    time segmentation once against frame 0's distinct echo times, then map
    every frame's samples onto rows of the shared coefficient table.

    Returns (b_by_echo, c, tl, unique_t_ms, pos_per_frame)."""
    Nt = omega.shape[-1]
    Ny, Nz = omega.shape[1], omega.shape[2]
    n_yz = Ny * Nz
    echo_times_flat = echo_times_yz.reshape(n_yz, Nt)  # (Ny*Nz,Nt), no Nx broadcast
    fit_times_flat = echo_times_flat if t_ref_s == 0.0 else echo_times_flat - t_ref_s
    b0_neg = (-b0map_hz).to(torch.float32)  # see module docstring's sign-convention note

    samp0 = omega[..., 0]
    idx0 = torch.nonzero(samp0.reshape(-1), as_tuple=False).squeeze(-1)
    t0_ms = (fit_times_flat[idx0 % n_yz, 0] * 1000).to(torch.float32)
    unique_t_ms = torch.unique(t0_ms, sorted=True)
    b_by_echo, c, tl = mri_exp_approx(b0_neg, nbins, L, unique_t_ms)
    _check_b_weight_row_sums(b_by_echo, "shared (frame 0's distinct echo times)")

    pos_per_frame = []
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
            f"build_sense_b0: frame {it}'s sample echo times aren't "
            "a subset of frame 0's distinct times -- the frame-invariant-timing "
            "assumption this function relies on doesn't hold for this dataset."
        )
        pos_per_frame.append(pos)
    return b_by_echo, c, tl, unique_t_ms, pos_per_frame


def build_sense_b0(
    smaps: torch.Tensor,
    omega: torch.Tensor,
    b0map_hz: torch.Tensor,
    echo_times_yz: torch.Tensor,
    L: int = 32,
    nbins: int = 128,
) -> BlockDiagonal:
    """smaps: (Nc,Nx,Ny,Nz) complex64. omega: (Nx,Ny,Nz,Nt) bool, same
    sample count K per frame (asserted by the caller, matching
    build_sense's own contract). b0map_hz: (Nx,Ny,Nz) real, Hz
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
    build_sense (including `.A[it].idx`) -- a drop-in
    replacement for it wherever an encoding operator is expected, so long as the
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
    one c_phasors tensor across every frame's SENSE_B0 (safe --
    _apply/_apply_adjoint only ever read it) fixes both.
    """
    N = tuple(smaps.shape[1:])
    b_by_echo, c, _tl, _t, pos_per_frame = _segment_fit(omega, b0map_hz, echo_times_yz, L, nbins)
    b_by_echo = b_by_echo.to(smaps.dtype)  # (n_unique_t,L), shared across every frame below
    # (Nvox,L)->(L,Nvox)->(L,*N)
    c_phasors = c.transpose(0, 1).reshape((L,) + N).to(smaps.dtype)
    frames = [
        SENSE_B0(smaps, omega[..., it], pos, b_by_echo, c_phasors)
        for it, pos in enumerate(pos_per_frame)
    ]
    return BlockDiagonal(frames)


def build_sense_b0_r2star(
    smaps: torch.Tensor,
    omega: torch.Tensor,
    b0map_hz: torch.Tensor,
    echo_times_yz: torch.Tensor,
    r2star_map: torch.Tensor,
    t_ref_s: float,
    L: int = 32,
    nbins: int = 128,
) -> BlockDiagonal:
    """build_sense_b0 plus R2* decay (see SENSE_B0_R2star). r2star_map:
    (Nx,Ny,Nz), 1/s, same grid as b0map_hz. t_ref_s: the nominal-TE echo's
    time since excitation (scan_info.mat's schedules[0,0,(ETL-1)//2,2]); echo
    times are measured from it. Other arguments as in build_sense_b0."""
    b_by_echo, _c, tl, _t, pos_per_frame = _segment_fit(
        omega, b0map_hz, echo_times_yz, L, nbins, t_ref_s=t_ref_s,
    )
    b_by_echo = b_by_echo.to(smaps.dtype)
    c_phasors = SENSE_B0_R2star.segment_phasors(b0map_hz, r2star_map, tl, smaps.dtype)
    frames = [
        SENSE_B0_R2star(smaps, omega[..., it], pos, b_by_echo, c_phasors)
        for it, pos in enumerate(pos_per_frame)
    ]
    return BlockDiagonal(frames)
