"""MRI encoding operators: image -> sampled multi-coil k-space.

    SENSE            smaps -> FFT -> sample
    SENSE_B0         B0 phase accrual -> smaps -> FFT -> sample (time-segmented)
    SENSE_B0_R2star  B0 phase accrual + R2* magnitude decay -> smaps -> FFT -> sample

Each class is one frame's operator; the build_* functions stack one per frame
into a BlockDiagonal mapping (Nx,Ny,Nz,Nt) -> (K,Nc,Nt). Outputs hold only the
K sampled k-space locations, not a zero-filled grid. See recon/README.md.

Sign convention: the forward model multiplies the image by exp(+i 2 pi df(r) t).
mri_exp_approx fits exp(-i 2 pi b t), so it is passed -b0map_hz.
"""

import math
import warnings

import torch
from mirtorch.linear import BlockDiagonal
from mirtorch.linear.linearmaps import LinearMap
from mirtorch.linear.mri import mri_exp_approx


class SENSE(LinearMap):
    """smaps -> centered ortho FFT -> keep sampled points.

    smaps: (Nc,*N) complex64. samp: (*N,) bool with K True entries.
    Maps (*N,) -> (K,Nc); `idx` holds the K flat (C-order) sample indices."""

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
        # Scatter along dim 1 so the reshape below is a view, not a (Nc,*N) copy.
        kc_full = torch.zeros(self.Nc, math.prod(self.N), dtype=y.dtype, device=y.device)
        kc_full[:, self.idx] = y.T
        kc_full = kc_full.reshape(self.Nc, *self.N)
        kc_shifted = torch.fft.ifftshift(kc_full, dim=self.dims)
        xc = torch.fft.fftshift(
            torch.fft.ifftn(kc_shifted, dim=self.dims, norm="ortho"), dim=self.dims
        )
        return (xc * self.smaps.conj()).sum(dim=0)


def build_sense(smaps: torch.Tensor, omega: torch.Tensor) -> BlockDiagonal:
    """(Nx,Ny,Nz,Nt) -> (K,Nc,Nt). smaps: (Nc,Nx,Ny,Nz). omega: (Nx,Ny,Nz,Nt)
    bool with the same K per frame. `.A[it].idx` are frame it's sample indices."""
    Nt = omega.shape[-1]
    frames = [SENSE(smaps, omega[..., it]) for it in range(Nt)]
    return BlockDiagonal(frames)


class SENSE_B0(SENSE):
    """SENSE with B0 phase accrual, by time segmentation:

        y = sum_l b_by_echo[pos, l] * SENSE(c_phasors[l] * x)

    b_by_echo: (n_times, L) interpolation weights, one row per distinct echo
        time. pos: (K,) row of b_by_echo for each sampled location.
    c_phasors: (L,*N) per-segment phasors.
    Both tables come from mri_exp_approx (see build_sense_b0) and are shared by
    every frame's operator; segments are looped, not batched, so memory stays
    at the single-segment footprint.
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

    This is the physical (decaying) model, and SENSE_B0's adjoint conjugates
    the phasors, which is exact for any complex phasor. Don't flip the R2* sign
    (an adjoint-only trick): .apply() would then model growth.

    Only the phasors differ from SENSE_B0 (segment_phasors); they're computed
    once and shared by every frame's operator.
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
        # The segmentation is fit on df alone; that's valid while R2*'s decay-time
        # product stays much smaller than df's bandwidth-time product.
        t_span_s = float((tl.max() - tl.min()).item())
        bt_df = float(b0map_hz.max() - b0map_hz.min()) * t_span_s
        bt_r2 = float(r2_hz.max()) * t_span_s
        print(f"  SENSE_B0_R2star: R2* decay-time product = {bt_r2:.4f} vs "
              f"B0 bandwidth-time product = {bt_df:.2f} (should be much smaller)")
        psi = 1j * 2 * math.pi * b0map_hz.to(torch.complex64) - r2_hz.to(torch.complex64)
        tl_c = tl.to(torch.complex64).reshape((-1,) + (1,) * len(N))
        return torch.exp(tl_c * psi[None, ...]).to(dtype)


def _check_b_weight_row_sums(b: torch.Tensor, frame_idx: int | str, tol: float = 0.1) -> None:
    """Warn if the segmentation fit looks ill-conditioned. Each row of b should
    sum to ~1 (the fit at frequency 0); erratic sums, caused by too few nbins
    for the field map's range, show up as signal loss and noise."""
    row_sums = b.sum(dim=1).abs()
    lo, hi = row_sums.min().item(), row_sums.max().item()
    if lo < 1 - tol or hi > 1 + tol:
        warnings.warn(
            f"build_sense_b0: frame {frame_idx}'s b_weights row sums "
            f"range [{lo:.4f}, {hi:.4f}] (want close to 1.0) -- the segmentation fit "
            f"looks ill-conditioned (nbins too coarse for b0map_hz's dynamic range is "
            f"the known cause; see recon/README.md). Reconstructing with "
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
    """Fit the time segmentation once on frame 0's distinct echo times (every
    frame uses the same set) and map each frame's samples to rows of the fit.
    Returns (b_by_echo, c, tl, unique_t_ms, pos_per_frame)."""
    Nt = omega.shape[-1]
    Ny, Nz = omega.shape[1], omega.shape[2]
    n_yz = Ny * Nz
    echo_times_flat = echo_times_yz.reshape(n_yz, Nt)  # echo time doesn't depend on kx
    fit_times_flat = echo_times_flat if t_ref_s == 0.0 else echo_times_flat - t_ref_s
    b0_neg = (-b0map_hz).to(torch.float32)  # sign convention: see module docstring

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
    """Like build_sense, with B0 phase accrual (see SENSE_B0).

    b0map_hz: (Nx,Ny,Nz) field map in Hz on the EPI grid. echo_times_yz:
    (Ny,Nz,Nt) seconds since excitation of each sampled (ky,kz). L: number of
    time segments (32 keeps forward-model error under 1% at ETL=60). nbins:
    histogram bins for the fit (the mirtorch default of 20 is ill-conditioned on
    real field maps). The spectral norm is not 1; estimate it before solving.
    """
    N = tuple(smaps.shape[1:])
    b_by_echo, c, _tl, _t, pos_per_frame = _segment_fit(omega, b0map_hz, echo_times_yz, L, nbins)
    b_by_echo = b_by_echo.to(smaps.dtype)
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
    (Nx,Ny,Nz) in 1/s. t_ref_s: nominal-TE echo time (utils.nominal_te_s)."""
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
