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
"""

import math

import torch
from mirtorch.linear import BlockDiagonal
from mirtorch.linear.linearmaps import LinearMap


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
