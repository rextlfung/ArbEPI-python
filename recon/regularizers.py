"""Regularizers g(x) and their proximal operators.

    MultiScaleLowRank  nuclear norm of space x time patches at one or more patch
                       scales (Ong & Lustig 2016). Has a closed-form prox: POGM.
    WaveletTV          L1-wavelet + total variation. No closed-form prox: PDHG.
    SpatioTemporalWaveletTV  WaveletTV on every frame jointly, plus optional
                       temporal TV (TemporalTVProx, its own prox). PDHG.
    TemporalHighPass   projector onto temporal frequencies above a cutoff, for the
                       smooth penalty (mu/2)||P x||^2 (any solver).
"""

import math

import pywt
import torch
import torch.nn.functional as F
from mirtorch.linear import Diffnd, Vstack
from mirtorch.linear.linearmaps import LinearMap
from mirtorch.prox import Prox


def _reg_weights(
    patch_sizes: list[tuple[int, int, int]], Nt: int, N_voxels: int, lambda_global: float
) -> list[float]:
    """Ong & Lustig eq. 4 times lambda_global: lambda_k = sqrt(p_k) + sqrt(Nt)
    + sqrt(log(N_voxels*Nt / max(p_k, Nt))), p_k = voxels per patch."""
    lambdas = []
    for ps in patch_sizes:
        p_k = math.prod(ps)
        lam = math.sqrt(p_k) + math.sqrt(Nt) + math.sqrt(math.log(N_voxels * Nt / max(p_k, Nt)))
        lambdas.append(lam * lambda_global)
    return lambdas


def _patch_starts(n: int, patch: int, stride: int) -> list[int]:
    nsteps = -(-(n - patch) // stride)  # ceil division
    return [min(i * stride, n - patch) for i in range(nsteps + 1)]


_DEFAULT_SVD_CHUNK_BYTES = 4_000_000_000  # per batched SVD call; see SVST


def _all_patch_starts(
    shape: tuple[int, int, int], patch_size: tuple[int, int, int], stride_size: tuple[int, int, int]
) -> tuple[list[tuple[int, int, int]], tuple[int, int, int]]:
    """All (sx,sy,sz) patch start positions over `shape`, and the patch size
    actually used (capped at the image size along each axis)."""
    Nx, Ny, Nz = shape
    psx, psy, psz = (min(p, n) for p, n in zip(patch_size, shape))
    starts_x = _patch_starts(Nx, psx, stride_size[0])
    starts_y = _patch_starts(Ny, psy, stride_size[1])
    starts_z = _patch_starts(Nz, psz, stride_size[2])
    starts = [(sx, sy, sz) for sz in starts_z for sy in starts_y for sx in starts_x]
    return starts, (psx, psy, psz)


def _chunk_size_for_budget(
    patch_size: tuple[int, int, int], Nt: int, element_size: int, max_chunk_bytes: int
) -> int:
    p_k = patch_size[0] * patch_size[1] * patch_size[2]
    return max(1, max_chunk_bytes // (element_size * p_k * Nt))


def _gather_patch_chunk(
    img: torch.Tensor, starts_chunk: list[tuple[int, int, int]], patch_size: tuple[int, int, int]
) -> torch.Tensor:
    """(len(starts_chunk), prod(patch_size), Nt) patches at the given starts."""
    psx, psy, psz = patch_size
    Nt = img.shape[-1]
    return torch.stack(
        [
            img[sx : sx + psx, sy : sy + psy, sz : sz + psz, :].reshape(psx * psy * psz, Nt)
            for sx, sy, sz in starts_chunk
        ],
        dim=0,
    )


def img2patches(img: torch.Tensor, patch_size, stride_size) -> torch.Tensor:
    """(Nx,Ny,Nz,Nt) -> (Np, prod(patch_size), Nt), one row per (space x time) patch."""
    Nx, Ny, Nz, Nt = img.shape
    if any(s <= 0 for s in stride_size):
        raise ValueError(f"stride_size elements must be positive, got {stride_size}")
    psx, psy, psz = (min(p, n) for p, n in zip(patch_size, (Nx, Ny, Nz)))

    starts_x = _patch_starts(Nx, psx, stride_size[0])
    starts_y = _patch_starts(Ny, psy, stride_size[1])
    starts_z = _patch_starts(Nz, psz, stride_size[2])

    patches = [
        img[sx : sx + psx, sy : sy + psy, sz : sz + psz, :].reshape(psx * psy * psz, Nt)
        for sz in starts_z
        for sy in starts_y
        for sx in starts_x
    ]
    return torch.stack(patches, dim=0)


def patches2img(P: torch.Tensor, patch_size, stride_size, og_size) -> torch.Tensor:
    """Inverse of img2patches: recombine via overlap-averaging."""
    _, _, Nt = P.shape
    Nx, Ny, Nz = og_size
    psx, psy, psz = (min(p, n) for p, n in zip(patch_size, og_size))

    starts_x = _patch_starts(Nx, psx, stride_size[0])
    starts_y = _patch_starts(Ny, psy, stride_size[1])
    starts_z = _patch_starts(Nz, psz, stride_size[2])

    img = torch.zeros(Nx, Ny, Nz, Nt, dtype=P.dtype, device=P.device)
    pcount = torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device=P.device)

    ip = 0
    for sz in starts_z:
        for sy in starts_y:
            for sx in starts_x:
                patch = P[ip].reshape(psx, psy, psz, Nt)
                img[sx : sx + psx, sy : sy + psy, sz : sz + psz, :] += patch
                pcount[sx : sx + psx, sy : sy + psy, sz : sz + psz] += 1.0
                ip += 1

    pcount.clamp_(min=1.0)
    return img / pcount.unsqueeze(-1)


def patch_nucnorm(P: torch.Tensor) -> torch.Tensor:
    """Sum of nuclear norms across all (space x time) patches in P: (Np, m, n)."""
    if P.ndim != 3:
        raise ValueError("P must be (patches, space, time)")
    return torch.linalg.svdvals(P).sum()


def _svst_batch(X: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """SVST of one batch of patches (see SVST)."""
    fro = torch.linalg.matrix_norm(X, ord="fro")
    zero_mask = fro <= beta
    mask_mnn = zero_mask[..., None, None]
    X_safe = torch.where(mask_mnn, torch.zeros_like(X), X)

    U, S, Vh = torch.linalg.svd(X_safe, full_matrices=False)
    s_thresh = torch.clamp(S - beta, min=0.0)
    recon = U @ (s_thresh.to(Vh.dtype).unsqueeze(-1) * Vh)
    reg = s_thresh.sum(dim=-1)

    recon = torch.where(mask_mnn, torch.zeros_like(recon), recon)
    reg = torch.where(zero_mask, torch.zeros_like(reg), reg)
    return recon, reg


def SVST(
    X: torch.Tensor, beta: float, max_chunk_bytes: int = _DEFAULT_SVD_CHUNK_BYTES
) -> tuple[torch.Tensor, torch.Tensor]:
    """Singular Value Soft-Thresholding, the proximal operator of beta * nuclear-norm.

    X: (Np, m, n), batched over patches. Returns (X_thresholded, reg),
    reg = per-batch-element sum(max(sigma - beta, 0)), the nuclear norm of the
    thresholded result -- a free byproduct of the SVD already computed.

    Patches with ||X||_F <= beta are set to exactly zero before the SVD (their
    result is zero anyway): cuSOLVER returns NaN on subnormal-magnitude input.

    max_chunk_bytes: patches are processed in batches of at most this many
    bytes, which bounds memory with no measurable slowdown.
    """
    item_bytes = X.element_size() * X.shape[-2] * X.shape[-1]
    chunk_size = max(1, max_chunk_bytes // item_bytes)
    if X.shape[0] <= chunk_size:
        return _svst_batch(X, beta)

    recons, regs = [], []
    for i in range(0, X.shape[0], chunk_size):
        recon_i, reg_i = _svst_batch(X[i : i + chunk_size], beta)
        recons.append(recon_i)
        regs.append(reg_i)
    return torch.cat(recons, dim=0), torch.cat(regs, dim=0)


def _unit_block_svst(img: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """1x1x1 patches: SVST of a 1 x Nt row is a soft-threshold of its norm."""
    norms = torch.linalg.vector_norm(img, dim=-1, keepdim=True)
    scale = torch.clamp(1.0 - beta / norms, min=0.0)  # beta/0=inf -> -inf -> clamped to 0
    result = img * scale
    reg = torch.clamp(norms.squeeze(-1) - beta, min=0.0).sum()
    return result, reg


def patchSVST(
    img: torch.Tensor, beta: float, patch_size, stride_size,
    max_chunk_bytes: int = _DEFAULT_SVD_CHUNK_BYTES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply patch-wise SVST to a 4-D image (Nx,Ny,Nz,Nt) with threshold beta.
    Returns (img_thresholded, reg), reg = nuclear norm of the result summed
    over all patches (sum of thresholded singular values), free from the SVD.

    Same result as img2patches -> SVST -> patches2img (overlaps averaged), but
    processes one memory-budgeted chunk of patches at a time instead of
    building the full patch tensor."""
    Nx, Ny, Nz, Nt = img.shape
    starts, (psx, psy, psz) = _all_patch_starts((Nx, Ny, Nz), patch_size, stride_size)
    if (psx, psy, psz) == (1, 1, 1):
        return _unit_block_svst(img, beta)

    chunk_size = _chunk_size_for_budget((psx, psy, psz), Nt, img.element_size(), max_chunk_bytes)
    img_out = torch.zeros_like(img)
    pcount = torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device=img.device)
    reg_total = torch.zeros((), dtype=torch.float32, device=img.device)

    for i in range(0, len(starts), chunk_size):
        chunk = starts[i : i + chunk_size]
        P_chunk = _gather_patch_chunk(img, chunk, (psx, psy, psz))
        result_chunk, reg_chunk = _svst_batch(P_chunk, beta)
        reg_total = reg_total + reg_chunk.sum()
        for j, (sx, sy, sz) in enumerate(chunk):
            patch = result_chunk[j].reshape(psx, psy, psz, Nt)
            img_out[sx : sx + psx, sy : sy + psy, sz : sz + psz, :] += patch
            pcount[sx : sx + psx, sy : sy + psy, sz : sz + psz] += 1.0

    pcount.clamp_(min=1.0)
    return img_out / pcount.unsqueeze(-1), reg_total


class SumScales(LinearMap):
    """(Nx,Ny,Nz,Nt,Nscales) -> (Nx,Ny,Nz,Nt): the image is the sum of its
    per-scale components, X_recon = X[...,0] + ... + X[...,Nscales-1]. Data
    consistency applies to this sum, so the MultiScaleLowRank data term is
    f(X) = 0.5 * ||A(SumScales(X)) - y||^2."""

    def __init__(self, img_shape: tuple[int, ...], Nscales: int):
        self.Nscales = Nscales
        super().__init__(tuple(img_shape) + (Nscales,), tuple(img_shape))

    def _apply(self, X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=-1)

    def _apply_adjoint(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1).expand(*x.shape, self.Nscales).clone()


class MultiScaleLowRank:
    """g(X) = sum_k lambda_k sum_patches ||patch_k(X[...,k])||_* (Ong & Lustig).

    X: (Nx,Ny,Nz,Nt,Nscales), one component per patch scale; the image is their
    sum (self.synthesis). One local scale is locally low rank; a whole-volume
    patch is globally low rank. lambda_k from _reg_weights.
    """

    def __init__(
        self,
        patch_sizes: list[tuple[int, int, int]],
        strides: list[tuple[int, int, int]],
        img_shape: tuple[int, int, int, int],
        lambda_global: float = 1.0,
    ):
        assert len(patch_sizes) == len(strides), "one stride per patch size"
        Nx, Ny, Nz, Nt = img_shape
        self.patch_sizes = [tuple(p) for p in patch_sizes]
        self.strides = [tuple(s) for s in strides]
        self.Nscales = len(patch_sizes)
        self.lambdas = _reg_weights(self.patch_sizes, Nt, Nx * Ny * Nz, lambda_global)
        self.synthesis = SumScales(img_shape, self.Nscales)
        self.last_cost = 0.0  # g(X) of the most recent prox output, free from its SVDs

    def cost(self, X: torch.Tensor) -> float:
        total = 0.0
        for k in range(self.Nscales):
            img_k = X[..., k]
            starts, ps = _all_patch_starts(img_k.shape[:3], self.patch_sizes[k], self.strides[k])
            chunk_size = _chunk_size_for_budget(
                ps, img_k.shape[-1], img_k.element_size(), _DEFAULT_SVD_CHUNK_BYTES
            )
            nuc = 0.0
            for i in range(0, len(starts), chunk_size):
                P_chunk = _gather_patch_chunk(img_k, starts[i : i + chunk_size], ps)
                nuc += patch_nucnorm(P_chunk).item()
            total += self.lambdas[k] * nuc
        return total

    def prox(self, X: torch.Tensor, c: float) -> torch.Tensor:
        """prox_{c*g}(X), computed scale by scale (g is separable across
        scales). Writes into X and returns it; also records g(result) in
        self.last_cost."""
        reg = 0.0
        for k in range(self.Nscales):
            result, cost = patchSVST(
                X[..., k], c * self.lambdas[k], self.patch_sizes[k], self.strides[k]
            )
            X[..., k] = result
            reg += self.lambdas[k] * cost.item()
        self.last_cost = reg
        return X


# ---------------------------------------------------------------- wavelet + TV


def _analysis_1d(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """One periodized orthogonal DWT level along the last axis of a real
    tensor: (..., n) -> (..., n) laid out as [approx (n/2) | detail (n/2)]."""
    n = x.shape[-1]
    L = lo.numel()
    batch = x.shape[:-1]
    xp = x.reshape(-1, 1, n)[..., torch.arange(n + L - 1, device=x.device) % n]  # circular pad
    w = torch.stack([lo, hi])[:, None, :]  # (2,1,L)
    y = F.conv1d(xp, w, stride=2)  # (B,2,n/2)
    return y.reshape(*batch, n)


def _synthesis_1d(y: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Exact adjoint (= inverse, since the transform is orthogonal) of _analysis_1d."""
    n = y.shape[-1]
    batch = y.shape[:-1]
    w = torch.stack([lo, hi])[:, None, :]  # (2,1,L)
    full = F.conv_transpose1d(y.reshape(-1, 2, n // 2), w, stride=2)  # (B,1,n+L-2)
    out = torch.zeros(full.shape[0], n, dtype=y.dtype, device=y.device)
    wrap = torch.arange(full.shape[-1], device=y.device) % n
    out.index_add_(1, wrap, full[:, 0])  # fold the circular wrap back in
    return out.reshape(*batch, n)


class Wavelet3D(LinearMap):
    """Multilevel separable orthogonal 3D DWT with periodized boundaries
    (torch, runs on GPU). mirtorch 0.3.1 only ships a 2D wavelet.

    Each axis is zero-padded up to a multiple of 2**levels (e.g. Nz=45 -> 48),
    so W = DWT o ZeroPad and W^H = Crop o IDWT. W^H W = I (W is an isometry),
    though not unitary when padding is needed. Output is the flattened
    Mallat-layout coefficient array. Complex input is transformed as separate
    real and imaginary parts."""

    def __init__(self, img_shape: tuple[int, int, int], wave: str = "db4", levels: int = 3):
        self.img_shape = tuple(img_shape)
        self.levels = levels
        m = 2 ** levels
        self.pad_shape = tuple(-(-n // m) * m for n in self.img_shape)
        w = pywt.Wavelet(wave)
        assert w.orthogonal, f"{wave} is not an orthogonal wavelet"
        self._lo = torch.tensor(w.dec_lo, dtype=torch.float32)
        self._hi = torch.tensor(w.dec_hi, dtype=torch.float32)
        super().__init__(self.img_shape, (math.prod(self.pad_shape),))

    def _filters(self, device):
        if self._lo.device != device:
            self._lo, self._hi = self._lo.to(device), self._hi.to(device)
        return self._lo, self._hi

    def _real(self, x: torch.Tensor, inverse: bool) -> torch.Tensor:
        lo, hi = self._filters(x.device)
        sizes = [tuple(n >> lev for n in self.pad_shape) for lev in range(self.levels)]
        order = list(range(self.levels))
        if inverse:
            order = order[::-1]
        x = x.clone()
        for lev in order:
            sx, sy, sz = sizes[lev]
            block = x[:sx, :sy, :sz]
            axes = [0, 1, 2] if not inverse else [2, 1, 0]
            for ax in axes:
                block = block.movedim(ax, -1)
                block = _synthesis_1d(block, lo, hi) if inverse else _analysis_1d(block, lo, hi)
                block = block.movedim(-1, ax)
            x[:sx, :sy, :sz] = block
        return x

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        pad = []
        for n, npad in zip(reversed(self.img_shape), reversed(self.pad_shape)):
            pad += [0, npad - n]
        parts = [x.real, x.imag] if x.is_complex() else [x]
        out = [self._real(F.pad(p.float(), pad), inverse=False) for p in parts]
        y = torch.complex(out[0], out[1]) if x.is_complex() else out[0]
        return y.reshape(-1)

    def _apply_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        y = y.reshape(self.pad_shape)
        parts = [y.real, y.imag] if y.is_complex() else [y]
        out = [self._real(p.float(), inverse=True) for p in parts]
        x = torch.complex(out[0], out[1]) if y.is_complex() else out[0]
        nx, ny, nz = self.img_shape
        return x[:nx, :ny, :nz]


class _Flatten(LinearMap):
    """op followed by a flatten to 1-D, so operators with differently shaped
    outputs can be stacked with Vstack(dim=0)."""

    def __init__(self, op: LinearMap):
        self.op = op
        super().__init__(tuple(op.size_in), (math.prod(op.size_out),))

    def _apply(self, x):
        return self.op.apply(x).reshape(-1)

    def _apply_adjoint(self, y):
        return self.op.adjoint(y.reshape(tuple(self.op.size_out)))


class SectionL1(Prox):
    """prox of sum_i lambdas[i] * ||v_i||_1, where v = [v_0; v_1; ...] is split
    into consecutive sections of the given sizes (mirtorch's prox.Stack only
    splits into equal sections)."""

    def __init__(self, lambdas: list[float], sizes: list[int]):
        super().__init__()
        self.lambdas = list(lambdas)
        self.sizes = list(sizes)

    def _apply(self, v: torch.Tensor, alpha) -> torch.Tensor:
        out = []
        for lam, section in zip(self.lambdas, torch.split(v, self.sizes)):
            thresh = float(alpha) * lam
            out.append(torch.sign(section) * torch.clamp(section.abs() - thresh, min=0))
        return torch.cat(out)


class WaveletTV:
    """g(x) = lamb_l1 * ||W x||_1 + lamb_tv * ||D x||_1 on one 3D frame, with W
    an orthogonal 3D wavelet and D the periodic finite difference along x, y, z
    (anisotropic TV). Written as h(G x) with G = [W; D], for solvers.pdhg's
    primal-dual solver (TV has no closed-form prox, so POGM doesn't apply)."""

    def __init__(
        self, img_shape: tuple[int, int, int], lamb_l1: float, lamb_tv: float,
        wave: str = "db4", levels: int = 3,
    ):
        W = Wavelet3D(img_shape, wave=wave, levels=levels)
        D = _Flatten(Diffnd(list(img_shape), dims=[0, 1, 2]))
        self.G = Vstack([W, D], dim=0)
        self.h_prox = SectionL1([lamb_l1, lamb_tv], [W.size_out[0], D.size_out[0]])
        # ||W||^2 = 1, ||D||^2 <= 4 per axis; inflated 5% as a safe upper bound.
        self.G_norm_squared = 1.05 * (1 + 4 * len(img_shape))

    def cost(self, x: torch.Tensor) -> float:
        Gx = self.G.apply(x)
        sw, sd = self.h_prox.sizes
        lw, ld = self.h_prox.lambdas
        return lw * Gx[:sw].abs().sum().item() + ld * Gx[sw:].abs().sum().item()


# ---------------------------------------------------------------- temporal regularization


class PerFrame(LinearMap):
    """A 3D LinearMap with 1-D (flattened) output applied to every frame of a
    (Nx,Ny,Nz,Nt) image; the frames' outputs are concatenated, frame-major."""

    def __init__(self, op: LinearMap, Nt: int):
        self.op = op
        self.Nt = Nt
        self.n_out = math.prod(op.size_out)
        super().__init__(tuple(op.size_in) + (Nt,), (self.n_out * Nt,))

    def _apply(self, x):
        return torch.cat([self.op.apply(x[..., t]).reshape(-1) for t in range(self.Nt)])

    def _apply_adjoint(self, y):
        y = y.reshape(self.Nt, self.n_out)
        return torch.stack(
            [self.op.adjoint(y[t].reshape(tuple(self.op.size_out))) for t in range(self.Nt)], -1
        )


def _tdiff(x: torch.Tensor) -> torch.Tensor:
    """Forward difference along the last (time) axis, non-periodic: (..., Nt) ->
    (..., Nt-1). Not periodic, so the first and last frames aren't tied together
    (a slow drift would otherwise pay for its full first-to-last change)."""
    return x[..., 1:] - x[..., :-1]


def _tdiff_adjoint(u: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(u[..., :1])
    return torch.cat([zero, u], -1) - torch.cat([u, zero], -1)


class TemporalTVProx(Prox):
    """prox of alpha lam ||D_t x||_1 (complex modulus of each frame-to-frame
    difference, summed): 1D TV denoising of every voxel's time series, by FISTA
    on its dual, min_{|u| <= alpha lam} 0.5 ||v - D_t^H u||^2 (x = v - D_t^H u,
    step 1/||D_t||^2 = 1/4). The dual is kept between calls and warm-starts the
    next one, so inside an outer solver a few inner steps (n_inner) suffice.

    Used as FBPD's proximable f rather than as a dual block of G: as a dual
    block, its dual variable starts at 0 and grows by sigma ||D_t x|| per outer
    iteration, which for a near-static image is tiny next to lam, so in 100
    iterations the clipping at lam never engaged and every lam above ~0.04 gave
    the same image (review item 263).

    Overrides __call__: mirtorch's Prox applies _apply to |v| and restores each
    element's phase, which is exact only for elementwise proxes; this one
    couples frames, so it works on the complex values directly."""

    def __init__(self, lam: float, n_inner: int = 10):
        super().__init__()
        self.lam, self.n_inner = float(lam), int(n_inner)
        self._u = None

    def __call__(self, v: torch.Tensor, alpha) -> torch.Tensor:
        return self._apply(v, alpha)

    def _apply(self, v: torch.Tensor, alpha) -> torch.Tensor:
        bound = float(alpha) * self.lam
        u = self._u
        if u is None or u.shape != v[..., 1:].shape or u.device != v.device:
            u = torch.zeros_like(v[..., 1:])
        w, t = u.clone(), 1.0
        for _ in range(self.n_inner):
            u_new = w + 0.25 * _tdiff(v - _tdiff_adjoint(w))
            u_new = u_new * torch.clamp(bound / u_new.abs().clamp_min(1e-30), max=1.0)
            t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
            w = u_new + ((t - 1) / t_new) * (u_new - u)
            u, t = u_new, t_new
        self._u = u
        return v - _tdiff_adjoint(u)

    def cost(self, x: torch.Tensor) -> float:
        return self.lam * _tdiff(x).abs().sum().item()


def dct_matrix(n: int, device=None) -> torch.Tensor:
    """Orthonormal DCT-II matrix C (n x n): C @ x is the DCT of x, C.T inverts it.
    Basis k is cos(pi k (t + 1/2) / n), frequency k / (2 n TR)."""
    t = torch.arange(n, dtype=torch.float64)
    k = t[:, None]
    C = torch.cos(math.pi * k * (t[None, :] + 0.5) / n) * math.sqrt(2.0 / n)
    C[0] /= math.sqrt(2.0)
    return C.to(device=device, dtype=torch.float32)


class TemporalHighPass:
    """Orthogonal projector P onto temporal frequencies above `cutoff_hz`, along
    the last axis, in the DCT-II basis: P = C^T diag(k / (2 Nt TR) > cutoff) C.
    The DCT, not the DFT, because it extends the series by reflection: a slow
    drift has no jump at the window edge, so it stays out of the penalized
    band (with the DFT, the first-to-last step leaks into every frequency).

    Used as the smooth penalty (mu/2) ||P x||^2, which is exactly zero on the
    kept band (<= cutoff_hz), so it can't bias a signal confined to it."""

    def __init__(self, Nt: int, tr_s: float, cutoff_hz: float, device=None):
        self.Nt, self.tr_s, self.cutoff_hz = Nt, tr_s, cutoff_hz
        freqs = torch.arange(Nt, dtype=torch.float64) / (2 * Nt * tr_s)
        self.keep = int((freqs <= cutoff_hz).sum())  # basis functions left unpenalized
        C = dct_matrix(Nt, device)
        self.P = (C[self.keep :].T @ C[self.keep :]).contiguous()  # (Nt, Nt), symmetric

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        P = self.P.to(x.device)
        if x.is_complex():
            return torch.complex(x.real @ P, x.imag @ P)  # P symmetric: x P = (P x^T)^T
        return x @ P

    def cost(self, x: torch.Tensor, mu: float) -> float:
        return 0.5 * mu * self.apply(x).abs().pow(2).sum().item()


class _Projector(LinearMap):
    """TemporalHighPass's P as a LinearMap with flattened output (P is
    symmetric, so it is its own adjoint)."""

    def __init__(self, hp: "TemporalHighPass", img_shape: tuple[int, ...]):
        self.hp = hp
        super().__init__(tuple(img_shape), (math.prod(img_shape),))

    def _apply(self, x):
        return self.hp.apply(x).reshape(-1)

    def _apply_adjoint(self, y):
        return self.hp.apply(y.reshape(tuple(self.size_in)))


class SectionProx(Prox):
    """prox of sum_i h_i(v_i) over consecutive sections v_i (sizes), each h_i
    either lam ||v_i||_1 ('l1') or (lam/2) ||v_i||^2 ('sq', whose prox is
    v / (1 + alpha lam))."""

    def __init__(self, kinds: list[str], lambdas: list[float], sizes: list[int]):
        super().__init__()
        self.kinds, self.lambdas, self.sizes = list(kinds), list(lambdas), list(sizes)

    def _apply(self, v: torch.Tensor, alpha) -> torch.Tensor:
        out = []
        for kind, lam, sec in zip(self.kinds, self.lambdas, torch.split(v, self.sizes)):
            if kind == "sq":
                out.append(sec / (1 + float(alpha) * lam))
            else:
                thresh = float(alpha) * lam
                out.append(torch.sign(sec) * torch.clamp(sec.abs() - thresh, min=0))
        return torch.cat(out)


class SpatioTemporalWaveletTV:
    """g(x) = lamb_l1 ||W x_t||_1 + lamb_tv ||D x_t||_1 summed over frames, plus
    lamb_ttv ||D_t x||_1 and (hp_weight/2) ||P x||^2: WaveletTV on every frame of
    a (Nx,Ny,Nz,Nt) image, jointly, with optional temporal TV and high-pass
    penalty (TemporalHighPass). G = [W; D; P] with h_prox for the dual blocks;
    temporal TV is f_prox (TemporalTVProx, None without it), FBPD's proximable
    term -- see TemporalTVProx for why not a block of G (review item 263).

    The high-pass penalty sits here, as a dual block with a closed-form prox,
    rather than in PDHG's smooth term: there its curvature hp_weight would cut
    the primal step to 1 / (1 + hp_weight), and at hp_weight = 30 100 iterations
    left the data term 20x above its converged value (review item 262)."""

    def __init__(
        self, img_shape: tuple[int, int, int, int], lamb_l1: float, lamb_tv: float,
        lamb_ttv: float = 0.0, wave: str = "db4", levels: int = 3,
        hp: "TemporalHighPass | None" = None, hp_weight: float = 0.0,
    ):
        sp, Nt = tuple(img_shape[:3]), img_shape[3]
        W = PerFrame(Wavelet3D(sp, wave=wave, levels=levels), Nt)
        D = PerFrame(_Flatten(Diffnd(list(sp), dims=[0, 1, 2])), Nt)
        ops, lams, kinds = [W, D], [lamb_l1, lamb_tv], ["l1", "l1"]
        norm2 = 1 + 4 * len(sp)
        self.f_prox = TemporalTVProx(lamb_ttv) if lamb_ttv > 0 else None
        if hp is not None and hp_weight > 0:
            ops.append(_Projector(hp, tuple(img_shape)))
            lams.append(hp_weight)
            kinds.append("sq")
            norm2 += 1  # P is an orthogonal projector
        self.G = Vstack(ops, dim=0)
        self.h_prox = SectionProx(kinds, lams, [op.size_out[0] for op in ops])
        self.G_norm_squared = 1.05 * norm2

    def cost(self, x: torch.Tensor) -> float:
        Gx = self.G.apply(x)
        p = self.h_prox
        c = sum(
            (0.5 * lam * sec.abs().pow(2).sum().item()) if kind == "sq"
            else lam * sec.abs().sum().item()
            for kind, lam, sec in zip(p.kinds, p.lambdas, torch.split(Gx, p.sizes))
        )
        return c + (self.f_prox.cost(x) if self.f_prox is not None else 0.0)
