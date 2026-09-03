"""Patch extraction/recombination and singular-value soft-thresholding (SVST)
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
"""

import torch

__all__ = ["img2patches", "patches2img", "patch_nucnorm", "SVST", "patchSVST"]


def _patch_starts(n: int, patch: int, stride: int) -> list[int]:
    nsteps = -(-(n - patch) // stride)  # ceil division
    return [min(i * stride, n - patch) for i in range(nsteps + 1)]


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


def SVST(X: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Singular Value Soft-Thresholding, the proximal operator of beta * nuclear-norm.

    X: (..., m, n), batched over any leading dims. Returns (X_thresholded, reg),
    reg = per-batch-element sum(max(sigma - beta, 0)), the nuclear norm of the
    thresholded result -- a free byproduct of the SVD already computed.

    Whenever ||X||_F <= beta, every singular value is <= beta too (sigma_max <=
    ||X||_F), so the result is exactly zero -- not an approximation. Forcing
    those entries to exact zero *before* the SVD (rather than just zeroing the
    output after) avoids feeding a near-zero-magnitude matrix through
    torch.linalg.svd: repeated soft-thresholding near this boundary can produce
    subnormal-magnitude patches, and the mslr-recon Julia port found cuSOLVER's
    GPU SVD returns all-NaN (not just imprecise) on those -- CPU LAPACK handles
    them fine, but the guard is applied on both backends here since it's cheap
    and exact either way.
    """
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


def _unit_block_svst(img: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """patch_size=[1,1,1]: SVST of each (1,Nt) voxel time series reduces to a
    vector soft-threshold (see recon.jl's derivation: SVD of a 1xNt row is
    U=[1], S=[||x||], Vh=x/||x||). Avoids ~Nvox individual 1x1 SVDs."""
    norms = torch.linalg.vector_norm(img, dim=-1, keepdim=True)
    scale = torch.clamp(1.0 - beta / norms, min=0.0)  # beta/0=inf -> -inf -> clamped to 0
    result = img * scale
    reg = torch.clamp(norms.squeeze(-1) - beta, min=0.0).sum()
    return result, reg


def patchSVST(
    img: torch.Tensor, beta: float, patch_size, stride_size
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply patch-wise SVST to a 4-D image (Nx,Ny,Nz,Nt) with threshold beta.
    Returns (img_thresholded, reg), reg = nuclear norm of the result summed
    over all patches (sum of thresholded singular values), free from the SVD."""
    Nx, Ny, Nz, _ = img.shape
    psx, psy, psz = (min(p, n) for p, n in zip(patch_size, (Nx, Ny, Nz)))
    if (psx, psy, psz) == (1, 1, 1):
        return _unit_block_svst(img, beta)

    P = img2patches(img, patch_size, stride_size)
    result, reg_per_patch = SVST(P, beta)
    img_out = patches2img(result, patch_size, stride_size, (Nx, Ny, Nz))
    return img_out, reg_per_patch.sum()
