"""Proximal gradient method with momentum (PGM/FPGM/POGM) and gradient
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
"""

import math
from typing import Callable, Literal

import torch

Momentum = Literal["pgm", "fpgm", "pogm"]
Restart = Literal["none", "gr", "fr"]

_EPS = torch.finfo(torch.float32).eps


def _gr_restart(fgrad: torch.Tensor, ynew_yold: torch.Tensor, restart_cutoff: float) -> bool:
    inner = -torch.vdot(fgrad.reshape(-1), ynew_yold.reshape(-1)).real.item()
    return inner <= restart_cutoff * fgrad.norm().item() * ynew_yold.norm().item()


def pogm_restart(
    x0: torch.Tensor,
    fcost: Callable[[torch.Tensor], float],
    fgrad_fn: Callable[[torch.Tensor], torch.Tensor],
    f_L: float,
    *,
    f_mu: float = 0.0,
    mom: Momentum = "pogm",
    restart: Restart = "gr",
    restart_cutoff: float = 0.0,
    bsig: float = 1.0,
    niter: int = 10,
    g_prox: Callable[[torch.Tensor, float], torch.Tensor] = lambda z, c: z,
    fun: Callable = lambda it, xk, yk, is_restart, fcostnew, rel_change: None,
    conv_tol: float = 0.0,
    conv_min_iter: int = 10,
):
    """x, out = pogm_restart(x0, fcost, fgrad_fn, f_L; ...)

    `fun(iter, xk, yk, is_restart, fcostnew, rel_change)` is called once per
    iteration (iter=0 for the initial point, rel_change=nan there) and its
    return values are collected into `out`, mirroring the Julia signature
    exactly (see mirt_mod.jl point 7)."""
    if mom not in ("pgm", "fpgm", "pogm"):
        raise ValueError(f"mom={mom}")
    if restart not in ("none", "gr", "fr"):
        raise ValueError(f"restart={restart}")
    if f_L < 0:
        raise ValueError(f"f_L={f_L} < 0")
    if f_mu < 0:
        raise ValueError(f"f_mu={f_mu} < 0")
    if bsig < 0:
        raise ValueError(f"bsig={bsig} < 0")
    if abs(restart_cutoff) >= 1:
        raise ValueError(f"restart_cutoff={restart_cutoff}")

    L, mu = float(f_L), float(f_mu)
    q = mu / L if L != 0 else 0.0

    told = sig = zetaold = 1.0
    xold = yold = uold = zold = x0
    Fcostold = fcost(x0)

    out = [None] * (niter + 1)
    out[0] = fun(0, x0, x0, False, Fcostold, float("nan"))
    if niter == 0:
        return x0, out[:1]

    niter_actual = niter
    last_restart_iter = 0
    adaptive_cmi = conv_min_iter

    xnew = ynew = None
    Fgrad = torch.zeros_like(x0)
    Fgradold = torch.zeros_like(x0) if mom == "pogm" else x0

    for it in range(1, niter + 1):
        alpha = (2.0 / (L + mu)) if (mom == "pgm" and mu != 0) else (1.0 / L)

        fgrad = fgrad_fn(xold)
        is_restart = False

        if mom in ("pgm", "fpgm"):
            ynew = g_prox(xold - alpha * fgrad, alpha)
            Fgrad = -(1.0 / alpha) * (ynew - xold)
            Fcostnew = fcost(ynew)

            if restart != "none":
                if (restart == "fr" and Fcostnew > Fcostold) or (
                    restart == "gr" and _gr_restart(Fgrad, ynew - yold, restart_cutoff)
                ):
                    told = 1.0
                    is_restart = True
                Fcostold = Fcostnew
        else:  # pogm
            unew = xold - alpha * fgrad

        beta = None
        tnew = told
        if mom == "fpgm" and mu != 0:
            beta = (1 - math.sqrt(q)) / (1 + math.sqrt(q))
        elif mom == "pogm" and mu != 0:
            beta = (2 + q - math.sqrt(q**2 + 8 * q)) ** 2 / (4 * (1 - q))
        elif mom != "pgm":
            tnew = (
                0.5 * (1 + math.sqrt(1 + 8 * told**2))
                if (mom == "pogm" and it == niter)
                else 0.5 * (1 + math.sqrt(1 + 4 * told**2))
            )
            beta = (told - 1) / tnew

        if mom == "pgm":
            xnew = ynew
        elif mom == "fpgm":
            xnew = ynew + beta * (ynew - yold)
        else:  # pogm
            gamma = (2 + q - math.sqrt(q**2 + 8 * q)) / 2 if mu != 0 else sig * told / tnew
            ba_z = beta * alpha / zetaold
            znew = unew + beta * (unew - uold) + gamma * (unew - xold) - ba_z * (xold - zold)
            zetanew = alpha * (1 + beta + gamma)
            # g_prox is allowed to mutate its argument in place (e.g.
            # recon/mslr.py's g_prox writes into X and returns it,
            # for a small transient memory win) -- clone znew first so
            # `xnew is znew` never holds. Without this, xnew - znew below
            # is always exactly zero regardless of what g_prox computed
            # (the two names alias the same mutated tensor), and zold at
            # :156 also ends up aliasing xold next iteration, silently
            # zeroing POGM's momentum correction terms every iteration.
            xnew = g_prox(znew.clone(), zetanew)

            iz = 1.0 / zetanew
            Fgrad = fgrad - iz * (xnew - znew)
            fgrad = None

            ynew = xold - alpha * Fgrad
            Fcostnew = fcost(xnew)

            if restart != "none":
                ynew_yold = ynew - yold
                if (restart == "fr" and Fcostnew > Fcostold) or (
                    restart == "gr" and _gr_restart(Fgrad, ynew_yold, restart_cutoff)
                ):
                    tnew = 1.0
                    sig = 1.0
                    is_restart = True
                elif torch.vdot(Fgrad.reshape(-1), Fgradold.reshape(-1)).real.item() < 0:
                    sig = bsig * sig
                Fcostold = Fcostnew
                Fgradold, Fgrad = Fgrad, Fgradold

            uold, zold, zetaold = unew, znew, zetanew

        recon_prev = xold if mom == "pogm" else yold
        recon_curr = xnew if mom == "pogm" else ynew
        rel_change = (recon_curr - recon_prev).norm().item() / (recon_prev.norm().item() + _EPS)

        if is_restart:
            last_restart_iter = it
            adaptive_cmi = max(1, adaptive_cmi - 1)

        if conv_tol > 0 and it >= last_restart_iter + adaptive_cmi and rel_change < conv_tol:
            out[it] = fun(it, xnew, ynew, is_restart, Fcostnew, rel_change)
            niter_actual = it
            break

        out[it] = fun(it, xnew, ynew, is_restart, Fcostnew, rel_change)
        xold, yold = xnew, ynew
        if mom != "pgm" and mu == 0:
            told = tnew

    result = xnew if mom == "pogm" else ynew
    return result, out[: niter_actual + 1]


def poweriter(
    apply_fwd: Callable[[torch.Tensor], torch.Tensor],
    apply_adj: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    niter: int = 200,
    tol: float = 1e-6,
) -> float:
    """Estimate the spectral norm sigma1 = ||A||_2 via power iteration on the
    normal operator A'A, given as its forward/adjoint applies."""
    x = x0.clone()
    ratio_old = float("inf")
    for _ in range(niter):
        Ax = apply_fwd(x)
        ratio = Ax.norm().item() / x.norm().item()
        if abs(ratio - ratio_old) / ratio < tol:
            return ratio
        ratio_old = ratio
        x = apply_adj(Ax)
        x = x / x.norm()
    return apply_fwd(x).norm().item() / x.norm().item()

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
