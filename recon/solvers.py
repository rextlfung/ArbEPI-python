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
            xnew = g_prox(znew, zetanew)

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
