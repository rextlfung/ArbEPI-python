"""Iterative solvers.

    pogm_restart  PGM / FPGM / POGM for min_x f(x) + g(x), f smooth, g prox-friendly
    pdhg          primal-dual for min_x f(x) + h(G x), f smooth, h prox-friendly
                  (for g = h o G with no closed-form prox, e.g. TV)
    cg            conjugate gradient on the normal equations A^H A x = A^H y
"""

import math
from typing import Callable, Literal

import torch
from mirtorch.alg import FBPD
from mirtorch.prox import Const

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
    """Minimize f(x) + g(x) by PGM, FPGM or POGM (Kim & Fessler) with restart.

    fcost, fgrad_fn: f and its gradient; f_L: Lipschitz constant of the
    gradient (step size 1/f_L). g_prox(z, c): prox of c*g. restart: "gr"
    (gradient), "fr" (function value) or "none". Stops early once the relative
    change of the iterate falls below conv_tol (0 disables this).
    fun(iter, xk, yk, is_restart, fcost, rel_change) is called every iteration,
    iter 0 being the start. Returns (x, list of fun's return values)."""
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
            # g_prox may write into its argument (MultiScaleLowRank.prox does), so
            # pass a copy: xnew must not alias znew.
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


def cg(
    A, ksp: torch.Tensor, shape: tuple[int, int, int, int], num_iter: int = 20,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, list[float]]:
    """Conjugate gradient on A^H A x = A^H ksp, starting from 0.

    A: an operator from recon/operators.py. ksp: (K,Nc,Nt). shape: (Nx,Ny,Nz,Nt).
    Returns (X, residuals), residuals[i] = ||r_i|| / ||r_0|| (residuals[0] = 1);
    stops early once below tol.
    """
    B = A.adjoint(ksp)  # (Nx,Ny,Nz,Nt)
    X = torch.zeros_like(B)
    R = B.clone()
    P = R.clone()
    rsold = torch.sum(torch.abs(R) ** 2).item()
    rs0 = rsold
    residuals = [1.0]

    if rs0 == 0:
        return X, residuals

    for _ in range(num_iter):
        AP = A.apply(P)
        AHAP = A.adjoint(AP)
        alpha = rsold / torch.sum(torch.real(torch.conj(P) * AHAP)).item()
        X = X + alpha * P
        R = R - alpha * AHAP
        rsnew = torch.sum(torch.abs(R) ** 2).item()
        rel_res = (rsnew / rs0) ** 0.5
        residuals.append(rel_res)
        if rel_res < tol:
            break
        beta = rsnew / rsold
        P = R + beta * P
        rsold = rsnew

    return X, residuals


def pdhg(
    grad_f: Callable[[torch.Tensor], torch.Tensor],
    f_L: float,
    prox_h,
    G,
    G_norm_squared: float,
    x0: torch.Tensor,
    niter: int = 100,
) -> torch.Tensor:
    """min_x f(x) + h(G x): the Condat-Vu primal-dual method, i.e. PDHG with a
    plain gradient step on the smooth term f, via mirtorch's FBPD.

    grad_f: gradient of f, f_L its Lipschitz constant. prox_h: a mirtorch Prox
    for h (FBPD applies it to h's convex conjugate via Moreau's identity).
    G: a mirtorch LinearMap, G_norm_squared an upper bound on ||G||^2 (sets the
    dual step size)."""
    solver = FBPD(grad_f, Const(), prox_h, g_L=f_L, G=G, G_norm_squared=G_norm_squared,
                  max_iter=niter)
    return solver.run(x0)
