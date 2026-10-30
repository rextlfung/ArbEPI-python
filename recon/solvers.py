"""Iterative solvers.

    pogm_restart  PGM / FPGM / POGM for min_x f(x) + g(x), f smooth, g prox-friendly
    pdhg          primal-dual for min_x f(x) + h(G x), f smooth, h prox-friendly
                  (for g = h o G with no closed-form prox, e.g. TV)
    cg            conjugate gradient on the normal equations A^H A x = A^H y
    ChunkedDataFidelity  0.5||A x - y||^2 and its gradient for a block-diagonal A, a few
                  frames of k-space on the device at a time (the series' k-space, prediction
                  and residual are each Nc / R times an image series: 2.3x at 23 coils, R = 10)
"""

import math
from typing import Callable, Literal

import numpy as np
import torch
from mirtorch.alg import FBPD
from mirtorch.prox import Const
from tqdm import tqdm

Momentum = Literal["pgm", "fpgm", "pogm"]
Restart = Literal["none", "gr", "fr"]
_EPS = torch.finfo(torch.float32).eps


def _gr_restart(fgrad: torch.Tensor, ynew_yold: torch.Tensor, restart_cutoff: float) -> bool:
    inner = -_vdot(fgrad, ynew_yold)
    return inner <= restart_cutoff * fgrad.norm().item() * ynew_yold.norm().item()


def _vdot(a: torch.Tensor, b: torch.Tensor) -> float:
    """Re <a, b> over all elements, without the elementwise temporaries of
    sum(real(conj(a) * b)) (each a full image series: 2.5 GB at 660 frames)."""
    return torch.vdot(a.reshape(-1), b.reshape(-1)).real.item()


def _bar(progress: bool | str, total: int, default_desc: str):
    """A tqdm bar over `total` iterations, or None when progress is off. A str
    `progress` is the label; leave=False so per-frame solves don't stack bars."""
    if not progress:
        return None
    desc = progress if isinstance(progress, str) else default_desc
    return tqdm(total=total, desc=desc, unit="it", leave=False)


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
    progress: bool | str = False,
):
    """Minimize f(x) + g(x) by PGM, FPGM or POGM (Kim & Fessler) with restart.

    fcost, fgrad_fn: f and its gradient; f_L: Lipschitz constant of the
    gradient (step size 1/f_L). g_prox(z, c): prox of c*g. restart: "gr"
    (gradient), "fr" (function value) or "none". Stops early once the relative
    change of the iterate falls below conv_tol (0 disables this).
    fun(iter, xk, yk, is_restart, fcost, rel_change) is called every iteration,
    iter 0 being the start. progress: show a tqdm bar over iterations with the
    cost and relative change (a str is the bar's label). Returns (x, list of
    fun's return values)."""
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

    bar = _bar(progress, niter, "POGM" if mom == "pogm" else mom.upper())
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
                elif _vdot(Fgrad, Fgradold) < 0:
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

        if bar is not None:
            bar.update(1)
            bar.set_postfix(cost=f"{Fcostnew:.4g}", dx=f"{rel_change:.2e}")

        if conv_tol > 0 and it >= last_restart_iter + adaptive_cmi and rel_change < conv_tol:
            out[it] = fun(it, xnew, ynew, is_restart, Fcostnew, rel_change)
            niter_actual = it
            break

        out[it] = fun(it, xnew, ynew, is_restart, Fcostnew, rel_change)
        xold, yold = xnew, ynew
        if mom != "pgm" and mu == 0:
            told = tnew

    if bar is not None:
        bar.close()
    result = xnew if mom == "pogm" else ynew
    return result, out[: niter_actual + 1]


class ChunkedDataFidelity:
    """f(x) = 0.5 ||scale A x - y||^2 for a block-diagonal A (one block per frame),
    with grad f = scale A^H (scale A x - y), computed `chunk` frames at a time.

    `blocks` is `A.A` of a `BlockDiagonal`, `ksp` the k-space in host memory as
    (Nt, K, Nc) (`utils.load_and_gather_ksp_host`). Only `chunk` frames of k-space
    are on the device at once, and each block is applied to its own frame exactly
    as `BlockDiagonal` does, so a chunk costs the same as the same frames in the
    whole-series product: the chunk size trades copy calls against device memory
    (three k-space-sized tensors per chunk frame) and has no effect on the result
    beyond the order in which the cost is summed.
    """

    def __init__(self, blocks, ksp: torch.Tensor, scale: float = 1.0, chunk: int = 8):
        if ksp.shape[0] != len(blocks):
            raise ValueError(f"ksp has {ksp.shape[0]} frames for {len(blocks)} blocks")
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1 frame, got {chunk}")
        self.blocks, self.ksp, self.scale, self.chunk = blocks, ksp, scale, int(chunk)

    def _chunks(self, device):
        for a in range(0, len(self.blocks), self.chunk):
            b = min(a + self.chunk, len(self.blocks))
            yield a, self.ksp[a:b].to(device, non_blocking=True)

    def cost_and_grad(self, x: torch.Tensor, grad: bool = True):
        """(0.5||scale A x - y||^2 as a float, its gradient or None); x is (..., Nt)."""
        g = torch.empty_like(x) if grad else None
        total = torch.zeros((), dtype=torch.float64, device=x.device)
        for a, y in self._chunks(x.device):
            for k in range(a, a + y.shape[0]):
                res = self.scale * self.blocks[k].apply(x[..., k]) - y[k - a]
                total += res.norm().double() ** 2
                if grad:
                    g[..., k] = self.scale * self.blocks[k].adjoint(res)
        return 0.5 * total.item(), g

    def adjoint_data(self, like: torch.Tensor) -> torch.Tensor:
        """scale A^H y, shaped and typed like `like` (Nx, Ny, Nz, Nt)."""
        out = torch.empty_like(like)
        for a, y in self._chunks(like.device):
            for k in range(a, a + y.shape[0]):
                out[..., k] = self.scale * self.blocks[k].adjoint(y[k - a])
        return out


def cg(
    A, ksp: torch.Tensor, shape: tuple[int, int, int, int], num_iter: int = 20,
    tol: float = 1e-6, reg_normal: Callable[[torch.Tensor], torch.Tensor] | None = None,
    progress: bool | str = False,
) -> tuple[torch.Tensor, list[float]]:
    """Conjugate gradient on (A^H A + Q) x = A^H ksp, starting from 0, where Q is
    reg_normal (a symmetric positive semidefinite map, e.g. mu P for a quadratic
    penalty (mu/2)||P x||^2; default none).

    A: an operator from recon/operators.py. ksp: (K,Nc,Nt). shape: (Nx,Ny,Nz,Nt).
    Returns (X, residuals), residuals[i] = ||r_i|| / ||r_0|| (residuals[0] = 1);
    stops early once below tol. progress: show a tqdm bar with the relative
    residual (a str is the bar's label).
    """
    B = A.adjoint(ksp)  # (Nx,Ny,Nz,Nt)
    X = torch.zeros_like(B)
    R = B.clone()
    P = R.clone()
    rsold = _vdot(R, R)
    rs0 = rsold
    residuals = [1.0]

    if rs0 == 0:
        return X, residuals

    bar = _bar(progress, num_iter, "CG")
    for _ in range(num_iter):
        AP = A.apply(P)
        AHAP = A.adjoint(AP)
        if reg_normal is not None:
            AHAP = AHAP + reg_normal(P)
        alpha = rsold / _vdot(P, AHAP)
        X.add_(P, alpha=alpha)
        R.sub_(AHAP, alpha=alpha)
        rsnew = _vdot(R, R)
        rel_res = (rsnew / rs0) ** 0.5
        residuals.append(rel_res)
        if bar is not None:
            bar.update(1)
            bar.set_postfix(res=f"{rel_res:.2e}")
        if rel_res < tol:
            break
        beta = rsnew / rsold
        P.mul_(beta).add_(R)
        rsold = rsnew

    if bar is not None:
        bar.close()
    return X, residuals


def pdhg(
    grad_f: Callable[[torch.Tensor], torch.Tensor],
    f_L: float,
    prox_h,
    G,
    G_norm_squared: float,
    x0: torch.Tensor,
    niter: int = 100,
    prox_g=None,
    trace_fun: Callable[[torch.Tensor], dict[str, float]] | None = None,
    trace_every: int = 1,
    progress: bool | str = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, np.ndarray]]:
    """min_x f(x) + g(x) + h(G x): the Condat-Vu primal-dual method, i.e. PDHG
    with a plain gradient step on the smooth term f, via mirtorch's FBPD.

    grad_f: gradient of f, f_L its Lipschitz constant. prox_h: a mirtorch Prox
    for h (FBPD applies it to h's convex conjugate via Moreau's identity).
    G: a mirtorch LinearMap, G_norm_squared an upper bound on ||G||^2 (sets the
    dual step size). prox_g: a Prox for a proximable g, applied in the primal
    step (default: g = 0). progress: show a tqdm bar over iterations (a str is
    the bar's label).

    trace_fun: x -> {name: float}, evaluated every `trace_every` iterations (and
    at the last). With it, returns (x, trace): trace['iter'] the 1-based
    iterations traced, trace[name] the values there, and trace['rel_change'] the
    relative primal change ||x_k - x_(k-1)|| / ||x_k|| at every iteration. A
    primal-dual method has no monotone cost: the primal objective can rise
    between iterations, so read the trend, not single steps."""
    solver = FBPD(grad_f, Const() if prox_g is None else prox_g, prox_h, g_L=f_L, G=G,
                  G_norm_squared=G_norm_squared, max_iter=niter)
    bar = _bar(progress, niter, "PDHG")
    if trace_fun is None and bar is None:
        return solver.run(x0)
    prev = x0.clone() if trace_fun is not None else None
    it, rel, iters, vals = 0, [], [], []

    def eval_func(x):
        # FBPD calls this once per iteration: the bar's tick and the trace point.
        nonlocal it
        it += 1
        if bar is not None:
            bar.update(1)
        if trace_fun is not None:
            n = torch.linalg.vector_norm(x).item()
            rel.append(torch.linalg.vector_norm(x - prev).item() / max(n, 1e-30))
            prev.copy_(x)
            if it % trace_every == 0 or it == niter:
                iters.append(it)
                vals.append(trace_fun(x))
        return 0.0

    solver.eval_func = eval_func
    try:
        x, _ = solver.run(x0)
    finally:
        if bar is not None:
            bar.close()
    if trace_fun is None:
        return x
    trace = {"iter": np.asarray(iters), "rel_change": np.asarray(rel)}
    for k in (vals[0] if vals else {}):
        trace[k] = np.asarray([v[k] for v in vals])
    return x, trace
