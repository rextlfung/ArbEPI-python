"""Multi-scale Locally Low-Rank (MSLR) fMRI reconstruction via decomposition.
Port of ../mslr-recon/scripts/reconstruct.jl (Ong & Lustig 2016), built on
mirtorch instead of MIRT.jl/LinearMapsAA -- see recon/operators.py,
recon/lowrank.py, recon/solvers.py for the individual pieces.

    X_final = X[...,0] + X[...,1] + ... + X[...,Nscales-1]

Each component X[...,k] is independently constrained to be locally low-rank
at its own patch scale (recon/lowrank.py's patchSVST); data consistency is
enforced on the sum. lambda_k set by the Ong & Lustig (2016) closed-form
formula (see _reg_weights below) -- no tuning needed beyond lambda_global.
"""

import math
import time
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch

from recon.lowrank import img2patches, patch_nucnorm, patchSVST
from recon.operators import build_encoding_operator, gather_ksp
from recon.solvers import pogm_restart


@dataclass
class ReconResult:
    X: torch.Tensor  # (Nx,Ny,Nz,Nt,Nscales)
    X_recon: torch.Tensor  # (Nx,Ny,Nz,Nt)
    omega: torch.Tensor  # (Nx,Ny,Nz,Nt) bool
    dc_costs: list[float]
    reg_costs: list[float]
    restarts: list[bool]
    rel_changes: list[float]
    R: float
    sigma1A: float
    L: float
    lambdas: list[float]
    runtime_s: float
    meta: dict = field(default_factory=dict)


def _load_array(fn: str, key: str) -> np.ndarray:
    """.h5 written in plain numpy order (e.g. this repo's own preprocessing/
    output, or mslr-recon's sigpy-export input path) -- no axis correction
    needed, unlike hdf5storage-written .mat files (see preprocessing/matio.py).

    Reads chunk-by-chunk along the last axis when the dataset is chunked
    there (this repo's own preprocessing/ writes ksp_epi_zf chunked one
    frame per chunk). A single whole-dataset `d[()]` call was measured at
    ~7 MB/s on real ArbEPI_epi_zf.h5 data (838M+ read syscalls for 7.5GB --
    an h5py/HDF5 pathology when the chunk cache doesn't fit even one chunk),
    versus ~500 MB/s reading one same-sized chunk slice at a time -- a
    two-orders-of-magnitude difference on an 11GB file that otherwise made
    this unusable."""
    with h5py.File(fn, "r") as f:
        d = f[key]
        if d.chunks is not None and d.chunks[-1] < d.shape[-1]:
            out = np.empty(d.shape, dtype=d.dtype)
            step = d.chunks[-1]
            for start in range(0, d.shape[-1], step):
                out[..., start : start + step] = d[..., start : start + step]
            return out
        return np.asarray(d[()])


def _reg_weights(
    patch_sizes: list[tuple[int, int, int]], Nt: int, N_voxels: int, lambda_global: float
) -> list[float]:
    """Ong & Lustig 2016 eq. (4): lambda_k = sqrt(p_k) + sqrt(Nt) +
    sqrt(log(N_voxels*Nt / max(p_k, Nt))), p_k = voxels per patch. Natural
    log (paper states the weight only up to a constant; lambda_global absorbs
    any rescaling -- see reconstruct.jl's own comment on this choice)."""
    lambdas = []
    for ps in patch_sizes:
        p_k = math.prod(ps)
        lam = math.sqrt(p_k) + math.sqrt(Nt) + math.sqrt(math.log(N_voxels * Nt / max(p_k, Nt)))
        lambdas.append(lam * lambda_global)
    return lambdas


def run_recon(
    *,
    fn_ksp: str,
    fn_smaps: str,
    patch_sizes: list[tuple[int, int, int]],
    strides: list[tuple[int, int, int]],
    niters: int = 200,
    sigma1A: float,
    device: torch.device | str = "cuda",
    mom: str = "fpgm",
    conv_tol: float = 1e-5,
    lambda_global: float = 1.0,
) -> ReconResult:
    device = torch.device(device)
    Nscales = len(patch_sizes)

    print("Loading sensitivity maps...")
    smaps_raw = torch.from_numpy(_load_array(fn_smaps, "smaps").astype(np.complex64)).to(device)
    smaps_rss = smaps_raw.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    smaps = smaps_raw / (smaps_rss + torch.finfo(torch.float32).eps)
    print(f"  Sensitivity maps: {tuple(smaps.shape)}")

    print("Loading k-space...")
    ksp0 = torch.from_numpy(_load_array(fn_ksp, "ksp_epi_zf").astype(np.complex64)).to(device)
    Nx, Ny, Nz, Nvc, Nt = ksp0.shape
    assert tuple(smaps.shape) == (Nx, Ny, Nz, Nvc), (
        f"smaps shape {tuple(smaps.shape)} doesn't match k-space dims ({Nx},{Ny},{Nz},{Nvc})"
    )

    omega = ksp0[:, :, :, 0, :] != 0
    R = (Nx * Ny * Nz) / omega[:, :, :, 0].sum().item()
    print(f"Acceleration factor R ~ {R:.2f}")
    for ic in range(1, Nvc):
        assert torch.equal(omega, ksp0[:, :, :, ic, :] != 0), f"Coil {ic} has a differing mask"
    counts = omega.sum(dim=(0, 1, 2))
    assert torch.all(counts == counts[0]), "Frames have differing sample counts"

    print("Building encoding operator...")
    smaps_chw = smaps.permute(3, 0, 1, 2).contiguous()  # (Nc,Nx,Ny,Nz)
    A = build_encoding_operator(smaps_chw, omega)
    ksp = gather_ksp(ksp0, A)  # (K,Nc,Nt) -- see operators.py for why gathered, not dense
    del ksp0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        free_gb, total_gb = (x / 1e9 for x in torch.cuda.mem_get_info())
        print(f"  VRAM free after freeing dense k-space: {free_gb:.2f} / {total_gb:.2f} GB")

    L = Nscales * sigma1A**2

    N_voxels = Nx * Ny * Nz
    lambdas = _reg_weights(patch_sizes, Nt, N_voxels, lambda_global)
    print(f"Regularization weights lambdas = {[round(lam, 6) for lam in lambdas]}")

    def image_sum(X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=-1)

    def dc_cost(X: torch.Tensor) -> float:
        res = A.apply(image_sum(X)) - ksp
        return 0.5 * res.norm().item() ** 2

    def dc_cost_grad(X: torch.Tensor) -> torch.Tensor:
        res = A.apply(image_sum(X)) - ksp
        g = A.adjoint(res)  # (Nx,Ny,Nz,Nt)
        return g.unsqueeze(-1).expand(-1, -1, -1, -1, Nscales).clone()

    def reg_cost(X: torch.Tensor) -> float:
        total = 0.0
        for k in range(Nscales):
            P = img2patches(X[..., k], patch_sizes[k], strides[k])
            total += lambdas[k] * patch_nucnorm(P).item()
        return total

    last_reg = [0.0]  # updated for free inside g_prox each iter; set once for iter 0 below

    def g_prox(X: torch.Tensor, c: float) -> torch.Tensor:
        reg = 0.0
        for k in range(Nscales):
            result, cost = patchSVST(X[..., k], c * lambdas[k], patch_sizes[k], strides[k])
            X[..., k] = result
            reg += lambdas[k] * cost.item()
        last_reg[0] = reg
        return X

    print("Initializing X0...")
    Atksp = A.adjoint(ksp) / Nscales  # (Nx,Ny,Nz,Nt)
    X0 = Atksp.unsqueeze(-1).expand(-1, -1, -1, -1, Nscales).clone()
    last_reg[0] = reg_cost(X0)

    dc_costs: list[float] = []
    reg_costs: list[float] = []
    restarts: list[bool] = []
    rel_changes: list[float] = []

    def logger(it, xk, yk, is_restart, fcostnew, rel_change):
        dc_costs.append(fcostnew)
        reg_costs.append(last_reg[0])
        restarts.append(is_restart)
        rel_changes.append(rel_change)

    print(
        f"\nIteratively reconstructing ({niters} iterations, {Nscales} scale(s), "
        f"mom={mom}, conv_tol={conv_tol})..."
    )
    t_start = time.time()
    X, _ = pogm_restart(
        X0,
        dc_cost,
        dc_cost_grad,
        L,
        mom=mom,
        niter=niters,
        g_prox=g_prox,
        fun=logger,
        conv_tol=conv_tol,
    )
    runtime_s = time.time() - t_start
    X_recon = image_sum(X)

    print(f"Wall-clock: {runtime_s:.1f} s, {runtime_s / max(len(dc_costs) - 1, 1):.2f} s/iter")

    return ReconResult(
        X=X,
        X_recon=X_recon,
        omega=omega,
        dc_costs=dc_costs,
        reg_costs=reg_costs,
        restarts=restarts,
        rel_changes=rel_changes,
        R=R,
        sigma1A=sigma1A,
        L=L,
        lambdas=lambdas,
        runtime_s=runtime_s,
        meta=dict(niters=niters, mom=mom, conv_tol=conv_tol, lambda_global=lambda_global,
                   patch_sizes=patch_sizes, strides=strides),
    )
