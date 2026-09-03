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
from recon.operators_b0 import build_encoding_operator_b0, estimate_spectral_norm
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


def _load_omega(
    fn_ksp: str, Nx: int, Ny: int, Nz: int, Nt: int, ksp0: torch.Tensor
) -> torch.Tensor:
    """(Nx,Ny,Nz,Nt) sampling mask, broadcast across the readout axis
    (kx doesn't affect which (ky,kz) locations were sampled).

    Prefers the authoritative 'omegas' dataset preprocess.py writes into
    the same file (preprocessing/preprocess.py's _build_omegas) over
    inferring the mask from which complex64 k-space values happen to be
    exactly zero: a real acquired sample that rounds to exactly 0+0j after
    phase correction would otherwise silently become "not acquired", and
    since that would be consistently wrong across every coil, a per-coil
    consistency check can't catch it either -- so that check is only worth
    doing in the fallback branch below, where it's actually load-bearing.
    Falls back to the `!= 0` derivation, logged, for recon files written
    before 'omegas' existed.
    """
    with h5py.File(fn_ksp, "r") as f:
        has_omegas = "omegas" in f
        if has_omegas:
            omegas_yzt = torch.from_numpy(np.asarray(f["omegas"][()])).to(ksp0.device)
    if has_omegas:
        assert tuple(omegas_yzt.shape) == (Ny, Nz, Nt), (
            f"omegas shape {tuple(omegas_yzt.shape)} doesn't match k-space dims ({Ny},{Nz},{Nt})"
        )
        return omegas_yzt.unsqueeze(0).expand(Nx, -1, -1, -1).bool()

    print(
        f"  '{fn_ksp}' has no 'omegas' dataset (written before preprocess.py added it) -- "
        "falling back to inferring the sampling mask from exact-zero k-space values."
    )
    omega = ksp0[:, :, :, 0, :] != 0
    for ic in range(1, ksp0.shape[3]):
        assert torch.equal(omega, ksp0[:, :, :, ic, :] != 0), f"Coil {ic} has a differing mask"
    return omega


def _load_echo_times(fn_ksp: str, device: torch.device) -> torch.Tensor:
    """(Ny,Nz,Nt) echo-time array (seconds since RF excitation), moved to
    device at its native shape -- shared by both B0-recon call sites
    (run_recon here and run_b0_recon.py) so neither has to duplicate the
    broadcast-to-(Nx,Ny,Nz,Nt) pattern build_encoding_operator_b0 no
    longer needs (see its docstring and docs/review-findings.md item 90)."""
    return torch.from_numpy(_load_array(fn_ksp, "echo_times").astype(np.float32)).to(device)


def _load_normalized_smaps(
    fn_smaps: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads smaps and RSS-normalizes it. Returns (smaps, smaps_chw):
    smaps is (Nx,Ny,Nz,Nc) complex64, each voxel's coil vector scaled to
    unit RSS; smaps_chw is (Nc,Nx,Ny,Nz), the layout
    build_encoding_operator{,_b0} expect. Shared by run_recon (here) and
    run_b0_recon.py, which used to duplicate this verbatim -- a real
    desync risk since run_b0_recon's whole purpose is measuring sigma1A
    for the operator run_recon builds moments later (see
    docs/review-findings.md item 94)."""
    smaps_raw = torch.from_numpy(_load_array(fn_smaps, "smaps").astype(np.complex64)).to(device)
    smaps_rss = smaps_raw.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    smaps = smaps_raw / (smaps_rss + torch.finfo(torch.float32).eps)
    smaps_chw = smaps.permute(3, 0, 1, 2).contiguous()
    return smaps, smaps_chw


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
    sigma1A: float | None = None,
    device: torch.device | str = "cuda",
    mom: str = "fpgm",
    conv_tol: float = 1e-5,
    lambda_global: float = 1.0,
    fn_b0map: str | None = None,
    L_b0: int = 32,
    nbins_b0: int = 128,
) -> ReconResult:
    """fn_b0map: optional path to a run_b0map.py output (<seqname>_b0map.h5,
    'b0map_hz' on the EPI grid -- see preprocessing/run_b0map.py). When
    given, builds the encoding operator with time-segmented B0 off-
    resonance correction (recon/operators_b0.py) instead of the plain
    encoding operator -- reads per-sample acquisition time from fn_ksp's
    'echo_times' dataset (preprocessing/preprocess.py's _build_echo_times;
    (Ny,Nz,Nt), broadcast across Nx here since kx doesn't affect echo
    time). L_b0/nbins_b0 are mri_exp_approx's segment count/histogram bins
    (see operators_b0.py's module docstring for the real-scale sweep that
    settled L_b0=32). sigma1A defaults to None, in which case it
    is measured here via power iteration (operators_b0.estimate_spectral_
    norm) on the operator actually built -- required when fn_b0map is set,
    since the B0-corrected operator's spectral norm is not guaranteed to
    match the uncorrected operator's, and supplying a too-small sigma1A
    silently makes POGM's step size too large (divergence, not a clean
    failure). Pass sigma1A explicitly to skip this measurement (e.g. when
    reusing a previously-measured value)."""
    device = torch.device(device)
    Nscales = len(patch_sizes)

    print("Loading sensitivity maps...")
    smaps, smaps_chw = _load_normalized_smaps(fn_smaps, device)
    print(f"  Sensitivity maps: {tuple(smaps.shape)}")

    print("Loading k-space...")
    ksp0 = torch.from_numpy(_load_array(fn_ksp, "ksp_epi_zf").astype(np.complex64)).to(device)
    Nx, Ny, Nz, Nvc, Nt = ksp0.shape
    assert tuple(smaps.shape) == (Nx, Ny, Nz, Nvc), (
        f"smaps shape {tuple(smaps.shape)} doesn't match k-space dims ({Nx},{Ny},{Nz},{Nvc})"
    )

    omega = _load_omega(fn_ksp, Nx, Ny, Nz, Nt, ksp0)
    R = (Nx * Ny * Nz) / omega[:, :, :, 0].sum().item()
    print(f"Acceleration factor R ~ {R:.2f}")
    counts = omega.sum(dim=(0, 1, 2))
    assert torch.all(counts == counts[0]), "Frames have differing sample counts"

    print("Building encoding operator...")  # smaps_chw already computed above
    if fn_b0map is not None:
        print(f"  Loading B0 field map from {fn_b0map} (L={L_b0}, nbins={nbins_b0})...")
        b0map_hz = torch.from_numpy(_load_array(fn_b0map, "b0map_hz").astype(np.float32)).to(device)
        assert tuple(b0map_hz.shape) == (Nx, Ny, Nz), (
            f"b0map_hz shape {tuple(b0map_hz.shape)} doesn't match k-space dims ({Nx},{Ny},{Nz})"
        )
        echo_times_yz = _load_echo_times(fn_ksp, device)
        A = build_encoding_operator_b0(
            smaps_chw, omega, b0map_hz, echo_times_yz, L=L_b0, nbins=nbins_b0
        )
    else:
        A = build_encoding_operator(smaps_chw, omega)

    if sigma1A is None:
        if fn_b0map is None:
            raise ValueError(
                "run_recon: sigma1A must be supplied when fn_b0map is not set -- "
                "the plain SENSE operator's spectral norm has no cheap closed-form "
                "estimate wired up here, so auto-estimation only covers the "
                "B0-corrected path."
            )
        print("  sigma1A not supplied -- measuring via power iteration...")
        x0 = torch.randn(Nx, Ny, Nz, Nt, dtype=torch.complex64, device=device)
        sigma1A = estimate_spectral_norm(A, x0)
        print(f"    sigma1A (B0-corrected) = {sigma1A:.6f}")
        del x0

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
