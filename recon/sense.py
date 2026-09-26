"""Iterative SENSE reconstruction: min_x f(x) + g(x).

f(x) = 0.5 * ||A x - y||^2 is data consistency, with A one of the encoding
operators in recon/operators.py (SENSE, or SENSE_B0 / SENSE_B0_R2star with
--B0 / --R2star). g(x) is the regularizer (recon/regularizers.py), and it
decides the solver (recon/solvers.py):

    --reg none        g = 0                        -> conjugate gradient
    --reg lowrank     multi-scale low-rank         -> POGM (or FPGM / PGM)
    --reg wavelet-tv  L1-wavelet + TV, per frame   -> PDHG (primal-dual)

    .venv-recon/bin/python -m recon.sense <datdir> <seqname> --reg lowrank \\
        --patch 6 6 6 --stride 3 3 3 [--B0 [--R2star]] [--frames 0,1,2] [--niter 200]

Reads <datdir>/recon/{<seqname>_epi_zf.h5, smaps_<seqname>_sigpy.h5,
<seqname>_b0map.h5} and writes <datdir>/recon/sense_<reg>[_b0|_b0r2star]/
<seqname>_recon.{h5,nii.gz,json}.
"""

import argparse
import math
import os
import time
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch

from recon.operators import build_sense, build_sense_b0, build_sense_b0_r2star
from recon.regularizers import MultiScaleLowRank, WaveletTV
from recon.solvers import cg, pdhg, pogm_restart
from recon.utils import (
    estimate_operator_noise_factor,
    estimate_spectral_norm,
    load_and_gather_ksp,
    load_array,
    load_echo_times,
    load_normalized_smaps,
    load_omega,
    resolve_device,
    save_result,
)

REGULARIZERS = ("none", "lowrank", "wavelet-tv")


@dataclass
class ReconResult:
    X: torch.Tensor  # (Nx,Ny,Nz,Nt,Nscales); Nscales=1 unless reg='lowrank'
    X_recon: torch.Tensor  # (Nx,Ny,Nz,Nt)
    omega: torch.Tensor  # (Nx,Ny,Nz,Nt) bool
    dc_costs: list[float]  # per iteration; CG: relative residual norms
    reg_costs: list[float]
    restarts: list[bool]
    rel_changes: list[float]
    R: float
    sigma1A: float
    L: float  # Lipschitz constant used for the step size (lowrank only, else nan)
    lambdas: list[float]
    runtime_s: float
    meta: dict = field(default_factory=dict)


def run_sense(
    *,
    fn_ksp: str,
    fn_smaps: str,
    reg: str = "lowrank",
    frames: list[int] | None = None,
    device: torch.device | str | None = None,
    fn_b0map: str | None = None,
    r2star_map: torch.Tensor | None = None,
    t_ref_s: float = 0.0,
    L_b0: int = 32,
    nbins_b0: int = 128,
    niters: int = 200,
    sigma1A: float | None = None,
    # lowrank
    patch_sizes: list[tuple[int, int, int]] | None = None,
    strides: list[tuple[int, int, int]] | None = None,
    lambda_global: float | None = 1.0,
    mom: str = "fpgm",
    conv_tol: float = 1e-5,
    normalize_noise: bool = True,
    # wavelet-tv
    lamb_l1: float = 0.005,
    lamb_tv: float = 0.005,
    wave: str = "db4",
    levels: int = 3,
    # none (CG)
    cg_tol: float = 1e-6,
) -> ReconResult:
    """Reconstruct the frames of fn_ksp with regularizer `reg`.

    fn_b0map: a preprocessing/run_b0map.py output; when given, A models B0
    phase accrual (SENSE_B0), using fn_ksp's per-sample 'echo_times'. With
    r2star_map (1/s, EPI grid) as well, A also models R2* decay relative to
    t_ref_s, the nominal-TE echo time (SENSE_B0_R2star).

    device: default "cuda" if available, else "cpu".
    sigma1A: spectral norm of A; measured by power iteration when None.
    lambda_global: lowrank weight scale; None means "use the acceleration
    factor R". normalize_noise (lowrank): rescale the data so image-domain
    thermal noise has unit variance, which the Ong & Lustig lambda formula
    assumes; undone on the returned image.
    """
    if reg not in REGULARIZERS:
        raise ValueError(f"reg={reg!r}, expected one of {REGULARIZERS}")
    if r2star_map is not None and fn_b0map is None:
        raise ValueError("r2star_map requires fn_b0map (R2* is modeled on top of B0)")
    device = resolve_device(device)

    print("Loading sensitivity maps...")
    smaps, smaps_chw = load_normalized_smaps(fn_smaps, device)
    print(f"  Sensitivity maps: {tuple(smaps.shape)}")
    Nx, Ny, Nz, Nvc = smaps.shape

    with h5py.File(fn_ksp, "r") as f:
        ksp_shape = f["ksp_epi_zf"].shape  # cheap metadata peek, no data read
    assert ksp_shape == (Nx, Ny, Nz, Nvc, ksp_shape[-1]), (
        f"smaps shape {tuple(smaps.shape)} doesn't match k-space dims {ksp_shape[:4]}"
    )
    frames = list(range(ksp_shape[-1])) if frames is None else list(frames)
    Nt = len(frames)

    omega = load_omega(fn_ksp, Nx, Ny, Nz, ksp_shape[-1], device)[..., frames]
    R = (Nx * Ny * Nz) / omega[:, :, :, 0].sum().item()
    print(f"Acceleration factor R ~ {R:.2f}")
    counts = omega.sum(dim=(0, 1, 2))
    assert torch.all(counts == counts[0]), "Frames have differing sample counts"

    print("Building encoding operator...")
    if fn_b0map is None:
        A = build_sense(smaps_chw, omega)
    else:
        print(f"  Loading B0 field map from {fn_b0map} (L={L_b0}, nbins={nbins_b0})...")
        b0map_hz = torch.from_numpy(load_array(fn_b0map, "b0map_hz").astype(np.float32)).to(device)
        assert tuple(b0map_hz.shape) == (Nx, Ny, Nz), (
            f"b0map_hz shape {tuple(b0map_hz.shape)} doesn't match k-space dims ({Nx},{Ny},{Nz})"
        )
        echo_times_yz = load_echo_times(fn_ksp, device)[..., frames]
        if r2star_map is None:
            A = build_sense_b0(smaps_chw, omega, b0map_hz, echo_times_yz, L=L_b0, nbins=nbins_b0)
        else:
            print(f"  R2* correction enabled (t_ref_s={t_ref_s * 1000:.3f} ms)...")
            assert tuple(r2star_map.shape) == (Nx, Ny, Nz), (
                f"r2star_map shape {tuple(r2star_map.shape)} != k-space grid ({Nx},{Ny},{Nz})"
            )
            A = build_sense_b0_r2star(
                smaps_chw,
                omega,
                b0map_hz,
                echo_times_yz,
                r2star_map,
                t_ref_s,
                L=L_b0,
                nbins=nbins_b0,
            )

    if sigma1A is None and reg != "none":
        print("  sigma1A not supplied -- measuring via power iteration...")
        x0 = torch.randn(Nx, Ny, Nz, dtype=torch.complex64, device=device)
        sigma1A = estimate_spectral_norm(A, x0)
        print(f"    sigma1A = {sigma1A:.6f}")
        del x0

    print("Loading k-space (gathered per frame, never materializing the dense array)...")
    ksp = load_and_gather_ksp(fn_ksp, A, device, frames=frames)  # (K,Nc,Nt)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        free_gb, total_gb = (x / 1e9 for x in torch.cuda.mem_get_info())
        print(f"  VRAM free after loading gathered k-space: {free_gb:.2f} / {total_gb:.2f} GB")

    common = dict(omega=omega, R=R, sigma1A=float("nan") if sigma1A is None else sigma1A)
    if reg == "none":
        return _solve_cg(A, ksp, (Nx, Ny, Nz, Nt), niters, cg_tol, common)
    if reg == "wavelet-tv":
        return _solve_wavelet_tv(
            A, ksp, (Nx, Ny, Nz, Nt), sigma1A, lamb_l1, lamb_tv, wave, levels, niters, common
        )
    if lambda_global is None:
        lambda_global = float(R)
    return _solve_lowrank(
        A,
        ksp,
        (Nx, Ny, Nz, Nt),
        sigma1A,
        patch_sizes,
        strides,
        lambda_global,
        mom,
        niters,
        conv_tol,
        normalize_noise,
        device,
        common,
    )


def _solve_cg(A, ksp, shape, niters, tol, common) -> ReconResult:
    print(f"\nRunning CG-SENSE ({niters} iterations max, tol={tol})...")
    t_start = time.time()
    X, residuals = cg(A, ksp, shape, num_iter=niters, tol=tol)
    runtime_s = time.time() - t_start
    print(
        f"Done in {runtime_s:.1f} s ({len(residuals) - 1} iterations, final relative residual "
        f"{residuals[-1]:.3e})."
    )
    return ReconResult(
        X=X.unsqueeze(-1),
        X_recon=X,
        dc_costs=residuals,
        reg_costs=[],
        restarts=[],
        rel_changes=[],
        L=float("nan"),
        lambdas=[],
        runtime_s=runtime_s,
        meta=dict(niters=niters, tol=tol),
        **common,
    )


def _solve_lowrank(
    A,
    ksp,
    shape,
    sigma1A,
    patch_sizes,
    strides,
    lambda_global,
    mom,
    niters,
    conv_tol,
    normalize_noise,
    device,
    common,
) -> ReconResult:
    g = MultiScaleLowRank(patch_sizes, strides, shape, lambda_global)
    S = g.synthesis  # (Nx,Ny,Nz,Nt,Nscales) -> (Nx,Ny,Nz,Nt)
    Nscales = g.Nscales

    noise_std = 1.0
    if normalize_noise:
        noise_std = estimate_operator_noise_factor(A, tuple(ksp.shape), ksp.dtype, device)
        print(
            f"  Operator noise-propagation factor = {noise_std:.4f} "
            f"-- rescaling ksp/X0 by 1/{noise_std:.4f}"
        )
        if not (math.isfinite(noise_std) and 1e-6 < noise_std < 1e9):
            raise ValueError(
                f"noise_std={noise_std} is non-finite or implausible; refusing to divide by it"
            )
        ksp = ksp / noise_std

    L = Nscales * sigma1A**2  # Lipschitz constant of grad f for f(X) = 0.5||A S X - y||^2
    print(f"Regularization weights lambdas = {[round(lam, 6) for lam in g.lambdas]}")

    def dc_cost(X):
        return 0.5 * (A.apply(S.apply(X)) - ksp).norm().item() ** 2

    def dc_grad(X):
        return S.adjoint(A.adjoint(A.apply(S.apply(X)) - ksp))

    print("Initializing X0...")
    X0 = S.adjoint(A.adjoint(ksp) / Nscales)
    g.last_cost = g.cost(X0)

    dc_costs, reg_costs, restarts, rel_changes = [], [], [], []

    def logger(it, xk, yk, is_restart, fcostnew, rel_change):
        dc_costs.append(fcostnew)
        reg_costs.append(g.last_cost)
        restarts.append(is_restart)
        rel_changes.append(rel_change)

    print(
        f"\nIteratively reconstructing ({niters} iterations, {Nscales} scale(s), "
        f"mom={mom}, conv_tol={conv_tol})..."
    )
    t_start = time.time()
    X, _ = pogm_restart(
        X0, dc_cost, dc_grad, L, mom=mom, niter=niters, g_prox=g.prox, fun=logger, conv_tol=conv_tol
    )
    runtime_s = time.time() - t_start
    X_recon = S.apply(X)
    print(f"Wall-clock: {runtime_s:.1f} s, {runtime_s / max(len(dc_costs) - 1, 1):.2f} s/iter")

    if normalize_noise:
        X = X * noise_std
        X_recon = X_recon * noise_std
    return ReconResult(
        X=X,
        X_recon=X_recon,
        dc_costs=dc_costs,
        reg_costs=reg_costs,
        restarts=restarts,
        rel_changes=rel_changes,
        L=L,
        lambdas=g.lambdas,
        runtime_s=runtime_s,
        meta=dict(
            niters=niters,
            mom=mom,
            conv_tol=conv_tol,
            lambda_global=lambda_global,
            patch_sizes=g.patch_sizes,
            strides=g.strides,
            normalize_noise=normalize_noise,
            noise_std=noise_std,
        ),
        **common,
    )


def _solve_wavelet_tv(
    A, ksp, shape, sigma1A, lamb_l1, lamb_tv, wave, levels, niters, common
) -> ReconResult:
    """Frame by frame. The operator is normalized to unit spectral norm and
    each frame's data to O(1) (its 99th-percentile magnitude), so lamb_l1/
    lamb_tv mean the same thing on every dataset; both scalings are undone on
    the returned image."""
    Nx, Ny, Nz, Nt = shape
    g = WaveletTV((Nx, Ny, Nz), lamb_l1, lamb_tv, wave=wave, levels=levels)
    X = torch.zeros(shape, dtype=torch.complex64, device=ksp.device)
    dc_costs, reg_costs = [], []
    print(
        f"\nReconstructing {Nt} frame(s) with L1-wavelet + TV (lamb_l1={lamb_l1}, "
        f"lamb_tv={lamb_tv}, {niters} PDHG iterations, operator normalized by 1/{sigma1A:.4f})..."
    )
    t_start = time.time()
    for it in range(Nt):
        t_frame = time.time()
        A_t = (1.0 / sigma1A) * A.A[it]
        y = ksp[:, :, it]
        # numpy, not torch.quantile, which rejects inputs over ~16M elements
        scale = 1.0 / float(np.percentile(y[y != 0].abs().cpu().numpy(), 99))
        y_s = y * scale

        def dc_grad(x, A_t=A_t, y_s=y_s):
            return A_t.adjoint(A_t.apply(x) - y_s)

        x0 = torch.zeros(Nx, Ny, Nz, dtype=torch.complex64, device=ksp.device)
        x = pdhg(dc_grad, 1.0, g.h_prox, g.G, g.G_norm_squared, x0, niter=niters)
        dc_costs.append(0.5 * (A_t.apply(x) - y_s).norm().item() ** 2)
        reg_costs.append(g.cost(x))
        X[..., it] = x / scale / sigma1A
        print(f"  frame {it + 1}/{Nt} done in {time.time() - t_frame:.1f}s")
    runtime_s = time.time() - t_start
    print(f"Wall-clock: {runtime_s:.1f}s ({runtime_s / Nt:.1f}s/frame)")
    return ReconResult(
        X=X.unsqueeze(-1),
        X_recon=X,
        dc_costs=dc_costs,
        reg_costs=reg_costs,
        restarts=[],
        rel_changes=[],
        L=float("nan"),
        lambdas=[lamb_l1, lamb_tv],
        runtime_s=runtime_s,
        meta=dict(niters=niters, wave=wave, levels=levels),
        **common,
    )


# ---------------------------------------------------------------- command line


def main(
    datdir: str,
    seqname: str,
    reg: str,
    *,
    B0: bool = False,
    fn_b0map: str | None = None,
    R2star: bool = False,
    zero_pad_z: bool = False,
    **kwargs,
) -> str:
    """Reconstruct <datdir>/recon/<seqname>_* and save the result. kwargs go
    to run_sense. For reg='lowrank', a GPU out-of-memory error falls back from
    POGM to FPGM to PGM (less solver state each time). Returns the output
    path without extension."""
    from preprocessing.config import load_config, load_seq_params, set_seq_paths
    from preprocessing.r2star_map import estimate_r2star_map_epi_grid
    from recon.utils import nominal_te_s

    recon_dir = os.path.join(datdir, "recon")
    fn_ksp = os.path.join(recon_dir, f"{seqname}_epi_zf.h5")
    fn_smaps = os.path.join(recon_dir, f"smaps_{seqname}_sigpy.h5")
    B0 = B0 or fn_b0map is not None
    if R2star and not B0:
        raise ValueError("R2star correction requires B0 correction")
    if B0 and fn_b0map is None:
        fn_b0map = os.path.join(recon_dir, f"{seqname}_b0map.h5")
    device = resolve_device(kwargs.get("device"))

    paths = set_seq_paths(load_config(datdir=datdir, seqnames=[seqname]), seqname)
    sp = load_seq_params(paths)
    if R2star:
        t_ref_s = nominal_te_s(paths.scan_info, sp.ETL)
        with h5py.File(fn_ksp, "r") as f:
            grid = f["ksp_epi_zf"].shape[:3]
        print(
            f"Estimating R2* map from dual-echo deGRE data (TE_nominal={t_ref_s * 1000:.3f} ms)..."
        )
        r2 = estimate_r2star_map_epi_grid(
            datdir, seqname, sp.fov_degre, sp.fov, grid, zero_pad_z=zero_pad_z
        )
        kwargs.update(r2star_map=torch.from_numpy(r2).to(device), t_ref_s=t_ref_s)

    moms = ["pogm", "fpgm", "pgm"]
    moms = moms[moms.index(kwargs.pop("mom", "pogm")) :] if reg == "lowrank" else [None]
    for mom in moms:
        try:
            result = run_sense(
                fn_ksp=fn_ksp,
                fn_smaps=fn_smaps,
                reg=reg,
                fn_b0map=fn_b0map,
                **({"mom": mom} if mom else {}),
                **kwargs,
            )
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if mom in (None, "pgm"):
                raise
            print(
                f"  mom={mom} ran out of GPU memory -- falling back to the next momentum method..."
            )

    suffix = "_b0r2star" if R2star else ("_b0" if B0 else "")
    out_dir = os.path.join(recon_dir, f"sense_{reg}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    frames = kwargs.get("frames")
    tag = "" if frames is None else "_frames" + _format_frames(frames)
    fn_out = os.path.join(out_dir, f"{seqname}_recon{tag}")
    save_result(
        fn_out,
        result,
        fov=sp.fov,
        seqname=seqname,
        reg=reg,
        b0_corrected=B0,
        r2star_corrected=R2star,
        L_b0=kwargs.get("L_b0", 32) if B0 else None,
    )
    return fn_out


def _parse_frames(text: str) -> list[int]:
    """'0,2,5' or '0-9' or '0-4,10' -> list of frame indices."""
    frames = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        frames += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return frames


def _format_frames(frames: list[int]) -> str:
    """Inverse of _parse_frames, compact: [0..9] -> '0-9', [0, 2] -> '0,2'."""
    if len(frames) > 1 and list(frames) == list(range(frames[0], frames[-1] + 1)):
        return f"{frames[0]}-{frames[-1]}"
    return ",".join(map(str, frames))


def _parse_patches(patches, strides, shape):
    """--patch/--stride values -> (patch_sizes, strides). 'full' = whole volume.
    A patch without a matching --stride gets stride = patch (no overlap)."""

    def parse(v):
        return tuple(shape) if v == ["full"] else tuple(int(n) for n in v)

    patch_sizes = [parse(p) for p in patches]
    strides = [parse(s) for s in (strides or [])]
    strides += patch_sizes[len(strides) :]
    return patch_sizes, strides


def _cli() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("datdir")
    p.add_argument("seqname")
    p.add_argument("--reg", choices=REGULARIZERS, required=True)
    p.add_argument("--B0", action="store_true", help="model B0 phase accrual (<seqname>_b0map.h5)")
    p.add_argument(
        "--b0map", default=None, help="B0 map .h5 to use instead of the default (implies --B0)"
    )
    p.add_argument("--R2star", action="store_true", help="also model R2* decay (needs --B0)")
    p.add_argument(
        "--zero-pad-z",
        action="store_true",
        help="(--R2star) zero-pad the R2* map where the EPI z-FOV exceeds deGRE's",
    )
    p.add_argument("--L", type=int, default=32, dest="L_b0", help="B0 time segments")
    p.add_argument("--nbins", type=int, default=128, dest="nbins_b0", help="B0 histogram bins")
    p.add_argument(
        "--niter",
        type=int,
        default=None,
        help="iterations (default: none 150, lowrank 200, wavelet-tv 100)",
    )
    p.add_argument("--frames", default=None, help="frame indices, e.g. 0,1,2 or 0-9 (default: all)")
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    lr = p.add_argument_group("lowrank")
    lr.add_argument(
        "--patch",
        nargs="+",
        action="append",
        metavar="N",
        help="patch size 'NX NY NZ' or 'full'; repeat for more scales (default 6 6 6)",
    )
    lr.add_argument(
        "--stride",
        nargs="+",
        action="append",
        metavar="N",
        help="stride per --patch, same format (default: 3 3 3 for the default patch)",
    )
    lr.add_argument(
        "--lambda-global", type=float, default=None, help="default: acceleration factor R"
    )
    lr.add_argument("--mom", choices=["pogm", "fpgm", "pgm"], default="pogm")
    lr.add_argument("--conv-tol", type=float, default=1e-5, help="<= 0 disables early stopping")
    wt = p.add_argument_group("wavelet-tv")
    wt.add_argument("--lamb-l1", type=float, default=0.005)
    wt.add_argument("--lamb-tv", type=float, default=0.005)
    wt.add_argument("--wave", default="db4")
    wt.add_argument("--levels", type=int, default=3)
    a = p.parse_args()
    if a.R2star and not (a.B0 or a.b0map):
        p.error("--R2star requires --B0")

    kwargs = dict(
        device=a.device,
        L_b0=a.L_b0,
        nbins_b0=a.nbins_b0,
        niters=a.niter or {"none": 150, "lowrank": 200, "wavelet-tv": 100}[a.reg],
        frames=_parse_frames(a.frames) if a.frames else None,
    )
    if a.reg == "lowrank":
        with h5py.File(os.path.join(a.datdir, "recon", f"{a.seqname}_epi_zf.h5"), "r") as f:
            shape = f["ksp_epi_zf"].shape[:3]
        patches, strides = (
            (a.patch, a.stride) if a.patch else ([["6", "6", "6"]], [["3", "3", "3"]])
        )
        patch_sizes, strides = _parse_patches(patches, strides, shape)
        kwargs.update(
            patch_sizes=patch_sizes,
            strides=strides,
            lambda_global=a.lambda_global,
            mom=a.mom,
            conv_tol=a.conv_tol,
        )
    elif a.reg == "wavelet-tv":
        kwargs.update(lamb_l1=a.lamb_l1, lamb_tv=a.lamb_tv, wave=a.wave, levels=a.levels)
    main(
        a.datdir,
        a.seqname,
        a.reg,
        B0=a.B0,
        fn_b0map=a.b0map,
        R2star=a.R2star,
        zero_pad_z=a.zero_pad_z,
        **kwargs,
    )


if __name__ == "__main__":
    _cli()
