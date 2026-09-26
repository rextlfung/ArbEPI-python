"""Everything in recon/ that isn't an operator, regularizer, solver or driver.

    I/O              read_frames_cropped, load_*, load_and_gather_ksp, nominal_te_s,
                     read_julia_mat, save_result
    operator norms   poweriter, estimate_spectral_norm, check_operator_unitary
    tSNR             object_mask, temporal_stability, tsnr_report
    analysis         sweep, benchmark, validate (one-off studies, not the production path)

    .venv-recon/bin/python -m recon.utils {tsnr,sweep,benchmark,validate} ...
"""

import argparse
import math
import os
import sys
import time
import warnings
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np
import torch
from mirtorch.linear import BlockDiagonal
from mirtorch.linear.mri import mri_exp_approx

from preprocessing.matio import read_mat
from preprocessing.nifti_io import save_recon_nifti
from recon.operators import SENSE, SENSE_B0, build_sense, build_sense_b0

if TYPE_CHECKING:
    from recon.sense import ReconResult


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    """device, or "cuda" if a GPU is available and "cpu" otherwise. Everything
    in recon/ runs on either; CPU is just much slower."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


# ---------------------------------------------------------------- I/O


def read_frames_cropped(
    fn: str, key: str, spatial_slices: tuple[slice, slice, slice] | None = None,
) -> np.ndarray:
    """Read an HDF5 dataset shaped (X, Y, Z, ..., T) one chunk at a time along
    T, optionally cropping each frame to spatial_slices (x, y, z) as it's read.
    Chunk-by-chunk reads are ~70x faster than a single d[()] on large files."""
    with h5py.File(fn, 'r') as f:
        d = f[key]
        chunked_by_frame = d.chunks is not None and d.chunks[-1] < d.shape[-1]

        if not chunked_by_frame:
            full = np.asarray(d[()])
            return full if spatial_slices is None else full[(*spatial_slices,)]

        step = d.chunks[-1]
        if spatial_slices is None:
            out = np.empty(d.shape, dtype=d.dtype)
            for start in range(0, d.shape[-1], step):
                out[..., start : start + step] = d[..., start : start + step]
            return out

        out = None
        for start in range(0, d.shape[-1], step):
            frame = d[..., start : start + step]  # one HDF5 chunk, full spatial extent
            cropped = frame[(*spatial_slices,)]
            if out is None:
                out = np.empty(cropped.shape[:-1] + (d.shape[-1],), dtype=d.dtype)
            out[..., start : start + step] = cropped
        return out


def load_array(fn: str, key: str) -> np.ndarray:
    """Whole dataset `key` from an .h5 written in numpy axis order."""
    return read_frames_cropped(fn, key)


def load_omega(
    fn_ksp: str, Nx: int, Ny: int, Nz: int, Nt: int, device: torch.device
) -> torch.Tensor:
    """(Nx,Ny,Nz,Nt) bool sampling mask from the file's 'omegas' (Ny,Nz,Nt),
    broadcast along kx. Files written before 'omegas' existed fall back to
    the nonzero pattern of the k-space itself (reads the whole array)."""
    with h5py.File(fn_ksp, "r") as f:
        has_omegas = "omegas" in f
        if has_omegas:
            omegas_yzt = torch.from_numpy(np.asarray(f["omegas"][()])).to(device)
    if has_omegas:
        assert tuple(omegas_yzt.shape) == (Ny, Nz, Nt), (
            f"omegas shape {tuple(omegas_yzt.shape)} doesn't match k-space dims ({Ny},{Nz},{Nt})"
        )
        return omegas_yzt.unsqueeze(0).expand(Nx, -1, -1, -1).bool()

    print(
        f"  '{fn_ksp}' has no 'omegas' dataset (written before preprocess.py added it) -- "
        "falling back to inferring the sampling mask from exact-zero k-space values."
    )
    ksp0 = torch.from_numpy(load_array(fn_ksp, "ksp_epi_zf").astype(np.complex64)).to(device)
    omega = ksp0[:, :, :, 0, :] != 0
    for ic in range(1, ksp0.shape[3]):
        assert torch.equal(omega, ksp0[:, :, :, ic, :] != 0), f"Coil {ic} has a differing mask"
    return omega


def load_echo_times(fn_ksp: str, device: torch.device) -> torch.Tensor:
    """(Ny,Nz,Nt) seconds since excitation of each sampled (ky,kz)."""
    return torch.from_numpy(load_array(fn_ksp, "echo_times").astype(np.float32)).to(device)


def load_normalized_smaps(
    fn_smaps: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sensitivity maps normalized to unit root-sum-of-squares per voxel.
    Returns (smaps (Nx,Ny,Nz,Nc), smaps_chw (Nc,Nx,Ny,Nz), the operators' layout)."""
    smaps_raw = torch.from_numpy(load_array(fn_smaps, "smaps").astype(np.complex64)).to(device)
    smaps_rss = smaps_raw.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    smaps = smaps_raw / (smaps_rss + torch.finfo(torch.float32).eps)
    smaps_chw = smaps.permute(3, 0, 1, 2).contiguous()
    return smaps, smaps_chw


def load_and_gather_ksp(
    fn_ksp: str, A: BlockDiagonal, device: torch.device, frames: list[int] | None = None,
) -> torch.Tensor:
    """(K,Nc,Nt) k-space at A's sampled locations, read one frame at a time so
    the dense (Nx,Ny,Nz,Nc,Nt) array is never in memory. frames: the file's
    frame index for each block of A (default: block it <- frame it)."""
    Nt = len(A.A)
    frames = list(range(Nt)) if frames is None else list(frames)
    Nc = A.A[0].Nc
    K = A.A[0].idx.numel()
    out = torch.empty(K, Nc, Nt, dtype=torch.complex64, device=device)
    with h5py.File(fn_ksp, "r") as f:
        d = f["ksp_epi_zf"]
        for it in range(Nt):
            frame = torch.from_numpy(np.asarray(d[..., frames[it]]).astype(np.complex64)).to(device)
            flat = frame.reshape(-1, Nc)  # spatial C-order flatten, matches SENSE
            out[:, :, it] = flat[A.A[it].idx, :]
    return out


def nominal_te_s(scan_info_path: str, etl: int) -> float:
    """Acquisition time (s since excitation) of the nominal-TE echo, the
    center echo of the train, from scan_info.mat's schedules."""
    schedules = read_mat(scan_info_path, ["schedules"])["schedules"]  # (Nframes,Nshots,ETL,3)
    return float(schedules[0, 0, (etl - 1) // 2, 2])


def read_julia_mat(path: str) -> dict:
    """Settings and results of a ../mslr-recon (Julia) run, from its v7.3 .mat.
    Arrays are stored axis-reversed and complex values as {real, imag}."""
    scalar_keys = ("R", "sigma1A", "L", "Nscales", "lambda_global", "Niters", "conv_tol")
    array_keys = ("dc_costs", "reg_costs", "lambdas")
    out = {}
    with h5py.File(path, "r") as f:
        for key in scalar_keys:
            out[key] = f[key][()].item()
        for key in array_keys:
            out[key] = np.asarray(f[key][()]).ravel()
        out["mom"] = f["mom"][()].tobytes().decode("utf-16-le").rstrip("\x00")
        raw = f["X_recon"][()]
        out["X_recon"] = (raw["real"] + 1j * raw["imag"]).astype(np.complex64).transpose()
        out["patch_sizes"] = [tuple(int(v) for v in f[ref][()]) for ref in f["patch_sizes"][()]]
        out["strides"] = [tuple(int(v) for v in f[ref][()]) for ref in f["strides"][()]]
    return out


def save_result(
    fn_base: str, result: "ReconResult", fov: tuple[float, float, float], **extra_attrs
) -> None:
    """Write <fn_base>.h5 (complex image, per-scale components, sampling mask,
    cost traces) and <fn_base>.nii.gz + .json (magnitude image, settings).
    The .h5 is written first, so a finished run is saved even if the rest fails."""
    X_recon_np = result.X_recon.detach().cpu().numpy()
    with h5py.File(f"{fn_base}.h5", "w") as f:
        f.create_dataset("X_recon", data=X_recon_np)
        f.create_dataset("X", data=result.X.detach().cpu().numpy())
        f.create_dataset("omega", data=result.omega.detach().cpu().numpy())
        f.create_dataset("dc_costs", data=np.asarray(result.dc_costs))
        f.create_dataset("reg_costs", data=np.asarray(result.reg_costs))
        f.create_dataset("restarts", data=np.asarray(result.restarts, dtype=bool))
        f.create_dataset("rel_changes", data=np.asarray(result.rel_changes))
        f.attrs["R"] = result.R
        f.attrs["sigma1A"] = result.sigma1A
        f.attrs["L"] = result.L
        f.attrs["runtime_s"] = result.runtime_s
    print(f"Wrote {fn_base}.h5")

    save_recon_nifti(
        fn_base,
        X_recon_np,
        fov=fov,
        R=result.R,
        sigma1A=result.sigma1A,
        L=result.L,
        lambdas=result.lambdas,
        runtime_s=result.runtime_s,
        n_iters=len(result.dc_costs) - 1,
        final_dc_cost=result.dc_costs[-1],
        final_reg_cost=result.reg_costs[-1] if result.reg_costs else None,
        **result.meta,
        **extra_attrs,
    )
    print(f"Wrote {fn_base}.nii.gz + .json")


# ---------------------------------------------------------------- operator norms


def poweriter(
    apply_fwd: Callable[[torch.Tensor], torch.Tensor],
    apply_adj: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    niter: int = 200,
    tol: float = 1e-6,
) -> float:
    """Spectral norm ||A||_2 by power iteration on A^H A, stopping once the
    estimate changes by less than tol (it converges from below)."""
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


def estimate_spectral_norm(A, x0: torch.Tensor, niter: int = 200, tol: float = 1e-6) -> float:
    """sigma1(A) by power iteration. For a BlockDiagonal (one block per frame)
    it is the max over the blocks, which keeps buffers at one frame's size.
    x0: nonzero start of one block's input shape (*N,)."""
    if isinstance(A, BlockDiagonal):
        return max(poweriter(block.apply, block.adjoint, x0, niter=niter, tol=tol) for block in A.A)
    return poweriter(A.apply, A.adjoint, x0, niter=niter, tol=tol)


def check_operator_unitary(
    A, x0: torch.Tensor, name: str = 'A', tol: float = 0.05, niter: int = 200,
    poweriter_tol: float = 1e-6,
) -> float:
    """Measure sigma1(A) and warn if it's not ~1. Regularization weights and
    step sizes tuned for a unitary operator don't transfer to one that isn't
    (e.g. the B0 operators, sigma1 ~1.2-1.9). Returns sigma1."""
    sigma1 = estimate_spectral_norm(A, x0, niter=niter, tol=poweriter_tol)
    if abs(sigma1 - 1.0) > tol:
        warnings.warn(
            f"check_operator_unitary: {name}'s sigma1(A) = {sigma1:.4f}, not close to 1.0 "
            f"(tol={tol}) -- this operator is not unitary. Any regularization weight "
            "(lambda_l1/lambda_tv/lambda_global/...) or step size tuned assuming a unitary "
            "operator (e.g. L1-wavelet_TV_B0_SENSE.py's lamb_l1=lamb_tv=0.005 default, tuned against a "
            "genuinely unitary sigpy.mri.linop.Sense) will not transfer directly -- expect to "
            "re-tune, or explicitly normalize the operator/data by sigma1(A) first.",
            stacklevel=2,
        )
    return sigma1


# ---------------------------------------------------------------- tSNR


def object_mask(img: np.ndarray, thresh_frac: float = 0.2) -> np.ndarray:
    """(Nx,Ny,Nz) bool: time-mean magnitude above thresh_frac of its maximum."""
    mean_img = img.mean(axis=-1)
    return mean_img > thresh_frac * mean_img.max()


def temporal_stability(img: np.ndarray, mask: np.ndarray, tr_s: float) -> dict:
    """img: [Nx, Ny, Nz, Nframes] magnitude. mask: [Nx, Ny, Nz] bool.

    Returns the standard phantom-stability measures:
    - tsnr_map: [Nx, Ny, Nz], mean/std over time (NaN outside mask)
    - roi_signal: [Nframes], spatial mean over mask per frame
    - percent_fluctuation: 100 * std(residual after linear detrend) / mean(roi_signal)
    - percent_drift: 100 * (linear fit endpoint - start) / mean(roi_signal)
    - roi_tsnr: mean(roi_signal) / std(roi_signal), without detrending
    """
    Nframes = img.shape[-1]
    t = np.arange(Nframes)

    voxels = img[mask]  # [Nvox, Nframes]
    mean_t = voxels.mean(axis=-1)
    std_t = voxels.std(axis=-1, ddof=1)
    tsnr_map = np.full(mask.shape, np.nan, dtype=np.float64)
    tsnr_map[mask] = np.divide(mean_t, std_t, out=np.zeros_like(mean_t), where=std_t > 0)

    roi_signal = img[mask].mean(axis=0)  # [Nframes]

    coeffs = np.polyfit(t, roi_signal, deg=1)
    fit = np.polyval(coeffs, t)
    residual = roi_signal - fit
    grand_mean = roi_signal.mean()

    percent_fluctuation = 100 * residual.std(ddof=1) / grand_mean
    percent_drift = 100 * (fit[-1] - fit[0]) / grand_mean
    roi_tsnr = grand_mean / roi_signal.std(ddof=1)

    return dict(
        tsnr_map=tsnr_map,
        roi_signal=roi_signal,
        time_s=t * tr_s,
        linear_fit=fit,
        percent_fluctuation=percent_fluctuation,
        percent_drift=percent_drift,
        roi_tsnr=roi_tsnr,
        median_voxel_tsnr=np.nanmedian(tsnr_map),
    )


def tsnr_report(
    fn_recons: list[str], tr_s: float, skip_frames: int = 0, thresh_frac: float = 0.2
) -> dict:
    """Temporal-stability report for one or more reconstructions.

    fn_recons: .nii.gz magnitude images [Nx,Ny,Nz,Nframes] as written by
    rss.py/sense.py. Prints the stability numbers for each and saves one
    figure (ROI signal + linear fit, central-slice tSNR map, last-minus-first
    frame; one column per input) next to the first input. Returns
    {label: stats}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    results = {}
    for fn in fn_recons:
        label = os.path.basename(fn).removesuffix(".nii.gz")
        img = np.asarray(nib.load(fn).dataobj)[..., skip_frames:]
        mask = object_mask(img, thresh_frac)
        stats = temporal_stability(img, mask, tr_s)
        results[label] = (img, mask, stats)
        print(f"\n=== {label} ===")
        print(f"  object mask: {mask.sum()} voxels ({100 * mask.sum() / mask.size:.1f}% of volume)")
        print(f"  percent fluctuation (detrended): {stats['percent_fluctuation']:.3f}%")
        span_s = (img.shape[-1] - 1) * tr_s
        print(f"  percent drift (linear, over {span_s:.0f}s): {stats['percent_drift']:.3f}%")
        print(f"  ROI tSNR (undetrended): {stats['roi_tsnr']:.1f}")
        print(f"  median per-voxel tSNR: {stats['median_voxel_tsnr']:.1f}")

    out_base = fn_recons[0].removesuffix(".nii.gz") + "_tsnr"
    labels = list(results)
    fig, axes = plt.subplots(3, len(labels), figsize=(5 * len(labels), 13), squeeze=False)
    for j, label in enumerate(labels):
        img, mask, stats = results[label]
        iz = img.shape[2] // 2
        ax = axes[0, j]
        ax.plot(stats["time_s"], stats["roi_signal"], "o-", label="ROI mean signal")
        ax.plot(stats["time_s"], stats["linear_fit"], "--", label="linear fit")
        ax.set_xlabel("time (s)")
        ax.set_title(f"{label}\nfluct={stats['percent_fluctuation']:.2f}%, "
                     f"drift={stats['percent_drift']:.2f}%, tSNR={stats['roi_tsnr']:.0f}")
        ax.legend()
        tsnr_slice = stats["tsnr_map"][:, :, iz].T
        im = axes[1, j].imshow(tsnr_slice, origin="lower", cmap="viridis", vmin=0)
        axes[1, j].set_title(f"tSNR, z={iz}")
        plt.colorbar(im, ax=axes[1, j], fraction=0.046)
        diff = img[:, :, iz, -1] - img[:, :, iz, 0]
        vmax = np.percentile(np.abs(diff[mask[:, :, iz]]), 99) if mask[:, :, iz].any() else 1
        im = axes[2, j].imshow(diff.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[2, j].set_title(f"last minus first frame, z={iz}")
        plt.colorbar(im, ax=axes[2, j], fraction=0.046)
        for ax in axes[1:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{out_base}.png", dpi=130)
    plt.close(fig)
    print(f"Wrote {out_base}.png")
    return {label: stats for label, (_, _, stats) in results.items()}


def _cli_tsnr() -> None:
    parser = argparse.ArgumentParser(description=tsnr_report.__doc__)
    parser.add_argument("fn_recons", nargs="+", help="recon .nii.gz file(s)")
    parser.add_argument("--tr", type=float, required=True, help="volume TR in seconds")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="drop this many leading frames (non-steady-state transient)")
    args = parser.parse_args()
    tsnr_report(args.fn_recons, args.tr, args.skip_frames)


# ---------------------------------------------------------------- analysis


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Real acquisition scale for the B0 studies below.
ETL = 60
DT_ECHO_S = 0.0012
B0_MIN_HZ, B0_MAX_HZ = -300.0, 70.0
TE_S = 0.030  # nominal TE the echo train is centered on; only shifts all t_per_ky uniformly
NBINS = 128  # production default


def _complex_randn(*shape, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    real = torch.randn(*shape, generator=g, device=DEVICE)
    imag = torch.randn(*shape, generator=g, device=DEVICE)
    return (real + 1j * imag).to(torch.complex64)


def _brute_force_time_varying_ksp(
    img: torch.Tensor, smaps: torch.Tensor, b0map_hz: torch.Tensor, t_per_ky: torch.Tensor
) -> torch.Tensor:
    """Exact time-varying B0 forward model (no segmentation): one full FFT per
    ky row with that row's phase, keeping only that row. (Nc,Nx,Ny,Nz)."""
    Nc, Nx, Ny, Nz = smaps.shape
    dims = (1, 2, 3)
    y_true = torch.zeros(Nc, Nx, Ny, Nz, dtype=torch.complex64, device=DEVICE)
    for iy in range(Ny):
        angle = (2 * math.pi * t_per_ky[iy].item()) * b0map_hz
        phasor = torch.exp(1j * angle).to(torch.complex64)
        demod = img * smaps * phasor
        k_full = torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(demod, dim=dims), dim=dims, norm="ortho"), dim=dims
        )
        y_true[:, :, iy, :] = k_full[:, :, iy, :]
    return y_true


def _setup_real_scale(seed: int = 100):
    """Small synthetic problem at the real echo-train length and field-map range,
    with its exact k-space. Returns (img, smaps, b0map_hz, t_per_ky, y_true_flat)."""
    Nx, Ny, Nz, Nc = 16, ETL, 8, 4  # Ny=ETL matches the real echo train exactly
    img = _complex_randn(Nx, Ny, Nz, seed=seed)
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=seed + 1)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-6)

    # Smooth field map over the real range, with a little curvature.
    yy = torch.linspace(0, 1, Ny, device=DEVICE).reshape(1, Ny, 1)
    zz = torch.linspace(-1, 1, Nz, device=DEVICE).reshape(1, 1, Nz)
    b0map_hz = (B0_MIN_HZ + (B0_MAX_HZ - B0_MIN_HZ) * yy + 15.0 * zz**2).expand(Nx, Ny, Nz)
    b0map_hz = b0map_hz.contiguous().clamp(B0_MIN_HZ, B0_MAX_HZ + 15.0)

    t_per_ky = TE_S + (torch.arange(Ny, device=DEVICE, dtype=torch.float32) - (Ny - 1) / 2) * DT_ECHO_S

    y_true = _brute_force_time_varying_ksp(img, smaps, b0map_hz, t_per_ky)
    y_true_flat = y_true.reshape(Nc, -1).T  # (K,Nc), C-order -- matches SENSE's own flatten
    return img, smaps, b0map_hz, t_per_ky, y_true_flat


def _build_operator(smaps: torch.Tensor, b0map_hz: torch.Tensor, t_frame_s: torch.Tensor, L: int, nbins: int):
    """Fully sampled single-frame SENSE_B0 with the segmentation fit on every
    sample's time."""
    Nx, Ny, Nz = smaps.shape[1:]
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)
    idx = torch.nonzero(full_mask.reshape(-1), as_tuple=False).squeeze(-1)
    t_ms = (t_frame_s.reshape(-1)[idx] * 1000).to(torch.float32)
    b0_neg = (-b0map_hz).to(torch.float32)  # sign convention: see operators.py
    b, c, _tl = mri_exp_approx(b0_neg, nbins, L, t_ms)
    N = (Nx, Ny, Nz)
    c = c.transpose(0, 1).reshape((L,) + N).to(smaps.dtype)
    pos = torch.arange(b.shape[0], device=b.device)  # identity gather (see SENSE_B0)
    return SENSE_B0(smaps, full_mask, pos, b.to(smaps.dtype), c)


def sweep(L_values: list[int], nbins: int = NBINS):
    """Forward-model error of SENSE_B0 vs the exact time-varying model, per L.
    Returns (error without B0 correction, [(L, error), ...])."""
    img, smaps, b0map_hz, t_per_ky, y_true_flat = _setup_real_scale()
    Nx, Ny, Nz = smaps.shape[1:]
    t_frame = t_per_ky.reshape(1, Ny, 1).expand(Nx, Ny, Nz).contiguous()
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)

    err_uncorrected = (
        (SENSE(smaps, full_mask).apply(img) - y_true_flat).norm().item()
        / y_true_flat.norm().item()
    )

    results = []
    for L in L_values:
        A = _build_operator(smaps, b0map_hz, t_frame, L, nbins)
        y_hat = A.apply(img)
        err = (y_hat - y_true_flat).norm().item() / y_true_flat.norm().item()
        results.append((L, err))
    return err_uncorrected, results


def _cli_sweep() -> None:
    L_values = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 56, 60]
    err_uncorrected, results = sweep(L_values)

    print(f"ETL={ETL}, dt_echo={DT_ECHO_S * 1000:.2f} ms, "
          f"b0 range=[{B0_MIN_HZ:.0f}, {B0_MAX_HZ:.0f}] Hz, nbins={NBINS}")
    print(f"BT (bandwidth-time product) = {(B0_MAX_HZ - B0_MIN_HZ) * ETL * DT_ECHO_S:.1f}\n")
    print(f"uncorrected (no B0 correction): rel_error = {err_uncorrected:.4f}\n")

    err6 = dict(results).get(6)
    print(f"{'L':>4} {'rel_error':>10} {'vs uncorr':>10} {'vs L=6':>10}")
    for L, err in results:
        vs_uncorr = err / err_uncorrected
        vs_l6 = err / err6 if err6 else float("nan")
        marker = "  <- current default" if L == 6 else ""
        print(f"{L:>4} {err:>10.4f} {vs_uncorr:>9.2%} {vs_l6:>9.2%}{marker}")

    target = 0.01
    converged_L = next((L for L, err in results if err < target), None)
    print(f"\nSmallest swept L with rel_error < {target:.0%}: "
          f"{converged_L if converged_L is not None else 'none in this sweep'}")


# Real acquisition scale for the benchmark.
_BENCH_SHAPE = (240, 240, 45, 18, 30)  # Nx, Ny, Nz, Nc, Nt
_BENCH_R = 9


def _build_inputs():
    """Random operator inputs at _BENCH_SHAPE (cost depends only on shapes)."""
    Nx, Ny, Nz, Nc, Nt = _BENCH_SHAPE
    R = _BENCH_R
    torch.manual_seed(0)
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=0)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-6)

    # K = Nx*Ny*Nz/R random samples per frame.
    K = (Nx * Ny * Nz) // R
    omega = torch.zeros(Nx, Ny, Nz, Nt, dtype=torch.bool, device=DEVICE)
    for it in range(Nt):
        perm = torch.randperm(Nx * Ny * Nz, device=DEVICE)[:K]
        flat = torch.zeros(Nx * Ny * Nz, dtype=torch.bool, device=DEVICE)
        flat[perm] = True
        omega[..., it] = flat.reshape(Nx, Ny, Nz)

    b0map_hz = (
        -300.0 + 370.0 * torch.linspace(0, 1, Ny, device=DEVICE).reshape(1, Ny, 1)
    ).expand(Nx, Ny, Nz).contiguous()

    # ETL distinct echo times, the same set in every frame (as build_sense_b0 requires).
    distinct_t_ms = torch.linspace(5.0, 5.0 + (ETL - 1) * 1.2, ETL, device=DEVICE)
    yz_idx = (
        torch.arange(Ny, device=DEVICE).reshape(Ny, 1) * Nz
        + torch.arange(Nz, device=DEVICE).reshape(1, Nz)
    ) % ETL
    t_yz_s = (distinct_t_ms[yz_idx] / 1000.0)  # (Ny,Nz)
    echo_times_s = t_yz_s.reshape(1, Ny, Nz, 1).expand(Nx, Ny, Nz, Nt).contiguous()

    return smaps, omega, b0map_hz, echo_times_s, K


def _time_forward_adjoint(A, x0, y0):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = A.apply(x0)
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = A.adjoint(y0)
    torch.cuda.synchronize()
    t_adj = time.perf_counter() - t0
    return t_fwd, t_adj


def benchmark(L_values: list[int]):
    """Time and GPU memory of one forward + adjoint of SENSE and of SENSE_B0 per L."""
    assert DEVICE == "cuda", "this benchmark is only meaningful on GPU"
    Nx, Ny, Nz, Nc, Nt = _BENCH_SHAPE
    smaps, omega, b0map_hz, echo_times_s, K = _build_inputs()
    x0 = _complex_randn(Nx, Ny, Nz, Nt, seed=1)
    y0 = _complex_randn(K, Nc, Nt, seed=2)

    print(f"scale: Nx,Ny,Nz,Nc,Nt={Nx},{Ny},{Nz},{Nc},{Nt}  K/frame={K}  ETL={ETL}  nbins={NBINS}\n")

    # Baseline: plain SENSE
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    A0 = build_sense(smaps, omega)
    t_fwd, t_adj = _time_forward_adjoint(A0, x0, y0)
    peak0 = torch.cuda.max_memory_allocated() / 1e9
    del A0
    torch.cuda.empty_cache()
    print(f"{'config':>18} {'build_mem_GB':>13} {'peak_apply_GB':>14} {'fwd_s':>8} {'adj_s':>8} {'fwd+adj_s':>10}")
    print(f"{'uncorrected':>18} {'-':>13} {peak0:>14.2f} {t_fwd:>8.3f} {t_adj:>8.3f} {t_fwd + t_adj:>10.3f}")

    results = []
    for L in L_values:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1e9
        A = build_sense_b0(smaps, omega, b0map_hz, echo_times_s, L=L, nbins=NBINS)
        torch.cuda.synchronize()
        mem_after_build = torch.cuda.memory_allocated() / 1e9
        build_mem = mem_after_build - mem_before

        torch.cuda.reset_peak_memory_stats()
        t_fwd, t_adj = _time_forward_adjoint(A, x0, y0)
        peak_apply = torch.cuda.max_memory_allocated() / 1e9

        print(f"{'L=' + str(L):>18} {build_mem:>13.2f} {peak_apply:>14.2f} {t_fwd:>8.3f} {t_adj:>8.3f} {t_fwd + t_adj:>10.3f}")
        results.append((L, build_mem, peak_apply, t_fwd, t_adj))
        del A
        torch.cuda.empty_cache()

    return (t_fwd, t_adj), results


def _cli_benchmark() -> None:
    L_values = [1, 6, 32, 60]
    benchmark(L_values)


def validate(fn_ksp: str, fn_smaps: str, fn_julia_mat: str) -> bool:
    """Re-run a ../mslr-recon (Julia) reconstruction with its own settings and
    compare costs, iteration count and image. Returns True if all checks pass."""
    from recon.sense import run_sense  # lazy: sense.py imports this module

    ref = read_julia_mat(fn_julia_mat)

    result = run_sense(
        reg="lowrank",
        normalize_operator=False,  # Julia uses A as is
        fn_ksp=fn_ksp,
        fn_smaps=fn_smaps,
        patch_sizes=ref["patch_sizes"],
        strides=ref["strides"],
        niters=int(ref["Niters"]),
        sigma1A=float(ref["sigma1A"]),
        mom=ref["mom"],
        conv_tol=float(ref["conv_tol"]),
        lambda_global=float(ref["lambda_global"]),
    )

    ok = True

    def check(name, got, want, rtol=1e-4):
        nonlocal ok
        rel = abs(got - want) / (abs(want) + 1e-30)
        status = "OK  " if rel <= rtol else "FAIL"
        ok &= rel <= rtol
        print(f"{status} {name}: python={got} julia={want} rel_diff={rel:.3e}")

    check("R", result.R, float(ref["R"]))
    check("sigma1A", result.sigma1A, float(ref["sigma1A"]))
    check("L", result.L, float(ref["L"]))
    check("n_iters (incl iter0)", len(result.dc_costs), len(ref["dc_costs"]), rtol=0)

    n = min(len(result.dc_costs), len(ref["dc_costs"]))
    dc_p, dc_j = np.array(result.dc_costs[:n]), np.array(ref["dc_costs"][:n])
    reg_p, reg_j = np.array(result.reg_costs[:n]), np.array(ref["reg_costs"][:n])
    check("dc_cost[-1]", dc_p[-1], dc_j[-1])
    # Multi-scale nuclear norms accumulate more float32 noise (~1-2e-4 measured).
    check("reg_cost[-1]", reg_p[-1], reg_j[-1], rtol=5e-4)
    dc_max_rel = float((np.abs(dc_p - dc_j) / np.abs(dc_j)).max())
    reg_max_rel = float((np.abs(reg_p - reg_j) / np.abs(reg_j)).max())
    print(f"     dc_cost max rel diff over all iters: {dc_max_rel:.3e}")
    print(f"     reg_cost max rel diff over all iters: {reg_max_rel:.3e}")
    ok &= dc_max_rel < 1e-3 and reg_max_rel < 1e-2

    Xj, Xp = ref["X_recon"], result.X_recon.cpu().numpy()
    rel_l2 = float(np.linalg.norm(Xj - Xp) / np.linalg.norm(Xj))
    pearson_r = float(np.corrcoef(np.abs(Xj).ravel(), np.abs(Xp).ravel())[0, 1])
    print(f"     X_recon relative L2 error: {rel_l2:.3e}")
    print(f"     X_recon Pearson r (magnitude): {pearson_r:.10f}")
    ok &= rel_l2 < 1e-2 and pearson_r > 0.999

    return ok


def _cli_validate() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fn_julia_mat")
    parser.add_argument("--ksp", default=None)
    parser.add_argument("--smaps", default=None)
    args = parser.parse_args()

    # <recon_dir>/mslr/<subdir>/<name>.mat -> <recon_dir>/ArbEPI_epi_zf.h5
    recon_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.fn_julia_mat)))
    fn_ksp = args.ksp or os.path.join(recon_dir, "ArbEPI_epi_zf.h5")
    fn_smaps = args.smaps or os.path.join(recon_dir, "smaps_ArbEPI_sigpy.h5")

    ok = validate(fn_ksp, fn_smaps, args.fn_julia_mat)
    sys.exit(0 if ok else 1)


_COMMANDS = {
    'tsnr': _cli_tsnr,
    'sweep': _cli_sweep,
    'benchmark': _cli_benchmark,
    'validate': _cli_validate,
}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        sys.exit('usage: python -m recon.utils {' + ','.join(_COMMANDS) + '} ...')
    _cmd = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {_cmd}'
    _COMMANDS[_cmd]()
