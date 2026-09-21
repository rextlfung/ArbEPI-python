"""One-off analysis and validation scripts (.venv-recon), not part of the production
path. Subcommands:

    sweep      time-segmentation count L accuracy sweep (produced the L=32 choice)
    benchmark  forward+adjoint cost vs L
    validate   field-by-field comparison against real ../mslr-recon (Julia) output

    .venv-recon/bin/python -m recon.analysis {sweep,benchmark,validate} ...

The sections below are the original module docstrings, kept verbatim.

Formerly recon/analysis/sweep_time_segments.py
----------------------------------------------
One-off analysis: sweep the time-segmentation count L (recon/operators.py)
against a synthetic ground truth scaled to this repo's REAL echo-train length
(ETL=60, ~1.2ms/echo -> ~72ms readout window) and REAL field-map range
(-300 to +70 Hz -- both numbers from operators.py's own module docstring)
-- not the 12-distinct-echo-time toy grid
tests/test_recon_operators_b0.py's test_more_segments_reduces_error_in_the_
realistic_regime uses. That test's own "L=16 is ~exact" finding is an
artifact of its toy grid having only 12 distinct echo times (L>=12 trivially
resolves every one exactly); it says nothing about whether L=6 (the current
production default, params.py-adjacent choice in operators.py/
run_recon.py) is adequate at the real ETL=60 scale, where there are up to
60 distinct echo times spanning a much larger bandwidth-time product
(BT = Delta_f_range * T_readout ~= 370 Hz * 0.072 s ~= 27).

Ground-truth construction mirrors tests/test_recon_b0_correction.py's
_brute_force_time_varying_ksp (brute-force, genuinely time-varying, one
dense FFT per echo/ky -- not a shortcut that could hide a segmentation
error) and tests/test_recon_operators_b0.py's mri_exp_approx-based operator
construction, reimplemented here (not imported from tests/) so this stays a
standalone recon/ analysis script, not a test-suite dependency. Nx/Nz/Nc are
kept small for speed -- this is a synthetic forward-model sweep, not a real
reconstruction -- only Ny=ETL and the field-map range/echo spacing need to
match real values, since the per-echo off-resonance phase model here only
depends on t_per_ky (one time per ky row, matching sequences/ArbEPI.py's
echo_times) and b0map_hz's spatial values, not on Nx/Nz/Nc.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.analysis sweep


Formerly recon/analysis/benchmark_b0_cost.py
--------------------------------------------
One-off analysis: measure the actual compute-time and GPU-memory cost of
recon/operators.py's GatheredSenseB0/build_encoding_operator_b0 as a
function of the time-segmentation count L, at this repo's REAL acquisition
scale (Nx,Ny,Nz,Nc,Nt = 240,240,45,18,30, R~9 -- see CLAUDE.md's recon/
section) -- not extrapolated from the smaller Julia/Python benchmark numbers
already documented there, which were measured for the *uncorrected*
GatheredSense operator only.

Uses synthetic (random) smaps/omega/b0map_hz/echo_times -- cost here depends
only on tensor shapes (FFT size, sample count K, L), not on real scan
content, matching recon/analysis.py's own reasoning for using
synthetic data.

Reports, for each swept L:
  - wall-clock time for one full forward (A.apply) + adjoint (A.adjoint)
    call over all 30 frames (one POGM gradient-step-equivalent)
  - peak GPU memory during that call (torch.cuda.max_memory_allocated)
  - the memory cost of building the operator itself -- dominated by the
    shared `(L,*N)` c_phasors tensor, which scales *linearly* with L (664
    MB at L=32); the per-frame `pos` index arrays (int64, one per frame)
    are the one genuinely L-independent piece, and small (69 MB total at
    this repo's real 30-frame/288000-sample-per-frame scale -- see
    recon/operators.py's GatheredSenseB0 docstring and
    docs/review-findings.md item 75, which this build_mem column measures
    the fix for)

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.analysis benchmark


Formerly recon/analysis/validate_against_mslr.py
------------------------------------------------
Validate recon/reconstruct.py against real ../mslr-recon (Julia/MIRT.jl)
output, field by field. Not a pytest test -- like seq2ge/validate_against_
matlab.py, this depends on real reference output that isn't committed to
this repo (machine-specific acquisition data + a completed mslr-recon run).

Usage (from repo root, using the .venv-recon environment):
    .venv-recon/bin/python -m recon.analysis validate <julia_reconstruct.mat>

The reference .mat is produced by ../mslr-recon's scripts/reconstruct.jl
(e.g. via experiments/20260822ball.jl -- run_recon's own `matwrite` call).
Every reconstruction parameter (fn_ksp/fn_smaps by convention -- see below,
patch_sizes, strides, sigma1A, lambda_global, conv_tol, niters, mom) is read
directly from the reference file rather than re-specified, so this always
replicates exactly what the reference run used, and fn_ksp/fn_smaps are
derived from the reference .mat's own directory layout
(<recon_dir>/mslr/<subdir>/<name>.mat -> <recon_dir>/ArbEPI_epi_zf.h5 +
<recon_dir>/smaps_ArbEPI_sigpy.h5 -- matching experiments/20260822ball.jl's
own `datasets` table), unless overridden with --ksp/--smaps.

Validated results (2026-08-25, RTX A6000, 20260822ball_radial dataset,
Nx,Ny,Nz,Nvc,Nt=240,240,45,18,30, R~9):

  config  iters  dc reldiff  reg reldiff  X_recon reldiff  Pearson r     py/julia s
  L       55     5.9e-7      3.0e-6       1.6e-5           0.9999999998  309/405
  G       56     1.6e-6      8.1e-6       3.8e-5           0.9999999989  96/134
  G+L     101    1.6e-6      2.1e-4       2.1e-5           0.9999999997  597/785

All three configs converge to the same iteration count as the Julia run
(confirming pogm_restart's early-stopping logic matches exactly) and match
to float32 summation-order noise -- the same class of ~1-ULP difference this
repo's seq2ge/ port already documents against real MATLAB output. Python
also runs consistently faster despite a simpler (fully-batched, not
hand-tuned) SVD/FFT implementation.
"""

import argparse
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
from mirtorch.linear.mri import mri_exp_approx

from recon.operators import (
    GatheredSense,
    GatheredSenseB0,
    build_encoding_operator,
    build_encoding_operator_b0,
)
from recon.reconstruct import run_recon

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Real values, both taken directly from operators.py's module docstring.
ETL = 60
DT_ECHO_S = 0.0012
B0_MIN_HZ, B0_MAX_HZ = -300.0, 70.0
TE_S = 0.030  # nominal TE the echo train is centered on; only shifts all t_per_ky uniformly
NBINS = 128  # matches operators.py/run_b0_recon.py's production default


def _complex_randn(*shape, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    real = torch.randn(*shape, generator=g, device=DEVICE)
    imag = torch.randn(*shape, generator=g, device=DEVICE)
    return (real + 1j * imag).to(torch.complex64)


def _brute_force_time_varying_ksp(
    img: torch.Tensor, smaps: torch.Tensor, b0map_hz: torch.Tensor, t_per_ky: torch.Tensor
) -> torch.Tensor:
    """Same construction as tests/test_recon_b0_correction.py's helper of the
    same name -- one dense 3D FFT per ky, keeping only that ky's slice, so
    the "ground truth" is genuinely time-varying rather than assembled from
    any segmented approximation."""
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
    Nx, Ny, Nz, Nc = 16, ETL, 8, 4  # Ny=ETL matches the real echo train exactly
    img = _complex_randn(Nx, Ny, Nz, seed=seed)
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=seed + 1)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-6)

    # A smooth field map spanning exactly the documented real range, plus a
    # little curvature (bounded so the range stays close to documented) --
    # smooth and structured, like a real B0 map, not white noise.
    yy = torch.linspace(0, 1, Ny, device=DEVICE).reshape(1, Ny, 1)
    zz = torch.linspace(-1, 1, Nz, device=DEVICE).reshape(1, 1, Nz)
    b0map_hz = (B0_MIN_HZ + (B0_MAX_HZ - B0_MIN_HZ) * yy + 15.0 * zz**2).expand(Nx, Ny, Nz)
    b0map_hz = b0map_hz.contiguous().clamp(B0_MIN_HZ, B0_MAX_HZ + 15.0)

    t_per_ky = TE_S + (torch.arange(Ny, device=DEVICE, dtype=torch.float32) - (Ny - 1) / 2) * DT_ECHO_S

    y_true = _brute_force_time_varying_ksp(img, smaps, b0map_hz, t_per_ky)
    y_true_flat = y_true.reshape(Nc, -1).T  # (K,Nc), C-order -- matches GatheredSense's own flatten
    return img, smaps, b0map_hz, t_per_ky, y_true_flat


def _build_operator(smaps: torch.Tensor, b0map_hz: torch.Tensor, t_frame_s: torch.Tensor, L: int, nbins: int):
    """Same construction as tests/test_recon_operators_b0.py's
    _build_b0_operator, reimplemented here to avoid a recon/ -> tests/
    import."""
    Nx, Ny, Nz = smaps.shape[1:]
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)
    idx = torch.nonzero(full_mask.reshape(-1), as_tuple=False).squeeze(-1)
    t_ms = (t_frame_s.reshape(-1)[idx] * 1000).to(torch.float32)
    b0_neg = (-b0map_hz).to(torch.float32)  # sign convention, see operators.py's module docstring (static-stage section)
    b, c, _tl = mri_exp_approx(b0_neg, nbins, L, t_ms)
    N = (Nx, Ny, Nz)
    c = c.transpose(0, 1).reshape((L,) + N).to(smaps.dtype)
    pos = torch.arange(b.shape[0], device=b.device)  # identity gather (see GatheredSenseB0)
    return GatheredSenseB0(smaps, full_mask, pos, b.to(smaps.dtype), c)


def sweep(L_values: list[int], nbins: int = NBINS):
    img, smaps, b0map_hz, t_per_ky, y_true_flat = _setup_real_scale()
    Nx, Ny, Nz = smaps.shape[1:]
    t_frame = t_per_ky.reshape(1, Ny, 1).expand(Nx, Ny, Nz).contiguous()
    full_mask = torch.ones(Nx, Ny, Nz, dtype=torch.bool, device=DEVICE)

    err_uncorrected = (
        (GatheredSense(smaps, full_mask).apply(img) - y_true_flat).norm().item()
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

assert DEVICE == "cuda", "this benchmark is only meaningful on GPU"

# Real scale, from CLAUDE.md's recon/ section.
Nx, Ny, Nz, Nc, Nt = 240, 240, 45, 18, 30
R = 9


def _build_inputs():
    torch.manual_seed(0)
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=0)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-6)

    # Same K per frame, K ~= Nx*Ny*Nz/R, built by keeping the first K flat
    # indices of a random permutation shared in *structure* (not values)
    # across frames -- exact sample locations don't matter for cost, only K.
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

    # ETL distinct echo times, tied to (iy,iz) mod ETL so every frame's
    # sampled times are a subset of the fixed pool -- matches
    # build_encoding_operator_b0's frame-invariant-timing assumption.
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
    _ = A.apply(x0)  # timed for wall-clock cost, result unused -- same as the adjoint call below
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = A.adjoint(y0)
    torch.cuda.synchronize()
    t_adj = time.perf_counter() - t0
    return t_fwd, t_adj


def benchmark(L_values: list[int]):
    smaps, omega, b0map_hz, echo_times_s, K = _build_inputs()
    x0 = _complex_randn(Nx, Ny, Nz, Nt, seed=1)
    y0 = _complex_randn(K, Nc, Nt, seed=2)

    print(f"scale: Nx,Ny,Nz,Nc,Nt={Nx},{Ny},{Nz},{Nc},{Nt}  K/frame={K}  ETL={ETL}  nbins={NBINS}\n")

    # Baseline: uncorrected GatheredSense (L=0 sentinel, no segmentation loop at all)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    A0 = build_encoding_operator(smaps, omega)
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
        A = build_encoding_operator_b0(smaps, omega, b0map_hz, echo_times_s, L=L, nbins=NBINS)
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

def _read_julia_mat(path: str) -> dict:
    """MAT.jl's matwrite output (v7.3, HDF5-backed): arrays are stored
    axis-reversed on disk like hdf5storage's Python-side v7.3 writer (see
    preprocessing/matio.py) -- reverse with .transpose(). Complex arrays are
    stored as a {real, imag} compound dtype rather than natively."""
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


def validate(fn_ksp: str, fn_smaps: str, fn_julia_mat: str) -> bool:
    ref = _read_julia_mat(fn_julia_mat)

    result = run_recon(
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
    # Multi-scale reg_cost (Nscales>1, summing nuclear norms across scales
    # including a giant whole-volume SVD) accumulates more floating-point
    # noise than a single-scale reg_cost -- measured ~1-2e-4 on both real
    # G+L runs (radial and laminar) vs ~1e-6 for single-scale L/G, so this
    # check alone gets a looser tolerance rather than loosening every check.
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

    # <recon_dir>/mslr/<subdir>/<name>.mat -> <recon_dir>/ArbEPI_epi_zf.h5,
    # matching experiments/20260822ball.jl's own `datasets` table.
    recon_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.fn_julia_mat)))
    fn_ksp = args.ksp or os.path.join(recon_dir, "ArbEPI_epi_zf.h5")
    fn_smaps = args.smaps or os.path.join(recon_dir, "smaps_ArbEPI_sigpy.h5")

    ok = validate(fn_ksp, fn_smaps, args.fn_julia_mat)
    sys.exit(0 if ok else 1)


_COMMANDS = {
    'sweep': _cli_sweep,
    'benchmark': _cli_benchmark,
    'validate': _cli_validate,
}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        sys.exit('usage: python -m recon.analysis {' + ','.join(_COMMANDS) + '} ...')
    _cmd = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {_cmd}'
    _COMMANDS[_cmd]()
