"""One-off analysis: measure the actual compute-time and GPU-memory cost of
recon/operators_b0.py's GatheredSenseB0/build_encoding_operator_b0 as a
function of the time-segmentation count L, at this repo's REAL acquisition
scale (Nx,Ny,Nz,Nc,Nt = 240,240,45,18,30, R~9 -- see CLAUDE.md's recon/
section) -- not extrapolated from the smaller Julia/Python benchmark numbers
already documented there, which were measured for the *uncorrected*
GatheredSense operator only.

Uses synthetic (random) smaps/omega/b0map_hz/echo_times -- cost here depends
only on tensor shapes (FFT size, sample count K, L), not on real scan
content, matching recon/sweep_time_segments.py's own reasoning for using
synthetic data.

Reports, for each swept L:
  - wall-clock time for one full forward (A.apply) + adjoint (A.adjoint)
    call over all 30 frames (one POGM gradient-step-equivalent)
  - peak GPU memory during that call (torch.cuda.max_memory_allocated)
  - the static, L-independent memory cost of building the operator itself
    (b_weights across all frames + the shared c_phasors tensor)

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.benchmark_b0_cost
"""

import time

import torch

from recon.operators import GatheredSense, build_encoding_operator
from recon.operators_b0 import build_encoding_operator_b0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "this benchmark is only meaningful on GPU"

# Real scale, from CLAUDE.md's recon/ section.
Nx, Ny, Nz, Nc, Nt = 240, 240, 45, 18, 30
R = 9
ETL = 60
NBINS = 128


def _complex_randn(*shape, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    real = torch.randn(*shape, generator=g, device=DEVICE)
    imag = torch.randn(*shape, generator=g, device=DEVICE)
    return (real + 1j * imag).to(torch.complex64)


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
    y = A.apply(x0)
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


if __name__ == "__main__":
    L_values = [1, 6, 32, 60]
    benchmark(L_values)
