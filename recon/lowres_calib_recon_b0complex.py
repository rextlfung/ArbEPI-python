"""Generalized-complex-field-map variant of recon/lowres_calib_recon_b0.py:
same fully-sampled calibration region, same adjoint-only philosophy, but
the time-segmented correction now accounts for a COMPLEX field combining
off-resonance and T2* decay, generalizing the real-only Δf(r) (Hz) that
recon/operators_b0.py's mri_exp_approx wraps (confirmed by reading
mirtorch's actual source: it raises TypeError on a complex b0 input, so
this generalization can't be done by just passing it a complex array --
it needs its own spatial-basis construction, below).

The *physical* forward-model exponent is psi(r) = i*2*pi*Δf(r) - R2*(r)
(signal decays as time since excitation increases). The array this module
actually builds and feeds to GatheredSenseB0, psi_recon = i*2*pi*Δf(r) +
R2*(r) (plus sign), is deliberately NOT that -- see
_build_calib_operator_b0_complex's inline comment for why: this
reconstruction only ever calls .adjoint(), never .apply(), and
GatheredSenseB0._apply_adjoint always conjugates c_phasors. Conjugating a
real quantity is a no-op, so building c_phasors from the physical -R2*
would make the adjoint apply the same decay a second time instead of
undoing it (confirmed empirically -- an earlier version with the physical
sign measured tSNR getting monotonically *worse* through uncorrected ->
phase-only -> this complex-field version, the opposite of the expected
direction). The true multiplicative inverse of exp(psi*t) is exp(-psi*t),
which only equals exp(conj(psi)*t) when Re(psi)=0 (pure rotation, the
phase-only case this pipeline already validated) -- psi_recon is chosen
so that conjugating it reproduces that true inverse instead.

Motivation (see CLAUDE.md's recon/ "B0 off-resonance correction" section
for the full derivation): lowres_calib_recon_b0.py's phase-only correction
demodulates off-resonance but leaves T2*/T1 amplitude decay uncorrected --
a real, partial cause of temporal instability confirmed directly by
preprocessing/lowres_calib_t2star_check.py (a given (ky,kz) calibration
location is acquired at a different echo index, hence a different amount
of decay, in different frames). Generalizing Δf(r) to psi(r) corrects
both simultaneously with the same L-segment machinery.

Reference time = TE_nominal, not t=0 (excitation): both the magnitude
(R2*) and phase (Δf) corrections are the real and imaginary parts of the
SAME complex exponent exp(psi(r)*t), so they share one time reference by
construction -- shifting the per-sample times fed into the segmentation
fit by -TE_nominal makes the reconstruction target "the image as it would
appear at the prescribed TE" (the standard GRE/EPI T2*-weighted
convention, and the convention used for phase in QSM/SWI-style imaging)
rather than "the undecayed image at the moment of excitation," which is
neither a standard nor numerically favorable reference (voxels with short
T2* would need very large amplification factors relative to t=0's much
earlier lead-in interval). TE_nominal is read directly from scan_info.mat's
schedules[...,2] at echo index (ETL-1)//2 -- the nominal-TE echo,
frame/shot-invariant by construction (see CLAUDE.md's mask2epi_radial
paragraph) -- not re-derived from the calibration region's own (possibly
incomplete) echo-time coverage.

Segmentation strategy: reuses mri_exp_approx's existing, already-tuned
(L=32) temporal interpolation weights (b_by_echo) and segment-placement
times (tl) UNCHANGED -- computed from Δf(r) alone, exactly as the
phase-only version does, just against TE-shifted times. Only the SPATIAL
basis functions are regeneralized: instead of mri_exp_approx's own
phase-only exp(i*2*pi*Δf(r)*tl[l]), this builds exp(psi(r)*tl[l]) directly
from the full, continuous per-voxel complex field (not the histogram-
binned reference values mri_exp_approx uses internally only to fit
b_by_echo). This is deliberately not a from-scratch joint (Δf, R2*)
segmentation fit: Δf's bandwidth-time product is what drove L=32 (see
CLAUDE.md's sweep finding, ~27 at this pipeline's real ETL=60/72ms scale);
R2*'s own decay-time product (R2*_typical * echo-train duration) is
printed by this script's main() specifically to check that it's small by
comparison, which is what justifies reusing Δf-only-tuned interpolation
weights for R2* too rather than re-deriving a joint fit.

R2*(r) comes from preprocessing/r2star_map.py's two-point estimate on the
same dual-echo deGRE data already used for Δf(r) -- see that module's
docstring.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.lowres_calib_recon_b0complex <datdir> [--seqname ArbEPI] [--device cuda]
"""

import argparse
import math
import os

import h5py
import numpy as np
import torch
from mirtorch.linear import BlockDiagonal
from mirtorch.linear.mri import mri_exp_approx

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.grid_resize import resize_to_epi_grid
from preprocessing.matio import read_mat
from preprocessing.nifti_io import save_recon_nifti
from preprocessing.r2star_map import estimate_r2star_map_epi_grid
from recon.lowres_calib_recon_b0 import compute_calib_mask, native_calib_grid
from recon.operators_b0 import GatheredSenseB0, _check_b_weight_row_sums
from recon.reconstruct import _load_array


def nominal_te_s(scan_info_path: str, etl: int) -> float:
    """The prescribed-TE echo's acquisition time (seconds since RF
    excitation), frame/shot-invariant by construction -- see module
    docstring. Read directly from scan_info.mat rather than derived from
    the calibration region's own echo-time coverage, which can be an
    incomplete subset of the full ETL (see lowres_calib_recon_b0.py's
    _build_calib_operator_b0 docstring)."""
    raw = read_mat(scan_info_path, ['schedules'])['schedules']  # (Nframes,Nshots,ETL,3)
    return float(raw[0, 0, (etl - 1) // 2, 2])


def _build_calib_operator_b0_complex(
    smaps_chw: torch.Tensor,
    calib_omega: torch.Tensor,
    b0map_hz: torch.Tensor,
    r2star_map: torch.Tensor,
    echo_times_s_shifted: torch.Tensor,
    L: int = 32,
    nbins: int = 128,
) -> BlockDiagonal:
    """Same construction as lowres_calib_recon_b0.py's
    _build_calib_operator_b0, generalized to a complex field -- see module
    docstring for what changes (spatial basis only) and what doesn't
    (temporal interpolation weights, segment placement, both still fit
    from Δf(r) alone via an unmodified mri_exp_approx call)."""
    Nt = calib_omega.shape[-1]
    N = tuple(smaps_chw.shape[1:])
    device = smaps_chw.device
    b0_neg = (-b0map_hz).to(torch.float32)

    idx_list, t_ms_list = [], []
    for it in range(Nt):
        samp = calib_omega[..., it]
        idx = torch.nonzero(samp.reshape(-1), as_tuple=False).squeeze(-1)
        t_ms = (echo_times_s_shifted[..., it].reshape(-1)[idx] * 1000).to(torch.float32)
        idx_list.append(idx)
        t_ms_list.append(t_ms)
    unique_t_ms = torch.unique(torch.cat(t_ms_list), sorted=True)

    b_by_echo, _c_phase_only, tl = mri_exp_approx(b0_neg, nbins, L, unique_t_ms)
    _check_b_weight_row_sums(b_by_echo, "shared complex-field fit (union of calib-region echo times, TE-referenced)")

    # Generalized complex spatial basis -- see module docstring. Reduces
    # exactly to mri_exp_approx's own phase-only c_phasors when
    # r2star_map == 0 (matches recon/operators_b0.py's confirmed sign
    # convention: exp(i*2*pi*b0map_hz(r)*tl[l])).
    #
    # Sign note (+R2*, not -R2*): GatheredSenseB0._apply_adjoint always
    # applies c_phasors[l].conj() -- the correct *inverse* of a pure
    # rotation (|exp(i*theta)|=1, so conj == 1/(.)), which is why the
    # phase-only version works via adjoint alone. Conjugating a REAL
    # quantity is a no-op, though, so a naive exp(i*2*pi*Δf*t - R2**t)
    # here would have the adjoint apply the SAME decay attenuation a
    # second time instead of undoing it (confirmed: an earlier version of
    # this line with a minus sign measured tSNR getting *worse* through
    # uncorrected -> phase-only -> complex-field, monotonically). The true
    # multiplicative inverse of exp(psi*t) is exp(-psi*t), not
    # exp(conj(psi)*t) -- those only coincide when Re(psi)=0. Feeding
    # GatheredSenseB0 a c_phasors built from psi_recon = i*2*pi*Δf + R2*
    # (note the sign flip on R2* relative to the physical forward-model
    # exponent) makes ITS conjugate equal exp(-psi_physical*t), the actual
    # decay-compensating inverse -- at the cost of this array no longer
    # being a physically faithful forward operator if .apply() were ever
    # called on it (it isn't, here: this reconstruction only ever calls
    # .adjoint()).
    psi_recon = (
        1j * 2 * math.pi * b0map_hz.to(torch.complex64)
        + r2star_map.to(torch.complex64)
    )  # (*N) complex
    tl_c = tl.to(torch.complex64).to(device)
    c_phasors = torch.exp(tl_c[:, None, None, None] * psi_recon[None, ...])  # (L,*N)

    frames = []
    for it in range(Nt):
        samp = calib_omega[..., it]
        t_ms = t_ms_list[it]
        pos = torch.searchsorted(unique_t_ms, t_ms).clamp(max=unique_t_ms.numel() - 1)
        assert torch.allclose(unique_t_ms[pos], t_ms, atol=1e-4), (
            f"_build_calib_operator_b0_complex: frame {it}'s sample echo times aren't in "
            "the union of every frame's calib-region echo times -- shouldn't be reachable "
            "by construction."
        )
        b = b_by_echo[pos].to(smaps_chw.dtype)
        frames.append(GatheredSenseB0(smaps_chw, samp, b, c_phasors))
    return BlockDiagonal(frames)


def run_b0complex_corrected_calib_recon(
    datdir: str, seqname: str = 'ArbEPI', L: int = 32, nbins: int = 128, device: str = 'cuda',
) -> dict:
    device_t = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    recon_dir = os.path.join(datdir, 'recon')
    fn_ksp = os.path.join(recon_dir, f'{seqname}_epi_zf.h5')
    fn_smaps = os.path.join(recon_dir, f'smaps_{seqname}_sigpy.h5')
    fn_b0map = os.path.join(recon_dir, f'{seqname}_b0map.h5')

    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    seq_params = load_seq_params(paths)
    fov, fov_degre = seq_params.fov, seq_params.fov_degre

    print(f'Loading smaps ({fn_smaps})...')
    smaps_raw = torch.from_numpy(_load_array(fn_smaps, 'smaps').astype(np.complex64))
    smaps_rss = smaps_raw.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    smaps = smaps_raw / (smaps_rss + torch.finfo(torch.float32).eps)  # (Nx,Ny,Nz,Nc), CPU
    Nx, Ny, Nz, _Nvc = smaps.shape

    print(f'Loading B0 field map ({fn_b0map})...')
    b0map_hz = torch.from_numpy(_load_array(fn_b0map, 'b0map_hz').astype(np.float32))
    assert tuple(b0map_hz.shape) == (Nx, Ny, Nz), (
        f'b0map_hz shape {tuple(b0map_hz.shape)} != smaps grid ({Nx},{Ny},{Nz})'
    )

    print('Estimating R2* map from dual-echo deGRE data...')
    r2star_hz = torch.from_numpy(
        estimate_r2star_map_epi_grid(datdir, seqname, fov_degre, fov, (Nx, Ny, Nz))
    )

    print(f'Loading echo times / sampling mask ({fn_ksp})...')
    echo_times_2d = _load_array(fn_ksp, 'echo_times').astype(np.float32)  # (Ny,Nz,Nt), numpy
    with h5py.File(fn_ksp, 'r') as f:
        omegas = f['omegas'][()]  # (Ny, Nz, Nt)
    Nt = omegas.shape[-1]
    calib_mask = compute_calib_mask(omegas)  # (Ny, Nz)
    n_calib = int(calib_mask.sum())
    print(f'Calibration region: {n_calib} / {calib_mask.size} (ky, kz) locations')
    mean_te_ms = echo_times_2d[calib_mask, :].mean(axis=0) * 1000  # (Nt,) ms

    te_nominal_s = nominal_te_s(paths.scan_info, seq_params.ETL)
    print(f'  TE_nominal = {te_nominal_s * 1000:.3f} ms (echo index {(seq_params.ETL - 1) // 2} of {seq_params.ETL})')

    grid = native_calib_grid(calib_mask, fov, Nx)
    xs, ys, zs = grid['x_slice'], grid['y_slice'], grid['z_slice']
    Nx_eff, Ny_eff, Nz_eff = grid['Nx_eff'], grid['Ny_eff'], grid['Nz_eff']
    voxel_mm = [1000 * fov[a] / n for a, n in enumerate((Nx_eff, Ny_eff, Nz_eff))]
    print(f'  Native grid: ({Nx_eff}, {Ny_eff}, {Nz_eff})  '
          f'(voxel size {voxel_mm[0]:.3f} x {voxel_mm[1]:.3f} x {voxel_mm[2]:.3f} mm)')

    n_target = (Nx_eff, Ny_eff, Nz_eff)
    smaps_native = torch.from_numpy(
        resize_to_epi_grid(smaps.numpy(), fov, fov, n_target, order=3).astype(np.complex64)
    ).to(device_t)
    smaps_chw = smaps_native.permute(3, 0, 1, 2).contiguous()  # (Nc,Nx_eff,Ny_eff,Nz_eff)
    b0map_hz_native = torch.from_numpy(
        resize_to_epi_grid(b0map_hz.numpy(), fov, fov, n_target, order=3).astype(np.float32)
    ).to(device_t)
    # R2* has much sharper local structure than smaps/b0map_hz (this
    # phantom's real air bubbles show up as T2* down to ~6ms in places),
    # and resize_to_epi_grid's cubic-spline zoom has no anti-aliasing
    # prefilter -- harmless for the smooth quantities above, but confirmed
    # (independent audit) to alias by up to ~14.7/s at this ~5x EPI-grid
    # -> native-grid downsample ratio, a ~23-25% local error in the T2*
    # correction factor at typical echo-time offsets from TE_nominal.
    # Gaussian-prefilter before the zoom (standard decimation
    # anti-aliasing, sigma set from the actual per-axis downsample ratio)
    # rather than touching grid_resize.py itself, which smaps/b0map_hz
    # both already use safely without one.
    from scipy.ndimage import gaussian_filter
    r2star_src = r2star_hz.numpy()
    ratios = [s / t for s, t in zip(r2star_src.shape, n_target)]
    sigmas = [max(r / 2, 0.0) for r in ratios]  # 0 where upsampling (ratio<1)
    r2star_prefiltered = gaussian_filter(r2star_src, sigma=sigmas)
    r2star_native = torch.from_numpy(
        np.clip(resize_to_epi_grid(r2star_prefiltered, fov, fov, n_target, order=3), 0.0, None).astype(np.float32)
    ).to(device_t)

    # Bandwidth-time-product sanity check (see module docstring): confirms
    # R2*'s decay-time product is small relative to Δf's, which is what
    # justifies reusing Δf-only-tuned interpolation weights for R2* too.
    echo_times_crop_raw = echo_times_2d[ys, zs, :]  # (Ny_eff, Nz_eff, Nt), pre-shift
    calib_mask_crop = calib_mask[ys, zs]  # (Ny_eff, Nz_eff)
    t_span_s = echo_times_crop_raw[calib_mask_crop, :].max() - echo_times_crop_raw[calib_mask_crop, :].min()
    df_range_hz = float(b0map_hz_native.max() - b0map_hz_native.min())
    r2_p95 = float(np.percentile(r2star_native.cpu().numpy(), 95))
    print(f'  bandwidth-time check: Δf range x echo-train span = {df_range_hz * t_span_s:.2f} '
          f'(dimensionless, ~27 at this pipeline\'s real ETL=60 scale per CLAUDE.md)')
    print(f'                        R2*(95th pct) x echo-train span = {r2_p95 * t_span_s:.4f} '
          '(dimensionless -- should be << the Δf figure above for the shared-weights reuse to be valid)')

    echo_times_crop = echo_times_crop_raw - te_nominal_s  # TE-referenced, both magnitude and phase
    echo_times_s = torch.from_numpy(echo_times_crop).to(device_t)
    echo_times_s = echo_times_s.unsqueeze(0).expand(Nx_eff, -1, -1, -1).contiguous()

    calib_mask_t = torch.from_numpy(calib_mask_crop).to(device_t)
    calib_omega = calib_mask_t[None, :, :, None].expand(Nx_eff, Ny_eff, Nz_eff, Nt)

    print(f'Building complex-field-corrected operator (L={L}, nbins={nbins})...')
    A = _build_calib_operator_b0_complex(
        smaps_chw, calib_omega, b0map_hz_native, r2star_native, echo_times_s, L=L, nbins=nbins,
    )
    idx_full = A.A[0].idx.cpu().numpy()
    for it in range(1, Nt):
        assert np.array_equal(A.A[it].idx.cpu().numpy(), idx_full), (
            'calibration-region sample indices differ across frames -- unexpected'
        )

    print(f'Loading k-space ({fn_ksp}) and gathering calibration-region samples...')
    ksp_epi_zf = _load_array(fn_ksp, 'ksp_epi_zf').astype(np.complex64)  # [Nx,Ny,Nz,Nc,Nt]
    ksp_epi_zf_crop = ksp_epi_zf[xs, ys, zs, :, :]  # [Nx_eff,Ny_eff,Nz_eff,Nc,Nt]
    del ksp_epi_zf
    from recon.lowres_calib_recon_b0 import gather_calib_ksp
    ksp_calib = torch.from_numpy(gather_calib_ksp(ksp_epi_zf_crop, idx_full)).to(device_t)  # [K,Nc,Nt]

    print('Reconstructing (adjoint only -- no iteration, no regularization)...')
    img = A.adjoint(ksp_calib)  # (Nx_eff, Ny_eff, Nz_eff, Nt) complex64
    img_np = img.detach().cpu().numpy()

    return dict(
        img_np=img_np, mean_te_ms=mean_te_ms, fov=fov, grid=grid, voxel_mm=voxel_mm,
        n_calib=n_calib, calib_mask=calib_mask, L=L, nbins=nbins, te_nominal_s=te_nominal_s,
    )


def main(
    datdir: str, seqname: str = 'ArbEPI', L: int = 32, nbins: int = 128, device: str = 'cuda',
) -> None:
    result = run_b0complex_corrected_calib_recon(datdir, seqname, L, nbins, device)
    img_np, fov = result['img_np'], result['fov']
    grid, voxel_mm, n_calib, calib_mask = result['grid'], result['voxel_mm'], result['n_calib'], result['calib_mask']
    Nx_eff, Ny_eff, Nz_eff = grid['Nx_eff'], grid['Ny_eff'], grid['Nz_eff']

    out_dir = os.path.join(datdir, 'recon', 'basic')
    os.makedirs(out_dir, exist_ok=True)
    fn_out = os.path.join(out_dir, f'{seqname}_recon_lowres_calib_b0complex')

    save_recon_nifti(
        fn_out, img_np, fov=fov, seqname=seqname,
        n_calib_samples=n_calib, n_ky_kz=int(calib_mask.size), L=L, nbins=nbins,
        te_nominal_s=result['te_nominal_s'],
        native_grid=[Nx_eff, Ny_eff, Nz_eff], native_voxel_mm=voxel_mm,
        note='Complex-field (off-resonance + T2*) corrected, TE-referenced, adjoint-only '
             '(no iteration/regularization) reconstruction of the fully-sampled k-space '
             'center only, at native (resolution-matched) grid size, not zero-padded',
    )
    print(f'Wrote {fn_out}.nii.gz + .json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdir')
    parser.add_argument('--seqname', default='ArbEPI')
    parser.add_argument('--L', type=int, default=32)
    parser.add_argument('--nbins', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args.datdir, args.seqname, args.L, args.nbins, args.device)
