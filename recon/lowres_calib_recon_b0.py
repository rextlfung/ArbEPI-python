"""B0-corrected variant of preprocessing/lowres_calib_recon.py -- same fully-
sampled (ky, kz) calibration region, same "no iteration, no regularization"
philosophy, but replacing the plain IFFT + smaps combine with the adjoint of
the time-segmented B0-corrected operator (recon/operators_b0.py).

Motivation: even though every frame samples the exact same (ky, kz)
calibration locations (see the other script's docstring), the *order* in
which a given frame's shots visit them differs -- so a given (ky, kz)
location is acquired at a different echo time (time since RF excitation)
in different frames (confirmed directly against this repo's own
`echo_times` array: sequences/ArbEPI.py's per-echo timing is frame-
invariant as a *set*, but which (ky,kz) location maps to which echo index
varies per frame, exactly like the full acquisition). Off-resonance phase
accrues with that time, so the same k-space location carries a different
B0-induced phase from frame to frame -- a source of temporal instability
in preprocessing/lowres_temporal_stability.py's measurements that has
nothing to do with per-frame system drift.

This applies the same single-adjoint-application philosophy as the
uncorrected script (`img = sum_c conj(smap_c) * ifft(ksp_c)`, no CG/POGM,
no low-rank patch regularization) but through
`recon.operators_b0.GatheredSenseB0`'s adjoint instead of a plain 3D IFFT,
so the per-sample conjugate-phase demodulation (Sutton/Noll/Fessler time-
segmentation) is included.

Does NOT reuse recon.operators_b0.build_encoding_operator_b0 directly: that
function's shared time-segmentation fit is built from frame 0's distinct
echo times alone, on the (correct, for its own use case) assumption that a
frame's full ETL-worth of samples covers every echo time the fit could ever
need. That assumption does not hold for the small calibration region alone
-- confirmed empirically (e.g. one dataset's frame 0 covers only 28 of the
33 distinct echo times used across all 30 frames within the calibration
region) -- so `_build_calib_operator_b0` below is the same construction
(matching this repo's actual current operators_b0.py contract: GatheredSenseB0
takes a per-sample-gathered (K,L) b_weights directly, and echo_times_s is
the full (Nx,Ny,Nz,Nt) broadcast, not the more compact (Ny,Nz,Nt) form) with
the union of echo times across every frame instead of frame 0's alone.

L defaults to 32 here (not operators_b0.py's own L=6 default) -- see
CLAUDE.md's recon/ "B0 off-resonance correction" section: a real-scale sweep
(recon/sweep_time_segments.py) found L=6 badly under-resolves this
pipeline's real ETL=60 bandwidth-time product (~35% error reduction only),
while L=32 is the smallest value that gets relative forward-model error
under 1%. (Also matches this repo's own pending, not-yet-landed fix for
that default -- see CLAUDE.md's item 82.)

Reconstructs at native (resolution-matched) grid size, not zero-padded to
the full (Nx,Ny,Nz) acquisition grid -- same reasoning and
`native_calib_grid` helper as preprocessing/lowres_calib_recon.py (see its
module docstring for the FOV/N resolution derivation); duplicated here
rather than imported, to keep this .venv-recon script off that module's
matplotlib dependency (not part of the `recon` extra). smaps and the B0
field map are both resized down to the native grid via
preprocessing/grid_resize.py's resize_to_epi_grid (the same FOV-preserving
resample the rest of this pipeline already uses between grids of different
resolution); k-space is spatially cropped in-memory after the usual
chunked-by-frame HDF5 read.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.lowres_calib_recon_b0 <datdir> [--seqname ArbEPI] [--device cuda]
"""

import argparse
import os

import h5py
import numpy as np
import torch
from mirtorch.linear import BlockDiagonal
from mirtorch.linear.mri import mri_exp_approx

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.grid_resize import resize_to_epi_grid
from preprocessing.nifti_io import save_recon_nifti
from recon.operators_b0 import GatheredSenseB0, _check_b_weight_row_sums
from recon.reconstruct import _load_array


def compute_calib_mask(omegas: np.ndarray) -> np.ndarray:
    """Same as preprocessing/lowres_calib_recon.py's -- duplicated rather
    than imported to keep this .venv-recon script from depending on
    preprocessing/lowres_calib_recon.py's module-level matplotlib import
    (not part of the `recon` extra)."""
    return np.all(omegas, axis=-1)


def native_calib_grid(calib_mask: np.ndarray, fov: tuple[float, float, float], Nx_full: int) -> dict:
    """Same as preprocessing/lowres_calib_recon.py's -- duplicated for the
    same reason as compute_calib_mask above."""
    ys, zs = np.nonzero(calib_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    z0, z1 = int(zs.min()), int(zs.max()) + 1
    Ny_eff, Nz_eff = y1 - y0, z1 - z0

    fov_x, fov_y, _fov_z = fov
    Nx_eff = round(Ny_eff * fov_x / fov_y)
    x0 = Nx_full // 2 - Nx_eff // 2
    x1 = x0 + Nx_eff

    return dict(
        Nx_eff=Nx_eff, Ny_eff=Ny_eff, Nz_eff=Nz_eff,
        x_slice=slice(x0, x1), y_slice=slice(y0, y1), z_slice=slice(z0, z1),
    )


def _build_calib_operator_b0(
    smaps_chw: torch.Tensor,
    calib_omega: torch.Tensor,
    b0map_hz: torch.Tensor,
    echo_times_s: torch.Tensor,
    L: int = 32,
    nbins: int = 128,
) -> BlockDiagonal:
    """Same construction as recon.operators_b0.build_encoding_operator_b0
    (this repo's current version: GatheredSenseB0 takes a per-sample-
    gathered (K,L) b_weights directly, echo_times_s is (Nx,Ny,Nz,Nt)),
    except the shared time-segmentation fit (mri_exp_approx) is built from
    the union of echo times across every frame's calibration-region
    samples, not frame 0's alone -- see module docstring for why frame 0
    alone is not a superset here."""
    Nt = calib_omega.shape[-1]
    N = tuple(smaps_chw.shape[1:])
    b0_neg = (-b0map_hz).to(torch.float32)

    idx_list, t_ms_list = [], []
    for it in range(Nt):
        samp = calib_omega[..., it]
        idx = torch.nonzero(samp.reshape(-1), as_tuple=False).squeeze(-1)
        t_ms = (echo_times_s[..., it].reshape(-1)[idx] * 1000).to(torch.float32)
        idx_list.append(idx)
        t_ms_list.append(t_ms)
    unique_t_ms = torch.unique(torch.cat(t_ms_list), sorted=True)

    b_by_echo, c, _tl = mri_exp_approx(b0_neg, nbins, L, unique_t_ms)
    _check_b_weight_row_sums(b_by_echo, "shared (union of calib-region echo times)")
    c_phasors = c.transpose(0, 1).reshape((L,) + N).to(smaps_chw.dtype)

    frames = []
    for it in range(Nt):
        samp = calib_omega[..., it]
        t_ms = t_ms_list[it]
        pos = torch.searchsorted(unique_t_ms, t_ms).clamp(max=unique_t_ms.numel() - 1)
        assert torch.allclose(unique_t_ms[pos], t_ms, atol=1e-4), (
            f"_build_calib_operator_b0: frame {it}'s sample echo times aren't in the "
            "union of every frame's calib-region echo times -- shouldn't be reachable "
            "by construction."
        )
        b = b_by_echo[pos].to(smaps_chw.dtype)
        frames.append(GatheredSenseB0(smaps_chw, samp, b, c_phasors))
    return BlockDiagonal(frames)


def gather_calib_ksp(ksp_epi_zf: np.ndarray, idx_full: np.ndarray) -> np.ndarray:
    """ksp_epi_zf: [Nx, Ny, Nz, Nc, Nt] numpy (plain, not torch -- kept off
    the GPU/out of any torch tensor since it's ~11GB and only ~1.6MB of it
    is actually needed). idx_full: flat C-order spatial indices into
    (Nx*Ny*Nz), identical for every frame (the calibration region is frame-
    invariant). Returns [K, Nc, Nt] complex64, matching GatheredSenseB0's
    forward/adjoint (K, Nc) per-frame convention -- see recon/operators.py's
    gather_ksp, which this mirrors but keeps ksp_epi_zf as plain numpy."""
    Nc, Nt = ksp_epi_zf.shape[3], ksp_epi_zf.shape[4]
    out = np.empty((idx_full.size, Nc, Nt), dtype=ksp_epi_zf.dtype)
    for it in range(Nt):
        flat = ksp_epi_zf[..., it].reshape(-1, Nc)
        out[:, :, it] = flat[idx_full, :]
    return out


def run_b0_corrected_calib_recon(
    datdir: str, seqname: str = 'ArbEPI', L: int = 32, nbins: int = 128, device: str = 'cuda',
) -> dict:
    """Core computation shared by main() (writes the nifti) and any other
    consumer of the B0-corrected calibration-region reconstruction (e.g.
    lowres_calib_freq_drift_check.py, which needs the same complex image
    plus each frame's mean calibration echo time, not just the saved
    magnitude-only nifti). Returns a dict: img_np [Nx_eff,Ny_eff,Nz_eff,Nt]
    complex64, mean_te_ms [Nt] (mean calibration-region echo time per frame,
    matching preprocessing/lowres_calib_t2star_check.py's
    calib_mean_echo_time_ms but computed from the same already-loaded
    echo_times_2d/calib_mask rather than a second file read), fov, grid
    (native_calib_grid's return value), voxel_mm, n_calib."""
    device_t = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    recon_dir = os.path.join(datdir, 'recon')
    fn_ksp = os.path.join(recon_dir, f'{seqname}_epi_zf.h5')
    fn_smaps = os.path.join(recon_dir, f'smaps_{seqname}_sigpy.h5')
    fn_b0map = os.path.join(recon_dir, f'{seqname}_b0map.h5')

    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    seq_params = load_seq_params(paths)
    fov = seq_params.fov

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

    print(f'Loading echo times / sampling mask ({fn_ksp})...')
    echo_times_2d = _load_array(fn_ksp, 'echo_times').astype(np.float32)  # (Ny,Nz,Nt), numpy
    with h5py.File(fn_ksp, 'r') as f:
        omegas = f['omegas'][()]  # (Ny, Nz, Nt)
    Nt = omegas.shape[-1]
    calib_mask = compute_calib_mask(omegas)  # (Ny, Nz)
    n_calib = int(calib_mask.sum())
    print(f'Calibration region: {n_calib} / {calib_mask.size} (ky, kz) locations')
    mean_te_ms = echo_times_2d[calib_mask, :].mean(axis=0) * 1000  # (Nt,) ms

    grid = native_calib_grid(calib_mask, fov, Nx)
    xs, ys, zs = grid['x_slice'], grid['y_slice'], grid['z_slice']
    Nx_eff, Ny_eff, Nz_eff = grid['Nx_eff'], grid['Ny_eff'], grid['Nz_eff']
    voxel_mm = [1000 * fov[a] / n for a, n in enumerate((Nx_eff, Ny_eff, Nz_eff))]
    print(f'  Native grid: ({Nx_eff}, {Ny_eff}, {Nz_eff})  '
          f'(voxel size {voxel_mm[0]:.3f} x {voxel_mm[1]:.3f} x {voxel_mm[2]:.3f} mm)')

    # smaps/b0map_hz are smooth, low-spatial-frequency quantities -- resize
    # in image space (same FOV-preserving resample the rest of this
    # pipeline uses between grids of different resolution) rather than
    # cropping k-space, which they were never sampled on in the first place.
    n_target = (Nx_eff, Ny_eff, Nz_eff)
    smaps_native = torch.from_numpy(
        resize_to_epi_grid(smaps.numpy(), fov, fov, n_target, order=3).astype(np.complex64)
    ).to(device_t)
    smaps_chw = smaps_native.permute(3, 0, 1, 2).contiguous()  # (Nc,Nx_eff,Ny_eff,Nz_eff)
    b0map_hz_native = torch.from_numpy(
        resize_to_epi_grid(b0map_hz.numpy(), fov, fov, n_target, order=3).astype(np.float32)
    ).to(device_t)

    # echo_times is a k-space-indexed array (acquisition time per sampled
    # (ky,kz) location), not an image -- crop it the same way k-space
    # itself is cropped below, not resized.
    echo_times_crop = echo_times_2d[ys, zs, :]  # (Ny_eff, Nz_eff, Nt)
    echo_times_s = torch.from_numpy(echo_times_crop).to(device_t)
    echo_times_s = echo_times_s.unsqueeze(0).expand(Nx_eff, -1, -1, -1).contiguous()

    calib_mask_crop = calib_mask[ys, zs]  # (Ny_eff, Nz_eff)
    calib_mask_t = torch.from_numpy(calib_mask_crop).to(device_t)
    calib_omega = calib_mask_t[None, :, :, None].expand(Nx_eff, Ny_eff, Nz_eff, Nt)

    print(f'Building B0-corrected operator (L={L}, nbins={nbins})...')
    A = _build_calib_operator_b0(smaps_chw, calib_omega, b0map_hz_native, echo_times_s, L=L, nbins=nbins)
    idx_full = A.A[0].idx.cpu().numpy()
    for it in range(1, Nt):
        assert np.array_equal(A.A[it].idx.cpu().numpy(), idx_full), (
            'calibration-region sample indices differ across frames -- unexpected'
        )

    print(f'Loading k-space ({fn_ksp}) and gathering calibration-region samples...')
    ksp_epi_zf = _load_array(fn_ksp, 'ksp_epi_zf').astype(np.complex64)  # [Nx,Ny,Nz,Nc,Nt]
    ksp_epi_zf_crop = ksp_epi_zf[xs, ys, zs, :, :]  # [Nx_eff,Ny_eff,Nz_eff,Nc,Nt]
    del ksp_epi_zf
    ksp_calib = torch.from_numpy(gather_calib_ksp(ksp_epi_zf_crop, idx_full)).to(device_t)  # [K,Nc,Nt]

    print('Reconstructing (adjoint only -- no iteration, no regularization)...')
    img = A.adjoint(ksp_calib)  # (Nx_eff, Ny_eff, Nz_eff, Nt) complex64
    img_np = img.detach().cpu().numpy()

    return dict(
        img_np=img_np, mean_te_ms=mean_te_ms, fov=fov, grid=grid, voxel_mm=voxel_mm,
        n_calib=n_calib, calib_mask=calib_mask, L=L, nbins=nbins,
    )


def main(
    datdir: str, seqname: str = 'ArbEPI', L: int = 32, nbins: int = 128, device: str = 'cuda',
) -> None:
    result = run_b0_corrected_calib_recon(datdir, seqname, L, nbins, device)
    img_np, fov = result['img_np'], result['fov']
    grid, voxel_mm, n_calib, calib_mask = result['grid'], result['voxel_mm'], result['n_calib'], result['calib_mask']
    Nx_eff, Ny_eff, Nz_eff = grid['Nx_eff'], grid['Ny_eff'], grid['Nz_eff']

    out_dir = os.path.join(datdir, 'recon', 'basic')
    os.makedirs(out_dir, exist_ok=True)
    fn_out = os.path.join(out_dir, f'{seqname}_recon_lowres_calib_b0')

    save_recon_nifti(
        fn_out, img_np, fov=fov, seqname=seqname,
        n_calib_samples=n_calib, n_ky_kz=int(calib_mask.size), L=L, nbins=nbins,
        native_grid=[Nx_eff, Ny_eff, Nz_eff], native_voxel_mm=voxel_mm,
        note='B0-corrected (time-segmented conjugate-phase, adjoint only, no '
             'iteration/regularization) IFFT + smaps combine of the fully-sampled '
             'k-space center only, reconstructed at native (resolution-matched) grid '
             'size, not zero-padded',
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
