"""Quick low-res sanity-check reconstruction from the fully-sampled k-space
calibration region, for a fast look at a dataset without running the full
iterative Stage-2 pipeline (recon_frames.py / recon/).

Every frame's (ky, kz) sampling mask (see sampling/pd_sample.py's
`calib_frac`) always includes a small, fully-sampled centered ellipse --
the calibration region ESPIRiT itself is calibrated from (via the deGRE
scan, not this one). That same guarantee holds for the *EPI* acquisition's
own per-frame mask: `omegas[..., t]` is `calib_mask | (extra incoherent
samples)` for every frame `t`, so `np.all(omegas, axis=-1)` (the
intersection across all frames) recovers exactly that calibration region --
no need to know `params.pd_calib_frac`/`R` ahead of time, or assume they
match the current defaults for a dataset acquired under different settings.
Verified empirically on both `20260822ball_*` datasets: 362/10800 (ky, kz)
locations, a centered ellipse, identical between the radial/laminar
variants (they share the same underlying (ky, kz) mask, only the per-frame
EPI shot ordering differs).

Since that region is exactly, not approximately, fully sampled, no
iterative reconstruction is needed: masking `ksp_epi_zf` down to it,
inverse-FFTing, and combining coils with the existing ESPIRiT sensitivity
maps (`recon/smaps_<seqname>_sigpy.h5`, already normalized so
`sum_c |s_c|^2 <= 1`, see smaps.py's process_smaps) is the correct linear
estimate directly -- `img = sum_c conj(s_c) * ifft(ksp_c)`, no
regularization or iteration. The image is low-resolution in (ky, kz) (only
the small calibration extent is used) but full-resolution along kx (the
readout direction, sampled in full on every echo) -- see CLAUDE.md's
sampling/pd_sample.py section.

Same centered-IFFT convention as run_rss.py's/gre_diagnostics.py's _ift3.

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_calib_recon <datdir> [seqname]
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.nifti_io import save_recon_nifti


def _ift3(d: np.ndarray) -> np.ndarray:
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(d, axes=axes), axes=axes), axes=axes)


def _load_chunked(f: h5py.File, key: str) -> np.ndarray:
    """Read a dataset chunk-by-chunk along its last axis when chunked there.
    A whole-dataset `d[()]` call on this repo's own ksp_epi_zf (chunked one
    frame per chunk) was measured at ~7 MB/s -- an h5py/HDF5 chunk-cache
    pathology, not a disk-speed limit -- versus ~500 MB/s reading one chunk
    at a time; see recon/reconstruct.py's `_load_array`, which this mirrors."""
    d = f[key]
    if d.chunks is not None and d.chunks[-1] < d.shape[-1]:
        out = np.empty(d.shape, dtype=d.dtype)
        step = d.chunks[-1]
        for start in range(0, d.shape[-1], step):
            out[..., start : start + step] = d[..., start : start + step]
        return out
    return np.asarray(d[()])


def compute_calib_mask(omegas: np.ndarray) -> np.ndarray:
    """omegas: [Ny, Nz, Nframes] bool. Returns [Ny, Nz] bool: the (ky, kz)
    locations sampled in *every* frame -- the fully-sampled calibration
    region every frame's mask is built around (see module docstring)."""
    return np.all(omegas, axis=-1)


def lowres_calib_recon(
    ksp_epi_zf: np.ndarray, calib_mask: np.ndarray, smaps: np.ndarray
) -> np.ndarray:
    """ksp_epi_zf: [Nx, Ny, Nz, Nvcoils, Nframes]. calib_mask: [Ny, Nz] bool.
    smaps: [Nx, Ny, Nz, Nvcoils] complex, sum_c|s_c|^2 <= 1 normalized.

    Returns [Nx, Ny, Nz, Nframes] complex: per-frame, sensitivity-map-
    weighted coil combination of the calibration-region-only image.
    """
    ksp_calib = ksp_epi_zf * calib_mask[None, :, :, None, None]
    img_coils = _ift3(ksp_calib)  # [Nx, Ny, Nz, Nvcoils, Nframes], batched over trailing axes
    return np.sum(np.conj(smaps)[..., None] * img_coils, axis=3)


def main(datdir: str, seqname: str = 'ArbEPI') -> None:
    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    seq_params = load_seq_params(paths)

    fn_epi_zf = paths.recon
    fn_smaps = os.path.join(datdir, 'recon', f'smaps_{seqname}_sigpy.h5')

    print(f'Loading {fn_epi_zf}...')
    with h5py.File(fn_epi_zf, 'r') as f:
        ksp_epi_zf = _load_chunked(f, 'ksp_epi_zf')  # [Nx, Ny, Nz, Nvcoils, Nframes]
        omegas = f['omegas'][()]  # [Ny, Nz, Nframes]

    calib_mask = compute_calib_mask(omegas)
    n_calib = int(calib_mask.sum())
    print(f'Calibration region: {n_calib} / {calib_mask.size} (ky, kz) locations')
    if n_calib == 0:
        raise RuntimeError(
            'lowres_calib_recon: empty calibration region (no (ky, kz) location is '
            'sampled in every frame) -- is this dataset actually fully sampled at '
            'k-space center?'
        )

    print(f'Loading {fn_smaps}...')
    with h5py.File(fn_smaps, 'r') as f:
        smaps = f['smaps'][()]  # [Nx, Ny, Nz, Nvcoils]

    print('Reconstructing...')
    img = lowres_calib_recon(ksp_epi_zf, calib_mask, smaps)  # [Nx, Ny, Nz, Nframes]

    out_dir = os.path.join(datdir, 'recon', 'basic')
    os.makedirs(out_dir, exist_ok=True)
    fn_out = os.path.join(out_dir, f'{seqname}_recon_lowres_calib')
    save_recon_nifti(
        fn_out, img, fov=seq_params.fov, seqname=seqname,
        n_calib_samples=n_calib, n_ky_kz=int(calib_mask.size),
        note='IFFT + smaps-weighted coil combine of the fully-sampled k-space center only',
    )
    print(f'Wrote {fn_out}.nii.gz + .json')

    Nx, Ny, Nz, Nframes = img.shape
    iz = Nz // 2
    mag = np.abs(img[:, :, iz, :])
    vmax = np.percentile(mag, 99.5)
    frame_idxs = np.linspace(0, Nframes - 1, min(6, Nframes)).astype(int)
    fig, axes = plt.subplots(1, len(frame_idxs), figsize=(3 * len(frame_idxs), 3.5))
    for ax, t in zip(np.atleast_1d(axes), frame_idxs):
        ax.imshow(mag[:, :, t].T, origin='lower', cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(f'frame {t}')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'{seqname}: low-res calib-region recon, z={iz}')
    plt.tight_layout()
    fn_png = os.path.join(out_dir, f'{seqname}_recon_lowres_calib_z{iz}.png')
    plt.savefig(fn_png, dpi=130)
    print(f'Wrote {fn_png}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdir')
    parser.add_argument('seqname', nargs='?', default='ArbEPI')
    args = parser.parse_args()
    main(args.datdir, args.seqname)
