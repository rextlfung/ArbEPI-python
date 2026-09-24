"""Root-sum-of-squares reconstruction: no smaps, no regularization, no B0
correction -- the plain baseline the other three drivers (recon/lowres_calib.py,
recon/L1-wavelet_TV_B0_SENSE.py, recon/run_recon.py) are compared against.

Revived (2026-09-22) after the recon/ consolidation removed the old
recon/run_rss.py + recon/recon_frames.py pair (they relied on MATLAB-literal
toppe.utils.ift3.m's fftshift-both-sides IFFT convention, since superseded
here by recon/operators.py's own verified ortho-normalized
fftshift(ifftn(ifftshift(.), norm='ortho')) adjoint -- the same convention
every other reconstruction in this repo now uses, so RSS is a fair,
consistent baseline rather than reproducing the old MATLAB port's own
odd-axis quirk (see the removed module's docstring, still in git history,
for that quirk's detail).

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.run_rss <datdir> <seqname>
"""

import argparse
import os

import numpy as np

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.nifti_io import save_recon_nifti
from recon.mslr import _load_array


def _ift3_ortho(kc: np.ndarray) -> np.ndarray:
    """Centered, ortho-normalized inverse 3D FFT over axes (0,1,2), matching
    recon/operators.py's GatheredSense._apply_adjoint's own per-coil IFFT
    exactly (fftshift(ifftn(ifftshift(.), norm='ortho')))."""
    axes = (0, 1, 2)
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kc, axes=axes), axes=axes, norm="ortho"), axes=axes
    )


def main_rss(datdir: str, seqname: str) -> None:
    recon_dir = os.path.join(datdir, "recon")
    fn_ksp = os.path.join(recon_dir, f"{seqname}_epi_zf.h5")

    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    sp = load_seq_params(paths)

    print("Loading k-space...")
    ksp = _load_array(fn_ksp, "ksp_epi_zf")  # (Nx,Ny,Nz,Nc,Nt)
    Nx, Ny, Nz, Nc, Nt = ksp.shape
    print(f"  {(Nx, Ny, Nz, Nc, Nt)}")

    img = np.empty((Nx, Ny, Nz, Nt), dtype=np.float32)
    for it in range(Nt):
        xc = _ift3_ortho(ksp[..., it])  # (Nx,Ny,Nz,Nc)
        img[..., it] = np.sqrt(np.sum(np.abs(xc) ** 2, axis=-1))
        print(f"  frame {it + 1}/{Nt} done")

    out_dir = os.path.join(recon_dir, "basic")
    os.makedirs(out_dir, exist_ok=True)
    fn_out = os.path.join(out_dir, f"{seqname}_recon_rss")
    save_recon_nifti(fn_out, img, fov=sp.fov, seqname=seqname)
    print(f"Wrote {fn_out}.nii.gz + .json")


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datdir")
    parser.add_argument("seqname")
    args = parser.parse_args()
    main_rss(args.datdir, args.seqname)


if __name__ == "__main__":
    _cli()
