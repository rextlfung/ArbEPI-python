"""Root-sum-of-squares reconstruction: zero-filled inverse FFT per coil, then
sqrt(sum_c |x_c|^2). Non-iterative, no sensitivity maps -- a quick first look.

Frames are batched onto the GPU (as many per batch as fit in max_batch_bytes)
and transformed together.

    .venv-recon/bin/python -m recon.rss <datdir> <seqname> [--device cuda]

Reads <datdir>/recon/<seqname>_epi_zf.h5 and writes
<datdir>/recon/rss/<seqname>_recon.{nii.gz,json}.
"""

import argparse
import os
import time

import h5py
import numpy as np
import torch

from preprocessing.nifti_io import save_recon_nifti


def rss(ksp: torch.Tensor) -> torch.Tensor:
    """ksp: (Nx,Ny,Nz,Nc,Nb) zero-filled k-space -> (Nx,Ny,Nz,Nb) magnitude.
    Uses the same centered, ortho-normalized IFFT as recon/mri_operator.py's SENSE."""
    dims = (0, 1, 2)
    x = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(ksp, dim=dims), dim=dims, norm="ortho"), dim=dims
    )
    return x.abs().pow(2).sum(dim=3).sqrt()


def run_rss(fn_ksp: str, device: str = "cuda", max_batch_bytes: float = 4e9) -> np.ndarray:
    """(Nx,Ny,Nz,Nt) float32 RSS image of every frame in fn_ksp."""
    device = torch.device(device)
    with h5py.File(fn_ksp, "r") as f:
        d = f["ksp_epi_zf"]
        Nx, Ny, Nz, Nc, Nt = d.shape
        frame_bytes = Nx * Ny * Nz * Nc * 8  # complex64
        batch = max(1, int(max_batch_bytes // (3 * frame_bytes)))  # input + FFT workspace
        img = np.empty((Nx, Ny, Nz, Nt), dtype=np.float32)
        for t0 in range(0, Nt, batch):
            t1 = min(t0 + batch, Nt)
            ksp = torch.from_numpy(np.asarray(d[..., t0:t1]).astype(np.complex64)).to(device)
            img[..., t0:t1] = rss(ksp).cpu().numpy()
            print(f"  frames {t0 + 1}-{t1}/{Nt} done")
    return img


def main(datdir: str, seqname: str, device: str = "cuda") -> str:
    from preprocessing.config import load_config, load_seq_params, set_seq_paths

    recon_dir = os.path.join(datdir, "recon")
    sp = load_seq_params(set_seq_paths(load_config(datdir=datdir, seqnames=[seqname]), seqname))
    t_start = time.time()
    img = run_rss(os.path.join(recon_dir, f"{seqname}_epi_zf.h5"), device)
    runtime_s = time.time() - t_start
    out_dir = os.path.join(recon_dir, "rss")
    os.makedirs(out_dir, exist_ok=True)
    fn_out = os.path.join(out_dir, f"{seqname}_recon")
    save_recon_nifti(fn_out, img, fov=sp.fov, seqname=seqname, runtime_s=runtime_s)
    print(f"Wrote {fn_out}.nii.gz + .json ({runtime_s:.1f}s)")
    return fn_out


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("datdir")
    parser.add_argument("seqname")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.datdir, args.seqname, args.device)


if __name__ == "__main__":
    _cli()
