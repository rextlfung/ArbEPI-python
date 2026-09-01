"""One-off diagnostic: reconstruct the dual-echo deGRE images from the
whitened+coil-compressed GRE cache (<seqname>_gre.h5's ksp_gre_echoes,
written by preprocess.py's STEP 2 -- see CLAUDE.md's preprocessing/ section)
into a viewable NIfTI, and dump PNG snapshots of both echo magnitudes plus
the B0 field-map pipeline's intermediate volumes (finit_hz, b0map_hz, mask)
-- for visually checking whether a persistent noisy/speckled reconstruction
artifact traces back to the GRE data itself or the field-map estimation,
rather than the B0-corrected recon operator (recon/operators_b0.py).

Same centered-IFFT convention as preprocessing/run_rss.py's _ift3
(fftshift(ifftn(fftshift(.))) per axis) and b0map.jl's own image-space
conversion (see its module docstring in CLAUDE.md).

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.gre_diagnostics <datdir> <seqname>
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


def main(datdir: str, seqname: str) -> None:
    recon_dir = os.path.join(datdir, "recon")
    fn_gre = os.path.join(recon_dir, f"{seqname}_gre.h5")
    fn_b0map = os.path.join(recon_dir, f"{seqname}_b0map.h5")
    out_dir = recon_dir

    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    sp = load_seq_params(paths)

    with h5py.File(fn_gre, "r") as f:
        ksp_echoes = f["ksp_gre_echoes"][()]  # (Nx,Ny,Nz,n_echoes,Nc)
        te_degre = f.attrs["TE_degre"]

    n_echoes = ksp_echoes.shape[3]
    print(f"ksp_gre_echoes: {ksp_echoes.shape}, TE_degre={te_degre}")

    img_echoes = np.stack(
        [
            np.sqrt(np.sum(np.abs(_ift3(ksp_echoes[:, :, :, ie, :])) ** 2, axis=-1))
            for ie in range(n_echoes)
        ],
        axis=-1,
    )  # (Nx,Ny,Nz,n_echoes) RSS-combined magnitude

    fn_out = os.path.join(out_dir, f"{seqname}_gre_echoes")
    save_recon_nifti(
        fn_out, img_echoes, fov=sp.fov_degre,
        seqname=seqname, TE_degre=list(map(float, te_degre)),
        note="RSS coil-combined magnitude, per echo (4th dim), on the deGRE grid",
    )
    print(f"Wrote {fn_out}.nii.gz + .json")

    with h5py.File(fn_b0map, "r") as f:
        finit_hz = f["finit_hz"][()]
        b0map_hz_degre = f["b0map_hz_degre"][()]
        mask_degre = f["mask_degre"][()]

    Nz = img_echoes.shape[2]
    iz = Nz // 2
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    im0 = axes[0, 0].imshow(img_echoes[:, :, iz, 0].T, origin="lower", cmap="gray")
    axes[0, 0].set_title(f"|GRE echo1| (TE={te_degre[0]*1e3:.2f}ms), z={iz}")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(img_echoes[:, :, iz, 1].T, origin="lower", cmap="gray")
    axes[0, 1].set_title(f"|GRE echo2| (TE={te_degre[1]*1e3:.2f}ms), z={iz}")
    plt.colorbar(im1, ax=axes[0, 1])

    ratio = img_echoes[:, :, iz, 1] / (img_echoes[:, :, iz, 0] + 1e-9)
    im2 = axes[0, 2].imshow(ratio.T, origin="lower", cmap="viridis", vmin=0, vmax=1.5)
    axes[0, 2].set_title("echo2/echo1 magnitude ratio (signal decay)")
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(finit_hz[:, :, iz].T, origin="lower", cmap="RdBu_r", vmin=-350, vmax=350)
    axes[1, 0].set_title("finit_hz (naive, wrapped-range init)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(b0map_hz_degre[:, :, iz].T, origin="lower", cmap="RdBu_r", vmin=-350, vmax=350)
    axes[1, 1].set_title("b0map_hz (MRIFieldmaps NCG output)")
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(mask_degre[:, :, iz].T, origin="lower", cmap="gray")
    axes[1, 2].set_title("mask")
    plt.colorbar(im5, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fn_png = os.path.join(out_dir, f"{seqname}_gre_b0_diagnostics_z{iz}.png")
    plt.savefig(fn_png, dpi=130)
    print(f"Wrote {fn_png}")

    # A second slice, off-center, in case the center slice hides something.
    iz2 = Nz // 4
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(img_echoes[:, :, iz2, 0].T, origin="lower", cmap="gray")
    axes[0].set_title(f"|GRE echo1|, z={iz2}")
    axes[1].imshow(finit_hz[:, :, iz2].T, origin="lower", cmap="RdBu_r", vmin=-350, vmax=350)
    axes[1].set_title(f"finit_hz, z={iz2}")
    axes[2].imshow(b0map_hz_degre[:, :, iz2].T, origin="lower", cmap="RdBu_r", vmin=-350, vmax=350)
    axes[2].set_title(f"b0map_hz, z={iz2}")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fn_png2 = os.path.join(out_dir, f"{seqname}_gre_b0_diagnostics_z{iz2}.png")
    plt.savefig(fn_png2, dpi=130)
    print(f"Wrote {fn_png2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datdir")
    parser.add_argument("seqname", nargs="?", default="ArbEPI")
    args = parser.parse_args()
    main(args.datdir, args.seqname)
