"""Checks whether a single, spatially-uniform (multiplicative) gain factor
per frame explains lowres_calib_recon.py's temporal fluctuation -- the
signature RF transmit / receiver gain drift leaves (every voxel scaled by
the same time-varying factor), as opposed to a spatially localized effect
(motion, a local susceptibility change, an artifact confined to one region).

Decomposes the object-ROI signal (per-voxel time-mean subtracted) via SVD:
if the dominant component's spatial loading is proportional to the object's
own time-mean image (correlation near 1), and that one component captures
most of the per-voxel variance, its temporal weights *are* the frame-to-
frame gain factor -- reported directly as a percent fluctuation, comparable
to preprocessing/lowres_temporal_stability.py's percent_fluctuation and
lowres_calib_t2star_check.py's TE-residual fluctuation.

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_calib_gain_drift_check <datdir> [datdir2 ...]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from preprocessing.lowres_temporal_stability import load_lowres_calib_recon, object_mask


def gain_drift_decomposition(img: np.ndarray, mask: np.ndarray, skip_frames: int = 1) -> dict:
    """img: [Nx,Ny,Nz,Nframes]. mask: [Nx,Ny,Nz] bool. Returns a dict with
    the SVD decomposition of the masked, per-voxel-demeaned signal and the
    derived gain-drift estimate (see module docstring)."""
    X = img[mask][:, skip_frames:]  # (Nvox, Nframes')
    mean_vox = X.mean(axis=1, keepdims=True)  # (Nvox, 1) -- each voxel's own time-mean
    Xc = X - mean_vox

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_explained = S**2 / np.sum(S**2)

    # Orient PC1 so its spatial loading correlates positively with the
    # object's own time-mean intensity (sign of an SVD component is
    # arbitrary otherwise).
    corr = np.corrcoef(U[:, 0], mean_vox[:, 0])[0, 1]
    sign = 1.0 if corr >= 0 else -1.0
    u1 = U[:, 0] * sign

    # PC1's contribution to the ROI-mean signal: spatial mean of its rank-1
    # reconstruction, sign-corrected. This is directly comparable to the
    # already-reported ROI-mean signal curve.
    c1 = u1.mean() * S[0] * Vt[0, :] * sign
    grand_mean = X.mean()

    return dict(
        var_explained_pc1=var_explained[0],
        var_explained_top3=var_explained[:3],
        pc1_corr_with_mean_image=abs(corr),
        pc1_gain_pct_fluctuation=100 * c1.std(ddof=1) / grand_mean,
        pc1_temporal=c1,
        grand_mean=grand_mean,
    )


def main(datdirs: list[str], seqname: str = 'ArbEPI', skip_frames: int = 1) -> None:
    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))
        img, _meta = load_lowres_calib_recon(datdir, seqname)  # uncorrected recon
        mask = object_mask(img)
        result = gain_drift_decomposition(img, mask, skip_frames)

        print(f'\n=== {label} ===')
        print(f'  PC1 variance explained: {100 * result["var_explained_pc1"]:.1f}% '
              f'(top 3: {[f"{100 * v:.1f}%" for v in result["var_explained_top3"]]})')
        print(f'  PC1 spatial-loading correlation with time-mean image: '
              f'{result["pc1_corr_with_mean_image"]:.3f}')
        print(f'  PC1-implied gain-drift fluctuation: {result["pc1_gain_pct_fluctuation"]:.3f}%')

        fig, ax = plt.subplots(figsize=(6, 4))
        t = np.arange(skip_frames, skip_frames + len(result['pc1_temporal']))
        ax.plot(t, 100 * result['pc1_temporal'] / result['grand_mean'], 'o-')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel('frame')
        ax.set_ylabel('PC1-implied gain (%)')
        ax.set_title(
            f'{label}: dominant spatially-uniform mode\n'
            f'{100 * result["var_explained_pc1"]:.0f}% of variance, '
            f'corr w/ mean image = {result["pc1_corr_with_mean_image"]:.2f}'
        )
        plt.tight_layout()
        out_dir = os.path.join(datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)
        fn_out = os.path.join(out_dir, f'lowres_calib_gain_drift_check_skip{skip_frames}.png')
        plt.savefig(fn_out, dpi=130)
        print(f'  Wrote {fn_out}')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdirs', nargs='+')
    parser.add_argument('--seqname', default='ArbEPI')
    parser.add_argument('--skip-frames', type=int, default=1)
    args = parser.parse_args()
    main(args.datdirs, args.seqname, args.skip_frames)
