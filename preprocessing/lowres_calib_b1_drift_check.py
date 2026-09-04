"""Re-runs lowres_calib_gain_drift_check.py's SVD gain-drift decomposition
on the *complex* reconstruction instead of the magnitude-only .nii.gz.

save_recon_nifti discards phase (np.abs(img), NIfTI has no complex dtype --
see nifti_io.py's module docstring), so the original gain-drift check --
built on load_lowres_calib_recon, which reads that saved nifti -- was blind
to phase from the start. A pure B1/flip-angle/receive-gain drift would
scale each voxel by the same *real, positive* factor every frame (a
magnitude-only effect), so it should show up identically whether or not
phase is included; if the dominant SVD mode instead carries most of its
fluctuation as a *phase* rotation rather than a magnitude change, that's
evidence against B1/gain drift and points toward something that shifts
phase instead (e.g. the frequency drift lowres_calib_freq_drift_check.py
tests directly).

Recomputes the complex image directly via lowres_calib_recon.lowres_calib_recon
(same pattern as lowres_calib_phase_ramp_check.py) rather than loading the
saved nifti. Decomposes the masked, per-voxel-demeaned *complex* signal via
SVD, phase-aligns the dominant component's temporal trace to a well-defined
reference (the component's own correlation with the time-mean image), then
splits that trace into a magnitude part (the gain-drift-like reading, same
units/definition as lowres_calib_gain_drift_check.py's pc1_gain_pct_fluctuation)
and a phase part (degrees) -- whichever dominates says which mechanism the
dominant fluctuation mode actually looks like.

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_calib_b1_drift_check <datdir> [datdir2 ...]
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.lowres_calib_recon import _load_chunked, compute_calib_mask, lowres_calib_recon
from preprocessing.lowres_temporal_stability import object_mask


def complex_gain_drift_decomposition(img: np.ndarray, mask: np.ndarray, skip_frames: int = 1) -> dict:
    """img: [Nx,Ny,Nz,Nframes] complex. mask: [Nx,Ny,Nz] bool. Returns the
    complex-SVD analogue of lowres_calib_gain_drift_check.py's
    gain_drift_decomposition -- see module docstring for the magnitude-vs-
    phase split."""
    eps = np.finfo(np.float64).eps
    X = img[mask][:, skip_frames:]  # (Nvox, Nframes') complex
    mean_vox = X.mean(axis=1, keepdims=True)  # (Nvox, 1), complex
    Xc = X - mean_vox

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_explained = S**2 / np.sum(S**2)

    # PC1's overall complex phase is arbitrary (any unit-modulus rotation of
    # U[:,0]/Vt[0,:] is an equally valid SVD solution) -- fix it by aligning
    # to the time-mean image's own phase, the same role sign correction
    # plays in the magnitude-only version.
    align = np.vdot(U[:, 0], mean_vox[:, 0])
    align_phase = align / (np.abs(align) + eps)
    u1 = U[:, 0] * np.conj(align_phase)
    corr_mag = np.abs(align) / (np.linalg.norm(U[:, 0]) * np.linalg.norm(mean_vox[:, 0]) + eps)

    c1 = u1.mean() * S[0] * Vt[0, :] * np.conj(align_phase)  # (Nframes',) complex temporal trace
    grand_mean = X.mean()

    gain_t = np.abs(c1)
    phase_t = np.angle(c1)

    return dict(
        var_explained_pc1=var_explained[0],
        var_explained_top3=var_explained[:3],
        pc1_corr_magnitude_with_mean_image=corr_mag,
        pc1_gain_pct_fluctuation=100 * gain_t.std(ddof=1) / np.abs(grand_mean),
        pc1_phase_std_deg=np.rad2deg(phase_t.std(ddof=1)),
        pc1_phase_range_deg=np.rad2deg(phase_t.max() - phase_t.min()),
        pc1_gain_temporal=gain_t,
        pc1_phase_temporal_deg=np.rad2deg(phase_t),
        grand_mean=grand_mean,
        # Diagnostic: c1's overall scale is u1.mean()*S[0] -- if |u1.mean()|
        # is small relative to u1's typical voxel magnitude (i.e. the
        # spatial pattern is close to zero-sum), that scalar's own phase is
        # noise-sensitive and can inflate/destabilize c1's apparent phase
        # swings independent of any real temporal effect.
        u1_mean_abs=np.abs(u1.mean()),
        u1_typical_abs=np.abs(u1).mean(),
    )


def main(datdirs: list[str], seqname: str = 'ArbEPI', skip_frames: int = 1) -> None:
    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))

        cfg = load_config(datdir=datdir, seqnames=[seqname])
        paths = set_seq_paths(cfg, seqname)
        seq_params = load_seq_params(paths)

        fn_epi_zf = paths.recon
        fn_smaps = os.path.join(datdir, 'recon', f'smaps_{seqname}_sigpy.h5')
        with h5py.File(fn_epi_zf, 'r') as f:
            ksp_epi_zf = _load_chunked(f, 'ksp_epi_zf')
            omegas = f['omegas'][()]
        calib_mask = compute_calib_mask(omegas)
        with h5py.File(fn_smaps, 'r') as f:
            smaps = f['smaps'][()]

        img, _grid = lowres_calib_recon(ksp_epi_zf, calib_mask, smaps, seq_params.fov)  # complex
        mask = object_mask(np.abs(img))
        result = complex_gain_drift_decomposition(img, mask, skip_frames)

        print(f'\n=== {label} ===')
        print(f'  PC1 variance explained (complex): {100 * result["var_explained_pc1"]:.1f}% '
              f'(top 3: {[f"{100 * v:.1f}%" for v in result["var_explained_top3"]]})')
        print(f'  PC1 spatial-loading magnitude-correlation with time-mean image: '
              f'{result["pc1_corr_magnitude_with_mean_image"]:.3f}')
        print(f'  PC1 magnitude (gain-like) fluctuation: {result["pc1_gain_pct_fluctuation"]:.3f}%')
        print(f'  PC1 phase fluctuation: std={result["pc1_phase_std_deg"]:.3f} deg, '
              f'range={result["pc1_phase_range_deg"]:.3f} deg')
        print(f'  scale-stability check: |u1.mean()| = {result["u1_mean_abs"]:.4g} vs. '
              f'typical |u1| voxel = {result["u1_typical_abs"]:.4g} '
              f'(ratio {result["u1_mean_abs"] / result["u1_typical_abs"]:.3f} -- '
              'near 0 means the spatial pattern nearly cancels in the mean, so the '
              'phase trace above is noise-amplified and untrustworthy)')
        which = 'magnitude (gain-like)' if result['pc1_gain_pct_fluctuation'] > \
            (result['pc1_phase_std_deg'] / 180 * 100) else 'phase (frequency-like)'
        print(f'  -> dominant mode looks more {which}-dominated')

        t = np.arange(skip_frames, skip_frames + len(result['pc1_gain_temporal']))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(t, 100 * (result['pc1_gain_temporal'] - np.abs(result['grand_mean'])) / np.abs(result['grand_mean']), 'o-')
        axes[0].axhline(0, color='gray', lw=0.5)
        axes[0].set_xlabel('frame')
        axes[0].set_ylabel('PC1 magnitude, relative to mean (%)')
        axes[0].set_title('gain-like component')
        axes[1].plot(t, result['pc1_phase_temporal_deg'], 'o-', color='C1')
        axes[1].axhline(0, color='gray', lw=0.5)
        axes[1].set_xlabel('frame')
        axes[1].set_ylabel('PC1 phase (deg)')
        axes[1].set_title('phase-like component')
        fig.suptitle(
            f'{label}: complex PC1 decomposition\n'
            f'{100 * result["var_explained_pc1"]:.0f}% of variance, '
            f'corr w/ mean image = {result["pc1_corr_magnitude_with_mean_image"]:.2f}'
        )
        plt.tight_layout()
        out_dir = os.path.join(datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)
        fn_out = os.path.join(out_dir, f'lowres_calib_b1_drift_check_skip{skip_frames}.png')
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
