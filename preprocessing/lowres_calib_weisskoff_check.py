"""Weisskoff radius-of-decorrelation test (Weisskoff, MRM 1996; the fBIRN
multicenter QA metric, Friedman & Glover, JMRI 2006) on lowres_calib_recon.py's
output -- a mechanism-agnostic diagnostic for whether the temporal
fluctuation this pipeline keeps measuring is dominated by ordinary thermal
noise or by spatially-correlated ("system") instability, without assuming
which physical mechanism is responsible.

For growing square ROIs (side length N voxels, centered on the object)
in the central slice, computes the same percent-fluctuation metric
lowres_temporal_stability.py uses (detrended std / mean of the ROI-mean
signal) as a function of N. Pure thermal noise averages down as 1/N (ROI
area N^2, fluctuation ~ 1/sqrt(area) = 1/N); spatially-correlated
instability (field drift, T2* effects tied to a smooth spatial pattern,
convection, coil/gain drift, etc. -- any effect that moves whole regions
together rather than each voxel independently) causes the curve to
flatten to a nonzero floor well before the ROI covers the whole object.
The ROI size where the measured curve departs from the theoretical 1/N
line is the "radius of decorrelation" (Rdc): a large plateau at high N
implies a real, spatially-structured, non-thermal effect is present,
independent of which physical mechanism it is.

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_calib_weisskoff_check <datdir> [datdir2 ...] [--variant '']
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from preprocessing.lowres_temporal_stability import load_lowres_calib_recon, object_mask


def weisskoff_curve(img_slice: np.ndarray, center: tuple[int, int], max_size: int) -> tuple[list[int], list[float]]:
    """img_slice: [Nx,Ny,Nframes] magnitude, one z-slice. center: (cx,cy)
    ROI center (object centroid). Returns (sizes, percent_fluctuation) for
    growing square ROIs of side 1..max_size voxels."""
    Nx, Ny, Nt = img_slice.shape
    cx, cy = center
    t = np.arange(Nt)
    sizes, fluctuations = [], []
    for size in range(1, max_size + 1):
        x0, y0 = cx - size // 2, cy - size // 2
        x1, y1 = x0 + size, y0 + size
        if x0 < 0 or y0 < 0 or x1 > Nx or y1 > Ny:
            break
        roi = img_slice[x0:x1, y0:y1, :].mean(axis=(0, 1))  # (Nt,)
        coeffs = np.polyfit(t, roi, 1)
        resid = roi - np.polyval(coeffs, t)
        fluctuations.append(100 * resid.std(ddof=1) / roi.mean())
        sizes.append(size)
    return sizes, fluctuations


def main(datdirs: list[str], seqname: str = 'ArbEPI', variant: str = '', skip_frames: int = 1) -> None:
    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))
        if variant:
            label = f'{label} ({variant})'
        img, _meta = load_lowres_calib_recon(datdir, seqname, variant)
        img = img[..., skip_frames:]
        mask = object_mask(img)

        Nx, Ny, Nz, _Nt = img.shape
        iz = Nz // 2
        mean_slice = img[:, :, iz, :].mean(axis=-1)
        mask_slice = mask[:, :, iz]
        ys, xs = np.nonzero(mask_slice) if mask_slice.any() else (np.array([Ny // 2]), np.array([Nx // 2]))
        # centroid of the object within this slice (intensity-weighted)
        w = mean_slice[mask_slice] if mask_slice.any() else np.array([1.0])
        cx = int(round(np.average(xs, weights=w))) if mask_slice.any() else Nx // 2
        cy = int(round(np.average(ys, weights=w))) if mask_slice.any() else Ny // 2

        max_size = min(Nx, Ny) - 2
        sizes, fluct = weisskoff_curve(img[:, :, iz, :], (cx, cy), max_size)

        # Theoretical pure-thermal-noise line, anchored at the smallest ROI.
        sizes_arr = np.array(sizes)
        theoretical = fluct[0] / sizes_arr

        # Radius of decorrelation: first size where measured/theoretical
        # ratio exceeds 2x (standard fBIRN convention is where the curve
        # departs from the fitted line by a fixed factor).
        ratio = np.array(fluct) / theoretical
        rdc_idx = np.argmax(ratio > 2.0) if np.any(ratio > 2.0) else len(sizes) - 1
        rdc = sizes[rdc_idx]

        print(f'\n=== {label}, z={iz}, centroid=({cx},{cy}) ===')
        print(f'  fluctuation at N=1 (single voxel): {fluct[0]:.2f}%')
        print(f'  fluctuation at N={sizes[-1]} (largest ROI): {fluct[-1]:.3f}% '
              f'(pure-thermal prediction: {theoretical[-1]:.3f}%)')
        print(f'  radius of decorrelation (Rdc, measured/theoretical > 2x): N={rdc} voxels')
        print(f'  plateau ratio at largest ROI (measured/theoretical): {ratio[-1]:.1f}x')

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(sizes, fluct, 'o-', label='measured')
        ax.loglog(sizes, theoretical, '--', label='pure thermal noise (1/N)')
        ax.axvline(rdc, color='gray', ls=':', label=f'Rdc = {rdc}')
        ax.set_xlabel('ROI size N (voxels per side)')
        ax.set_ylabel('percent fluctuation (%)')
        ax.set_title(f'{label}: Weisskoff radius-of-decorrelation')
        ax.legend()
        plt.tight_layout()
        out_dir = os.path.join(datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)
        suffix = f'_{variant}' if variant else ''
        fn_out = os.path.join(out_dir, f'lowres_calib_weisskoff_check{suffix}.png')
        plt.savefig(fn_out, dpi=130)
        print(f'  Wrote {fn_out}')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdirs', nargs='+')
    parser.add_argument('--seqname', default='ArbEPI')
    parser.add_argument('--variant', default='')
    parser.add_argument('--skip-frames', type=int, default=1)
    args = parser.parse_args()
    main(args.datdirs, args.seqname, args.variant, args.skip_frames)
