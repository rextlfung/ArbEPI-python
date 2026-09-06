"""Directly measures what fraction of the total complex-valued temporal
fluctuation in lowres_calib_recon.py's output shows up as *magnitude*
variation (visible in the saved, magnitude-only .nii.gz) versus being pure
phase variation (invisible there).

This is a different, more direct question than lowres_calib_b1_drift_check.py
answers. That script's SVD decomposition found the dominant mode's *shared
temporal coefficient* c1(t) is phase-dominated (its own magnitude barely
fluctuates, its own phase sweeps ~100+ degrees) -- but that is a statement
about one scalar shared across every voxel, not about what happens at any
individual voxel. Each voxel's actual value is
    X(v,t) = mean(v) + u1(v)*c1(t) + ...
and whether a rotation of the shared c1(t) shows up as a *magnitude* change
at voxel v depends entirely on how u1(v)'s own phase relates to mean(v)'s
own phase: the component of the perturbation parallel to mean(v)'s phase
moves |X(v,t)|, the component perpendicular to it (to first order) doesn't.
Since u1's spatial phase is not uniformly aligned with the object's own
phase (confirmed: the magnitude-correlation numbers in the b1 check are
well below 1), a phase-rotating shared coefficient generically produces
real, voxel-local magnitude swings wherever that alignment isn't exactly
90 degrees -- which is the typical case, not a special one. So "the shared
coefficient is phase-dominated" does NOT imply "the visible images barely
change in magnitude" -- and this script checks that directly on the raw
data instead of arguing it from the SVD's structure.

For each voxel, decomposes the complex deviation d(v,t) = X(v,t) - mean(v)
into the component parallel to mean(v)'s own phase (drives magnitude
change, to first order) and the component perpendicular to it (drives
phase change, magnitude-neutral to first order), sums the squared energy
of each across every voxel and frame, and reports both that first-order
split and the *exact* (not first-order-approximated) magnitude variance,
each as a fraction of the total complex-valued deviation energy -- the
direct answer to "how much of the total fluctuation this pipeline measures
is actually visible as magnitude change."

Usage (from repo root, .venv-preprocessing):
    .venv-preprocessing/bin/python -m preprocessing.lowres_calib_magnitude_variance_fraction_check <datdir> [datdir2 ...]
"""

import argparse
import os

import h5py
import numpy as np

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.lowres_calib_recon import _load_chunked, compute_calib_mask, lowres_calib_recon
from preprocessing.lowres_temporal_stability import object_mask


def magnitude_variance_fraction(img: np.ndarray, mask: np.ndarray, skip_frames: int = 1) -> dict:
    """img: [Nx,Ny,Nz,Nframes] complex. mask: [Nx,Ny,Nz] bool. See module
    docstring for the parallel/perpendicular (radial/tangential) split."""
    X = img[mask][:, skip_frames:]  # (Nvox, Nframes') complex
    mean_vox = X.mean(axis=1, keepdims=True)  # (Nvox, 1)
    d = X - mean_vox  # complex deviation from each voxel's own time-mean

    theta = np.angle(mean_vox)  # (Nvox, 1) -- each voxel's own steady-state phase
    d_rot = d * np.exp(-1j * theta)  # rotate into that voxel's own phase frame
    d_radial = d_rot.real  # component parallel to mean(v)'s phase -- moves |X|
    d_tangential = d_rot.imag  # component perpendicular -- moves phase, not |X| (1st order)

    var_radial = float(np.sum(d_radial**2))
    var_tangential = float(np.sum(d_tangential**2))
    var_total = var_radial + var_tangential  # == sum|d|^2, the total complex deviation energy

    # Exact (not first-order-approximated) magnitude variance, for comparison.
    mag = np.abs(X)
    mag_mean = mag.mean(axis=1, keepdims=True)
    var_magnitude_exact = float(np.sum((mag - mag_mean) ** 2))

    return dict(
        var_radial=var_radial, var_tangential=var_tangential, var_total=var_total,
        frac_radial_1st_order=var_radial / var_total,
        var_magnitude_exact=var_magnitude_exact,
        frac_magnitude_exact=var_magnitude_exact / var_total,
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
        r = magnitude_variance_fraction(img, mask, skip_frames)

        print(f'\n=== {label} ===')
        print('  1st-order, per-voxel-phase-frame decomposition of total complex deviation energy:')
        print(f'    magnitude-changing (radial) fraction:  {100 * r["frac_radial_1st_order"]:.1f}%')
        print(f'    phase-changing (tangential) fraction:  {100 * (1 - r["frac_radial_1st_order"]):.1f}%')
        print(f'  exact |X(v,t)| variance as a fraction of total complex deviation energy: '
              f'{100 * r["frac_magnitude_exact"]:.1f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdirs', nargs='+')
    parser.add_argument('--seqname', default='ArbEPI')
    parser.add_argument('--skip-frames', type=int, default=1)
    args = parser.parse_args()
    main(args.datdirs, args.seqname, args.skip_frames)
