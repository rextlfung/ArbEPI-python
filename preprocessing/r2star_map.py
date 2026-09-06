"""Estimates a T2*/R2* map from the same dual-echo deGRE data already used
for B0 field-map estimation (run_b0map.py), for the generalized complex
field-map correction in recon/lowres_calib_recon_b0complex.py.

Motivation: recon/operators_b0.py's time-segmented correction only
demodulates *phase* (off-resonance) -- it leaves T2*/T1 amplitude decay
completely uncorrected, which lowres_calib_t2star_check.py measured as a
real, partial cause of this pipeline's temporal instability (a given
(ky,kz) calibration location is acquired at a different echo index --
hence a different amount of T2* decay -- in different frames). The fix is
to generalize the real field map Δf(r) (Hz) to a complex field
ψ(r) = i*2π*Δf(r) - R2*(r), so the same segmented-exponential machinery
corrects magnitude decay alongside phase. See CLAUDE.md's recon/ "B0
off-resonance correction" section for the generalization's derivation and
the reference time discussion (why the correction targets TE_nominal, not
t=0 -- both magnitude and phase share the same reference by construction,
since they're the real and imaginary parts of the same complex exponent).

R2* estimation: a plain two-point log-ratio,
    R2*(r) = ln(|S(TE1,r)| / |S(TE2,r)|) / (TE2 - TE1)
using the SAME two GRE echoes already used for Δf(r) -- no new acquisition
needed. This is much cruder than MRIFieldmaps.jl's regularized NCG fit for
Δf (a closed-form 2-point estimate, not an optimization), so it reuses
run_b0map.py's own `mask_degre` (magnitude threshold ANDed with an ESPIRiT
eigenvalue mask, already computed and stored in `<seqname>_b0map.h5`) to
exclude background/noise-only voxels where the log-ratio is meaningless,
and clips to a physically sane R2* range to avoid a handful of low-SNR
voxels producing wild outliers that would otherwise poison the complex
field's segmentation fit.

Per-echo image reconstruction matches preprocessing/gre_diagnostics.py's
own RSS coil combine exactly (same _ift3 convention, same per-echo
sqrt(sum_c|.|^2)), so the R2* map is consistent with the same GRE images
that script already renders for visual QA.

Returns the map on the EPI grid (Nx,Ny,Nz), matching how run_b0map.py
already resizes b0map_hz from the deGRE grid up to the EPI grid -- same
recipe (mask-before-resize, cubic spline) applied here to R2* instead.

Usage: import estimate_r2star_map_epi_grid and call directly; this module
has no __main__ (it's a small helper, not a standalone diagnostic driver).
"""

import os

import h5py
import numpy as np

from preprocessing.grid_resize import resize_to_epi_grid


def _ift3(d: np.ndarray) -> np.ndarray:
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(d, axes=axes), axes=axes), axes=axes)


def _rss_echo_images(ksp_echoes: np.ndarray) -> np.ndarray:
    """ksp_echoes: (Nx,Ny,Nz,n_echoes,Nc). Returns (Nx,Ny,Nz,n_echoes) RSS
    magnitude per echo -- identical computation to gre_diagnostics.py's
    img_echoes, kept in sync deliberately (both read the same cache)."""
    n_echoes = ksp_echoes.shape[3]
    return np.stack(
        [
            np.sqrt(np.sum(np.abs(_ift3(ksp_echoes[:, :, :, ie, :])) ** 2, axis=-1))
            for ie in range(n_echoes)
        ],
        axis=-1,
    )


def estimate_r2star_map_degre_grid(
    datdir: str, seqname: str = 'ArbEPI', r2star_max: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (r2star_degre, mask_degre): R2* in 1/s, masked to 0 outside
    run_b0map.py's own deGRE-grid mask, clipped to [0, r2star_max] (200/s
    corresponds to T2*=5ms, an extremely short outlier floor -- real
    tissue/phantom T2* at this field strength is far longer). Requires
    <seqname>_gre.h5 (STEP 2's ksp_gre_echoes/TE_degre) and
    <seqname>_b0map.h5 (run_b0map.py's mask_degre) to already exist."""
    recon_dir = os.path.join(datdir, 'recon')
    fn_gre = os.path.join(recon_dir, f'{seqname}_gre.h5')
    fn_b0map = os.path.join(recon_dir, f'{seqname}_b0map.h5')

    with h5py.File(fn_gre, 'r') as f:
        ksp_echoes = f['ksp_gre_echoes'][()]  # (Nx,Ny,Nz,n_echoes,Nc)
        te_degre = np.asarray(f.attrs['TE_degre'], dtype=np.float64)  # (n_echoes,) s
    if ksp_echoes.shape[3] < 2:
        raise ValueError(
            f'estimate_r2star_map_degre_grid: {fn_gre} has only '
            f'{ksp_echoes.shape[3]} echo(es) -- R2* needs at least 2.'
        )

    with h5py.File(fn_b0map, 'r') as f:
        mask_degre = f['mask_degre'][()].astype(bool)

    img_echoes = _rss_echo_images(ksp_echoes)  # (Nx,Ny,Nz,n_echoes)
    s1, s2 = img_echoes[..., 0], img_echoes[..., 1]
    dte = float(te_degre[1] - te_degre[0])

    eps = np.finfo(np.float64).eps
    valid = mask_degre & (s1 > eps) & (s2 > eps)
    r2star_degre = np.zeros(img_echoes.shape[:3], dtype=np.float32)
    r2star_degre[valid] = (np.log(s1[valid] / s2[valid]) / dte).astype(np.float32)
    # Physical R2* can't be negative (spins only decay, never gain signal
    # between echoes) -- a negative value here is pure noise, and clipping
    # it (rather than leaving it) keeps the complex field's segmentation
    # fit from being pulled around by voxels where the 2-point estimate is
    # unreliable (mask_degre already excludes background, but doesn't
    # guarantee every in-mask voxel has good echo-to-echo SNR).
    r2star_degre = np.clip(r2star_degre, 0.0, r2star_max)
    r2star_degre[~valid] = 0.0
    return r2star_degre, mask_degre


def estimate_r2star_map_epi_grid(
    datdir: str, seqname: str, fov_degre: tuple[float, float, float],
    fov: tuple[float, float, float], n_target: tuple[int, int, int],
) -> np.ndarray:
    """Same mask-then-resize recipe run_b0map.py uses for b0map_hz (order=3
    cubic spline, masked before resizing so interpolation doesn't blend in
    a background fill value at the mask boundary) -- see that module for
    the original. Returns (Nx,Ny,Nz) float32, 1/s, on the EPI grid."""
    r2star_degre, mask_degre = estimate_r2star_map_degre_grid(datdir, seqname)
    r2star = resize_to_epi_grid(
        r2star_degre * mask_degre, fov_degre, fov, n_target, order=3,
    ).astype(np.float32)
    return np.clip(r2star, 0.0, None)  # cubic-spline overshoot can dip slightly negative
