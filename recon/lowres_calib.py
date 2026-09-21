"""Low-resolution reconstruction of the fully sampled k-space calibration region,
with optional B0-informed correction, and its temporal-stability check.
Subcommands:

    calib      reconstruct the calibration region (plain by default)
    stability  temporal-stability analysis of `calib`'s output

`calib` is a plain IFFT + smaps-weighted coil combine unless a B0 field map is
provided: `--b0` (uses <datdir>/recon/<seqname>_b0map.h5) or `--b0map PATH`
switch to the time-segmented, adjoint-only B0-corrected reconstruction, and
`--r2star` (with either) additionally corrects T2* amplitude decay via the
complex-field variant. torch/mirtorch are imported lazily, only on the B0
paths, so the plain path runs in .venv-preprocessing (no torch) while the B0
paths need .venv-recon:

    .venv-preprocessing/bin/python -m recon.lowres_calib calib <datdir> [seqname ...]
    .venv-recon/bin/python -m recon.lowres_calib calib <datdir> [seqname] --b0 [--r2star] [--L 32] [--nbins 128] [--device cuda]
    .venv-preprocessing/bin/python -m recon.lowres_calib stability <datdir> [<datdir> ...] [--variant {,b0,b0complex}]

The sections below are the original module docstrings, kept verbatim (the two
B0 sections' `python -m recon.lowres_calib {b0,b0complex}` usage is now
`calib --b0` / `calib --b0 --r2star`; their note that compute_calib_mask/
native_calib_grid are duplicated no longer applies -- there is one copy).

Formerly recon/lowres_calib/lowres_calib_recon.py
-------------------------------------------------
Quick low-res sanity-check reconstruction from the fully-sampled k-space
calibration region, for a fast look at a dataset without running the full
iterative Stage-2 pipeline (sigpy_recon.py / recon/).

Every frame's (ky, kz) sampling mask (see sampling/pd_sample.py's
`calib_frac`) always includes a small, fully-sampled centered region --
the calibration region ESPIRiT itself is calibrated from (via the deGRE
scan, not this one). That same guarantee holds for the *EPI* acquisition's
own per-frame mask: `omegas[..., t]` is `calib_mask | (extra incoherent
samples)` for every frame `t`, so `np.all(omegas, axis=-1)` (the
intersection across all frames) recovers exactly that calibration region --
no need to know `params.pd_calib_frac`/`R` ahead of time, or assume they
match the current defaults for a dataset acquired under different settings,
and no dependence on the region's shape either (a plain set intersection,
which works the same whether `calib_mask` is an ellipse or a rectangle).
Verified empirically on both `20260822ball_*` datasets: 362/10800 (ky, kz)
locations, a centered ellipse, identical between the radial/laminar
variants (they share the same underlying (ky, kz) mask, only the per-frame
EPI shot ordering differs) -- both acquired under `pd_sample.py`'s original
area-matched-ellipse `calib_frac` semantics (fraction of the sample
budget); a since-reverted intermediate version briefly made calib_frac a
fixed fraction of k-space instead (independent of R), but that let the
calibration region consume nearly the entire sample budget at high
acceleration, so `_calib_side_frac` restored the fraction-of-sample-budget
sizing -- now realized as a rectangle rather than the original ellipse
(see that module's docstring and docs/review-findings.md item 195). This
paragraph's specific numbers are a historical record of those two
(ellipse-shaped) datasets, not a current claim about the shape a fresh
acquisition's calibration region will have.

Since that region is exactly, not approximately, fully sampled, no
iterative reconstruction is needed: masking `ksp_epi_zf` down to it,
inverse-FFTing, and combining coils with the existing ESPIRiT sensitivity
maps (`recon/smaps_<seqname>_sigpy.h5`, already normalized so
`sum_c |s_c|^2 <= 1`, see smaps.py's process_smaps) is the correct linear
estimate directly -- `img = sum_c conj(s_c) * ifft(ksp_c)`, no
regularization or iteration.

**Reconstructs at native resolution, not zero-padded to the full (Nx, Ny,
Nz) acquisition grid.** Standard Cartesian MRI relation: resolution =
FOV/N (Delta_k = 1/FOV, and N samples span a k-space extent of N*Delta_k =
N/FOV). The calibration region's (ky, kz) bounding box -- 49 x 10 samples
on both `20260822ball_*` datasets, out of the full 240 x 45 -- caps the
achievable in-plane resolution at FOV_y/49 = 4.41 mm and FOV_z/10 = 4.05 mm,
far coarser than the full acquisition's 0.9 mm. Zero-padding that region up
to the full (Nx, Ny, Nz) grid before IFFT (an earlier version of this
script did exactly that) is pure sinc interpolation -- it doesn't add any
real information, and it makes neighboring voxels highly correlated by
construction (heavily oversampled relative to the true resolution), which
inflates variance-based diagnostics run on the result (e.g. an SVD/PCA
decomposition's "fraction of variance in the top component" -- see
preprocessing/lowres_calib_gain_drift_check.py). Reconstructing directly at
the native grid size gives the same true image content without the
redundant interpolation. `kx` (the readout direction, fully sampled on
every echo, not calibration-limited) is *also* cropped to match, to the
same effective sample count as `ky` -- `Nx_eff = round(Ny_eff * FOV_x /
FOV_y)`, which on these two datasets (FOV_x == FOV_y) works out to exactly
49, matching `Ny_eff` -- an explicit choice to keep the two in-plane axes
at matched resolution rather than leaving `kx` at full resolution while
`ky`/`kz` are calibration-limited.

Same centered-IFFT convention as sigpy_recon.py's/gre_diagnostics.py's _ift3,
and the same FOV-preserving resize (`grid_resize.resize_to_epi_grid`) the
rest of this pipeline already uses to move smaps between grids of the same
FOV at different resolutions.

Usage (from repo root, .venv-preprocessing -- this module needs sigpy/h5py/
matplotlib, not torch/mirtorch, despite living under recon/ alongside the
torch-based MSLR pipeline; see CLAUDE.md's recon/ section for the venv
split rationale):
    .venv-preprocessing/bin/python -m recon.lowres_calib calib <datdir> [seqname]


Formerly recon/lowres_calib/lowres_temporal_stability.py
--------------------------------------------------------
Temporal stability analysis of lowres_calib.py's output.

That reconstruction masks every frame down to the same fixed, fully-sampled
(ky, kz) calibration region before IFFT + smaps combine (see its module
docstring) -- so unlike a full-resolution reconstruction, there is no
frame-varying undersampling mask/trajectory in the signal path here at all.
Any temporal variation this script measures is therefore attributable to
the object/system itself (thermal noise, scanner drift, motion) and *not*
to which (ky, kz) locations a given frame happened to sample -- the
standard NEMA/fBIRN-style phantom stability decomposition (percent
fluctuation + linear drift from a per-frame ROI-mean signal curve, plus a
per-voxel tSNR map) is used to quantify that directly.

variant='' reads recon/lowres_calib.py's plain output;
variant='b0' reads recon/lowres_calib.py's B0-corrected output --
see that module's docstring for why per-frame B0-induced phase, not just
system drift/noise, is expected to show up here as apparent instability
for a static object.

Usage (from repo root, .venv-preprocessing -- matplotlib/nibabel, not
torch/mirtorch, despite comparing recon/lowres_calib.py's
.venv-recon-only output; see CLAUDE.md's recon/ section "not a
single-venv package" note):
    .venv-preprocessing/bin/python -m recon.lowres_calib stability <datdir> [--seqname ArbEPI] [--variant b0]


Formerly recon/lowres_calib/lowres_calib_recon_b0.py
----------------------------------------------------
B0-corrected variant of recon/lowres_calib.py: same fully-sampled
(ky, kz) calibration region, same "no iteration, no regularization"
philosophy (`img = sum_c conj(smap_c) * ifft(ksp_c)`), but through
recon/operators.py's GatheredSenseB0 adjoint instead of a plain 3D
IFFT, so time-segmented conjugate-phase (Sutton/Noll/Fessler) B0
demodulation is included.

Motivation: even though every frame samples the exact same (ky, kz)
calibration locations (see lowres_calib.py's module docstring), the
*order* in which a given frame's shots visit them differs, so a given
(ky, kz) location is acquired at a different echo time (time since RF
excitation) in different frames -- confirmed against this repo's own
`echo_times` array: per-echo timing is frame-invariant as a *set*
(sequences/ArbEPI.py), but which (ky,kz) location maps to which echo
index varies per frame, exactly like the full acquisition. Off-resonance
phase accrues with that time, so the same k-space location carries a
different B0-induced phase from frame to frame -- for a genuinely static
object (a phantom), that is a source of *apparent* temporal instability
that has nothing to do with real signal change.

Does NOT reuse recon.operators.build_encoding_operator_b0 directly:
that function's shared time-segmentation fit is built from frame 0's
distinct echo times alone, on the (correct, for its own use case)
assumption that a frame's full ETL-worth of samples covers every echo
time the fit could ever need. That doesn't hold for the small calibration
region alone -- a given frame's calibration-region samples can miss some
of the echo times other frames' calibration samples use -- so
`_build_calib_operator_b0` below takes the union of echo times across
every frame's calibration-region samples instead of frame 0's alone.
Otherwise mirrors build_encoding_operator_b0's current construction
exactly, including GatheredSenseB0's (smaps, samp, pos, b_by_echo,
c_phasors) contract (b_by_echo shared across frames as one (n_unique_t,L)
table, each frame supplies only its own row-index vector `pos` -- not a
per-frame pre-gathered (K,L) copy).

L defaults to 32 here (not operators.py's own L=6 default) -- see
CLAUDE.md's recon/ "B0 off-resonance correction" section: a real-scale
sweep (recon/analysis.py) found L=6 badly under-resolves this
pipeline's real ETL=60 bandwidth-time product, while L=32 is the smallest
value that gets relative forward-model error under 1%.

Reconstructs at native (resolution-matched) grid size, not zero-padded to
the full (Nx,Ny,Nz) acquisition grid -- see lowres_calib.py's module
docstring for the FOV/N resolution derivation. compute_calib_mask/
native_calib_grid are duplicated from there rather than imported, to keep
this .venv-recon-only module off that file's module-level matplotlib
import (not part of the `recon` pyproject extra -- see CLAUDE.md's recon/
section "not a single-venv package" note).

Sign convention: this reconstruction only ever calls `.adjoint()`, never
`.apply()`. For the phase-only field this module builds (no R2* term),
that needs no special handling -- `c_phasors` here is a pure rotation
(|exp(i*theta)| = 1), so its conjugate *is* its own multiplicative
inverse, and GatheredSenseB0._apply_adjoint's existing `.conj()` already
does the right thing. (A complex field generalizing this to also correct
T2*/T1 decay, as recon/operators.py's r2star_map parameter supports for
the bidirectional/iterative path, needs a different, deliberately-flipped
sign for an adjoint-only reconstruction -- see
build_encoding_operator_b0's own docstring for why -- and isn't
implemented here.)

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.lowres_calib calib --b0 <datdir> \
        [--seqname ArbEPI] [--device cuda]


Formerly recon/lowres_calib/lowres_calib_recon_b0complex.py
-----------------------------------------------------------
Generalized-complex-field-map variant of recon/lowres_calib.py:
same fully-sampled calibration region, same adjoint-only philosophy, but
the time-segmented correction now accounts for a COMPLEX field combining
off-resonance and T2* decay, generalizing the real-only Δf(r) (Hz) that
recon/operators.py's mri_exp_approx wraps (mirtorch's own source raises
TypeError on a complex b0 input, so this generalization can't be done by
just passing it a complex array -- it needs its own spatial-basis
construction, below).

The *physical* forward-model exponent is psi(r) = i*2*pi*Δf(r) - R2*(r)
(signal decays as time since excitation increases). The array this module
actually builds and feeds to GatheredSenseB0, psi_recon = i*2*pi*Δf(r) +
R2*(r) (plus sign), is deliberately NOT that -- see
_build_calib_operator_b0_complex's inline comment for why: this
reconstruction only ever calls .adjoint(), never .apply(), and
GatheredSenseB0._apply_adjoint always conjugates c_phasors. Conjugating a
real quantity is a no-op, so building c_phasors from the physical -R2*
would make the adjoint apply the same decay a second time instead of
undoing it (confirmed empirically in the exploratory branch this was
ported from -- a version with the physical sign measured tSNR getting
monotonically *worse* through uncorrected -> phase-only -> this
complex-field version, the opposite of the expected direction). The true
multiplicative inverse of exp(psi*t) is exp(-psi*t), which only equals
exp(conj(psi)*t) when Re(psi)=0 (pure rotation, the phase-only case
lowres_calib.py already validated) -- psi_recon is chosen so
that conjugating it reproduces that true inverse instead.

Does NOT reuse recon.operators.build_encoding_operator_b0 directly, for
the same reason lowres_calib.py doesn't -- see that module's
docstring. Also does NOT copy the worktree exploratory branch's
GatheredSenseB0 call verbatim: that branch predates the current
(smaps, samp, pos, b_by_echo, c_phasors) contract (docs/review-findings.md
item 75) and passed a pre-gathered per-frame `b` instead -- adapted here to
build `pos` (a row-index vector into the shared `b_by_echo` table) exactly
like lowres_calib.py's `_build_calib_operator_b0` does.

Motivation (see CLAUDE.md's recon/ "B0 off-resonance correction" section
for the full derivation): lowres_calib.py's phase-only correction
demodulates off-resonance but leaves T2*/T1 amplitude decay uncorrected --
a given (ky,kz) calibration location is acquired at a different echo
index, hence a different amount of decay, in different frames, exactly
the same TE-scrambling mechanism that motivates the phase correction.
Generalizing Δf(r) to psi(r) corrects both simultaneously with the same
L-segment machinery.

Reference time = TE_nominal, not t=0 (excitation): both the magnitude
(R2*) and phase (Δf) corrections are the real and imaginary parts of the
SAME complex exponent exp(psi(r)*t), so they share one time reference by
construction -- shifting the per-sample times fed into the segmentation
fit by -TE_nominal makes the reconstruction target "the image as it would
appear at the prescribed TE" (the standard GRE/EPI T2*-weighted
convention) rather than "the undecayed image at the moment of excitation,"
which is neither standard nor numerically favorable (voxels with short
T2* would need very large amplification factors relative to t=0's much
earlier lead-in interval). TE_nominal is read directly from scan_info.mat's
schedules[...,2] at echo index (ETL-1)//2 -- the nominal-TE echo,
frame/shot-invariant by construction (see CLAUDE.md's mask2epi_radial
paragraph) -- not re-derived from the calibration region's own (possibly
incomplete) echo-time coverage.

Segmentation strategy: reuses mri_exp_approx's existing, already-tuned
(L=32) temporal interpolation weights (b_by_echo) and segment-placement
times (tl) UNCHANGED -- computed from Δf(r) alone, exactly as the
phase-only version does, just against TE-shifted times. Only the SPATIAL
basis functions are regeneralized: instead of mri_exp_approx's own
phase-only exp(i*2*pi*Δf(r)*tl[l]), this builds exp(psi(r)*tl[l]) directly
from the full, continuous per-voxel complex field (not the histogram-
binned reference values mri_exp_approx uses internally only to fit
b_by_echo). This is deliberately not a from-scratch joint (Δf, R2*)
segmentation fit: Δf's bandwidth-time product is what drove L=32 (see
CLAUDE.md's sweep finding, ~27 at this pipeline's real ETL=60/72ms scale);
R2*'s own decay-time product (R2*_typical * echo-train duration) is
printed by this script's main() specifically to check that it's small by
comparison, which is what justifies reusing Δf-only-tuned interpolation
weights for R2* too rather than re-deriving a joint fit.

R2*(r) comes from preprocessing/r2star_map.py's two-point estimate on the
same dual-echo deGRE data already used for Δf(r) -- see that module's
docstring.

Usage (from repo root, .venv-recon):
    .venv-recon/bin/python -m recon.lowres_calib calib --b0 --r2star <datdir> \
        [--seqname ArbEPI] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import TYPE_CHECKING

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from preprocessing.config import PreprocessingConfig, load_config, load_seq_params, set_seq_paths
from preprocessing.grid_resize import resize_to_epi_grid
from preprocessing.matio import read_mat
from preprocessing.nifti_io import save_recon_nifti
from preprocessing.r2star_map import estimate_r2star_map_epi_grid
from recon.hdf5_chunked_io import read_frames_cropped

if TYPE_CHECKING:
    import torch
    from mirtorch.linear import BlockDiagonal


def _ift3(d: np.ndarray) -> np.ndarray:
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(d, axes=axes), axes=axes), axes=axes)


def compute_calib_mask(omegas: np.ndarray) -> np.ndarray:
    """omegas: [Ny, Nz, Nframes] bool. Returns [Ny, Nz] bool: the (ky, kz)
    locations sampled in *every* frame -- the fully-sampled calibration
    region every frame's mask is built around (see module docstring)."""
    return np.all(omegas, axis=-1)


def native_calib_grid(
    calib_mask: np.ndarray, fov: tuple[float, float, float], Nx_full: int
) -> dict:
    """calib_mask: [Ny, Nz] bool. fov: (fx, fy, fz) m, the full acquisition
    FOV (unchanged by any of this -- only resolution/N changes). Nx_full:
    the full readout matrix size (240 on these datasets).

    Returns the native (resolution-matched, not zero-padded) grid size and
    the centered crop slices into the full (Nx, Ny, Nz) k-space array --
    see module docstring for the FOV/N resolution relation and why kx is
    cropped too, to Ny's effective sample count."""
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


def lowres_calib_recon(
    ksp_crop: np.ndarray, calib_mask: np.ndarray, grid: dict, smaps: np.ndarray,
    fov: tuple[float, float, float],
) -> np.ndarray:
    """ksp_crop: [Nx_eff, Ny_eff, Nz_eff, Nvcoils, Nframes] -- already
    cropped to `grid`'s slices. Callers compute `grid` (via
    native_calib_grid) from `calib_mask` *before* reading k-space, so the
    HDF5 read itself can crop to this bounding box (recon/hdf5_chunked_io.py's
    read_frames_cropped) instead of loading the full dense array and
    cropping after -- see docs/review-findings.md item 204 for why that
    matters (the full array is ~201GB at this pipeline's real 0.8mm/R~93.5
    scale, vs. a few hundred calibration-region samples actually needed).
    calib_mask: [Ny, Nz] bool, full-grid shape (cropped internally via
    grid's y/z slices). smaps: [Nx, Ny, Nz, Nvcoils] complex, sum_c|s_c|^2
    <= 1 normalized, full grid. fov: (fx, fy, fz) m.

    Returns img: [Nx_eff, Ny_eff, Nz_eff, Nframes] complex -- per-frame,
    sensitivity-map-weighted coil combination of the calibration-region-only
    image, reconstructed at native resolution (see module docstring) rather
    than zero-padded to the full grid.
    """
    ys, zs = grid['y_slice'], grid['z_slice']
    # [Ny_eff, Nz_eff] -- calib_mask's own shape within its bounding box. A
    # no-op for a rectangular calib_mask (its bounding box is itself, fully
    # true), but real masking for e.g. an older ellipse-shaped one, whose
    # own bounding box has corners outside the region -- kept generic since
    # calib_mask's shape isn't assumed here.
    calib_mask_crop = calib_mask[ys, zs]
    ksp_crop = ksp_crop * calib_mask_crop[None, :, :, None, None]  # zero any stray non-calib sample

    img_coils = _ift3(ksp_crop)  # [Nx_eff, Ny_eff, Nz_eff, Nvcoils, Nframes]

    n_target = (grid['Nx_eff'], grid['Ny_eff'], grid['Nz_eff'])
    smaps_native = resize_to_epi_grid(smaps, fov, fov, n_target, order=3)
    img = np.sum(np.conj(smaps_native)[..., None] * img_coils, axis=3)
    return img


def _recon_one(cfg: PreprocessingConfig, seqname: str) -> None:
    datdir = cfg.datdir
    paths = set_seq_paths(cfg, seqname)
    seq_params = load_seq_params(paths)

    fn_epi_zf = paths.recon
    fn_smaps = os.path.join(datdir, 'recon', f'smaps_{seqname}_sigpy.h5')

    print(f'Loading sampling mask ({fn_epi_zf})...')
    with h5py.File(fn_epi_zf, 'r') as f:
        omegas = f['omegas'][()]  # [Ny, Nz, Nframes]
        Nx_full = f['ksp_epi_zf'].shape[0]

    calib_mask = compute_calib_mask(omegas)
    n_calib = int(calib_mask.sum())
    print(f'Calibration region: {n_calib} / {calib_mask.size} (ky, kz) locations')
    if n_calib == 0:
        raise RuntimeError(
            'lowres_calib_recon: empty calibration region (no (ky, kz) location is '
            'sampled in every frame) -- is this dataset actually fully sampled at '
            'k-space center?'
        )

    # Computed before the k-space load (not after, as an earlier version of
    # this function did internally) so the HDF5 read itself can crop to
    # this bounding box -- see docs/review-findings.md item 204.
    grid = native_calib_grid(calib_mask, seq_params.fov, Nx_full)
    xs, ys, zs = grid['x_slice'], grid['y_slice'], grid['z_slice']

    print(f'Loading {fn_epi_zf} (cropped to the calibration-region bounding box)...')
    ksp_crop = read_frames_cropped(
        fn_epi_zf, 'ksp_epi_zf', spatial_slices=(xs, ys, zs)
    )  # [Nx_eff, Ny_eff, Nz_eff, Nvcoils, Nframes]

    print(f'Loading {fn_smaps}...')
    with h5py.File(fn_smaps, 'r') as f:
        smaps = f['smaps'][()]  # [Nx, Ny, Nz, Nvcoils]

    print('Reconstructing at native (resolution-matched) grid size...')
    img = lowres_calib_recon(
        ksp_crop, calib_mask, grid, smaps, seq_params.fov
    )  # [Nx_eff, Ny_eff, Nz_eff, Nframes]
    voxel_mm = [1000 * seq_params.fov[a] / img.shape[a] for a in range(3)]
    print(f'  Native grid: {img.shape[:3]}  (voxel size {voxel_mm[0]:.3f} x '
          f'{voxel_mm[1]:.3f} x {voxel_mm[2]:.3f} mm)')

    out_dir = os.path.join(datdir, 'recon', 'lowres_calib')
    os.makedirs(out_dir, exist_ok=True)
    fn_out = os.path.join(out_dir, f'{seqname}_recon_lowres_calib')
    save_recon_nifti(
        fn_out, img, fov=seq_params.fov, seqname=seqname,
        n_calib_samples=n_calib, n_ky_kz=int(calib_mask.size),
        native_grid=list(img.shape[:3]), native_voxel_mm=voxel_mm,
        note='IFFT + smaps-weighted coil combine of the fully-sampled k-space center only, '
             'reconstructed at native (resolution-matched) grid size, not zero-padded',
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
    plt.close(fig)
    print(f'Wrote {fn_png}')


def run_lowres_calib_recon(cfg: PreprocessingConfig) -> None:
    """Batch driver, mirroring sigpy_recon.py/run_preprocessing.py's per-sequence
    try/except pattern -- runs _recon_one for every cfg.seqnames, continuing
    past a single sequence's failure rather than aborting the batch."""
    print(f'Batch: {len(cfg.seqnames)} sequence(s) in {cfg.datdir}')
    for i, seqname in enumerate(cfg.seqnames, start=1):
        print(f'\n[{i}/{len(cfg.seqnames)}] {seqname}')
        try:
            _recon_one(cfg, seqname)
        except Exception as e:  # noqa: BLE001 -- mirrors sigpy_recon.py's per-sequence try/catch
            print(f"ERROR [{seqname}]: {e}\nSkipping...")
    print('\nBatch complete.')


def _build_calib_operator_b0(
    smaps_chw: torch.Tensor,
    calib_omega: torch.Tensor,
    b0map_hz: torch.Tensor,
    echo_times_s: torch.Tensor,
    L: int = 32,
    nbins: int = 128,
) -> BlockDiagonal:
    """Same construction as recon.operators.build_encoding_operator_b0's
    current (post item-75) contract -- GatheredSenseB0(smaps, samp, pos,
    b_by_echo, c_phasors), b_by_echo shared across frames -- except the
    shared time-segmentation fit (mri_exp_approx) is built from the union
    of echo times across every frame's calibration-region samples, not
    frame 0's alone. See module docstring for why frame 0 alone isn't a
    superset here."""
    import torch
    from mirtorch.linear import BlockDiagonal
    from mirtorch.linear.mri import mri_exp_approx

    from recon.operators import GatheredSenseB0, _check_b_weight_row_sums

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
    _check_b_weight_row_sums(b_by_echo, 'shared (union of calib-region echo times)')
    b_by_echo = b_by_echo.to(smaps_chw.dtype)  # (n_unique_t, L)
    c_phasors = c.transpose(0, 1).reshape((L,) + N).to(smaps_chw.dtype)

    frames = []
    for it in range(Nt):
        samp = calib_omega[..., it]
        t_ms = t_ms_list[it]
        pos = torch.searchsorted(unique_t_ms, t_ms).clamp(max=unique_t_ms.numel() - 1)
        assert torch.allclose(unique_t_ms[pos], t_ms, atol=1e-4), (
            f"_build_calib_operator_b0: frame {it}'s sample echo times aren't in the "
            "union of every frame's calib-region echo times -- shouldn't be reachable "
            'by construction.'
        )
        frames.append(GatheredSenseB0(smaps_chw, samp, pos, b_by_echo, c_phasors))
    return BlockDiagonal(frames)


def gather_calib_ksp(ksp_epi_zf: np.ndarray, idx_full: np.ndarray) -> np.ndarray:
    """ksp_epi_zf: [Nx, Ny, Nz, Nc, Nt] numpy (plain, not torch -- kept off
    the GPU/out of any torch tensor since it can be many GB and only a
    small fraction of it is the calibration region). idx_full: flat
    C-order spatial indices into (Nx*Ny*Nz), identical for every frame
    (the calibration region is frame-invariant). Returns [K, Nc, Nt]
    complex64, matching GatheredSenseB0's forward/adjoint (K, Nc)
    per-frame convention -- see recon/operators.py's gather_ksp, which
    this mirrors but keeps ksp_epi_zf as plain numpy."""
    Nc, Nt = ksp_epi_zf.shape[3], ksp_epi_zf.shape[4]
    out = np.empty((idx_full.size, Nc, Nt), dtype=ksp_epi_zf.dtype)
    for it in range(Nt):
        flat = ksp_epi_zf[..., it].reshape(-1, Nc)
        out[:, :, it] = flat[idx_full, :]
    return out


def run_b0_corrected_calib_recon(
    datdir: str, seqname: str, L: int = 32, nbins: int = 128, device: str = 'cuda',
    fn_b0map: str | None = None,
) -> dict:
    """Core computation shared by main_b0() (writes the nifti) and any other
    consumer wanting the B0-corrected calibration-region reconstruction
    directly. fn_b0map: field-map .h5 (default <datdir>/recon/<seqname>_b0map.h5). Returns a dict: img_np [Nx_eff,Ny_eff,Nz_eff,Nt] complex64,
    fov, grid (native_calib_grid's return value), voxel_mm, n_calib,
    calib_mask, L, nbins."""
    import torch

    device_t = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    recon_dir = os.path.join(datdir, 'recon')
    fn_ksp = os.path.join(recon_dir, f'{seqname}_epi_zf.h5')
    fn_smaps = os.path.join(recon_dir, f'smaps_{seqname}_sigpy.h5')
    if fn_b0map is None:
        fn_b0map = os.path.join(recon_dir, f'{seqname}_b0map.h5')

    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    seq_params = load_seq_params(paths)
    fov = seq_params.fov

    print(f'Loading smaps ({fn_smaps})...')
    smaps_raw = torch.from_numpy(read_frames_cropped(fn_smaps, 'smaps').astype(np.complex64))
    smaps_rss = smaps_raw.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    smaps = smaps_raw / (smaps_rss + torch.finfo(torch.float32).eps)  # (Nx,Ny,Nz,Nc), CPU
    Nx, Ny, Nz, _Nvc = smaps.shape

    print(f'Loading B0 field map ({fn_b0map})...')
    b0map_hz = torch.from_numpy(read_frames_cropped(fn_b0map, 'b0map_hz').astype(np.float32))
    assert tuple(b0map_hz.shape) == (Nx, Ny, Nz), (
        f'b0map_hz shape {tuple(b0map_hz.shape)} != smaps grid ({Nx},{Ny},{Nz})'
    )

    print(f'Loading echo times / sampling mask ({fn_ksp})...')
    echo_times_2d = read_frames_cropped(fn_ksp, 'echo_times').astype(np.float32)  # (Ny,Nz,Nt), numpy
    with h5py.File(fn_ksp, 'r') as f:
        omegas = f['omegas'][()]  # (Ny, Nz, Nt)
    Nt = omegas.shape[-1]
    calib_mask = compute_calib_mask(omegas)  # (Ny, Nz)
    n_calib = int(calib_mask.sum())
    print(f'Calibration region: {n_calib} / {calib_mask.size} (ky, kz) locations')
    if n_calib == 0:
        raise RuntimeError(
            'run_b0_corrected_calib_recon: empty calibration region (no (ky, kz) '
            'location is sampled in every frame) -- is this dataset fully sampled '
            'at k-space center?'
        )

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
    A = _build_calib_operator_b0(
        smaps_chw, calib_omega, b0map_hz_native, echo_times_s, L=L, nbins=nbins
    )
    idx_full = A.A[0].idx.cpu().numpy()
    for it in range(1, Nt):
        assert np.array_equal(A.A[it].idx.cpu().numpy(), idx_full), (
            'calibration-region sample indices differ across frames -- unexpected'
        )

    print(f'Loading k-space ({fn_ksp}), cropped to the calibration-region bounding box, '
          'and gathering calibration-region samples...')
    # Cropped during the HDF5 read itself (recon/hdf5_chunked_io.py's
    # read_frames_cropped), not after loading the full dense array -- see
    # docs/review-findings.md item 204 (the full array is ~201GB at this
    # pipeline's real 0.8mm/R~93.5 scale, vs. a few hundred calibration-
    # region samples actually needed).
    ksp_epi_zf_crop = read_frames_cropped(
        fn_ksp, 'ksp_epi_zf', spatial_slices=(xs, ys, zs)
    ).astype(np.complex64)  # [Nx_eff,Ny_eff,Nz_eff,Nc,Nt]
    ksp_gathered = gather_calib_ksp(ksp_epi_zf_crop, idx_full)  # [K, Nc, Nt]
    ksp_calib = torch.from_numpy(ksp_gathered).to(device_t)

    print('Reconstructing (adjoint only -- no iteration, no regularization)...')
    img = A.adjoint(ksp_calib)  # (Nx_eff, Ny_eff, Nz_eff, Nt) complex64
    img_np = img.detach().cpu().numpy()

    return dict(
        img_np=img_np, fov=fov, grid=grid, voxel_mm=voxel_mm,
        n_calib=n_calib, calib_mask=calib_mask, L=L, nbins=nbins,
    )


def main_b0(
    datdir: str, seqname: str, L: int = 32, nbins: int = 128, device: str = 'cuda',
    fn_b0map: str | None = None,
) -> None:
    result = run_b0_corrected_calib_recon(datdir, seqname, L, nbins, device, fn_b0map)
    img_np, fov = result['img_np'], result['fov']
    grid, voxel_mm = result['grid'], result['voxel_mm']
    n_calib, calib_mask = result['n_calib'], result['calib_mask']
    Nx_eff, Ny_eff, Nz_eff = grid['Nx_eff'], grid['Ny_eff'], grid['Nz_eff']

    out_dir = os.path.join(datdir, 'recon', 'lowres_calib')
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


def nominal_te_s(scan_info_path: str, etl: int) -> float:
    """The prescribed-TE echo's acquisition time (seconds since RF
    excitation), frame/shot-invariant by construction -- see module
    docstring. Read directly from scan_info.mat rather than derived from
    the calibration region's own echo-time coverage, which can be an
    incomplete subset of the full ETL (see lowres_calib.py's
    _build_calib_operator_b0 docstring)."""
    raw = read_mat(scan_info_path, ['schedules'])['schedules']  # (Nframes,Nshots,ETL,3)
    return float(raw[0, 0, (etl - 1) // 2, 2])


def _build_calib_operator_b0_complex(
    smaps_chw: torch.Tensor,
    calib_omega: torch.Tensor,
    b0map_hz: torch.Tensor,
    r2star_map: torch.Tensor,
    echo_times_s_shifted: torch.Tensor,
    L: int = 32,
    nbins: int = 128,
) -> BlockDiagonal:
    """Same construction as lowres_calib.py's
    _build_calib_operator_b0, generalized to a complex field -- see module
    docstring for what changes (spatial basis only) and what doesn't
    (temporal interpolation weights, segment placement, both still fit
    from Δf(r) alone via an unmodified mri_exp_approx call)."""
    import torch
    from mirtorch.linear import BlockDiagonal
    from mirtorch.linear.mri import mri_exp_approx

    from recon.operators import GatheredSenseB0, _check_b_weight_row_sums

    Nt = calib_omega.shape[-1]
    device = smaps_chw.device
    b0_neg = (-b0map_hz).to(torch.float32)

    idx_list, t_ms_list = [], []
    for it in range(Nt):
        samp = calib_omega[..., it]
        idx = torch.nonzero(samp.reshape(-1), as_tuple=False).squeeze(-1)
        t_ms = (echo_times_s_shifted[..., it].reshape(-1)[idx] * 1000).to(torch.float32)
        idx_list.append(idx)
        t_ms_list.append(t_ms)
    unique_t_ms = torch.unique(torch.cat(t_ms_list), sorted=True)

    b_by_echo, _c_phase_only, tl = mri_exp_approx(b0_neg, nbins, L, unique_t_ms)
    _check_b_weight_row_sums(
        b_by_echo, 'shared complex-field fit (union of calib-region echo times, TE-referenced)'
    )
    b_by_echo = b_by_echo.to(smaps_chw.dtype)  # (n_unique_t, L)

    # Generalized complex spatial basis -- see module docstring. Reduces
    # exactly to mri_exp_approx's own phase-only c_phasors when
    # r2star_map == 0 (matches recon/operators.py's confirmed sign
    # convention: exp(i*2*pi*b0map_hz(r)*tl[l])).
    #
    # Sign note (+R2*, not -R2*): GatheredSenseB0._apply_adjoint always
    # applies c_phasors[l].conj() -- the correct *inverse* of a pure
    # rotation (|exp(i*theta)|=1, so conj == 1/(.)), which is why the
    # phase-only version works via adjoint alone. Conjugating a REAL
    # quantity is a no-op, though, so a naive exp(i*2*pi*Δf*t - R2**t)
    # here would have the adjoint apply the SAME decay attenuation a
    # second time instead of undoing it (see module docstring). The true
    # multiplicative inverse of exp(psi*t) is exp(-psi*t), not
    # exp(conj(psi)*t) -- those only coincide when Re(psi)=0. Feeding
    # GatheredSenseB0 a c_phasors built from psi_recon = i*2*pi*Δf + R2*
    # (note the sign flip on R2* relative to the physical forward-model
    # exponent) makes ITS conjugate equal exp(-psi_physical*t), the actual
    # decay-compensating inverse -- at the cost of this array no longer
    # being a physically faithful forward operator if .apply() were ever
    # called on it (it isn't, here: this reconstruction only ever calls
    # .adjoint()).
    psi_recon = (
        1j * 2 * math.pi * b0map_hz.to(torch.complex64)
        + r2star_map.to(torch.complex64)
    )  # (*N) complex
    tl_c = tl.to(torch.complex64).to(device)
    c_phasors = torch.exp(tl_c[:, None, None, None] * psi_recon[None, ...])  # (L,*N)
    c_phasors = c_phasors.to(smaps_chw.dtype)

    frames = []
    for it in range(Nt):
        samp = calib_omega[..., it]
        t_ms = t_ms_list[it]
        pos = torch.searchsorted(unique_t_ms, t_ms).clamp(max=unique_t_ms.numel() - 1)
        assert torch.allclose(unique_t_ms[pos], t_ms, atol=1e-4), (
            f"_build_calib_operator_b0_complex: frame {it}'s sample echo times aren't in "
            "the union of every frame's calib-region echo times -- shouldn't be reachable "
            'by construction.'
        )
        frames.append(GatheredSenseB0(smaps_chw, samp, pos, b_by_echo, c_phasors))
    return BlockDiagonal(frames)


def run_b0complex_corrected_calib_recon(
    datdir: str, seqname: str, L: int = 32, nbins: int = 128, device: str = 'cuda',
    zero_pad_z: bool = False, fn_b0map: str | None = None,
) -> dict:
    """Core computation shared by main_b0complex() and any other consumer wanting
    the complex-field-corrected calibration-region reconstruction
    directly. Returns a dict: img_np [Nx_eff,Ny_eff,Nz_eff,Nt] complex64,
    fov, grid (native_calib_grid's return value), voxel_mm, n_calib,
    calib_mask, L, nbins, te_nominal_s.

    zero_pad_z: forwarded to estimate_r2star_map_epi_grid's deGRE-grid ->
    EPI-grid resize -- see that function's docstring (docs/review-findings.md
    item 203). Needed for this session's 5.4mm variant, whose EPI z-FOV
    (145.8mm) exceeds deGRE's fixed 144mm slab -- the same condition
    smaps.py/run_b0map.py already need it for, since R2* estimation reads
    from the same deGRE acquisition."""
    import torch

    device_t = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    recon_dir = os.path.join(datdir, 'recon')
    fn_ksp = os.path.join(recon_dir, f'{seqname}_epi_zf.h5')
    fn_smaps = os.path.join(recon_dir, f'smaps_{seqname}_sigpy.h5')
    if fn_b0map is None:
        fn_b0map = os.path.join(recon_dir, f'{seqname}_b0map.h5')

    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    seq_params = load_seq_params(paths)
    fov, fov_degre = seq_params.fov, seq_params.fov_degre

    print(f'Loading smaps ({fn_smaps})...')
    smaps_raw = torch.from_numpy(read_frames_cropped(fn_smaps, 'smaps').astype(np.complex64))
    smaps_rss = smaps_raw.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    smaps = smaps_raw / (smaps_rss + torch.finfo(torch.float32).eps)  # (Nx,Ny,Nz,Nc), CPU
    Nx, Ny, Nz, _Nvc = smaps.shape

    print(f'Loading B0 field map ({fn_b0map})...')
    b0map_hz = torch.from_numpy(read_frames_cropped(fn_b0map, 'b0map_hz').astype(np.float32))
    assert tuple(b0map_hz.shape) == (Nx, Ny, Nz), (
        f'b0map_hz shape {tuple(b0map_hz.shape)} != smaps grid ({Nx},{Ny},{Nz})'
    )

    print('Estimating R2* map from dual-echo deGRE data...')
    r2star_hz = torch.from_numpy(
        estimate_r2star_map_epi_grid(
            datdir, seqname, fov_degre, fov, (Nx, Ny, Nz), zero_pad_z=zero_pad_z,
        )
    )

    print(f'Loading echo times / sampling mask ({fn_ksp})...')
    echo_times_2d = read_frames_cropped(fn_ksp, 'echo_times').astype(np.float32)  # (Ny,Nz,Nt), numpy
    with h5py.File(fn_ksp, 'r') as f:
        omegas = f['omegas'][()]  # (Ny, Nz, Nt)
    Nt = omegas.shape[-1]
    calib_mask = compute_calib_mask(omegas)  # (Ny, Nz)
    n_calib = int(calib_mask.sum())
    print(f'Calibration region: {n_calib} / {calib_mask.size} (ky, kz) locations')
    if n_calib == 0:
        raise RuntimeError(
            'run_b0complex_corrected_calib_recon: empty calibration region (no (ky, kz) '
            'location is sampled in every frame) -- is this dataset fully sampled '
            'at k-space center?'
        )

    te_nominal_s = nominal_te_s(paths.scan_info, seq_params.ETL)
    print(f'  TE_nominal = {te_nominal_s * 1000:.3f} ms '
          f'(echo index {(seq_params.ETL - 1) // 2} of {seq_params.ETL})')

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
    # R2* has much sharper local structure than smaps/b0map_hz (this
    # phantom's real air bubbles show up as T2* down to ~6ms in places),
    # and resize_to_epi_grid's cubic-spline zoom has no anti-aliasing
    # prefilter -- harmless for the smooth quantities above, but a real
    # ~5x EPI-grid -> native-grid downsample ratio here can alias by
    # ~23-25% locally in the T2* correction factor at typical echo-time
    # offsets from TE_nominal if left unfiltered. Gaussian-prefilter
    # before the zoom (standard decimation anti-aliasing, sigma set from
    # the actual per-axis downsample ratio) rather than touching
    # grid_resize.py itself, which smaps/b0map_hz both already use safely
    # without one.
    r2star_src = r2star_hz.numpy()
    ratios = [s / t for s, t in zip(r2star_src.shape, n_target)]
    sigmas = [max(r / 2, 0.0) for r in ratios]  # 0 where upsampling (ratio<1)
    r2star_prefiltered = gaussian_filter(r2star_src, sigma=sigmas)
    r2star_resized = resize_to_epi_grid(r2star_prefiltered, fov, fov, n_target, order=3)
    r2star_native = torch.from_numpy(
        np.clip(r2star_resized, 0.0, None).astype(np.float32)
    ).to(device_t)

    # echo_times is a k-space-indexed array (acquisition time per sampled
    # (ky,kz) location), not an image -- crop it the same way k-space
    # itself is cropped below, not resized.
    echo_times_crop_raw = echo_times_2d[ys, zs, :]  # (Ny_eff, Nz_eff, Nt), pre-shift
    calib_mask_crop = calib_mask[ys, zs]  # (Ny_eff, Nz_eff)

    # Bandwidth-time-product sanity check (see module docstring): confirms
    # R2*'s decay-time product is small relative to Δf's, which is what
    # justifies reusing Δf-only-tuned interpolation weights for R2* too.
    te_calib = echo_times_crop_raw[calib_mask_crop, :]
    t_span_s = te_calib.max() - te_calib.min()
    df_range_hz = float(b0map_hz_native.max() - b0map_hz_native.min())
    r2_p95 = float(np.percentile(r2star_native.cpu().numpy(), 95))
    print(f'  bandwidth-time check: Δf range x echo-train span = {df_range_hz * t_span_s:.2f} '
          "(dimensionless, ~27 at this pipeline's real ETL=60 scale per CLAUDE.md)")
    print(f'                        R2*(95th pct) x echo-train span = {r2_p95 * t_span_s:.4f} '
          '(dimensionless -- should be << the Δf figure above for the '
          'shared-weights reuse to be valid)')

    echo_times_crop = echo_times_crop_raw - te_nominal_s  # TE-referenced, both magnitude and phase
    echo_times_s = torch.from_numpy(echo_times_crop).to(device_t)
    echo_times_s = echo_times_s.unsqueeze(0).expand(Nx_eff, -1, -1, -1).contiguous()

    calib_mask_t = torch.from_numpy(calib_mask_crop).to(device_t)
    calib_omega = calib_mask_t[None, :, :, None].expand(Nx_eff, Ny_eff, Nz_eff, Nt)

    print(f'Building complex-field-corrected operator (L={L}, nbins={nbins})...')
    A = _build_calib_operator_b0_complex(
        smaps_chw, calib_omega, b0map_hz_native, r2star_native, echo_times_s, L=L, nbins=nbins,
    )
    idx_full = A.A[0].idx.cpu().numpy()
    for it in range(1, Nt):
        assert np.array_equal(A.A[it].idx.cpu().numpy(), idx_full), (
            'calibration-region sample indices differ across frames -- unexpected'
        )

    print(f'Loading k-space ({fn_ksp}), cropped to the calibration-region bounding box, '
          'and gathering calibration-region samples...')
    # Cropped during the HDF5 read itself (recon/hdf5_chunked_io.py's
    # read_frames_cropped), not after loading the full dense array -- see
    # docs/review-findings.md item 213.
    ksp_epi_zf_crop = read_frames_cropped(
        fn_ksp, 'ksp_epi_zf', spatial_slices=(xs, ys, zs)
    ).astype(np.complex64)  # [Nx_eff,Ny_eff,Nz_eff,Nc,Nt]
    ksp_gathered = gather_calib_ksp(ksp_epi_zf_crop, idx_full)  # [K, Nc, Nt]
    ksp_calib = torch.from_numpy(ksp_gathered).to(device_t)

    print('Reconstructing (adjoint only -- no iteration, no regularization)...')
    img = A.adjoint(ksp_calib)  # (Nx_eff, Ny_eff, Nz_eff, Nt) complex64
    img_np = img.detach().cpu().numpy()

    return dict(
        img_np=img_np, fov=fov, grid=grid, voxel_mm=voxel_mm,
        n_calib=n_calib, calib_mask=calib_mask, L=L, nbins=nbins, te_nominal_s=te_nominal_s,
    )


def main_b0complex(
    datdir: str, seqname: str, L: int = 32, nbins: int = 128, device: str = 'cuda',
    zero_pad_z: bool = False, fn_b0map: str | None = None,
) -> None:
    result = run_b0complex_corrected_calib_recon(
        datdir, seqname, L, nbins, device, zero_pad_z, fn_b0map,
    )
    img_np, fov = result['img_np'], result['fov']
    grid, voxel_mm = result['grid'], result['voxel_mm']
    n_calib, calib_mask = result['n_calib'], result['calib_mask']
    Nx_eff, Ny_eff, Nz_eff = grid['Nx_eff'], grid['Ny_eff'], grid['Nz_eff']

    out_dir = os.path.join(datdir, 'recon', 'lowres_calib')
    os.makedirs(out_dir, exist_ok=True)
    fn_out = os.path.join(out_dir, f'{seqname}_recon_lowres_calib_b0complex')

    save_recon_nifti(
        fn_out, img_np, fov=fov, seqname=seqname,
        n_calib_samples=n_calib, n_ky_kz=int(calib_mask.size), L=L, nbins=nbins,
        te_nominal_s=result['te_nominal_s'],
        native_grid=[Nx_eff, Ny_eff, Nz_eff], native_voxel_mm=voxel_mm,
        note='Complex-field (off-resonance + T2*) corrected, TE-referenced, adjoint-only '
             '(no iteration/regularization) reconstruction of the fully-sampled k-space '
             'center only, at native (resolution-matched) grid size, not zero-padded',
    )
    print(f'Wrote {fn_out}.nii.gz + .json')


def main_calib(
    datdir: str, seqname: str = 'ArbEPI', b0: bool = False, fn_b0map: str | None = None,
    r2star: bool = False, L: int = 32, nbins: int = 128, device: str = 'cuda',
    zero_pad_z: bool = False,
) -> None:
    """Single-sequence entry point. Plain IFFT + smaps combine by default
    (numpy only); with b0=True (or an explicit fn_b0map, which implies it)
    the time-segmented B0-corrected adjoint reconstruction, and with
    r2star=True additionally the complex-field (off-resonance + T2*)
    variant -- both need torch/mirtorch (.venv-recon), imported lazily.
    A single-dataset failure raises directly instead of being
    caught-and-printed, useful for interactive/ad hoc use."""
    b0 = b0 or fn_b0map is not None
    if r2star and not b0:
        raise ValueError('main_calib: r2star correction requires a B0 map (b0=True or fn_b0map)')
    if not b0:
        _recon_one(load_config(datdir=datdir, seqnames=[seqname]), seqname)
    elif r2star:
        main_b0complex(datdir, seqname, L, nbins, device, zero_pad_z, fn_b0map)
    else:
        main_b0(datdir, seqname, L, nbins, device, fn_b0map)


def load_lowres_calib_recon(
    datdir: str, seqname: str = 'ArbEPI', variant: str = ''
) -> tuple[np.ndarray, dict]:
    """variant: '' for the plain (uncorrected) recon, 'b0' for
    recon/lowres_calib.py's B0-corrected output."""
    suffix = f'_{variant}' if variant else ''
    fn_base = os.path.join(datdir, 'recon', 'lowres_calib', f'{seqname}_recon_lowres_calib{suffix}')
    img = np.asarray(nib.load(f'{fn_base}.nii.gz').dataobj)  # [Nx, Ny, Nz, Nframes], magnitude
    with open(f'{fn_base}.json') as f:
        meta = json.load(f)
    return img, meta


def object_mask(img: np.ndarray, thresh_frac: float = 0.2) -> np.ndarray:
    """[Nx, Ny, Nz] bool, thresholded on the time-mean magnitude -- the
    smaps eigenvalue mask already zeroes the true background (see
    lowres_calib.py), so a simple relative threshold on what's left
    cleanly separates object from noise-only voxels."""
    mean_img = img.mean(axis=-1)
    return mean_img > thresh_frac * mean_img.max()


def temporal_stability(img: np.ndarray, mask: np.ndarray, tr_s: float) -> dict:
    """img: [Nx, Ny, Nz, Nframes] magnitude. mask: [Nx, Ny, Nz] bool.

    Returns a dict of the standard phantom-stability decomposition:
    - tsnr_map: [Nx, Ny, Nz], mean/std over time (NaN outside mask)
    - roi_signal: [Nframes], spatial mean over mask per frame
    - percent_fluctuation: 100 * std(residual after linear detrend) / mean(roi_signal)
    - percent_drift: 100 * (linear fit endpoint - start) / mean(roi_signal)
    - roi_tsnr: mean(roi_signal) / std(roi_signal) (no detrend -- the raw,
      undetrended ROI-average tSNR, for comparison against the
      detrend-and-decompose numbers above)
    """
    Nframes = img.shape[-1]
    t = np.arange(Nframes)

    voxels = img[mask]  # [Nvox, Nframes]
    mean_t = voxels.mean(axis=-1)
    std_t = voxels.std(axis=-1, ddof=1)
    tsnr_map = np.full(mask.shape, np.nan, dtype=np.float64)
    tsnr_map[mask] = np.divide(mean_t, std_t, out=np.zeros_like(mean_t), where=std_t > 0)

    roi_signal = img[mask].mean(axis=0)  # [Nframes]

    coeffs = np.polyfit(t, roi_signal, deg=1)
    fit = np.polyval(coeffs, t)
    residual = roi_signal - fit
    grand_mean = roi_signal.mean()

    percent_fluctuation = 100 * residual.std(ddof=1) / grand_mean
    percent_drift = 100 * (fit[-1] - fit[0]) / grand_mean
    roi_tsnr = grand_mean / roi_signal.std(ddof=1)

    return dict(
        tsnr_map=tsnr_map,
        roi_signal=roi_signal,
        time_s=t * tr_s,
        linear_fit=fit,
        percent_fluctuation=percent_fluctuation,
        percent_drift=percent_drift,
        roi_tsnr=roi_tsnr,
        median_voxel_tsnr=np.nanmedian(tsnr_map),
    )


def main_stability(
    datdirs: list[str], seqname: str = 'ArbEPI', tr_s: float = 2.0, skip_frames: int = 0,
    variant: str = '',
) -> None:
    results = {}
    for datdir in datdirs:
        label = os.path.basename(os.path.normpath(datdir))
        if variant:
            label = f'{label} ({variant})'
        img, meta = load_lowres_calib_recon(datdir, seqname, variant)
        if skip_frames:
            print(f'(dropping first {skip_frames} frame(s) as a non-steady-state transient)')
            img = img[..., skip_frames:]
        mask = object_mask(img)
        stats = temporal_stability(img, mask, tr_s)
        results[label] = (img, mask, stats)

        print(f'\n=== {label} ===')
        print(f'  object mask: {mask.sum()} voxels ({100 * mask.sum() / mask.size:.1f}% of volume)')
        print(f'  ROI-mean signal: {stats["roi_signal"].mean():.4g} +/- {stats["roi_signal"].std(ddof=1):.4g}')
        print(f'  percent fluctuation (detrended): {stats["percent_fluctuation"]:.3f}%')
        print(f'  percent drift (linear, over {(img.shape[-1] - 1) * tr_s:.0f}s): {stats["percent_drift"]:.3f}%')
        print(f'  ROI tSNR (undetrended): {stats["roi_tsnr"]:.1f}')
        print(f'  median per-voxel tSNR: {stats["median_voxel_tsnr"]:.1f}')

    # Comparison figures (one panel per dataset) are saved next to each
    # dataset's own .nii.gz -- i.e. a copy in every datdir's recon/lowres_calib/,
    # not a shared parent directory, so each dataset's folder stays
    # self-contained even when the figure itself compares multiple datasets.
    out_dirs = [os.path.join(d, 'recon', 'lowres_calib') for d in datdirs]
    labels = list(results.keys())
    suffix = (f'_{variant}' if variant else '') + (f'_skip{skip_frames}' if skip_frames else '')

    def _save_fig(name: str) -> None:
        fn = f'lowres_temporal_stability_{name}{suffix}.png'
        for out_dir in out_dirs:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, fn), dpi=130)
        print(f'Wrote {fn} to: {", ".join(out_dirs)}')

    # --- Figure 1: per-frame ROI-mean signal + linear fit, one panel per dataset ---
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 4), squeeze=False)
    for ax, label in zip(axes[0], labels):
        _, _, stats = results[label]
        ax.plot(stats['time_s'], stats['roi_signal'], 'o-', label='ROI mean signal')
        ax.plot(stats['time_s'], stats['linear_fit'], '--', label='linear fit')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('ROI mean signal (a.u.)')
        ax.set_title(
            f'{label}\nfluct={stats["percent_fluctuation"]:.2f}%, '
            f'drift={stats["percent_drift"]:.2f}%, tSNR={stats["roi_tsnr"]:.0f}'
        )
        ax.legend()
    plt.tight_layout()
    _save_fig('roi_signal')

    # --- Figure 2: tSNR map, central slice, one panel per dataset ---
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4.5), squeeze=False)
    for ax, label in zip(axes[0], labels):
        img, mask, stats = results[label]
        iz = img.shape[2] // 2
        im = ax.imshow(stats['tsnr_map'][:, :, iz].T, origin='lower', cmap='viridis', vmin=0)
        ax.set_title(f'{label}: tSNR, z={iz}')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    _save_fig('tsnr_map')

    # --- Figure 3: frame-to-frame difference vs frame 0, central slice ---
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4.5), squeeze=False)
    for ax, label in zip(axes[0], labels):
        img, mask, stats = results[label]
        iz = img.shape[2] // 2
        diff = img[:, :, iz, -1] - img[:, :, iz, 0]
        vmax = np.percentile(np.abs(diff[mask[:, :, iz]]), 99) if mask[:, :, iz].any() else 1
        im = ax.imshow(diff.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{label}: frame -1 minus frame 0, z={iz}')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    _save_fig('last_minus_first')


def _cli_calib() -> None:
    parser = argparse.ArgumentParser(
        description="Low-res calibration-region reconstruction (plain, or B0-informed with --b0/--b0map)."
    )
    parser.add_argument('datdir')
    parser.add_argument('seqname', nargs='*', default=['ArbEPI'],
                        help='one or more seqnames to batch (default: ArbEPI)')
    parser.add_argument('--b0', action='store_true',
                        help='apply B0-informed correction using <datdir>/recon/<seqname>_b0map.h5 (needs torch)')
    parser.add_argument('--b0map', default=None,
                        help='B0 field map .h5 to use instead of the default (implies --b0; single seqname only)')
    parser.add_argument('--r2star', action='store_true',
                        help='with B0 correction, also correct T2* amplitude decay (complex field)')
    parser.add_argument('--L', type=int, default=32)
    parser.add_argument('--nbins', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--zero-pad-z', action='store_true',
        help='(with --r2star) zero-pad the deGRE-grid R2* estimate where the EPI z-FOV exceeds '
             "deGRE's fixed z-FOV, instead of raising (see docs/review-findings.md item 203)",
    )
    args = parser.parse_args()
    b0 = args.b0 or args.b0map is not None
    if args.r2star and not b0:
        parser.error('--r2star requires --b0 or --b0map')
    if args.b0map is not None and len(args.seqname) != 1:
        parser.error('--b0map PATH requires exactly one seqname')
    if len(args.seqname) > 1 and not b0:
        run_lowres_calib_recon(load_config(datdir=args.datdir, seqnames=args.seqname))
        return
    for seqname in args.seqname:
        main_calib(
            args.datdir, seqname, b0=b0, fn_b0map=args.b0map, r2star=args.r2star,
            L=args.L, nbins=args.nbins, device=args.device, zero_pad_z=args.zero_pad_z,
        )


def _cli_stability() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('datdirs', nargs='+')
    parser.add_argument('--seqname', default='ArbEPI')
    parser.add_argument('--tr', type=float, default=2.0, help='volume TR in seconds')
    parser.add_argument(
        '--skip-frames', type=int, default=0,
        help='drop this many leading frames before computing stats (non-steady-state transient)',
    )
    parser.add_argument(
        '--variant', default='',
        help="'' for the plain recon, 'b0' or 'b0complex' for the B0-corrected outputs of `calib`",
    )
    args = parser.parse_args()
    main_stability(args.datdirs, args.seqname, args.tr, args.skip_frames, args.variant)


_COMMANDS = {
    'calib': _cli_calib,
    'stability': _cli_stability,
}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        sys.exit('usage: python -m recon.lowres_calib {' + ','.join(_COMMANDS) + '} ...')
    _cmd = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {_cmd}'
    _COMMANDS[_cmd]()
