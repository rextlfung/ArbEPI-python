"""Coil sensitivity map estimation and post-processing.

Ports makeSmaps.m's 'bart' branch -- now sigpy.mri.app.EspiritCalib, BART's
ecalib was dropped, see CLAUDE.md -- and process_smaps.m's 4-step
mask/crop/resize/normalize pipeline. makeSmaps.m's 'pisco' branch is not
ported: SENSEmethod is effectively always sigpy-ESPIRiT in this port, and
PISCO (a separate, large MATLAB toolbox) has no available Python port.

ESPIRiT: Uecker M, Lai P, Murphy MJ, et al. "ESPIRiT -- an eigenvalue
approach to autocalibrating parallel MRI: where SENSE meets GRAPPA." Magn
Reson Med. 2014;71(3):990-1001.

Convention throughout this repo is coils-last ([..., Ncoils]); sigpy's
EspiritCalib expects coils-first ([Ncoils, ...]), so estimate_smaps
transposes at its boundary rather than propagating that convention further.
"""

import os

import h5py
import numpy as np
import sigpy as sp
import sigpy.mri.app as mri_app
from scipy import ndimage

from preprocessing.config import PreprocessingConfig, SeqParams, SeqPaths
from preprocessing.grid_resize import resize_to_epi_grid
from preprocessing.nifti_io import save_recon_nifti


def _default_device() -> sp.Device:
    """GPU if sigpy/cupy sees one, else CPU. ESPIRiT's per-voxel eigen
    decomposition (the `AHA @ x` power iteration in sigpy's own
    EspiritCalib) is a batched, embarrassingly parallel linear-algebra op
    over every calibration-grid voxel at once -- exactly the kind of
    workload that's much faster on GPU -- so there's no reason to force CPU
    when hardware is present. Falls back silently (not an error) when cupy
    isn't installed or no device is visible, matching this repo's general
    stance on optional acceleration (e.g. GERecon/Julia are also
    opportunistic, not hard requirements -- see CLAUDE.md)."""
    if sp.config.cupy_enabled:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return sp.Device(0)
    return sp.cpu_device


def estimate_smaps(
    ksp_gre: np.ndarray,
    calib_width: int = 24,
    thresh: float = 0.02,
    crop: float = 0.95,
    cal_size: int = 24,
    device: sp.Device | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """ESPIRiT sensitivity map estimation from fully-sampled GRE k-space.

    ksp_gre: [Nx, Ny, Nz, Ncoils] complex.
    Returns (smaps, emap): smaps [cal_size, cal_size, cal_size, Ncoils]
    complex; emap [cal_size, cal_size, cal_size] float, the dominant-
    eigenvalue map (sigpy's EspiritCalib "currently only supports
    outputting one set of maps", so unlike BART's ecalib there's no
    emaps(...,end)-style selection among several map sets to do).

    crop: sigpy's own EspiritCalib default (0.95) -- a stricter minimum-
        eigenvalue threshold than BART ecalib's own default of 0.8, which
        `makeSmaps.m`'s `bart('ecalib', ksp)` call (passing no `-c` flag)
        implicitly used. This repo previously pinned 0.8 to match that
        MATLAB/BART reference exactly; explicit decision (2026-09-14) to
        use sigpy's own default instead, accepting a smaller (more tightly
        cropped) retained sensitivity-map support than the original
        MATLAB pipeline produced.

        This is now the *only* eigenvalue threshold in the whole
        smaps pipeline -- `process_smaps` takes this same value (as its own
        `crop` parameter) to build its object-support mask, rather than a
        second, independently-configured threshold. Two separate knobs used
        to exist here (`process_smaps`' own `threshold_mask`, default 0.2)
        and actively fought each other: since `threshold_mask < crop`
        always held, `process_smaps`' mask was strictly looser than what
        `crop` had already zeroed inside ESPIRiT, so the boundary annulus
        between the two thresholds survived as tiny cubic-spline-interpolated
        residuals -- which the final RSS-normalization step then amplified
        back up to full unit magnitude. Net effect: the exported object mask
        tracked `threshold_mask` alone (measured ~80-85% of the calibration
        volume across four real datasets) and was completely insensitive to
        `crop` (bit-identical masks at crop=0.8 vs. crop=0.95) -- ESPIRiT's
        own crop threshold was silently neutered by the time it reached the
        actual output. At this pipeline's `cal_size=24` resolution, `emap`
        is itself nearly saturated (>0.99) over most of the calibration cube
        with only the outermost ~1-2 voxels rolling off, so a low threshold
        like the old 0.2 default barely constrains anything: measured mask
        fraction was ~80% vs. the true object's ~32-38% (from thresholding
        the actual reconstructed EPI magnitude image) -- a ~2-2.5x oversized
        mask. Using `crop` alone (0.95) instead cuts the mask to ~43% of the
        calibration volume, much closer to the true object extent, and
        removes the two-parameters-fighting failure mode entirely by
        construction (single source of truth for "is this voxel inside the
        object").

        Residual, deliberately accepted, ~11%-by-volume oversizing at
        crop=0.95 (investigated 2026-09-14, after the mask-fighting and
        blocky-boundary fixes above): on `01_fullsamp_4p55mm` (the *only*
        one of this session's four sequences with a trustworthy ground
        truth -- see below), the final mask covers 43.3% of the volume vs.
        38.9% for the true object (magnitude-thresholded EPI reconstruction),
        roughly 1-2 voxels (~5-9mm) too far out on each side. Two candidate
        fixes were tested directly against that ground truth and one was
        ruled out: raising `cal_size` (24->32->48) does not shrink the
        margin at all (if anything it grows slightly, 43.3%->43.9%->44.4%)
        while costing far more compute (11s->17s->47s for one sequence) --
        the residual isn't a calibration-*resolution* artifact. Raising
        `crop` itself does close the gap: 0.97 -> 40.2%, 0.98 -> 38.6%
        (a near-exact match to the true 38.9%), but 0.99 overshoots the
        other way and starts excluding real signal (edge offsets flip
        positive, i.e. the mask edge sits *inside* the true object
        boundary on multiple lines). Explicit decision: keep 0.95 rather
        than chase the closer 0.98 match, because the two failure
        directions are not symmetric for a SENSE encoding operator -- a
        mask that's too large costs nothing (the extra voxels have near-zero
        true coil sensitivity anyway, so RSS-normalizing them contributes
        no real signal), while a mask that's too tight discards real,
        unrecoverable k-space-encoded information at the object boundary.
        0.95 errs on the safe side of that asymmetry.

        The `02_11x_2mm`/`03_ultrafast_1shot_2mm`/`04_ultrafast_2shot_2mm`
        sequences could *not* be used to validate any of the above: they're
        accelerated (R~11 to ~142), and root-sum-of-squares reconstruction
        of an accelerated acquisition with no unfolding produces severe
        aliasing -- confirmed directly (attempting the same ground-truth
        comparison on `02_11x_2mm`'s RSS recon gave a nonsensical "true
        object" covering 72% of the volume, an aliasing artifact, not real
        anatomy). Only a fully-sampled (R=1) sequence's RSS reconstruction
        is a valid ground truth for this kind of check.

    cal_size: resize ksp_gre's spatial dims to this matrix size (per axis)
        before running ESPIRiT, rather than passing the full acquisition
        grid -- a center-*crop* only on axes where the source is larger
        than cal_size; at this repo's real deGRE dims (Nz_degre=21 < 24)
        sigpy's `sp.resize` zero-*pads* z instead, so the calibration
        region there is 24 samples wide with 3 synthetic-zero rows, a
        small real effect on the fit rather than a true no-op (see below).
        sigpy's EspiritCalib allocates a coil-covariance array sized to
        its *entire* input k-space's spatial shape (not just calib_width
        -- `AHA = zeros(image_shape + (Ncoils, Ncoils))` in its own
        source), so passing a full 3D acquisition grid directly is
        impractical: confirmed against real project data, a
        108^3/12-coil GRE volume allocated 2.9GB for that one array alone
        and, combined with a non-vectorized per-calibration-kernel Python
        loop, thrashed 14GB+ of memory and never completed in over an hour
        (killed). Resizing k-space this way reduces resolution while
        preserving FOV -- process_smaps() already resizes the result up
        to the EPI target grid afterward regardless of what resolution
        ESPIRiT itself ran at, so there's no need to calibrate at full
        acquisition resolution. Default matches calib_width (24) so
        EspiritCalib's own internal calibration-region crop becomes a
        no-op on x/y (where the source is >= 24) and the covariance/
        eigenmap stage runs at the same small size throughout; z is the
        one axis where that "no-op" claim doesn't fully hold, per above.

    device: sigpy Device (or a plain int -- sigpy's own convention, -1 for
        CPU, >=0 for a GPU index) to run ESPIRiT's calibration/power
        iteration on. None (default) auto-selects a GPU via
        `_default_device()` when sigpy/cupy sees one, else CPU -- ESPIRiT's
        per-voxel eigen decomposition is a large batched linear-algebra op
        that benefits substantially from GPU parallelism. `ksp_gre` itself
        stays plain numpy regardless -- sigpy's EspiritCalib moves only the
        (small, cal_size-shaped) calibration region onto `device` internally
        -- and this function always returns plain numpy arrays, converting
        back off the compute device if one was used, so callers never need
        to know whether GPU acceleration happened.
    """
    device = _default_device() if device is None else sp.Device(device)
    ksp_coils_first = np.moveaxis(ksp_gre, -1, 0)
    if cal_size is not None:
        ncoils = ksp_coils_first.shape[0]
        ksp_coils_first = sp.resize(ksp_coils_first, (ncoils, cal_size, cal_size, cal_size))
    calib = mri_app.EspiritCalib(
        ksp_coils_first,
        calib_width=calib_width,
        thresh=thresh,
        crop=crop,
        device=device,
        show_pbar=False,
        output_eigenvalue=True,
    )
    mps, emap = calib.run()
    mps = sp.to_device(mps, sp.cpu_device)
    emap = sp.to_device(emap, sp.cpu_device)
    smaps = np.moveaxis(mps, 0, -1)
    return smaps, emap[0]


def _masked_gaussian_smooth(
    smaps: np.ndarray, mask: np.ndarray, sigma_vox: np.ndarray
) -> np.ndarray:
    """Per-coil, mask-normalized Gaussian smoothing: blur `smaps * mask` and
    the mask itself with the same kernel, then divide, so the background
    (exact zero) never bleeds into the blurred estimate near the boundary
    (the standard trick for smoothing up to a mask edge without a dark
    halo -- equivalent to a local weighted average that renormalizes for
    however much of the kernel's support fell outside the mask). Real part
    and imaginary part are blurred separately (`ndimage.gaussian_filter`
    has no native complex support). Does *not* re-apply `mask` itself --
    callers combine this with an exact re-mask afterward (see
    process_smaps), since the normalized blur intentionally extends a
    smoothed estimate slightly past the mask boundary by design.
    """
    weight = mask.astype(np.float64)
    denom = ndimage.gaussian_filter(weight, sigma_vox)
    denom[denom < 1e-6] = 1
    out = np.empty_like(smaps)
    for c in range(smaps.shape[-1]):
        num_real = ndimage.gaussian_filter(smaps[..., c].real * weight, sigma_vox)
        num_imag = ndimage.gaussian_filter(smaps[..., c].imag * weight, sigma_vox)
        out[..., c] = (num_real + 1j * num_imag) / denom
    return out


def process_smaps(
    smaps_raw: np.ndarray,
    emap: np.ndarray,
    fov_gre: tuple[float, float, float],
    fov: tuple[float, float, float],
    n_target: tuple[int, int, int],
    crop: float,
    smooth_sigma_mm: float = 6.0,
) -> np.ndarray:
    """Mask, z-crop, resize, smooth, and RSS-normalize raw sensitivity maps
    to the EPI acquisition grid. Ports process_smaps.m's 'bart' eigenvalue
    convention (high eigenvalue = inside object) -- the only convention
    relevant here since PISCO isn't ported; the smoothing step has no
    process_smaps.m equivalent (see below for why it was added).

    Assumption carried over from process_smaps.m: the GRE and EPI
    acquisitions share the same isocenter, so a symmetric z-crop is valid.

    smaps_raw: [Nx_gre, Ny_gre, Nz_gre, Ncoils]
    emap: [Nx_gre, Ny_gre, Nz_gre]
    fov_gre, fov: (fx, fy, fz) in meters
    n_target: (Nx, Ny, Nz), the EPI acquisition grid
    crop: the same eigenvalue threshold passed to `estimate_smaps`'s
        `EspiritCalib` call -- must be the identical value the caller used
        there. This function used to take its own independent
        `threshold_mask` (default 0.2) and apply it to `smaps_raw` directly
        (a distinct pre-resize masking step, since removed -- see below);
        that was a real bug (see `estimate_smaps`'s `crop` docstring for the
        full measurement), since a threshold looser than ESPIRiT's own
        `crop` never actually constrained anything -- `smaps_raw` already
        had zeros below `crop`, and RSS-normalization erased the difference
        between "just above threshold_mask" and "an interpolated near-zero
        residual" by renormalizing either back to unit magnitude. `crop` is
        used here only to rebuild `eig_mask` for the post-resize re-mask
        below -- see there for why a *pre*-resize mask on `smaps_raw` is no
        longer applied at all.
    smooth_sigma_mm: Gaussian smoothing sigma in mm, applied on the target
        grid (0 disables). See below for why this exists.
    """
    # `smaps_raw` is assumed already zero wherever `emap <= crop` -- true by
    # construction for this function's only real caller (`estimate_smaps`,
    # via sigpy's own `EspiritCalib.crop`, the identical `emap > crop`
    # comparison), confirmed bit-exact on real project data. A pre-resize
    # `smaps_raw *= eig_mask` line used to sit here to enforce this
    # defensively, but multiplying an array by a mask it's already exactly
    # zero outside of is a no-op -- removed per this repo's convention of
    # not validating invariants a caller already guarantees.

    # Crop z to match EPI FOV, then interpolate (cubic spline) to the EPI
    # grid -- see grid_resize.py's module docstring for why this
    # deGRE-grid-to-EPI-grid crop+resize is shared with run_b0map.py.
    smaps = resize_to_epi_grid(smaps_raw, fov_gre, fov, n_target, order=3)

    # Object mask, built by interpolating the *continuous* emap (cubic
    # spline, same as smaps_raw above) to the target grid and thresholding
    # at `crop` there -- not by binarizing emap at cal_size=24 resolution
    # first and nearest-neighbor-resizing that already-binary field (this
    # function's earlier approach). The earlier approach preserved each
    # coarse calibration-grid voxel's own 0/1 value as a solid block
    # spanning the full zoom factor (Nx/24, ~1.7-4.5x at this pipeline's
    # real target grids) on the target grid -- visibly blocky/stair-stepped
    # on real data even though the true object boundary is round (confirmed:
    # a real 24-voxel calibration's emap already has a smooth eigenvalue
    # roll-off over its outermost ~2-3 voxels near the object edge, so
    # binarizing before resizing throws that smooth sub-cal_size-voxel
    # information away for good). Interpolating the continuous field first
    # and thresholding on the fine grid instead recovers a much rounder
    # boundary at essentially the same mask size (<0.2 percentage point
    # difference in total mask fraction, measured on real data) -- this is
    # also already what run_b0map.py/b0map.jl does with its own `emap_degre`
    # ESPIRiT-eigenvalue mask, so this brings process_smaps' mask in line
    # with the more careful approach already used elsewhere in this
    # pipeline. Cubic spline can overshoot near a hard step the same way it
    # does for smaps_raw (see below) -- irrelevant here since only which
    # side of `crop` each interpolated value lands on matters, not its
    # exact value.
    #
    # This masking step is still needed at all (whether built this way or
    # the old way) because cubic spline's prefilter is a global (IIR)
    # operation: `smaps_raw`'s hard zero boundary leaks small (~1e-6 to
    # 1e-9) nonzero values into a halo just outside the object after its own
    # resize above -- invisible on its own, but amplified straight back up
    # to unit magnitude by the RSS normalization below (whose `rss < eps`
    # clamp only catches values under ~2e-16, not this leakage) and again by
    # recon/reconstruct.py's own RSS-renormalization on load, silently
    # erasing the mask everywhere except exact-zero voxels. An exact 0/1
    # re-mask (thresholding, not interpolating, the final decision) after
    # both resizes guarantees background is exactly zero regardless.
    emap_resized = resize_to_epi_grid(emap, fov_gre, fov, n_target, order=3)
    target_mask = emap_resized > crop
    smaps = smaps * target_mask[..., None]

    # Smooth away the resolution-limited blocky/rippling texture visible
    # near the object edge -- confirmed (by re-running estimate_smaps at
    # cal_size up to 48, and on the fully unmasked resize) to be a genuine
    # ESPIRiT-at-cal_size artifact, not an interpolation or masking bug: it
    # shows up identically with no masking applied at all, and does not
    # improve with a larger calibration grid (only with much higher
    # runtime cost, ~6x at cal_size=48 vs. the cal_size=24 default) or a
    # looser/tighter eigenvalue crop. Real coil sensitivity profiles vary
    # smoothly over centimeters, so this noise is safe to remove with a
    # mild low-pass filter without discarding real anatomical information.
    # `_masked_gaussian_smooth` blurs up to the mask boundary without
    # pulling in the (zero) background; converting to voxel units per
    # axis (rather than a fixed voxel-count sigma) keeps the *physical*
    # smoothing extent the same regardless of the target grid's own
    # resolution. Re-masking after is required: the mask-normalized blur
    # deliberately smears an estimate slightly past the true boundary
    # (that's what avoids a dark halo), so it must be cut back to the same
    # exact support computed above, or the "background is exactly zero"
    # guarantee above would be undone again.
    if smooth_sigma_mm > 0:
        vox_mm = np.array(fov) / np.array(n_target) * 1000
        sigma_vox = smooth_sigma_mm / vox_mm
        smaps = _masked_gaussian_smooth(smaps, target_mask, sigma_vox)
        smaps = smaps * target_mask[..., None]

    # 4. Normalize: divide by the cross-coil RSS so sum(|s_c|^2) <= 1
    # everywhere, matching the ESPIRiT convention regularized SENSE recon
    # (sigpy or otherwise) expects when no explicit step size is given.
    rss = np.sqrt(np.sum(np.abs(smaps) ** 2, axis=-1))
    rss[rss < np.finfo(rss.dtype).eps] = 1
    return smaps / rss[..., None]


def _estimate_uncompressed_smaps(
    ksp_gre_uncompressed: np.ndarray,
    fov_degre: tuple[float, float, float],
    n_target_degre: tuple[int, int, int],
    cfg: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ESPIRiT calibration + deGRE-grid mask/resize/smooth/normalize on
    whitened-but-not-PCA-compressed GRE k-space (preprocess.py's STEP 2
    `ksp_gre_uncompressed`) -- a true per-physical-coil (e.g. 32-channel)
    calibration, run independently of the Nvcoils-compressed calibration
    above rather than derived from it (`cc_matrix` is rank-reducing, so a
    PCA-compressed sensitivity map set can't be projected back into
    per-physical-coil profiles after the fact). Returns
    (smaps_raw_uncompressed, emap_uncompressed, smaps_degre_uncompressed);
    `emap_uncompressed` must stay paired with `smaps_raw_uncompressed`
    (not swapped for the Nvcoils-compressed `emap`) -- process_smaps
    assumes `smaps_raw` is already exactly zero wherever its own paired
    `emap <= crop` (see process_smaps' docstring), and only a
    self-consistent pair guarantees that.
    """
    smaps_raw_unc, emap_unc = estimate_smaps(ksp_gre_uncompressed, crop=cfg.crop)
    smaps_degre_unc = process_smaps(
        smaps_raw_unc, emap_unc, fov_degre, fov_degre, n_target_degre, cfg.crop,
        smooth_sigma_mm=cfg.smaps_smooth_sigma_mm,
    )
    return smaps_raw_unc, emap_unc, smaps_degre_unc


def _backfill_uncompressed_smaps(
    fn_smaps: str,
    fn_gre: str,
    fov_degre: tuple[float, float, float],
    n_target_degre: tuple[int, int, int],
    cfg: PreprocessingConfig,
) -> np.ndarray | None:
    """Best-effort (re)fill of `smaps_degre_uncompressed` into an existing
    smaps cache -- either it predates the dataset entirely, or its cached
    `Ncoils` attr has gone stale relative to the current
    `ksp_gre_uncompressed` (mirroring the outer `Nvcoils` staleness check
    this function's caller already does for the compressed set). Needs
    `ksp_gre_uncompressed` in the GRE cache (preprocess.py's STEP 2, only
    written from this feature onward) -- returns None with a printed
    warning, rather than raising, when that input isn't available (a
    `<seqname>_gre.h5` written before this feature, or already cleaned
    up): the same tolerance load_seq_params already extends to durable,
    non-regeneratable per-acquisition records that predate a later field
    (see its n_echoes_degre/TE_degre defaults).
    """
    if not os.path.exists(fn_gre):
        print(
            f"  WARNING: '{fn_gre}' not found -- skipping uncompressed-coil "
            'smaps backfill (re-run preprocess() to populate it).'
        )
        return None
    with h5py.File(fn_gre, 'r') as f:
        if 'ksp_gre_uncompressed' not in f:
            print(
                f"  WARNING: '{fn_gre}' has no 'ksp_gre_uncompressed' dataset "
                '(written by a pre-uncompressed-smaps preprocess() run) -- '
                'skipping uncompressed-coil smaps backfill; re-run preprocess() '
                'to populate it.'
            )
            return None
        ksp_gre_uncompressed = f['ksp_gre_uncompressed'][()]
    print(f'  Backfilling uncompressed-coil deGRE-grid smaps into {fn_smaps}...')
    smaps_raw_unc, emap_unc, smaps_degre_unc = _estimate_uncompressed_smaps(
        ksp_gre_uncompressed, fov_degre, n_target_degre, cfg,
    )
    with h5py.File(fn_smaps, 'a') as f:
        # del-if-present rather than a bare create_dataset: this can also
        # run to *repair* a cache whose Ncoils has gone stale (see the
        # cache-valid branch's staleness check above), in which case these
        # three names already exist and create_dataset would raise.
        for name in ('smaps_raw_uncompressed', 'emap_uncompressed', 'smaps_degre_uncompressed'):
            if name in f:
                del f[name]
        f.create_dataset('smaps_raw_uncompressed', data=smaps_raw_unc)
        f.create_dataset('emap_uncompressed', data=emap_unc)
        f.create_dataset('smaps_degre_uncompressed', data=smaps_degre_unc)
        f.attrs['Ncoils'] = ksp_gre_uncompressed.shape[-1]
    return smaps_degre_unc


def load_smaps(
    cfg: PreprocessingConfig, paths: SeqPaths, seq_params: SeqParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray | None]:
    """(smaps, smaps_degre, emap_degre, nvcoils, smaps_degre_uncompressed):
    sensitivity maps on the EPI grid (the SENSE encoding operator's own
    grid), the *deGRE* grid (for preprocessing/julia/b0map.jl's `smap`
    argument -- see its module docstring for why passing real smaps
    there, instead of leaving B0 field-map estimation to MRIFieldmaps'
    phase-contrast coil-combine fallback, is expected to reduce field-map
    noise in this pipeline's real low-per-coil-SNR object-center regions),
    and a *second*, independently-calibrated deGRE-grid set estimated from
    the whitened-but-not-PCA-compressed GRE k-space (the real physical
    receive-coil count -- e.g. 32 -- not Nvcoils). `emap_degre` is
    ESPIRiT's own dominant-eigenvalue map, also resized to the deGRE grid,
    for an optional ESPIRiT-informed image-support mask in b0map.jl
    (thresholded the same way process_smaps already does for `eig_mask`)
    -- a complement to its own magnitude-based mask, not a guaranteed fix
    for the same reason a raw magnitude threshold already has known
    limits near low-SNR/partial-volume voxels (see CLAUDE.md's recon/
    section).

    `smaps_degre_uncompressed` is a true per-physical-coil calibration
    for a consumer that needs real per-coil profiles rather than a
    PCA-compressed subspace -- it is *not* derivable from `smaps_degre`
    after the fact (`cc_matrix` is rank-reducing: `cc_matrix.conj().T @
    smaps_degre` would only be a rank-Nvcoils approximation of the true
    per-coil maps, not the real thing), so it always comes from its own
    ESPIRiT calibration on `ksp_gre_uncompressed` (see
    `_estimate_uncompressed_smaps`). It is `None` when the
    `<seqname>_gre.h5` cache predates `ksp_gre_uncompressed` (a
    pre-this-feature preprocess() run) and can't be backfilled -- re-run
    preprocess() to populate it; this is the only field in this return
    tuple that can come back None. Cached under its own `Ncoils` attr
    (independent of `Nvcoils`, which only guards the compressed set) --
    a coil-compression-setting change moves `Nvcoils` without touching
    `Ncoils`, and vice versa for a different physical coil array or a
    re-run archive, so both are checked and backfilled independently.

    The two deGRE-grid-Nvcoils arrays (`smaps_degre`/`emap_degre`) are
    resized from the same `cal_size`-cropped ESPIRiT calibration
    (`smaps_raw`/`emap`, see `estimate_smaps`) as `smaps` via
    `process_smaps`/`resize_to_epi_grid` -- deGRE-grid uses `fov_gre` as
    both source *and* target FOV (only resolution changes, no z-crop),
    EPI-grid is the existing crop+resize. Loads from/writes
    `<datdir>/recon/smaps_<seqname>_sigpy.h5` (was `recon_frames.py`'s
    private `_load_smaps` -- moved here, and extended with the
    `smaps_degre`/`emap_degre` datasets, so `run_b0map.py` can reuse the
    same cache instead of re-running ESPIRiT). An older cache written
    before these existed is backfilled in place rather than re-estimating
    from scratch (`smaps_raw`/`emap` are already cached); the uncompressed
    set is backfilled the same way when its own input is available.
    """
    fn_smaps = os.path.join(cfg.datdir, 'recon', f'smaps_{paths.seqname}_sigpy.h5')
    fn_smaps_nifti = fn_smaps[: -len('.h5')] + '.nii.gz'
    fn_gre = os.path.join(cfg.datdir, 'recon', f'{paths.seqname}_gre.h5')
    fov_degre = tuple(seq_params.fov_degre)
    n_target_degre = (seq_params.Nx_degre, seq_params.Ny_degre, seq_params.Nz_degre)

    # Guard against a stale cache from a run with a different Nvcoils (e.g.
    # coil-compression settings changed since the cache was written) --
    # matching preprocess.py's own smaps_valid check. Falls through to
    # re-estimation below on mismatch, same as if fn_smaps didn't exist. If
    # fn_gre isn't available to check against (e.g. already cleaned up),
    # trust the existing cache rather than failing outright.
    smaps_cache_valid = os.path.exists(fn_smaps)
    if smaps_cache_valid and os.path.exists(fn_gre):
        with h5py.File(fn_smaps, 'r') as f:
            cached_nvcoils = int(f.attrs['Nvcoils'])
        with h5py.File(fn_gre, 'r') as f:
            current_nvcoils = f['ksp_gre'].shape[-1]
        smaps_cache_valid = cached_nvcoils == current_nvcoils

    if smaps_cache_valid:
        print(f'Loading precomputed sensitivity maps from {fn_smaps}')
        with h5py.File(fn_smaps, 'r') as f:
            smaps, nvcoils = f['smaps'][()], int(f.attrs['Nvcoils'])
            has_degre = 'smaps_degre' in f and 'emap_degre' in f
            if has_degre:
                smaps_degre, emap_degre = f['smaps_degre'][()], f['emap_degre'][()]
            else:
                smaps_raw, emap = f['smaps_raw'][()], f['emap'][()]
            has_uncompressed = 'smaps_degre_uncompressed' in f
            if has_uncompressed:
                cached_ncoils = int(f.attrs.get('Ncoils', -1))
                smaps_degre_uncompressed = f['smaps_degre_uncompressed'][()]
        # Ncoils staleness check, mirroring the outer Nvcoils one above --
        # independent axes (a coil-compression-setting change moves
        # Nvcoils; a different physical coil array or a re-run archive
        # moves Ncoils), so Nvcoils matching says nothing about whether
        # the cached uncompressed set still matches the current GRE cache.
        if has_uncompressed and os.path.exists(fn_gre):
            with h5py.File(fn_gre, 'r') as f:
                if 'ksp_gre_uncompressed' in f:
                    has_uncompressed = cached_ncoils == f['ksp_gre_uncompressed'].shape[-1]
        if not has_degre:
            print(f'  Backfilling deGRE-grid smaps/emap into {fn_smaps}...')
            smaps_degre = process_smaps(
                smaps_raw, emap, fov_degre, fov_degre, n_target_degre, cfg.crop,
                smooth_sigma_mm=cfg.smaps_smooth_sigma_mm,
            )
            emap_degre = resize_to_epi_grid(emap, fov_degre, fov_degre, n_target_degre, order=3)
            with h5py.File(fn_smaps, 'a') as f:
                f.create_dataset('smaps_degre', data=smaps_degre)
                f.create_dataset('emap_degre', data=emap_degre)
        if not has_uncompressed:
            smaps_degre_uncompressed = _backfill_uncompressed_smaps(
                fn_smaps, fn_gre, fov_degre, n_target_degre, cfg,
            )
        if not os.path.exists(fn_smaps_nifti):
            # Backfill: cache was written before the NIfTI export existed.
            save_recon_nifti(
                fn_smaps[: -len('.h5')], smaps, fov=seq_params.fov,
                seqname=paths.seqname, Nvcoils=nvcoils,
            )
        return smaps, smaps_degre, emap_degre, nvcoils, smaps_degre_uncompressed

    if os.path.exists(fn_smaps):
        print(f'Cached sensitivity maps at {fn_smaps} have stale Nvcoils -- re-estimating.')
    print('Sensitivity maps not found. Estimating via sigpy ESPIRiT...')
    with h5py.File(fn_gre, 'r') as f:
        ksp_gre = f['ksp_gre'][()]
        ksp_gre_uncompressed = (
            f['ksp_gre_uncompressed'][()] if 'ksp_gre_uncompressed' in f else None
        )
    nvcoils = ksp_gre.shape[-1]
    smaps_raw, emap = estimate_smaps(ksp_gre, crop=cfg.crop)
    smaps = process_smaps(
        smaps_raw, emap, fov_degre, tuple(seq_params.fov),
        (seq_params.Nx, seq_params.Ny, seq_params.Nz), cfg.crop,
        smooth_sigma_mm=cfg.smaps_smooth_sigma_mm,
    )
    smaps_degre = process_smaps(
        smaps_raw, emap, fov_degre, fov_degre, n_target_degre, cfg.crop,
        smooth_sigma_mm=cfg.smaps_smooth_sigma_mm,
    )
    emap_degre = resize_to_epi_grid(emap, fov_degre, fov_degre, n_target_degre, order=3)

    if ksp_gre_uncompressed is not None:
        smaps_raw_unc, emap_unc, smaps_degre_uncompressed = _estimate_uncompressed_smaps(
            ksp_gre_uncompressed, fov_degre, n_target_degre, cfg,
        )
    else:
        print(
            f"  WARNING: '{fn_gre}' has no 'ksp_gre_uncompressed' dataset "
            '(written by a pre-uncompressed-smaps preprocess() run) -- '
            'skipping uncompressed-coil smaps; re-run preprocess() to populate it.'
        )
        smaps_raw_unc = emap_unc = smaps_degre_uncompressed = None

    with h5py.File(fn_smaps, 'w') as f:
        f.create_dataset('smaps_raw', data=smaps_raw)
        f.create_dataset('emap', data=emap)
        f.create_dataset('smaps', data=smaps)
        f.create_dataset('smaps_degre', data=smaps_degre)
        f.create_dataset('emap_degre', data=emap_degre)
        if smaps_degre_uncompressed is not None:
            f.create_dataset('smaps_raw_uncompressed', data=smaps_raw_unc)
            f.create_dataset('emap_uncompressed', data=emap_unc)
            f.create_dataset('smaps_degre_uncompressed', data=smaps_degre_uncompressed)
            f.attrs['Ncoils'] = ksp_gre_uncompressed.shape[-1]
        f.attrs['Nvcoils'] = nvcoils
    # Coil axis stands in for save_recon_nifti's "frames" axis -- FSLeyes'
    # volume slider then scrolls through per-coil maps, magnitude-only
    # (NIfTI has no complex dtype; see nifti_io module docstring).
    save_recon_nifti(
        fn_smaps[: -len('.h5')], smaps, fov=seq_params.fov, seqname=paths.seqname, Nvcoils=nvcoils,
    )
    return smaps, smaps_degre, emap_degre, nvcoils, smaps_degre_uncompressed
