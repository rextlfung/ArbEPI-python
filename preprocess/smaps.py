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

import numpy as np
import sigpy as sp
import sigpy.mri.app as mri_app
from scipy import ndimage

from preprocess.grid_resize import resize_to_epi_grid


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
    zero_pad_z: bool = False,
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
    zero_pad_z: forwarded to resize_to_epi_grid -- set True to zero-fill
        (rather than raise on) the EPI grid's outermost z-slices when the
        EPI acquisition's own z-FOV exceeds fov_gre's (deGRE's fixed z-FOV
        not covering one particular EPI resolution variant's rounding-
        driven z-FOV, a real case this was added for -- see
        resize_to_epi_grid's own docstring and docs/review-findings.md
        item 196). Leave False unless the caller has made that deliberate
        call for a specific already-acquired dataset.
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
    smaps = resize_to_epi_grid(smaps_raw, fov_gre, fov, n_target, order=3, zero_pad_z=zero_pad_z)

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
    # recon/mslr.py's own RSS-renormalization on load, silently
    # erasing the mask everywhere except exact-zero voxels. An exact 0/1
    # re-mask (thresholding, not interpolating, the final decision) after
    # both resizes guarantees background is exactly zero regardless.
    emap_resized = resize_to_epi_grid(emap, fov_gre, fov, n_target, order=3, zero_pad_z=zero_pad_z)
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
