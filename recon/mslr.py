"""Multi-scale Locally Low-Rank (MSLR) fMRI reconstruction via decomposition.
Port of ../mslr-recon/scripts/reconstruct.jl (Ong & Lustig 2016), built on
mirtorch instead of MIRT.jl/LinearMapsAA -- see recon/operators.py and
recon/mslr.py for the individual pieces.

    X_final = X[...,0] + X[...,1] + ... + X[...,Nscales-1]

Each component X[...,k] is independently constrained to be locally low-rank
at its own patch scale (recon/mslr.py's patchSVST); data consistency is
enforced on the sum. lambda_k set by the Ong & Lustig (2016) closed-form
formula (see _reg_weights below) -- no tuning needed beyond lambda_global.

save_result (below; formerly its own module, recon/save_result.py) is what
actually persists a ReconResult -- run_recon() itself only returns one in
memory. It writes two files per result: `<fn_base>.nii.gz` + `.json`
(magnitude image + metadata sidecar, via preprocessing/nifti_io.py's
save_recon_nifti -- same format/convention every other reconstructed image
in this pipeline uses) and `<fn_base>.h5` (full-precision complex
X_recon/X plus the solver convergence trace, plain numpy-order h5py -- no
MATLAB consumer, matching this repo's own .h5-not-.mat convention for
internal artifacts).


Formerly recon/solvers.py
-------------------------
Proximal gradient method with momentum (PGM/FPGM/POGM) and gradient
restart. Port of ../mslr-recon/src/mirt_mod.jl's `pogm_restart`, itself a
modified port of MIRT.pogm_restart (Kim & Fessler, 2017/2018) adding
GPU-memory-safe scalar typing, in-place buffer reuse, and early stopping via
`conv_tol`.

The scalar-typing and buffer-aliasing machinery in the Julia version exists
specifically to avoid Float64 promotion of CuArray{ComplexF32} and to fit a
48GB-VRAM budget under Julia's broadcast-allocates-a-new-array semantics
(see mirt_mod.jl's module docstring, points 1-5). None of that applies here:
Python floats multiplied against a complex64 tensor stay complex64 (PyTorch's
weak-scalar type promotion), and PyTorch's caching allocator reuses freed
blocks without needing manual aliasing -- so this port keeps the exact
momentum/restart math (points 6-7 of that docstring: early stopping and the
extended `fun` callback) but drops the Julia-GPU-specific mechanics.


Formerly recon/lowrank.py
-------------------------
Patch extraction/recombination and singular-value soft-thresholding (SVST)
for locally-low-rank (LLR) regularization of a 4-D image time series
(Nx, Ny, Nz, Nt). Port of ../mslr-recon/src/recon.jl (Ong & Lustig 2016).

Unlike the Julia original (which loops over patches with @threads on CPU or
sequential CUSOLVER calls on GPU, to avoid a huge intermediate tensor and
work around a GPU-only cuSOLVER NaN bug), this port batches every patch into
one tensor and calls a single batched `torch.linalg.svd` -- PyTorch already
parallelizes a batched SVD internally (cuSOLVER batched routines on GPU,
multi-threaded LAPACK on CPU), so there is no need to loop by hand. The one
piece of Julia's SVST that IS still needed here is the exact-zero shortcut
below (see SVST docstring) -- it's a correctness safeguard, not a Julia-GPU
memory optimization, so it survives the port.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import h5py
import numpy as np
import torch

from preprocessing.nifti_io import save_recon_nifti
from recon.hdf5_chunked_io import read_frames_cropped
from recon.operators import (
    build_encoding_operator,
    build_encoding_operator_b0,
    estimate_spectral_norm,
    load_and_gather_ksp,
)


@dataclass
class ReconResult:
    X: torch.Tensor  # (Nx,Ny,Nz,Nt,Nscales)
    X_recon: torch.Tensor  # (Nx,Ny,Nz,Nt)
    omega: torch.Tensor  # (Nx,Ny,Nz,Nt) bool
    dc_costs: list[float]
    reg_costs: list[float]
    restarts: list[bool]
    rel_changes: list[float]
    R: float
    sigma1A: float
    L: float
    lambdas: list[float]
    runtime_s: float
    meta: dict = field(default_factory=dict)


def _load_array(fn: str, key: str) -> np.ndarray:
    """.h5 written in plain numpy order (e.g. this repo's own preprocessing/
    output, or mslr-recon's sigpy-export input path) -- no axis correction
    needed, unlike hdf5storage-written .mat files (see preprocessing/matio.py).

    Thin wrapper around recon/hdf5_chunked_io.py's read_frames_cropped
    (shared with recon/lowres_calib.py, which needs the same
    chunk-by-chunk-along-the-last-axis logic without pulling in this
    module's torch/mirtorch imports -- see docs/review-findings.md item
    200) -- returns the full array, since run_recon processes every frame
    and has no crop to apply here."""
    return read_frames_cropped(fn, key)


def _load_omega(
    fn_ksp: str, Nx: int, Ny: int, Nz: int, Nt: int, device: torch.device
) -> torch.Tensor:
    """(Nx,Ny,Nz,Nt) sampling mask, broadcast across the readout axis
    (kx doesn't affect which (ky,kz) locations were sampled).

    Prefers the authoritative 'omegas' dataset preprocess.py writes into
    the same file (preprocessing/preprocess.py's _build_omegas) over
    inferring the mask from which complex64 k-space values happen to be
    exactly zero: a real acquired sample that rounds to exactly 0+0j after
    phase correction would otherwise silently become "not acquired", and
    since that would be consistently wrong across every coil, a per-coil
    consistency check can't catch it either -- so that check is only worth
    doing in the fallback branch below, where it's actually load-bearing.
    Falls back to the `!= 0` derivation, logged, for recon files written
    before 'omegas' existed -- that fallback loads the whole dense ksp0
    itself (there's no way around it, needing every coil's exact-zero
    pattern), unlike the normal (has_omegas) path, which takes only
    `device` and never touches ksp_epi_zf at all so callers can build the
    encoding operator (and therefore load_and_gather_ksp's memory-bounded
    per-frame path) before ever loading real k-space data.
    """
    with h5py.File(fn_ksp, "r") as f:
        has_omegas = "omegas" in f
        if has_omegas:
            omegas_yzt = torch.from_numpy(np.asarray(f["omegas"][()])).to(device)
    if has_omegas:
        assert tuple(omegas_yzt.shape) == (Ny, Nz, Nt), (
            f"omegas shape {tuple(omegas_yzt.shape)} doesn't match k-space dims ({Ny},{Nz},{Nt})"
        )
        return omegas_yzt.unsqueeze(0).expand(Nx, -1, -1, -1).bool()

    print(
        f"  '{fn_ksp}' has no 'omegas' dataset (written before preprocess.py added it) -- "
        "falling back to inferring the sampling mask from exact-zero k-space values."
    )
    ksp0 = torch.from_numpy(_load_array(fn_ksp, "ksp_epi_zf").astype(np.complex64)).to(device)
    omega = ksp0[:, :, :, 0, :] != 0
    for ic in range(1, ksp0.shape[3]):
        assert torch.equal(omega, ksp0[:, :, :, ic, :] != 0), f"Coil {ic} has a differing mask"
    return omega


def _load_echo_times(fn_ksp: str, device: torch.device) -> torch.Tensor:
    """(Ny,Nz,Nt) echo-time array (seconds since RF excitation), moved to
    device at its native shape -- shared by both B0-recon call sites
    (run_recon here and recon/run_recon.py's mslr-ref) so neither has to duplicate the
    broadcast-to-(Nx,Ny,Nz,Nt) pattern build_encoding_operator_b0 no
    longer needs (see its docstring and docs/review-findings.md item 90)."""
    return torch.from_numpy(_load_array(fn_ksp, "echo_times").astype(np.float32)).to(device)


def _load_normalized_smaps(
    fn_smaps: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads smaps and RSS-normalizes it. Returns (smaps, smaps_chw):
    smaps is (Nx,Ny,Nz,Nc) complex64, each voxel's coil vector scaled to
    unit RSS; smaps_chw is (Nc,Nx,Ny,Nz), the layout
    build_encoding_operator{,_b0} expect. Shared by run_recon (here) and
    run_recon.py, which used to duplicate this verbatim -- a real
    desync risk since run_b0_recon's whole purpose is measuring sigma1A
    for the operator run_recon builds moments later (see
    docs/review-findings.md item 94)."""
    smaps_raw = torch.from_numpy(_load_array(fn_smaps, "smaps").astype(np.complex64)).to(device)
    smaps_rss = smaps_raw.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    smaps = smaps_raw / (smaps_rss + torch.finfo(torch.float32).eps)
    smaps_chw = smaps.permute(3, 0, 1, 2).contiguous()
    return smaps, smaps_chw


def estimate_noise_std(X: torch.Tensor, bg_frac: float = 0.25) -> float:
    """Robust image-domain thermal-noise std estimate, from the lowest-
    magnitude bg_frac of voxels (assumed background/air, dominated by noise
    rather than signal -- no explicit object mask needed). Real and
    imaginary parts of complex Gaussian noise share the same std, so the
    real part alone (not the rectified magnitude, which is biased) is used.

    Added 2026-09-18 after finding this port's real reconstructions have
    thermal noise far from the unit-variance-in-image-space Ong & Lustig's
    lambda_k formula assumes (see run_recon's normalize_noise docstring) --
    ../mslr-recon/scripts/reconstruct.jl's own comment states this
    explicitly ("Formula assumes unit-variance noise in image space. BART
    prewhitening gives sigma_ksp ~= 1 and A is approximately unitary, so
    sigma_image ~= 1 -- no correction needed") but that assumption doesn't
    hold here: this repo's own k-space whitening (preprocessing/coils.py)
    and PCA coil compression are both verified variance-preserving in
    isolation (unit-tested; compression's rows are orthonormal, which
    provably preserves noise variance), so the source of the mismatch is
    elsewhere in the pipeline (EPI ramp-sampling regridding is the leading
    suspect, not yet isolated) -- measuring and normalizing directly avoids
    needing to find it. Measured on real 2_6x_2.4mm data: a single A^H(ksp)
    (X0, before any iteration) has background std ~184 -- ~150-200x away
    from the ~1 the formula assumes, not a small correction.

    Thresholding on the lowest bg_frac of |X| (Rayleigh-distributed for
    i.i.d. complex Gaussian noise, scale sigma) truncates the real part's
    own distribution too, biasing a naive std() of the selected real parts
    low -- confirmed both analytically and by direct Monte Carlo (e.g.
    bg_frac=0.25 measures ~0.37*sigma, not sigma). Exact closed-form
    correction: M^2/sigma^2 is Exp(2) (Rayleigh -> exponential), so with
    t = -ln(1-bg_frac) (the truncation point in those units) and using
    E[real^2|M<=m_p] = E[M^2|M<=m_p]/2 (real/imag symmetry) with
    E[M^2|M<=m_p] = 2*sigma^2*(1-(1+t)(1-bg_frac))/bg_frac (incomplete-
    gamma integral of the truncated exponential), the selected subset's
    true real-part variance is sigma^2*(1-(1+t)(1-bg_frac))/bg_frac --
    correction = 1/sqrt(that ratio). Verified: this closed form matches
    a 2M-sample Monte Carlo to 4 significant figures at every bg_frac
    tested (0.1-0.5) -- see tests/test_recon_mslr.py.

    Exact-zero voxels are excluded before the bg_frac selection -- a real
    bug, not a hypothetical, caught on 2026-09-19: GatheredSense's smaps
    combine (recon/operators.py) forces every masked-out-background voxel
    (process_smaps' own exact-zero-background invariant -- see its
    docstring) to be *identically* zero, not small-magnitude, in every
    A^H(ksp). For a real ball phantom this repo's own smaps arrays are
    ~57% exact zero, comfortably exceeding bg_frac=0.25, so the naive
    version's lowest-bg_frac selection landed entirely on these
    zero-by-construction (not thermal-noise) voxels and returned a hard
    0.0 -- normalize_noise then divided ksp by that 0.0, propagating
    inf/NaN through the whole solve until it surfaced downstream as an
    unrelated-looking SVD convergence failure in patchSVST, many
    iterations and over an hour of GPU time later. The lesson generalizes:
    always test this kind of background-statistics estimator against a
    masked (exact-zero-outside-support) operator, not just a fully dense
    one -- a fully random/dense synthetic smaps (as this function's first
    round of tests used) cannot reproduce this failure mode at all.
    """
    nonzero = X[X.abs() > 0]
    if nonzero.numel() < 100:
        raise ValueError(
            f'estimate_noise_std: only {nonzero.numel()} nonzero-magnitude voxels in X -- '
            'too few to estimate a noise floor from (or the masked-out-background convention '
            'has changed; see this function\'s docstring for the exact-zero-masking bug this '
            'check guards against).'
        )
    mag = nonzero.abs()
    k = max(1, int(bg_frac * mag.numel()))
    thresh = torch.kthvalue(mag, k).values
    bg_vals = nonzero[mag <= thresh]
    measured = bg_vals.real.std().item()
    if not (measured > 0):
        raise ValueError(
            f'estimate_noise_std: measured std={measured} (non-positive/NaN) even after '
            'excluding exact zeros -- the background selection is still degenerate; '
            'investigate before trusting normalize_noise on this data.'
        )

    t = -math.log(1 - bg_frac)
    ratio = (1 - (1 + t) * (1 - bg_frac)) / bg_frac
    correction = 1 / math.sqrt(ratio)
    return measured * correction


def estimate_operator_noise_factor(
    A, size_out: tuple, dtype: torch.dtype, device: torch.device, n_trials: int = 20,
) -> float:
    """Mean image-domain std from propagating synthetic unit-variance
    complex Gaussian k-space noise through A.adjoint(), averaged over
    n_trials independent draws. A clean, signal-free measurement of the
    operator's own noise-propagation factor -- no real data involved, so
    no risk of the real-signal contamination that made estimate_noise_std
    unreliable on real, tightly-masked data (see its docstring's
    2026-09-19 bug writeup). size_out: A's k-space-domain shape (e.g.
    (K,Nc,Nt) for the full multi-frame BlockDiagonal `run_recon` uses).

    std is computed over the *nonzero* output only (excludes exact-zero,
    masked-out-of-coil-support voxels -- see recon/operators.py's smaps
    combine): std() over the whole array would otherwise be diluted by
    whatever fraction of the grid the coil-sensitivity mask excludes (as
    large as ~57% on a real ball phantom), underestimating the actual
    noise level *within* the region normalize_noise's calibration actually
    needs to be correct for (the masked-out region is exactly zero in
    every reconstruction anyway, regardless of any regularization
    threshold, so its "noise level" is neither real nor relevant).
    """
    stds = []
    for _ in range(n_trials):
        noise_k = (
            torch.randn(*size_out, dtype=torch.float32, device=device)
            + 1j * torch.randn(*size_out, dtype=torch.float32, device=device)
        ) / math.sqrt(2)
        img_noise = A.adjoint(noise_k.to(dtype))
        nonzero = img_noise[img_noise.abs() > 0]
        stds.append(nonzero.std().item() if nonzero.numel() > 0 else img_noise.std().item())
    return float(np.mean(stds))


def estimate_kspace_noise_std(
    ksp_frame: torch.Tensor, idx: torch.Tensor, N: tuple[int, int, int], outer_frac: float = 0.1,
) -> float:
    """Empirical per-sample std, measured directly in k-space from the
    outer_frac highest-radius (ky,kz) shell of one frame's gathered
    samples. Despite the name, this is NOT a reliable measure of thermal
    noise alone on real acquired data, and is no longer used by run_recon's
    normalize_noise path (removed 2026-09-22, see that docstring) -- real
    k-space at high (ky,kz) can carry substantial structured, non-Gaussian
    content (shot-to-shot inconsistency across a multi-shot echo train,
    sharp-edge signal for a fully-sampled acquisition) that pre-whitening
    was never meant to remove, and this estimator has no way to separate
    that from the i.i.d. Gaussian thermal-noise floor. Kept as a
    diagnostic for exactly that real-vs-Gaussian question (large values
    here indicate real structured high-k content, not necessarily a
    bug) and because it correctly recovers a *known, injected* noise std
    on synthetic data with no structured content -- see this module's
    tests. radius is computed in (ky,kz) only (idx unraveled against
    N=(Nx,Ny,Nz)) -- kx is always fully sampled per shot (whole readout
    line), so it doesn't discriminate high vs low spatial frequency
    content the way undersampled ky,kz do.

    ksp_frame: (K,Nc) complex, this frame's gathered k-space (recon/
    operators.py's gather_ksp convention). idx: (K,) int64, that same
    frame's flat sample indices into the C-order-flattened (Nx,Ny,Nz) grid
    (e.g. GatheredSense.idx) -- same row order as ksp_frame.
    """
    Nx, Ny, Nz = N
    kx = idx // (Ny * Nz)
    rem = idx - kx * (Ny * Nz)
    ky = (rem // Nz).float() - Ny / 2
    kz = (rem % Nz).float() - Nz / 2
    r = torch.sqrt(ky**2 + kz**2)
    thresh = torch.quantile(r.float(), 1 - outer_frac)
    outer = ksp_frame[r >= thresh]
    return outer.real.std().item()


def _reg_weights(
    patch_sizes: list[tuple[int, int, int]], Nt: int, N_voxels: int, lambda_global: float
) -> list[float]:
    """Ong & Lustig 2016 eq. (4): lambda_k = sqrt(p_k) + sqrt(Nt) +
    sqrt(log(N_voxels*Nt / max(p_k, Nt))), p_k = voxels per patch. Natural
    log (paper states the weight only up to a constant; lambda_global absorbs
    any rescaling -- see reconstruct.jl's own comment on this choice)."""
    lambdas = []
    for ps in patch_sizes:
        p_k = math.prod(ps)
        lam = math.sqrt(p_k) + math.sqrt(Nt) + math.sqrt(math.log(N_voxels * Nt / max(p_k, Nt)))
        lambdas.append(lam * lambda_global)
    return lambdas


def run_recon(
    *,
    fn_ksp: str,
    fn_smaps: str,
    patch_sizes: list[tuple[int, int, int]],
    strides: list[tuple[int, int, int]],
    niters: int = 200,
    sigma1A: float | None = None,
    device: torch.device | str = "cuda",
    mom: str = "fpgm",
    conv_tol: float = 1e-5,
    lambda_global: float = 1.0,
    fn_b0map: str | None = None,
    L_b0: int = 32,
    nbins_b0: int = 128,
    normalize_noise: bool = True,
    r2star_map: torch.Tensor | None = None,
    t_ref_s: float = 0.0,
) -> ReconResult:
    """fn_b0map: optional path to a run_b0map.py output (<seqname>_b0map.h5,
    'b0map_hz' on the EPI grid -- see preprocessing/run_b0map.py). When
    given, builds the encoding operator with time-segmented B0 off-
    resonance correction (recon/operators.py) instead of the plain
    encoding operator -- reads per-sample acquisition time from fn_ksp's
    'echo_times' dataset (preprocessing/preprocess.py's _build_echo_times;
    (Ny,Nz,Nt), broadcast across Nx here since kx doesn't affect echo
    time). L_b0/nbins_b0 are mri_exp_approx's segment count/histogram bins
    (see operators.py's module docstring for the real-scale sweep that
    settled L_b0=32). sigma1A defaults to None, in which case it
    is measured here via power iteration (operators_b0.estimate_spectral_
    norm) on the operator actually built -- required when fn_b0map is set,
    since the B0-corrected operator's spectral norm is not guaranteed to
    match the uncorrected operator's, and supplying a too-small sigma1A
    silently makes POGM's step size too large (divergence, not a clean
    failure). Pass sigma1A explicitly to skip this measurement (e.g. when
    reusing a previously-measured value).

    r2star_map/t_ref_s: optional -- forwarded to build_encoding_operator_b0
    to also correct T2*/T1 amplitude decay (see that function's docstring
    for the complex-field ψ(r) = i*2*pi*Δf(r) - R2*(r) generalization, the
    sign-convention divergence from the worktree-lowres-calib-recon
    branch's adjoint-only calib script, and why t_ref_s should be the
    nominal-TE echo's acquisition time). Ignored (and must be left None)
    when fn_b0map is None -- R2* correction only makes sense layered on
    top of the B0-corrected operator, not the plain one.

    normalize_noise: rescale `ksp` (and therefore X0 and every POGM
    iterate) by 1/estimate_operator_noise_factor before the solve, undoing
    it on the returned X/X_recon only -- _reg_weights' lambda_k formula
    (Ong & Lustig 2016 eq. 4) assumes unit-variance thermal noise in image
    space (see ../mslr-recon/scripts/reconstruct.jl's own comment to that
    effect, and arXiv:1507.08751 Section II-A/IV for the underlying i.i.d.
    Gaussian noise model the formula is calibrated against).

    The k-space-side factor is NOT measured empirically from real acquired
    data (a real per-dataset outer-(ky,kz)-shell std, formerly via
    estimate_kspace_noise_std, was tried and removed 2026-09-22): real
    k-space is `Y = sum_i X_i + X_Z` (same eq. as the paper above), and an
    outer-shell std estimator has no way to separate the i.i.d. Gaussian
    X_Z pre-whitening is meant to normalize from real structured,
    non-Gaussian content that also lands at high (ky,kz) -- shot-to-shot
    inconsistency across a multi-shot echo train (T1/steady-state drift,
    eddy currents, off-resonance accrual), or genuine sharp-edge signal
    for a fully-sampled (R~1) acquisition. Measured directly on real
    20260918ball data: a real-noise-scan-derived whitening matrix, real
    coil-compression matrix, and real k-space trajectory, run end to end
    through preprocessing/preprocess.py's actual whitening -> coil
    compression -> regridding -> scatter chain on REAL noise-scan samples
    (not synthetic, so the true coil covariance is preserved) reproduces
    unit variance to within ~1-3% (0.997 on 1_1x_5.4mm, 0.973 on
    2_6x_2.4mm) -- i.e. the pipeline provably preserves thermal-noise
    variance by construction. Meanwhile the *same* real dataset's actual
    acquired-data outer-shell std reads ~72-101 (1_1x_5.4mm, R~1) and ~73
    pre-epi_gridding-fix / ~7.7 predicted post-fix (2_6x_2.4mm, R~6) --
    proving that gap is real structured k-space content, not a
    normalization bug, and that the outer-shell estimator was never a
    valid proxy for the paper's X_Z term to begin with. A hard
    `[0.1, 10]`-band guardrail built on that estimator (2026-09-21) is
    removed for the same reason: it would (and did) block correctly-
    processed, fully-sampled acquisitions no differently than a genuine
    absolute-scale bug, and passing it for an undersampled acquisition was
    coincidental, not a validation. estimate_kspace_noise_std itself is
    kept (still independently tested) as a diagnostic for exactly this
    real-vs-Gaussian-noise question, just no longer called from here.

    dc_costs/reg_costs in the returned ReconResult are measured in this
    normalized scale, not the original data's physical units -- expect much
    smaller numbers than an unnormalized run at the same lambda_global."""
    device = torch.device(device)
    Nscales = len(patch_sizes)

    print("Loading sensitivity maps...")
    smaps, smaps_chw = _load_normalized_smaps(fn_smaps, device)
    print(f"  Sensitivity maps: {tuple(smaps.shape)}")
    Nx, Ny, Nz, Nvc = smaps.shape

    with h5py.File(fn_ksp, "r") as f:
        ksp_shape = f["ksp_epi_zf"].shape  # cheap metadata peek, no data read
    assert ksp_shape == (Nx, Ny, Nz, Nvc, ksp_shape[-1]), (
        f"smaps shape {tuple(smaps.shape)} doesn't match k-space dims {ksp_shape[:4]}"
    )
    Nt = ksp_shape[-1]

    omega = _load_omega(fn_ksp, Nx, Ny, Nz, Nt, device)
    R = (Nx * Ny * Nz) / omega[:, :, :, 0].sum().item()
    print(f"Acceleration factor R ~ {R:.2f}")
    counts = omega.sum(dim=(0, 1, 2))
    assert torch.all(counts == counts[0]), "Frames have differing sample counts"

    print("Building encoding operator...")  # smaps_chw already computed above
    if fn_b0map is not None:
        print(f"  Loading B0 field map from {fn_b0map} (L={L_b0}, nbins={nbins_b0})...")
        b0map_hz = torch.from_numpy(_load_array(fn_b0map, "b0map_hz").astype(np.float32)).to(device)
        assert tuple(b0map_hz.shape) == (Nx, Ny, Nz), (
            f"b0map_hz shape {tuple(b0map_hz.shape)} doesn't match k-space dims ({Nx},{Ny},{Nz})"
        )
        echo_times_yz = _load_echo_times(fn_ksp, device)
        if r2star_map is not None:
            print(f"  R2* correction enabled (t_ref_s={t_ref_s * 1000:.3f} ms)...")
            assert tuple(r2star_map.shape) == (Nx, Ny, Nz), (
                f"r2star_map shape {tuple(r2star_map.shape)} doesn't match k-space dims "
                f"({Nx},{Ny},{Nz})"
            )
        A = build_encoding_operator_b0(
            smaps_chw, omega, b0map_hz, echo_times_yz, L=L_b0, nbins=nbins_b0,
            r2star_map=r2star_map, t_ref_s=t_ref_s,
        )
    else:
        assert r2star_map is None, (
            "run_recon: r2star_map requires fn_b0map (R2* correction is layered on top of "
            "the B0-corrected operator, not the plain one)"
        )
        A = build_encoding_operator(smaps_chw, omega)

    if sigma1A is None:
        if fn_b0map is None:
            raise ValueError(
                "run_recon: sigma1A must be supplied when fn_b0map is not set -- "
                "the plain SENSE operator's spectral norm has no cheap closed-form "
                "estimate wired up here, so auto-estimation only covers the "
                "B0-corrected path."
            )
        print("  sigma1A not supplied -- measuring via power iteration...")
        x0 = torch.randn(Nx, Ny, Nz, Nt, dtype=torch.complex64, device=device)
        sigma1A = estimate_spectral_norm(A, x0)
        print(f"    sigma1A (B0-corrected) = {sigma1A:.6f}")
        del x0

    print("Loading k-space (gathered per frame, never materializing the dense array)...")
    ksp = load_and_gather_ksp(fn_ksp, A, device)  # (K,Nc,Nt)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        free_gb, total_gb = (x / 1e9 for x in torch.cuda.mem_get_info())
        print(f"  VRAM free after loading gathered k-space: {free_gb:.2f} / {total_gb:.2f} GB")

    # See normalize_noise's docstring above: the k-space-side factor is not
    # measured from real (structured-content-contaminated) acquired data --
    # the pipeline's own whitening + coil-compression + regridding chain is
    # verified (real-noise-through-pipeline test, see docstring) to preserve
    # unit-variance thermal noise by construction, so only the operator's
    # own noise-propagation factor needs measuring here.
    noise_std = 1.0
    if normalize_noise:
        op_factor = estimate_operator_noise_factor(A, tuple(ksp.shape), ksp.dtype, device)
        print(f"  Operator noise-propagation factor = {op_factor:.4f}")
        noise_std = op_factor
        print(f"  Estimated thermal-noise std (image domain, pre-normalization) = "
              f"{noise_std:.4f} -- rescaling ksp/X0 by 1/{noise_std:.4f}")
        if not (math.isfinite(noise_std) and 1e-6 < noise_std < 1e9):
            raise ValueError(
                f"run_recon: noise_std={noise_std} is non-finite or outside a plausible "
                "range -- refusing to divide ksp by this (see normalize_noise's docstring "
                "for the real crash this guard exists to catch immediately, instead of "
                "silently propagating inf/NaN through the whole solve)."
            )
        ksp = ksp / noise_std

    L = Nscales * sigma1A**2

    N_voxels = Nx * Ny * Nz
    lambdas = _reg_weights(patch_sizes, Nt, N_voxels, lambda_global)
    print(f"Regularization weights lambdas = {[round(lam, 6) for lam in lambdas]}")

    def image_sum(X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=-1)

    def dc_cost(X: torch.Tensor) -> float:
        res = A.apply(image_sum(X)) - ksp
        return 0.5 * res.norm().item() ** 2

    def dc_cost_grad(X: torch.Tensor) -> torch.Tensor:
        res = A.apply(image_sum(X)) - ksp
        g = A.adjoint(res)  # (Nx,Ny,Nz,Nt)
        return g.unsqueeze(-1).expand(-1, -1, -1, -1, Nscales).clone()

    def reg_cost(X: torch.Tensor) -> float:
        # Chunked gather (patchSVST's own _all_patch_starts/_gather_patch_chunk),
        # not img2patches directly -- the same full-P-tensor OOM risk applies
        # here (called once, for X0, but still needs to not crash before the
        # solve even starts on a large grid/fine patch_size).
        total = 0.0
        for k in range(Nscales):
            img_k = X[..., k]
            starts, ps = _all_patch_starts(img_k.shape[:3], patch_sizes[k], strides[k])
            chunk_size = _chunk_size_for_budget(ps, img_k.shape[-1], img_k.element_size(), _DEFAULT_SVD_CHUNK_BYTES)
            nuc = 0.0
            for i in range(0, len(starts), chunk_size):
                P_chunk = _gather_patch_chunk(img_k, starts[i : i + chunk_size], ps)
                nuc += patch_nucnorm(P_chunk).item()
            total += lambdas[k] * nuc
        return total

    last_reg = [0.0]  # updated for free inside g_prox each iter; set once for iter 0 below

    def g_prox(X: torch.Tensor, c: float) -> torch.Tensor:
        reg = 0.0
        for k in range(Nscales):
            result, cost = patchSVST(X[..., k], c * lambdas[k], patch_sizes[k], strides[k])
            X[..., k] = result
            reg += lambdas[k] * cost.item()
        last_reg[0] = reg
        return X

    print("Initializing X0...")
    Atksp = A.adjoint(ksp) / Nscales  # (Nx,Ny,Nz,Nt)
    X0 = Atksp.unsqueeze(-1).expand(-1, -1, -1, -1, Nscales).clone()
    last_reg[0] = reg_cost(X0)

    dc_costs: list[float] = []
    reg_costs: list[float] = []
    restarts: list[bool] = []
    rel_changes: list[float] = []

    def logger(it, xk, yk, is_restart, fcostnew, rel_change):
        dc_costs.append(fcostnew)
        reg_costs.append(last_reg[0])
        restarts.append(is_restart)
        rel_changes.append(rel_change)

    print(
        f"\nIteratively reconstructing ({niters} iterations, {Nscales} scale(s), "
        f"mom={mom}, conv_tol={conv_tol})..."
    )
    t_start = time.time()
    X, _ = pogm_restart(
        X0,
        dc_cost,
        dc_cost_grad,
        L,
        mom=mom,
        niter=niters,
        g_prox=g_prox,
        fun=logger,
        conv_tol=conv_tol,
    )
    runtime_s = time.time() - t_start
    X_recon = image_sum(X)

    print(f"Wall-clock: {runtime_s:.1f} s, {runtime_s / max(len(dc_costs) - 1, 1):.2f} s/iter")

    if normalize_noise:
        X = X * noise_std
        X_recon = X_recon * noise_std

    return ReconResult(
        X=X,
        X_recon=X_recon,
        omega=omega,
        dc_costs=dc_costs,
        reg_costs=reg_costs,
        restarts=restarts,
        rel_changes=rel_changes,
        R=R,
        sigma1A=sigma1A,
        L=L,
        lambdas=lambdas,
        runtime_s=runtime_s,
        meta=dict(niters=niters, mom=mom, conv_tol=conv_tol, lambda_global=lambda_global,
                   patch_sizes=patch_sizes, strides=strides,
                   normalize_noise=normalize_noise, noise_std=noise_std),
    )


def save_result(
    fn_base: str, result: ReconResult, fov: tuple[float, float, float], **extra_attrs
) -> None:
    # Raw complex data first, deliberately: a run_recon() call can take tens
    # of minutes, and save_recon_nifti (below) needs a plain numpy array,
    # not a CUDA tensor -- getting that boundary wrong once already lost a
    # completed real reconstruction (see git history / session notes), so
    # the full-precision .h5 -- needing no such conversion care beyond the
    # explicit .cpu().numpy() already here -- goes to disk before anything
    # else gets a chance to fail.
    X_recon_np = result.X_recon.detach().cpu().numpy()
    with h5py.File(f"{fn_base}.h5", "w") as f:
        f.create_dataset("X_recon", data=X_recon_np)
        f.create_dataset("X", data=result.X.detach().cpu().numpy())
        f.create_dataset("omega", data=result.omega.detach().cpu().numpy())
        f.create_dataset("dc_costs", data=np.asarray(result.dc_costs))
        f.create_dataset("reg_costs", data=np.asarray(result.reg_costs))
        f.create_dataset("restarts", data=np.asarray(result.restarts, dtype=bool))
        f.create_dataset("rel_changes", data=np.asarray(result.rel_changes))
        f.attrs["R"] = result.R
        f.attrs["sigma1A"] = result.sigma1A
        f.attrs["L"] = result.L
        f.attrs["runtime_s"] = result.runtime_s
    print(f"Wrote {fn_base}.h5")

    save_recon_nifti(
        fn_base,
        X_recon_np,
        fov=fov,
        R=result.R,
        sigma1A=result.sigma1A,
        L=result.L,
        lambdas=result.lambdas,
        runtime_s=result.runtime_s,
        n_iters=len(result.dc_costs) - 1,
        final_dc_cost=result.dc_costs[-1],
        final_reg_cost=result.reg_costs[-1],
        **result.meta,
        **extra_attrs,
    )
    print(f"Wrote {fn_base}.nii.gz + .json")

Momentum = Literal["pgm", "fpgm", "pogm"]
Restart = Literal["none", "gr", "fr"]

_EPS = torch.finfo(torch.float32).eps


def _gr_restart(fgrad: torch.Tensor, ynew_yold: torch.Tensor, restart_cutoff: float) -> bool:
    inner = -torch.vdot(fgrad.reshape(-1), ynew_yold.reshape(-1)).real.item()
    return inner <= restart_cutoff * fgrad.norm().item() * ynew_yold.norm().item()


def pogm_restart(
    x0: torch.Tensor,
    fcost: Callable[[torch.Tensor], float],
    fgrad_fn: Callable[[torch.Tensor], torch.Tensor],
    f_L: float,
    *,
    f_mu: float = 0.0,
    mom: Momentum = "pogm",
    restart: Restart = "gr",
    restart_cutoff: float = 0.0,
    bsig: float = 1.0,
    niter: int = 10,
    g_prox: Callable[[torch.Tensor, float], torch.Tensor] = lambda z, c: z,
    fun: Callable = lambda it, xk, yk, is_restart, fcostnew, rel_change: None,
    conv_tol: float = 0.0,
    conv_min_iter: int = 10,
):
    """x, out = pogm_restart(x0, fcost, fgrad_fn, f_L; ...)

    `fun(iter, xk, yk, is_restart, fcostnew, rel_change)` is called once per
    iteration (iter=0 for the initial point, rel_change=nan there) and its
    return values are collected into `out`, mirroring the Julia signature
    exactly (see mirt_mod.jl point 7)."""
    if mom not in ("pgm", "fpgm", "pogm"):
        raise ValueError(f"mom={mom}")
    if restart not in ("none", "gr", "fr"):
        raise ValueError(f"restart={restart}")
    if f_L < 0:
        raise ValueError(f"f_L={f_L} < 0")
    if f_mu < 0:
        raise ValueError(f"f_mu={f_mu} < 0")
    if bsig < 0:
        raise ValueError(f"bsig={bsig} < 0")
    if abs(restart_cutoff) >= 1:
        raise ValueError(f"restart_cutoff={restart_cutoff}")

    L, mu = float(f_L), float(f_mu)
    q = mu / L if L != 0 else 0.0

    told = sig = zetaold = 1.0
    xold = yold = uold = zold = x0
    Fcostold = fcost(x0)

    out = [None] * (niter + 1)
    out[0] = fun(0, x0, x0, False, Fcostold, float("nan"))
    if niter == 0:
        return x0, out[:1]

    niter_actual = niter
    last_restart_iter = 0
    adaptive_cmi = conv_min_iter

    xnew = ynew = None
    Fgrad = torch.zeros_like(x0)
    Fgradold = torch.zeros_like(x0) if mom == "pogm" else x0

    for it in range(1, niter + 1):
        alpha = (2.0 / (L + mu)) if (mom == "pgm" and mu != 0) else (1.0 / L)

        fgrad = fgrad_fn(xold)
        is_restart = False

        if mom in ("pgm", "fpgm"):
            ynew = g_prox(xold - alpha * fgrad, alpha)
            Fgrad = -(1.0 / alpha) * (ynew - xold)
            Fcostnew = fcost(ynew)

            if restart != "none":
                if (restart == "fr" and Fcostnew > Fcostold) or (
                    restart == "gr" and _gr_restart(Fgrad, ynew - yold, restart_cutoff)
                ):
                    told = 1.0
                    is_restart = True
                Fcostold = Fcostnew
        else:  # pogm
            unew = xold - alpha * fgrad

        beta = None
        tnew = told
        if mom == "fpgm" and mu != 0:
            beta = (1 - math.sqrt(q)) / (1 + math.sqrt(q))
        elif mom == "pogm" and mu != 0:
            beta = (2 + q - math.sqrt(q**2 + 8 * q)) ** 2 / (4 * (1 - q))
        elif mom != "pgm":
            tnew = (
                0.5 * (1 + math.sqrt(1 + 8 * told**2))
                if (mom == "pogm" and it == niter)
                else 0.5 * (1 + math.sqrt(1 + 4 * told**2))
            )
            beta = (told - 1) / tnew

        if mom == "pgm":
            xnew = ynew
        elif mom == "fpgm":
            xnew = ynew + beta * (ynew - yold)
        else:  # pogm
            gamma = (2 + q - math.sqrt(q**2 + 8 * q)) / 2 if mu != 0 else sig * told / tnew
            ba_z = beta * alpha / zetaold
            znew = unew + beta * (unew - uold) + gamma * (unew - xold) - ba_z * (xold - zold)
            zetanew = alpha * (1 + beta + gamma)
            # g_prox is allowed to mutate its argument in place (e.g.
            # recon/mslr.py's g_prox writes into X and returns it,
            # for a small transient memory win) -- clone znew first so
            # `xnew is znew` never holds. Without this, xnew - znew below
            # is always exactly zero regardless of what g_prox computed
            # (the two names alias the same mutated tensor), and zold at
            # :156 also ends up aliasing xold next iteration, silently
            # zeroing POGM's momentum correction terms every iteration.
            xnew = g_prox(znew.clone(), zetanew)

            iz = 1.0 / zetanew
            Fgrad = fgrad - iz * (xnew - znew)
            fgrad = None

            ynew = xold - alpha * Fgrad
            Fcostnew = fcost(xnew)

            if restart != "none":
                ynew_yold = ynew - yold
                if (restart == "fr" and Fcostnew > Fcostold) or (
                    restart == "gr" and _gr_restart(Fgrad, ynew_yold, restart_cutoff)
                ):
                    tnew = 1.0
                    sig = 1.0
                    is_restart = True
                elif torch.vdot(Fgrad.reshape(-1), Fgradold.reshape(-1)).real.item() < 0:
                    sig = bsig * sig
                Fcostold = Fcostnew
                Fgradold, Fgrad = Fgrad, Fgradold

            uold, zold, zetaold = unew, znew, zetanew

        recon_prev = xold if mom == "pogm" else yold
        recon_curr = xnew if mom == "pogm" else ynew
        rel_change = (recon_curr - recon_prev).norm().item() / (recon_prev.norm().item() + _EPS)

        if is_restart:
            last_restart_iter = it
            adaptive_cmi = max(1, adaptive_cmi - 1)

        if conv_tol > 0 and it >= last_restart_iter + adaptive_cmi and rel_change < conv_tol:
            out[it] = fun(it, xnew, ynew, is_restart, Fcostnew, rel_change)
            niter_actual = it
            break

        out[it] = fun(it, xnew, ynew, is_restart, Fcostnew, rel_change)
        xold, yold = xnew, ynew
        if mom != "pgm" and mu == 0:
            told = tnew

    result = xnew if mom == "pogm" else ynew
    return result, out[: niter_actual + 1]


def _patch_starts(n: int, patch: int, stride: int) -> list[int]:
    nsteps = -(-(n - patch) // stride)  # ceil division
    return [min(i * stride, n - patch) for i in range(nsteps + 1)]


_DEFAULT_SVD_CHUNK_BYTES = 4_000_000_000  # see SVST's docstring for the benchmark this is based on


def _all_patch_starts(
    shape: tuple[int, int, int], patch_size: tuple[int, int, int], stride_size: tuple[int, int, int]
) -> tuple[list[tuple[int, int, int]], tuple[int, int, int]]:
    """Every (sx,sy,sz) patch start position over `shape`, plus the clamped
    (psx,psy,psz) patch size actually used (patch_size capped to each
    axis' own image size) -- shared by img2patches/patches2img (unchanged,
    still used directly by tests and reg_cost's chunked path below) and
    patchSVST's own chunked gather/scatter (which never materializes the
    full per-scale patch tensor -- see patchSVST's docstring)."""
    Nx, Ny, Nz = shape
    psx, psy, psz = (min(p, n) for p, n in zip(patch_size, shape))
    starts_x = _patch_starts(Nx, psx, stride_size[0])
    starts_y = _patch_starts(Ny, psy, stride_size[1])
    starts_z = _patch_starts(Nz, psz, stride_size[2])
    starts = [(sx, sy, sz) for sz in starts_z for sy in starts_y for sx in starts_x]
    return starts, (psx, psy, psz)


def _chunk_size_for_budget(
    patch_size: tuple[int, int, int], Nt: int, element_size: int, max_chunk_bytes: int
) -> int:
    p_k = patch_size[0] * patch_size[1] * patch_size[2]
    return max(1, max_chunk_bytes // (element_size * p_k * Nt))


def _gather_patch_chunk(
    img: torch.Tensor, starts_chunk: list[tuple[int, int, int]], patch_size: tuple[int, int, int]
) -> torch.Tensor:
    """(len(starts_chunk), prod(patch_size), Nt) -- img2patches' own gather,
    restricted to one chunk of patch positions."""
    psx, psy, psz = patch_size
    Nt = img.shape[-1]
    return torch.stack(
        [
            img[sx : sx + psx, sy : sy + psy, sz : sz + psz, :].reshape(psx * psy * psz, Nt)
            for sx, sy, sz in starts_chunk
        ],
        dim=0,
    )


def img2patches(img: torch.Tensor, patch_size, stride_size) -> torch.Tensor:
    """(Nx,Ny,Nz,Nt) -> (Np, prod(patch_size), Nt), one row per (space x time) patch."""
    Nx, Ny, Nz, Nt = img.shape
    if any(s <= 0 for s in stride_size):
        raise ValueError(f"stride_size elements must be positive, got {stride_size}")
    psx, psy, psz = (min(p, n) for p, n in zip(patch_size, (Nx, Ny, Nz)))

    starts_x = _patch_starts(Nx, psx, stride_size[0])
    starts_y = _patch_starts(Ny, psy, stride_size[1])
    starts_z = _patch_starts(Nz, psz, stride_size[2])

    patches = [
        img[sx : sx + psx, sy : sy + psy, sz : sz + psz, :].reshape(psx * psy * psz, Nt)
        for sz in starts_z
        for sy in starts_y
        for sx in starts_x
    ]
    return torch.stack(patches, dim=0)


def patches2img(P: torch.Tensor, patch_size, stride_size, og_size) -> torch.Tensor:
    """Inverse of img2patches: recombine via overlap-averaging."""
    _, _, Nt = P.shape
    Nx, Ny, Nz = og_size
    psx, psy, psz = (min(p, n) for p, n in zip(patch_size, og_size))

    starts_x = _patch_starts(Nx, psx, stride_size[0])
    starts_y = _patch_starts(Ny, psy, stride_size[1])
    starts_z = _patch_starts(Nz, psz, stride_size[2])

    img = torch.zeros(Nx, Ny, Nz, Nt, dtype=P.dtype, device=P.device)
    pcount = torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device=P.device)

    ip = 0
    for sz in starts_z:
        for sy in starts_y:
            for sx in starts_x:
                patch = P[ip].reshape(psx, psy, psz, Nt)
                img[sx : sx + psx, sy : sy + psy, sz : sz + psz, :] += patch
                pcount[sx : sx + psx, sy : sy + psy, sz : sz + psz] += 1.0
                ip += 1

    pcount.clamp_(min=1.0)
    return img / pcount.unsqueeze(-1)


def patch_nucnorm(P: torch.Tensor) -> torch.Tensor:
    """Sum of nuclear norms across all (space x time) patches in P: (Np, m, n)."""
    if P.ndim != 3:
        raise ValueError("P must be (patches, space, time)")
    return torch.linalg.svdvals(P).sum()


def _svst_batch(X: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One un-chunked SVST batch -- see SVST's docstring for the algorithm;
    this is exactly its former body, factored out so SVST can call it
    per-chunk without duplicating the math."""
    fro = torch.linalg.matrix_norm(X, ord="fro")
    zero_mask = fro <= beta
    mask_mnn = zero_mask[..., None, None]
    X_safe = torch.where(mask_mnn, torch.zeros_like(X), X)

    U, S, Vh = torch.linalg.svd(X_safe, full_matrices=False)
    s_thresh = torch.clamp(S - beta, min=0.0)
    recon = U @ (s_thresh.to(Vh.dtype).unsqueeze(-1) * Vh)
    reg = s_thresh.sum(dim=-1)

    recon = torch.where(mask_mnn, torch.zeros_like(recon), recon)
    reg = torch.where(zero_mask, torch.zeros_like(reg), reg)
    return recon, reg


def SVST(
    X: torch.Tensor, beta: float, max_chunk_bytes: int = _DEFAULT_SVD_CHUNK_BYTES
) -> tuple[torch.Tensor, torch.Tensor]:
    """Singular Value Soft-Thresholding, the proximal operator of beta * nuclear-norm.

    X: (Np, m, n), batched over patches. Returns (X_thresholded, reg),
    reg = per-batch-element sum(max(sigma - beta, 0)), the nuclear norm of the
    thresholded result -- a free byproduct of the SVD already computed.

    Whenever ||X||_F <= beta, every singular value is <= beta too (sigma_max <=
    ||X||_F), so the result is exactly zero -- not an approximation. Forcing
    those entries to exact zero *before* the SVD (rather than just zeroing the
    output after) avoids feeding a near-zero-magnitude matrix through
    torch.linalg.svd: repeated soft-thresholding near this boundary can produce
    subnormal-magnitude patches, and the mslr-recon Julia port found cuSOLVER's
    GPU SVD returns all-NaN (not just imprecise) on those -- CPU LAPACK handles
    them fine, but the guard is applied on both backends here since it's cheap
    and exact either way.

    max_chunk_bytes: torch.linalg.svd runs on Np patches at once, chunked
    along the patch (batch) dimension so no single call needs more than
    ~max_chunk_bytes for X's own storage (U's output is comparable size,
    so peak memory is a small multiple of this, not of the full Np-patch
    batch). A single un-chunked call scales memory linearly with patch
    count and can exceed real GPU capacity at fine resolution/large grids
    -- measured for this repo's real 4_93.5x_0.8mm dataset at its
    patch_size=(18,18,18): ~90GB unchunked (15979 patches) against a 49GB
    GPU. Chunking was benchmarked (2026-09-22, same GPU, dataset-3-scale
    23958x729x60 patches) at chunk sizes 1000-8000 patches: wall-clock
    time was within noise of the unchunked call (0.95-1.00x) -- cuSOLVER's
    batched SVD is already compute-saturated at these chunk sizes, so this
    is a real fix for memory with no meaningful runtime cost, not a
    speed/memory tradeoff. 4GB keeps every chunk far below the point where
    the benchmark showed any slowdown, for any patch_size/Nt this repo
    uses today.
    """
    item_bytes = X.element_size() * X.shape[-2] * X.shape[-1]
    chunk_size = max(1, max_chunk_bytes // item_bytes)
    if X.shape[0] <= chunk_size:
        return _svst_batch(X, beta)

    recons, regs = [], []
    for i in range(0, X.shape[0], chunk_size):
        recon_i, reg_i = _svst_batch(X[i : i + chunk_size], beta)
        recons.append(recon_i)
        regs.append(reg_i)
    return torch.cat(recons, dim=0), torch.cat(regs, dim=0)


def _unit_block_svst(img: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """patch_size=[1,1,1]: SVST of each (1,Nt) voxel time series reduces to a
    vector soft-threshold (see recon.jl's derivation: SVD of a 1xNt row is
    U=[1], S=[||x||], Vh=x/||x||). Avoids ~Nvox individual 1x1 SVDs."""
    norms = torch.linalg.vector_norm(img, dim=-1, keepdim=True)
    scale = torch.clamp(1.0 - beta / norms, min=0.0)  # beta/0=inf -> -inf -> clamped to 0
    result = img * scale
    reg = torch.clamp(norms.squeeze(-1) - beta, min=0.0).sum()
    return result, reg


def patchSVST(
    img: torch.Tensor, beta: float, patch_size, stride_size,
    max_chunk_bytes: int = _DEFAULT_SVD_CHUNK_BYTES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply patch-wise SVST to a 4-D image (Nx,Ny,Nz,Nt) with threshold beta.
    Returns (img_thresholded, reg), reg = nuclear norm of the result summed
    over all patches (sum of thresholded singular values), free from the SVD.

    Unlike img2patches+SVST+patches2img (mathematically identical, and
    still what small/test-scale callers use), this never materializes the
    full (Np, prod(patch_size), Nt) patch tensor -- it gathers, SVST's, and
    scatters one memory-budgeted chunk of patches at a time (same budget/
    rationale as SVST's own max_chunk_bytes -- see its docstring for the
    benchmark showing this costs no meaningful wall-clock time). Needed for
    real large-grid/fine-resolution reconstructions: this repo's real
    4_93.5x_0.8mm dataset at patch_size=(18,18,18) would need ~45GB just
    for the unchunked P tensor alone (before SVST's own U/Vh), against a
    49GB GPU."""
    Nx, Ny, Nz, Nt = img.shape
    starts, (psx, psy, psz) = _all_patch_starts((Nx, Ny, Nz), patch_size, stride_size)
    if (psx, psy, psz) == (1, 1, 1):
        return _unit_block_svst(img, beta)

    chunk_size = _chunk_size_for_budget((psx, psy, psz), Nt, img.element_size(), max_chunk_bytes)
    img_out = torch.zeros_like(img)
    pcount = torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device=img.device)
    reg_total = torch.zeros((), dtype=torch.float32, device=img.device)

    for i in range(0, len(starts), chunk_size):
        chunk = starts[i : i + chunk_size]
        P_chunk = _gather_patch_chunk(img, chunk, (psx, psy, psz))
        result_chunk, reg_chunk = _svst_batch(P_chunk, beta)
        reg_total = reg_total + reg_chunk.sum()
        for j, (sx, sy, sz) in enumerate(chunk):
            patch = result_chunk[j].reshape(psx, psy, psz, Nt)
            img_out[sx : sx + psx, sy : sy + psy, sz : sz + psz, :] += patch
            pcount[sx : sx + psx, sy : sy + psy, sz : sz + psz] += 1.0

    pcount.clamp_(min=1.0)
    return img_out / pcount.unsqueeze(-1), reg_total
