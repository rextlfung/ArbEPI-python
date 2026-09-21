"""End-to-end smoke test for recon/reconstruct.py's run_recon: simulates a
small synthetic multi-coil Cartesian acquisition, writes it out in the same
.h5 layout preprocessing/ produces (ksp_epi_zf, smaps), and checks that MSLR
reconstruction runs to completion with a monotonically-behaved cost and no
NaNs -- not a golden-output comparison (see the plan's real-data validation
for that), just confirmation the full pipeline (operators + lowrank + solvers
+ I/O) is wired together correctly.
"""

import math

import h5py
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from recon.operators import build_encoding_operator  # noqa: E402
from recon.reconstruct import (  # noqa: E402
    _load_omega,
    estimate_kspace_noise_std,
    estimate_noise_std,
    estimate_operator_noise_factor,
    run_recon,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _complex_randn(*shape, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    real = torch.randn(*shape, generator=g, device=DEVICE)
    imag = torch.randn(*shape, generator=g, device=DEVICE)
    return (real + 1j * imag).to(torch.complex64)


def _write_synthetic_dataset(tmp_path, Nx, Ny, Nz, Nc, Nt, R, fn_prefix="ksp"):
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=0)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)

    x_true = _complex_randn(Nx, Ny, Nz, Nt, seed=1)
    omega = torch.stack(
        [torch.rand(Nx, Ny, Nz, device=DEVICE) > (1 - 1 / R) for _ in range(Nt)], dim=-1
    )
    # Guarantee every frame samples the same count (run_recon asserts this),
    # and every frame includes at least one sample.
    counts = omega.sum(dim=(0, 1, 2))
    k = counts.min().item()
    assert k > 0
    omega = omega & (torch.cumsum(omega.reshape(-1, Nt), dim=0) <= k).reshape(Nx, Ny, Nz, Nt)

    A = build_encoding_operator(smaps, omega)
    ksp_gathered = A.apply(x_true)  # (K,Nc,Nt)

    # Real ksp_epi_zf.h5 files (this repo's own preprocessing/ output, or
    # mslr-recon's sigpy-export path) are dense zero-filled arrays, not
    # gathered -- scatter back using each frame's own operator indices so
    # run_recon's internal gather_ksp() recovers exactly ksp_gathered.
    ksp_dense = torch.zeros(Nx, Ny, Nz, Nc, Nt, dtype=torch.complex64, device=DEVICE)
    ksp_dense_flat = ksp_dense.reshape(-1, Nc, Nt)
    for it in range(Nt):
        ksp_dense_flat[A.A[it].idx, :, it] = ksp_gathered[:, :, it]

    ksp_np = ksp_dense.cpu().numpy()
    smaps_np = smaps.permute(1, 2, 3, 0).contiguous().cpu().numpy()  # -> (Nx,Ny,Nz,Nc)

    fn_ksp = tmp_path / f"{fn_prefix}_epi_zf.h5"
    fn_smaps = tmp_path / f"{fn_prefix}_smaps.h5"
    with h5py.File(fn_ksp, "w") as f:
        f.create_dataset("ksp_epi_zf", data=ksp_np)
    with h5py.File(fn_smaps, "w") as f:
        f.create_dataset("smaps", data=smaps_np)
    return str(fn_ksp), str(fn_smaps)


def test_load_omega_prefers_omegas_dataset_over_exact_zero_inference(tmp_path):
    """A sample that rounds to exactly 0+0j after phase correction is still
    a real, acquired sample -- inferring the mask from `ksp != 0` would
    silently mark it 'not acquired'. Write a ksp_epi_zf.h5 with exactly
    this case (a sampled voxel whose k-space value is exact zero) alongside
    the authoritative 'omegas' dataset preprocess.py writes, and confirm
    _load_omega trusts 'omegas' rather than being fooled by the exact
    zero."""
    Nx, Ny, Nz, Nc, Nt = 4, 3, 3, 2, 2
    ksp_np = _complex_randn(Nx, Ny, Nz, Nc, Nt, seed=0).cpu().numpy()
    omegas_np = torch.zeros(Ny, Nz, Nt, dtype=torch.bool).numpy()
    omegas_np[0, 0, :] = True  # the one sampled location
    ksp_np[:, 0, 0, :, :] = 0.0  # ...whose k-space value happens to be exact zero

    fn_ksp = tmp_path / "ksp_with_omegas.h5"
    with h5py.File(fn_ksp, "w") as f:
        f.create_dataset("ksp_epi_zf", data=ksp_np)
        f.create_dataset("omegas", data=omegas_np)

    ksp0 = torch.from_numpy(ksp_np).to(DEVICE)
    omega = _load_omega(str(fn_ksp), Nx, Ny, Nz, Nt, ksp0)
    assert omega[:, 0, 0, :].all(), "the sampled-but-zero-valued location must read as sampled"
    assert not omega[:, 1:, 1:, :].any()


def test_load_omega_falls_back_to_exact_zero_inference_without_omegas(tmp_path):
    """Recon files written before preprocess.py added 'omegas' must still
    work, via the `!= 0` fallback."""
    Nx, Ny, Nz, Nc, Nt = 4, 3, 3, 2, 2
    ksp_np = _complex_randn(Nx, Ny, Nz, Nc, Nt, seed=1).cpu().numpy()

    fn_ksp = tmp_path / "ksp_no_omegas.h5"
    with h5py.File(fn_ksp, "w") as f:
        f.create_dataset("ksp_epi_zf", data=ksp_np)

    ksp0 = torch.from_numpy(ksp_np).to(DEVICE)
    omega = _load_omega(str(fn_ksp), Nx, Ny, Nz, Nt, ksp0)
    expected = ksp0[:, :, :, 0, :] != 0
    assert torch.equal(omega, expected)


def test_run_recon_smoke(tmp_path):
    """With lambda_global=1.0 (uncalibrated to this synthetic data's scale,
    unlike the real Ong & Lustig unit-noise-variance assumption), the very
    first prox step can legitimately zero the whole image -- FISTA/POGM only
    guarantee the *total* objective (dc_cost + reg_cost) is non-increasing,
    not dc_cost alone. So this only checks the actual guarantee plus basic
    well-formedness; see test_run_recon_recovers_signal_without_regularization
    for a real fidelity check."""
    Nx, Ny, Nz, Nc, Nt = 12, 12, 8, 4, 5
    fn_ksp, fn_smaps = _write_synthetic_dataset(tmp_path, Nx, Ny, Nz, Nc, Nt, R=3.0)

    result = run_recon(
        fn_ksp=fn_ksp,
        fn_smaps=fn_smaps,
        patch_sizes=[(Nx, Ny, Nz), (4, 4, 4)],
        strides=[(Nx, Ny, Nz), (2, 2, 2)],
        niters=15,
        sigma1A=1.0,
        device=DEVICE,
        mom="fpgm",
        conv_tol=0.0,
        lambda_global=1.0,
    )

    assert torch.isfinite(result.X_recon.abs()).all()
    dc = torch.tensor(result.dc_costs)
    reg = torch.tensor(result.reg_costs)
    assert torch.isfinite(dc).all() and torch.isfinite(reg).all()
    total = dc + reg
    assert total[-1] < total[0]
    assert len(result.dc_costs) == 16  # niters + 1 (iter 0 is logged too)
    assert result.X.shape == (Nx, Ny, Nz, Nt, 2)


def test_run_recon_recovers_signal_without_regularization(tmp_path):
    """lambda_global=0 reduces MSLR to unregularized per-frame SENSE, whose
    minimizer is unique and known: with A well-conditioned (moderate R),
    reconstruction should converge close to the ground-truth image used to
    simulate the k-space."""
    Nx, Ny, Nz, Nc, Nt = 10, 10, 6, 4, 4
    torch.manual_seed(42)
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=10)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    x_true = _complex_randn(Nx, Ny, Nz, Nt, seed=11)
    omega = torch.stack(
        [torch.rand(Nx, Ny, Nz, device=DEVICE) > 0.3 for _ in range(Nt)], dim=-1
    )
    counts = omega.sum(dim=(0, 1, 2))
    k = counts.min().item()
    omega = omega & (torch.cumsum(omega.reshape(-1, Nt), dim=0) <= k).reshape(Nx, Ny, Nz, Nt)

    A = build_encoding_operator(smaps, omega)
    ksp_gathered = A.apply(x_true)  # (K,Nc,Nt)
    ksp_dense = torch.zeros(Nx, Ny, Nz, Nc, Nt, dtype=torch.complex64, device=DEVICE)
    ksp_dense_flat = ksp_dense.reshape(-1, Nc, Nt)
    for it in range(Nt):
        ksp_dense_flat[A.A[it].idx, :, it] = ksp_gathered[:, :, it]
    ksp_np = ksp_dense.cpu().numpy()
    smaps_np = smaps.permute(1, 2, 3, 0).contiguous().cpu().numpy()

    fn_ksp = tmp_path / "ksp2.h5"
    fn_smaps = tmp_path / "smaps2.h5"
    with h5py.File(fn_ksp, "w") as f:
        f.create_dataset("ksp_epi_zf", data=ksp_np)
    with h5py.File(fn_smaps, "w") as f:
        f.create_dataset("smaps", data=smaps_np)

    sigma1A = 1.0  # full-rank random mask here is close to full sampling; over-estimate is safe
    result = run_recon(
        fn_ksp=str(fn_ksp),
        fn_smaps=str(fn_smaps),
        patch_sizes=[(1, 1, 1)],
        strides=[(1, 1, 1)],
        niters=300,
        sigma1A=sigma1A,
        device=DEVICE,
        mom="pogm",
        conv_tol=0.0,
        lambda_global=0.0,
    )

    rel_err = (result.X_recon - x_true).norm().item() / x_true.norm().item()
    assert rel_err < 0.05


def test_estimate_noise_std_recovers_known_scale():
    """A large bright 'object' region embedded in a much bigger 'background'
    of known-std noise -- estimate_noise_std's bottom-bg_frac magnitude
    selection should recover that known std, not be pulled toward the
    object's much larger scale."""
    known_std = 37.5
    X = known_std * _complex_randn(24, 24, 24, seed=99)
    X[8:16, 8:16, 8:16] += 5000.0  # object: far above the noise floor, small fraction of volume

    measured = estimate_noise_std(X, bg_frac=0.5)
    assert abs(measured - known_std) / known_std < 0.15


def test_run_recon_normalize_noise_round_trip_preserves_recovery(tmp_path):
    """normalize_noise=True internally rescales ksp/X0 by 1/estimate_noise_std
    and undoes it on the returned X/X_recon -- with real-scale (not O(1))
    synthetic data, recovery should still match the same ground truth
    within the same tolerance as the unregularized-recovery test above,
    confirming the scale round-trip doesn't introduce a bias.

    x_true is a smooth, low-frequency-concentrated object (not white noise)
    at real-scanner-unit-like amplitude, with a *separate*, realistically-
    scaled (~unit-variance-ish) complex Gaussian noise term added directly
    to k-space -- mirroring real whitened data, where signal concentrates
    at low spatial frequency and noise is present at every frequency
    including the outer shell run_recon's guardrail (2026-09-21) inspects.
    A white-noise x_true (this test's original design) has no such
    separation -- its own "signal" already looks like O(scale) noise
    everywhere, including the outer shell, which made the new guardrail
    correctly refuse it; that guardrail firing on real 2_6x_2.4mm data
    with a genuine ~70-800x deviation is exactly what it's for, so the fix
    here is a more realistic synthetic setup, not a weaker guardrail.

    Ny,Nz are deliberately larger than this file's other tiny synthetic
    grids (not e.g. 10x6): estimate_kspace_noise_std's default outer_frac
    (0.1) needs enough (ky,kz) samples in that outer shell for a stable
    std estimate -- an earlier 10x6 attempt gave only 8 outer samples,
    producing a wildly unstable (not signal-contaminated) measurement that
    tripped the new guardrail spuriously.

    smaps must ALSO be smooth (not this file's usual white-noise
    _complex_randn maps): a smooth image times a white-noise sensitivity
    profile is itself white in k-space, which defeats the whole "signal
    concentrates at low k, outer shell is noise-only" premise this test
    relies on -- measured directly during debugging: the real operator's
    A.apply(x_true) alone (zero injected noise) produced an outer-shell
    estimate_kspace_noise_std reading of ~294, versus ~277 with real noise
    added on top, proving the reading was ~100% signal leakage through
    white-noise smaps, not noise. Real coil sensitivities are smooth, so
    four fixed, well-separated smooth Gaussian bumps (not random per-voxel
    values) reproduces that -- well-separated so SENSE stays well
    conditioned at this test's R~1.5 (four bumps at random close-together
    offsets can come out nearly collinear and blow up the *recovery*
    assertion below instead)."""
    Nx, Ny, Nz, Nc, Nt = 8, 30, 24, 4, 3
    torch.manual_seed(43)

    xx, yy, zz = torch.meshgrid(
        torch.linspace(-1, 1, Nx, device=DEVICE),
        torch.linspace(-1, 1, Ny, device=DEVICE),
        torch.linspace(-1, 1, Nz, device=DEVICE),
        indexing="ij",
    )
    coil_centers = [(0.0, 1.2, 0.0), (0.0, -1.2, 0.0), (0.0, 0.0, 1.2), (0.0, 0.0, -1.2)]
    smaps = torch.stack([
        torch.exp(-1.5 * ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)).to(torch.complex64)
        * torch.exp(1j * (cx * xx + cy * yy + cz * zz))
        for cx, cy, cz in coil_centers
    ], dim=0).to(DEVICE)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)

    blob = torch.exp(-4 * (xx**2 + yy**2 + zz**2))  # smooth, low-k-concentrated
    x_true = 5000.0 * blob.unsqueeze(-1).expand(Nx, Ny, Nz, Nt).to(torch.complex64)

    # omega undersamples only (ky,kz), broadcasting fully-sampled kx across
    # every frame -- the real acquisition convention estimate_kspace_
    # noise_std's (ky,kz)-radius calculation assumes (see its docstring).
    # A per-voxel-random omega (this test's earlier design) breaks that
    # assumption and made the outer-shell selection meaningless -- not
    # smaller-than-expected due to sample count, but structurally wrong.
    omega_yz = torch.stack(
        [torch.rand(Ny, Nz, device=DEVICE) > 0.3 for _ in range(Nt)], dim=-1
    )
    counts = omega_yz.sum(dim=(0, 1))
    k = counts.min().item()
    omega_yz = omega_yz & (torch.cumsum(omega_yz.reshape(-1, Nt), dim=0) <= k).reshape(Ny, Nz, Nt)
    omega = omega_yz.unsqueeze(0).expand(Nx, -1, -1, -1)

    A = build_encoding_operator(smaps, omega)
    true_noise_std = 1.5
    K = A.A[0].idx.numel()  # k (per-(ky,kz)) * Nx, once kx is fully sampled
    ksp_gathered = A.apply(x_true) + true_noise_std * _complex_randn(K, Nc, Nt, seed=22)
    ksp_dense = torch.zeros(Nx, Ny, Nz, Nc, Nt, dtype=torch.complex64, device=DEVICE)
    ksp_dense_flat = ksp_dense.reshape(-1, Nc, Nt)
    for it in range(Nt):
        ksp_dense_flat[A.A[it].idx, :, it] = ksp_gathered[:, :, it]

    # Sanity-check the synthetic setup itself before handing it to run_recon:
    # the outer-shell estimator should recover true_noise_std to within a
    # modest tolerance now that smaps is smooth. This is the exact check
    # that would have caught the white-noise-smaps signal-leakage bug this
    # test's docstring describes (it silently passed the old [0.1, 10]
    # guardrail band at ~277 without it).
    kspace_std_check = estimate_kspace_noise_std(ksp_gathered[:, :, 0], A.A[0].idx, (Nx, Ny, Nz))
    assert abs(kspace_std_check - true_noise_std) / true_noise_std < 0.3, (
        f"synthetic setup's own outer-shell noise estimate ({kspace_std_check:.4f}) "
        f"strayed too far from the injected true_noise_std ({true_noise_std}) -- "
        "likely signal leaking into the outer (ky,kz) shell again"
    )

    ksp_np = ksp_dense.cpu().numpy()
    smaps_np = smaps.permute(1, 2, 3, 0).contiguous().cpu().numpy()

    fn_ksp = tmp_path / "ksp3.h5"
    fn_smaps = tmp_path / "smaps3.h5"
    with h5py.File(fn_ksp, "w") as f:
        f.create_dataset("ksp_epi_zf", data=ksp_np)
    with h5py.File(fn_smaps, "w") as f:
        f.create_dataset("smaps", data=smaps_np)

    result = run_recon(
        fn_ksp=str(fn_ksp),
        fn_smaps=str(fn_smaps),
        patch_sizes=[(1, 1, 1)],
        strides=[(1, 1, 1)],
        niters=300,
        sigma1A=1.0,
        device=DEVICE,
        mom="pogm",
        conv_tol=0.0,
        lambda_global=0.0,
        normalize_noise=True,
    )

    rel_err = (result.X_recon - x_true).norm().item() / x_true.norm().item()
    assert rel_err < 0.15  # a modicum of real noise is now present, unlike the noiseless case above
    assert result.meta["normalize_noise"] is True
    assert result.meta["noise_std"] > 1.0  # real-scanner-unit-like scale, not already ~1


def test_estimate_noise_std_ignores_exact_zero_masked_background():
    """Regression test for a real 2026-09-19 crash: process_smaps masks
    every voxel outside the coil-sensitivity support to *exactly* zero
    (recon/operators.py's GatheredSense combine then forces A^H(ksp) to be
    exactly zero there too, in every frame -- not small-magnitude thermal
    noise, an exact zero by construction). On real ball-phantom data ~57%
    of voxels are this kind of exact zero, comfortably exceeding
    bg_frac=0.25's default -- estimate_noise_std's lowest-bg_frac selection
    landed entirely on these masked (not noisy) voxels and returned a hard
    0.0, which normalize_noise (at the time) divided ksp by, propagating
    inf/NaN through an entire real ~2.5 GPU-hour run before it surfaced as
    an unrelated-looking SVD failure deep in patchSVST. A fully random/
    dense smaps (this function's original test) cannot reproduce this at
    all -- a masked object support is required.

    This only checks "doesn't silently return 0" -- estimate_noise_std's
    own accuracy on real masked data turned out to still be unreliable
    (varies ~3x with bg_frac depending on how much real signal the
    lowest-bg_frac selection catches, since ESPIRiT's support mask leaves
    little genuine noise-only-but-nonzero region to sample), which is why
    run_recon's normalize_noise no longer calls this function at all --
    see estimate_operator_noise_factor/estimate_kspace_noise_std instead.
    estimate_noise_std is kept (fixed, tested) for whatever other use may
    still want a rough background estimate, not as the normalization path."""
    known_std = 200.0
    X = torch.zeros(30, 30, 30, dtype=torch.complex64, device=DEVICE)
    # ~65% of the volume exactly zero (masked background) -- exceeds the
    # default bg_frac=0.25, so an unguarded lowest-bg_frac selection would
    # land entirely inside this region.
    object_mask = torch.zeros(30, 30, 30, dtype=torch.bool, device=DEVICE)
    object_mask[9:20, 9:20, 9:20] = True
    noise = known_std * _complex_randn(30, 30, 30, seed=101)
    X[object_mask] = 5000.0 + noise[object_mask]  # signal + noise, inside support
    X[~object_mask] = 0.0  # exactly zero, not noise -- outside support

    measured = estimate_noise_std(X, bg_frac=0.25)
    assert measured > 0, "must not silently return 0 for a masked-background operator"
    assert math.isfinite(measured)


def test_estimate_noise_std_raises_on_fully_degenerate_input():
    """Defense in depth: if the nonzero region is too small (or the
    background selection is somehow still all-zero) this must raise
    loudly, not silently return 0.0 for normalize_noise to divide by."""
    X = torch.zeros(10, 10, 10, dtype=torch.complex64, device=DEVICE)
    with pytest.raises(ValueError):
        estimate_noise_std(X)


def test_estimate_operator_noise_factor_is_stable_and_matches_unitary_case():
    """The operator's own noise-propagation factor (run_recon's actual
    normalize_noise mechanism, replacing the unreliable image-domain
    background estimate above) should be (a) reproducible across
    independent random draws, and (b) exactly 1.0 for a fully-sampled,
    RSS-normalized-smaps operator, which is unitary by construction (see
    tests/test_recon_operators.py's own near-unity spectral-norm check --
    the same invariant, just probed via noise propagation instead of power
    iteration)."""
    Nx, Ny, Nz, Nc = 10, 10, 6, 4
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=200)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    full_mask = torch.ones(Nx, Ny, Nz, 1, dtype=torch.bool, device=DEVICE)
    A = build_encoding_operator(smaps, full_mask)

    factors = [
        estimate_operator_noise_factor(A, (Nx * Ny * Nz, Nc, 1), torch.complex64, DEVICE, n_trials=10)
        for _ in range(3)
    ]
    assert max(factors) - min(factors) < 0.02, f"unstable across draws: {factors}"
    assert abs(factors[0] - 1.0) < 0.05


def test_estimate_kspace_noise_std_recovers_known_scale_even_with_masked_smaps():
    """Unlike estimate_noise_std (image-domain background, broken by
    ESPIRiT's exact-zero masking -- see the tests above), this measures
    directly in k-space, where there's no masking at all: every gathered
    k-space sample is a real acquired value. Build a scenario with a
    *masked* smaps (mimicking the real ~50-60% exact-zero case) to confirm
    that masking -- which broke the image-domain approach -- has no effect
    here."""
    Nx, Ny, Nz, Nc = 12, 12, 8, 4
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=201)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    smaps[:, :4, :4, :4] = 0  # mask out a chunk, like a real ESPIRiT support crop
    full_mask = torch.ones(Nx, Ny, Nz, 1, dtype=torch.bool, device=DEVICE)
    A = build_encoding_operator(smaps, full_mask)

    known_std = 500.0
    x_true = _complex_randn(Nx, Ny, Nz, 1, seed=202)
    ksp = A.apply(x_true)  # (K,Nc,1), K = Nx*Ny*Nz here (fully sampled)
    kspace_noise = known_std * _complex_randn(*ksp.shape, seed=203)
    ksp_noisy = ksp + kspace_noise

    measured = estimate_kspace_noise_std(ksp_noisy[:, :, 0], A.A[0].idx, (Nx, Ny, Nz), outer_frac=0.3)
    assert abs(measured - known_std) / known_std < 0.15
