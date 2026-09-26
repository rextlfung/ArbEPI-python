"""End-to-end tests for recon/sense.py's run_sense: simulates a
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

from recon.operators import build_sense  # noqa: E402
from recon.sense import run_sense  # noqa: E402
from recon.utils import estimate_operator_noise_factor, load_omega  # noqa: E402

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
    # Guarantee every frame samples the same count (run_sense asserts this),
    # and every frame includes at least one sample.
    counts = omega.sum(dim=(0, 1, 2))
    k = counts.min().item()
    assert k > 0
    omega = omega & (torch.cumsum(omega.reshape(-1, Nt), dim=0) <= k).reshape(Nx, Ny, Nz, Nt)

    A = build_sense(smaps, omega)
    ksp_gathered = A.apply(x_true)  # (K,Nc,Nt)

    # Real ksp_epi_zf.h5 files (this repo's own preprocessing/ output, or
    # mslr-recon's sigpy-export path) are dense zero-filled arrays, not
    # gathered -- scatter back using each frame's own operator indices so
    # run_sense's internal gather_ksp() recovers exactly ksp_gathered.
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
    load_omega trusts 'omegas' rather than being fooled by the exact
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

    omega = load_omega(str(fn_ksp), Nx, Ny, Nz, Nt, DEVICE)
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
    omega = load_omega(str(fn_ksp), Nx, Ny, Nz, Nt, DEVICE)
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

    result = run_sense(
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

    A = build_sense(smaps, omega)
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
    result = run_sense(
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


def test_run_recon_normalize_noise_round_trip_preserves_recovery(tmp_path):
    """normalize_noise=True internally rescales ksp/X0 by
    1/estimate_operator_noise_factor(A) and undoes it on the returned
    X/X_recon -- this factor is a property of the encoding operator alone
    (see run_sense's normalize_noise docstring for why it's no longer
    measured from the k-space data itself, 2026-09-22), so the round trip
    should recover the same ground truth regardless of the data's own
    scale or injected noise level, confirming un-normalizing doesn't
    introduce a bias."""
    Nx, Ny, Nz, Nc, Nt = 10, 10, 6, 4, 4
    torch.manual_seed(42)
    smaps = _complex_randn(Nc, Nx, Ny, Nz, seed=10)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    x_true = 5000.0 * _complex_randn(Nx, Ny, Nz, Nt, seed=11)  # real-scanner-unit-like amplitude
    omega = torch.stack(
        [torch.rand(Nx, Ny, Nz, device=DEVICE) > 0.3 for _ in range(Nt)], dim=-1
    )
    counts = omega.sum(dim=(0, 1, 2))
    k = counts.min().item()
    omega = omega & (torch.cumsum(omega.reshape(-1, Nt), dim=0) <= k).reshape(Nx, Ny, Nz, Nt)

    A = build_sense(smaps, omega)
    K = A.A[0].idx.numel()
    true_noise_std = 1.5
    ksp_gathered = A.apply(x_true) + true_noise_std * _complex_randn(K, Nc, Nt, seed=22)
    ksp_dense = torch.zeros(Nx, Ny, Nz, Nc, Nt, dtype=torch.complex64, device=DEVICE)
    ksp_dense_flat = ksp_dense.reshape(-1, Nc, Nt)
    for it in range(Nt):
        ksp_dense_flat[A.A[it].idx, :, it] = ksp_gathered[:, :, it]
    ksp_np = ksp_dense.cpu().numpy()
    smaps_np = smaps.permute(1, 2, 3, 0).contiguous().cpu().numpy()

    fn_ksp = tmp_path / "ksp3.h5"
    fn_smaps = tmp_path / "smaps3.h5"
    with h5py.File(fn_ksp, "w") as f:
        f.create_dataset("ksp_epi_zf", data=ksp_np)
    with h5py.File(fn_smaps, "w") as f:
        f.create_dataset("smaps", data=smaps_np)

    result = run_sense(
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
    assert math.isfinite(result.meta["noise_std"]) and result.meta["noise_std"] > 0


def test_estimate_operator_noise_factor_is_stable_and_matches_unitary_case():
    """The operator's own noise-propagation factor (run_sense's actual
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
    A = build_sense(smaps, full_mask)

    factors = [
        estimate_operator_noise_factor(A, (Nx * Ny * Nz, Nc, 1), torch.complex64, DEVICE, n_trials=10)
        for _ in range(3)
    ]
    assert max(factors) - min(factors) < 0.02, f"unstable across draws: {factors}"
    assert abs(factors[0] - 1.0) < 0.05


def _write_dataset(tmp_path, smaps, omega, x_true, tag):
    """Dense zero-filled ksp_epi_zf.h5 + smaps.h5 for a known image, in the
    layout preprocessing/ writes. smaps (Nc,Nx,Ny,Nz) must be RSS-normalized
    (run_sense normalizes whatever it loads)."""
    Nc, Nx, Ny, Nz = smaps.shape
    Nt = omega.shape[-1]
    A = build_sense(smaps, omega)
    ksp_g = A.apply(x_true)
    dense = torch.zeros(Nx * Ny * Nz, Nc, Nt, dtype=torch.complex64, device=DEVICE)
    for it in range(Nt):
        dense[A.A[it].idx, :, it] = ksp_g[:, :, it]
    fn_ksp, fn_smaps = tmp_path / f"{tag}_epi_zf.h5", tmp_path / f"{tag}_smaps.h5"
    with h5py.File(fn_ksp, "w") as f:
        f.create_dataset("ksp_epi_zf", data=dense.reshape(Nx, Ny, Nz, Nc, Nt).cpu().numpy())
    with h5py.File(fn_smaps, "w") as f:
        f.create_dataset("smaps", data=smaps.permute(1, 2, 3, 0).contiguous().cpu().numpy())
    return str(fn_ksp), str(fn_smaps)


def _phantom_setup(tmp_path, undersample=False, Nt=1):
    """Piecewise-constant block phantom, 4 smooth Gaussian coils."""
    nx, ny, nz = 16, 16, 4
    xs, ys = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing="ij")
    sens = torch.stack([torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 10**2))
                        for cx, cy in [(3, 3), (3, 13), (13, 3), (13, 13)]])
    sens = sens[..., None].expand(-1, -1, -1, nz).to(torch.complex64).to(DEVICE)
    sens = sens / sens.abs().pow(2).sum(0, keepdim=True).sqrt()
    omega = torch.ones(nx, ny, nz, Nt, dtype=torch.bool, device=DEVICE)
    if undersample:
        omega[:, 1::2] = False
    x_true = torch.zeros(nx, ny, nz, Nt, dtype=torch.complex64, device=DEVICE)
    x_true[4:12, 4:12] = 1.0
    fn_ksp, fn_smaps = _write_dataset(tmp_path, sens, omega, x_true, "phantom")
    return x_true, fn_ksp, fn_smaps


def test_run_sense_wavelet_tv_converges_to_truth_as_lambda_shrinks_fully_sampled(tmp_path):
    x_true, fn_ksp, fn_smaps = _phantom_setup(tmp_path)
    errors = []
    for lamda in (1e-2, 1e-3, 1e-5):
        r = run_sense(fn_ksp=fn_ksp, fn_smaps=fn_smaps, reg="wavelet-tv", lamb_l1=lamda,
                      lamb_tv=lamda, niters=150, device=DEVICE)
        errors.append((r.X_recon - x_true).abs().max().item())
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-2


def test_run_sense_wavelet_tv_handles_undersampled_multiframe_data(tmp_path):
    x_true, fn_ksp, fn_smaps = _phantom_setup(tmp_path, undersample=True, Nt=2)
    r = run_sense(fn_ksp=fn_ksp, fn_smaps=fn_smaps, reg="wavelet-tv", lamb_l1=1e-4, lamb_tv=1e-4,
                  niters=50, device=DEVICE, frames=[1])
    assert r.X_recon.shape == (16, 16, 4, 1)
    assert torch.isfinite(r.X_recon.abs()).all()


def test_run_sense_none_runs_cg_and_recovers_fully_sampled_image(tmp_path):
    x_true, fn_ksp, fn_smaps = _phantom_setup(tmp_path, Nt=2)
    r = run_sense(fn_ksp=fn_ksp, fn_smaps=fn_smaps, reg="none", niters=30, device=DEVICE)
    assert r.dc_costs[0] == 1.0 and r.dc_costs[-1] < 1e-5
    torch.testing.assert_close(r.X_recon, x_true, atol=1e-3, rtol=1e-3)


def test_run_sense_rejects_r2star_without_b0(tmp_path):
    _, fn_ksp, fn_smaps = _phantom_setup(tmp_path)
    with pytest.raises(ValueError, match="requires fn_b0map"):
        run_sense(fn_ksp=fn_ksp, fn_smaps=fn_smaps, reg="none", device=DEVICE,
                  r2star_map=torch.zeros(16, 16, 4, device=DEVICE))
