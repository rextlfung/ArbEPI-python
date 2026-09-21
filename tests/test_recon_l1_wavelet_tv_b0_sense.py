import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")
pytest.importorskip("sigpy")
pytest.importorskip("nibabel")

from recon.operators import build_encoding_operator  # noqa: E402

# The module name has a hyphen, so it can't be used in an `import` statement.
mod = importlib.import_module("recon.L1-wavelet_TV_B0_SENSE")


def _setup(nx=16, ny=16, nz=4, undersample=False):
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    centers = [(3, 3), (3, 13), (13, 3), (13, 13)]
    sens = np.stack(
        [np.exp(-((xs - c[0]) ** 2 + (ys - c[1]) ** 2) / (2 * 10**2)) for c in centers]
    )
    sens = np.tile(sens[:, :, :, None], (1, 1, 1, nz)).astype(np.complex64)  # (Nc,Nx,Ny,Nz)
    omega = np.ones((nx, ny, nz, 1), dtype=bool)
    if undersample:
        omega[:, 1::2, :, :] = False
    img_true = np.zeros((nx, ny, nz), dtype=np.complex64)
    img_true[4:12, 4:12, :] = 1.0

    A = build_encoding_operator(torch.from_numpy(sens), torch.from_numpy(omega))
    op = A.A[0]
    y = op.apply(torch.from_numpy(img_true)).numpy()  # (K, Nc)
    return img_true, mod.TorchLinopBridge(op, torch.device("cpu")), y


def test_torch_linop_bridge_adjoint_is_consistent():
    img_true, A, y = _setup()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(img_true.shape) + 1j * rng.standard_normal(img_true.shape)
    w = rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)

    lhs = np.vdot(w, A(x.astype(np.complex64)))
    rhs = np.vdot(A.H(w.astype(np.complex64)), x)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-4)


def test_wavelet_tv_recon_b0_converges_to_truth_as_lambda_shrinks_fully_sampled():
    img_true, A, y = _setup()

    errors = []
    for lamda in (1e-2, 1e-3, 1e-5):
        recon = mod.wavelet_tv_recon_b0(A, y, lamda, lamda, num_iter=150)
        errors.append(np.abs(recon - img_true).max())

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-2


def test_wavelet_tv_recon_b0_handles_undersampled_data_without_error():
    img_true, A, y = _setup(undersample=True)
    recon = mod.wavelet_tv_recon_b0(A, y, 1e-4, 1e-4, num_iter=50)
    assert recon.shape == img_true.shape
    assert np.all(np.isfinite(recon))
