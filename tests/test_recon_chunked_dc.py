"""The chunked data-consistency term (`--dc-chunk`, `solvers.ChunkedDataFidelity`) is the
whole-series product computed a few frames at a time, not a different objective."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mirtorch")

from recon.operators import build_sense  # noqa: E402
from recon.sense import run_sense  # noqa: E402
from recon.solvers import ChunkedDataFidelity  # noqa: E402
from tests._helpers import DEVICE, complex_randn  # noqa: E402
from tests.test_recon_sense import _write_synthetic_dataset  # noqa: E402


def _problem(Nx=8, Ny=8, Nz=6, Nc=3, Nt=5, R=3.0):
    smaps = complex_randn(Nc, Nx, Ny, Nz, seed=0)
    smaps = smaps / (smaps.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    omega = torch.stack(
        [torch.rand(Nx, Ny, Nz, device=DEVICE) > (1 - 1 / R) for _ in range(Nt)], dim=-1
    )
    k = omega.sum(dim=(0, 1, 2)).min().item()
    omega = omega & (torch.cumsum(omega.reshape(-1, Nt), dim=0) <= k).reshape(Nx, Ny, Nz, Nt)
    A = build_sense(smaps, omega)
    x = complex_randn(Nx, Ny, Nz, Nt, seed=1)
    y = A.apply(complex_randn(Nx, Ny, Nz, Nt, seed=2)) + 0.1 * A.apply(x)  # (K,Nc,Nt)
    return A, x, y


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 8])
def test_chunked_cost_gradient_and_adjoint_match_the_whole_series_product(chunk):
    A, x, y = _problem()
    s = 0.7
    cost = 0.5 * (s * A.apply(x) - y).norm().item() ** 2
    grad = s * A.adjoint(s * A.apply(x) - y)
    adj = s * A.adjoint(y)

    ksp = y.permute(2, 0, 1).contiguous().cpu()  # (Nt,K,Nc), host
    dc = ChunkedDataFidelity(A.A, ksp, scale=s, chunk=chunk)
    c, g = dc.cost_and_grad(x)
    assert abs(c - cost) < 1e-5 * cost
    assert (g - grad).norm() < 1e-5 * grad.norm()
    assert (dc.adjoint_data(torch.empty_like(x)) - adj).norm() < 1e-5 * adj.norm()
    assert dc.cost_and_grad(x, grad=False)[1] is None


def test_chunked_data_fidelity_rejects_bad_arguments():
    A, x, y = _problem()
    ksp = y.permute(2, 0, 1).contiguous().cpu()
    with pytest.raises(ValueError, match="chunk"):
        ChunkedDataFidelity(A.A, ksp, chunk=0)
    with pytest.raises(ValueError, match="frames"):
        ChunkedDataFidelity(A.A, ksp[:-1], chunk=2)


@pytest.mark.parametrize("hp_weight", [0.0, 0.5])
@pytest.mark.parametrize("mom", ["pgm", "fpgm"])
def test_run_sense_dc_chunk_reproduces_the_unchunked_recon(tmp_path, mom, hp_weight):
    Nx, Ny, Nz, Nc, Nt = 12, 12, 8, 4, 4
    fn_ksp, fn_smaps = _write_synthetic_dataset(tmp_path, Nx, Ny, Nz, Nc, Nt, R=3.0)
    kw = dict(fn_ksp=fn_ksp, fn_smaps=fn_smaps, patch_sizes=[(4, 4, 4)], strides=[(4, 4, 4)],
              sigma1A=1.3, device=DEVICE, mom=mom, conv_tol=0.0, lambda_global=0.05, niters=6,
              hp_weight=hp_weight, volume_tr_s=1.0, cycle_spin_seed=3)
    whole = run_sense(**kw)
    chunked = run_sense(dc_chunk=3, **kw)  # 4 frames in chunks of 3 and 1
    rel = (chunked.X_recon - whole.X_recon).norm() / whole.X_recon.norm()
    assert rel < 1e-5
    for a, b in zip(chunked.dc_costs, whole.dc_costs):
        assert abs(a - b) < 1e-4 * abs(b)
    assert chunked.meta["dc_chunk"] == 3


def test_dc_chunk_is_only_for_mslr(tmp_path):
    fn_ksp, fn_smaps = _write_synthetic_dataset(tmp_path, 8, 8, 6, 3, 2, R=3.0)
    with pytest.raises(ValueError, match="mslr"):
        run_sense(fn_ksp=fn_ksp, fn_smaps=fn_smaps, reg="wavelet-tv", niters=1, sigma1A=1.0,
                  device=DEVICE, dc_chunk=2)
