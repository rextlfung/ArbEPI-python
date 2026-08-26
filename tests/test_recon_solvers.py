"""Convergence sanity checks for recon/solvers.py's pogm_restart, on a
trivial least-squares problem (A=I) with a known closed-form solution,
across all three momentum variants."""

import pytest

torch = pytest.importorskip("torch")

from recon.solvers import pogm_restart  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("mom", ["pgm", "fpgm", "pogm"])
def test_pogm_restart_converges_on_identity_least_squares(mom):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    n = 50
    real = torch.randn(n, generator=g, device=DEVICE)
    imag = torch.randn(n, generator=g, device=DEVICE)
    b = (real + 1j * imag).to(torch.complex64)
    x0 = torch.zeros(n, dtype=torch.complex64, device=DEVICE)

    x, out = pogm_restart(
        x0,
        lambda x: 0.5 * (x - b).norm().item() ** 2,
        lambda x: x - b,
        1.0,
        mom=mom,
        niter=200,
        conv_tol=1e-8,
        fun=lambda it, xk, yk, is_restart, fcostnew, rel_change: fcostnew,
    )
    rel_err = (x - b).norm().item() / b.norm().item()
    assert rel_err < 1e-4
    assert out[-1] < out[0]


def test_pogm_restart_niter_zero_returns_initial_point():
    x0 = torch.zeros(5, dtype=torch.complex64, device=DEVICE)
    x, out = pogm_restart(x0, lambda x: 0.0, lambda x: x, 1.0, niter=0)
    assert torch.equal(x, x0)
    assert len(out) == 1


def test_pogm_restart_rejects_invalid_mom():
    x0 = torch.zeros(5, dtype=torch.complex64, device=DEVICE)
    with pytest.raises(ValueError):
        pogm_restart(x0, lambda x: 0.0, lambda x: x, 1.0, mom="bogus")
