"""Helpers shared by the torch-based recon tests (import only after
pytest.importorskip("torch"))."""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def complex_randn(*shape, seed):
    """Seeded standard complex-normal tensor (complex64) on DEVICE."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    real = torch.randn(*shape, generator=g, device=DEVICE)
    imag = torch.randn(*shape, generator=g, device=DEVICE)
    return (real + 1j * imag).to(torch.complex64)
