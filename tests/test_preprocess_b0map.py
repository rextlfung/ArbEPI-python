"""preprocess/b0map.py and julia/b0map.jl, on synthetic dual-echo GRE data.

The julia tests need a `julia` on PATH with preprocess/julia instantiated
(`julia --project=preprocess/julia -e 'import Pkg; Pkg.instantiate()'`) and
are skipped otherwise.
"""

import os
import shutil
import subprocess

import h5py
import numpy as np
import pytest

pytest.importorskip('scipy')

from preprocess.b0map import (  # noqa: E402
    JULIA_DIR,
    JULIA_SCRIPT,
    estimate_b0map,
    fit_mask,
    resize_to_epi,
)

needs_julia = pytest.mark.skipif(shutil.which('julia') is None, reason='julia not on PATH')


def _fft3c(x):
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)


def _synthetic_echoes(rng, f0_true, amp, te, ncoils=2):
    """[Nx, Ny, Nz, n_echoes, Nc] k-space of `amp` with field map f0_true (Hz)."""
    shape = f0_true.shape
    coil_sens = rng.normal(size=shape + (ncoils,)) + 1j * rng.normal(size=shape + (ncoils,))
    ksp = np.zeros(shape + (len(te), ncoils), dtype=np.complex64)
    for e, t in enumerate(te):
        base = amp * np.exp(1j * 2 * np.pi * f0_true * t)
        for c in range(ncoils):
            noise = 0.01 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
            ksp[..., e, c] = _fft3c(base * coil_sens[..., c] + noise)
    return ksp


def _grid(n):
    return np.meshgrid(*(np.linspace(-1, 1, k) for k in n), indexing='ij')


@needs_julia
def test_estimate_b0map_recovers_known_field_map():
    rng = np.random.default_rng(0)
    xx, yy, zz = _grid((16, 16, 8))
    mask = np.sqrt(xx**2 + yy**2 + zz**2) < 0.8
    f0_true = 60.0 * xx
    te = np.array([0.0015, 0.0035])
    r = estimate_b0map(_synthetic_echoes(rng, f0_true, mask.astype(float), te), te)

    # numpy axis order (the HDF5.jl reversal round trip is what this checks)
    assert r['b0_map'].shape == f0_true.shape
    diff = r['b0_map'][mask] - f0_true[mask]
    assert np.sqrt(np.mean(diff**2)) < 5.0  # the true map spans +-48 Hz in the mask
    assert np.corrcoef(r['b0_map'][mask], f0_true[mask])[0, 1] > 0.98
    assert r['mask'][mask].mean() > 0.95


@needs_julia
def test_estimate_b0map_unwraps_beyond_the_naive_unambiguous_range():
    """+-450 Hz exceeds a two-point estimate's +-1/(2 dTE) = +-250 Hz; ROMEO's
    unwrapped initialization is what gets b0map to the right 2pi branch
    (a plain wrapped start gives ~207 Hz RMSE here). Smooth Gaussian magnitude:
    a hard-edged mask breaks ROMEO's smoothness assumption unrealistically."""
    rng = np.random.default_rng(0)
    xx, yy, zz = _grid((48, 48, 24))
    amp = np.exp(-3 * (xx**2 + yy**2 + zz**2))
    f0_true = 450.0 * xx
    te = np.array([0.0015, 0.0035])
    r = estimate_b0map(_synthetic_echoes(rng, f0_true, amp, te), te)
    m = r['mask']
    assert np.abs(r['finit_hz'][m]).max() > 300
    assert np.sqrt(np.mean((r['b0_map'][m] - f0_true[m]) ** 2)) < 60.0
    assert np.corrcoef(r['b0_map'][m], f0_true[m])[0, 1] > 0.95


@needs_julia
def test_estimate_b0map_uses_smaps_and_their_eigenvalue_mask():
    rng = np.random.default_rng(2)
    xx, yy, zz = _grid((16, 16, 8))
    mask = np.sqrt(xx**2 + yy**2 + zz**2) < 0.8
    f0_true = 40.0 * yy
    te = np.array([0.0015, 0.0035])
    ksp = _synthetic_echoes(rng, f0_true, mask.astype(float), te, ncoils=3)
    smaps = np.ones((16, 16, 8, 3), dtype=np.complex64) / np.sqrt(3)
    emap = np.where(xx > 0, 1.0, 0.0)  # excludes half the volume
    r = estimate_b0map(ksp, te, smaps, emap, crop=0.5)
    assert not r['mask'][xx <= 0].any()


def test_resize_to_epi_moves_field_map_and_mask_onto_the_epi_grid():
    xx, _, _ = _grid((16, 16, 8))
    b0 = 60.0 * xx
    mask = np.abs(xx) < 0.7
    n_epi = (32, 32, 16)
    b0_epi, m_epi = resize_to_epi(b0, mask, (0.2, 0.2, 0.1), (0.2, 0.2, 0.08), n_epi)
    assert b0_epi.shape == n_epi and m_epi.shape == n_epi
    assert m_epi.dtype == bool
    assert np.abs(b0_epi).max() <= 1.5 * np.abs(b0 * mask).max() + 1.0


def test_fit_mask_combines_magnitude_and_eigenvalue_masks():
    img = np.array([[[1.0, 0.05, 0.5]]])
    emap = np.array([[[0.99, 0.99, 0.5]]])
    np.testing.assert_array_equal(fit_mask(img, 0.1), [[[True, False, True]]])
    np.testing.assert_array_equal(fit_mask(img, 0.1, emap, 0.95), [[[True, False, False]]])


@needs_julia
def test_b0map_jl_errors_without_te_degre_attr(tmp_path):
    gre_path = tmp_path / 'no_te_gre.h5'
    with h5py.File(gre_path, 'w') as f:
        f.create_dataset('ksp_gre_echoes', data=np.zeros((4, 4, 4, 2, 1), dtype=np.complex64))
    result = subprocess.run(
        ['julia', f'--project={JULIA_DIR}', JULIA_SCRIPT, str(gre_path), str(tmp_path / 'o.h5')],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'TE_degre' in result.stderr
    assert os.path.basename(JULIA_SCRIPT) == 'b0map.jl'
