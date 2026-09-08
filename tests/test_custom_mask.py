"""Unit tests for sampling/external_mask.py's resolve_custom_omegas -- the
validation/broadcast logic behind params.py's custom_mask_path option (see
README.md's "Using custom ky-kz-t sampling masks" section). These write
small v5 .mat fixtures directly (scipy.io.savemat) rather than building a
full sequence, matching this repo's "unit tests on algorithm invariants"
testing philosophy (see CLAUDE.md's Commands section).
"""

import numpy as np
import pytest
import scipy.io as sio

from sampling.external_mask import resolve_custom_omegas

Ny, Nz, ETL = 8, 6, 4


def _write_mask(path, mask, key='samp'):
    sio.savemat(str(path), {key: mask.astype(np.uint8)})


def _flat_mask(n_samples, rng):
    """A single (Ny, Nz) frame with exactly n_samples True entries."""
    flat = np.zeros(Ny * Nz, dtype=bool)
    flat[rng.choice(Ny * Nz, size=n_samples, replace=False)] = True
    return flat.reshape(Ny, Nz)


def test_2d_mask_is_broadcast_across_frames(tmp_path):
    rng = np.random.default_rng(0)
    mask = _flat_mask(n_samples=8, rng=rng)  # divisible by ETL=4 -> Nshots=2
    path = tmp_path / 'mask.mat'
    _write_mask(path, mask)

    omegas, Nshots, R = resolve_custom_omegas(str(path), Ny, Nz, Nframes=5, ETL=ETL)

    assert omegas.shape == (Ny, Nz, 5)
    for f in range(5):
        np.testing.assert_array_equal(omegas[:, :, f], mask)
    assert Nshots == 2
    assert R == pytest.approx(Ny * Nz / 8)


def test_3d_mask_passes_through_and_sets_Nframes(tmp_path):
    rng = np.random.default_rng(1)
    n_samples = 8
    mask3d = np.stack([_flat_mask(n_samples, rng) for _ in range(3)], axis=-1)
    path = tmp_path / 'mask3d.mat'
    _write_mask(path, mask3d)

    omegas, Nshots, R = resolve_custom_omegas(str(path), Ny, Nz, Nframes=3, ETL=ETL)

    np.testing.assert_array_equal(omegas, mask3d)
    assert Nshots == n_samples // ETL


def test_custom_mask_key_is_respected(tmp_path):
    rng = np.random.default_rng(2)
    mask = _flat_mask(n_samples=4, rng=rng)
    path = tmp_path / 'mask_customkey.mat'
    _write_mask(path, mask, key='my_mask')

    omegas, Nshots, R = resolve_custom_omegas(
        str(path), Ny, Nz, Nframes=1, ETL=ETL, key='my_mask'
    )
    assert Nshots == 1


def test_3d_mask_frame_count_mismatch_raises(tmp_path):
    rng = np.random.default_rng(3)
    n_samples = 8
    mask3d = np.stack([_flat_mask(n_samples, rng) for _ in range(3)], axis=-1)
    path = tmp_path / 'mask_badframes.mat'
    _write_mask(path, mask3d)

    with pytest.raises(ValueError, match='time frames'):
        resolve_custom_omegas(str(path), Ny, Nz, Nframes=5, ETL=ETL)


def test_uneven_per_frame_sample_counts_raises(tmp_path):
    rng = np.random.default_rng(4)
    frame0 = _flat_mask(8, rng)
    frame1 = _flat_mask(12, rng)
    mask3d = np.stack([frame0, frame1], axis=-1)
    path = tmp_path / 'mask_uneven.mat'
    _write_mask(path, mask3d)

    with pytest.raises(ValueError, match='same number'):
        resolve_custom_omegas(str(path), Ny, Nz, Nframes=2, ETL=ETL)


def test_sample_count_not_divisible_by_etl_raises(tmp_path):
    rng = np.random.default_rng(5)
    mask = _flat_mask(n_samples=9, rng=rng)  # not divisible by ETL=4
    path = tmp_path / 'mask_bad_etl.mat'
    _write_mask(path, mask)

    with pytest.raises(ValueError, match='divisible by ETL'):
        resolve_custom_omegas(str(path), Ny, Nz, Nframes=1, ETL=ETL)
