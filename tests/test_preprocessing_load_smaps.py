"""load_smaps: cache read/write, backfill, and the uncompressed-coil
(true per-physical-coil) deGRE-grid smaps set added alongside the
existing Nvcoils-compressed one.
"""

import h5py
import numpy as np
import pytest

pytest.importorskip("sigpy")
pytest.importorskip("nibabel")

from preprocessing.config import PreprocessingConfig, SeqParams, SeqPaths  # noqa: E402
from preprocessing.smaps import load_smaps  # noqa: E402


def _gaussian_coil_sens(n, centers, sigma):
    xs, ys, zs = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    return np.stack(
        [
            np.exp(-((xs - c[0]) ** 2 + (ys - c[1]) ** 2 + (zs - c[2]) ** 2) / (2 * sigma**2))
            for c in centers
        ],
        axis=-1,
    ).astype(complex)


def _synthetic_gre_ksp(n, ncoils, seed=0):
    """A small 3D GRE k-space volume with a centered object and
    Gaussian-blob coil sensitivities, in the same style as
    test_preprocessing_smaps.py's shape/support checks -- real enough for
    ESPIRiT to fit a nontrivial calibration, small enough to run fast."""
    rng = np.random.default_rng(seed)
    centers = [(n * f[0], n * f[1], n * f[2]) for f in
               [(0.2, 0.2, 0.5), (0.2, 0.8, 0.5), (0.8, 0.2, 0.5), (0.8, 0.8, 0.5)][:ncoils]]
    while len(centers) < ncoils:
        centers.append((rng.uniform(0, n), rng.uniform(0, n), rng.uniform(0, n)))
    sens = _gaussian_coil_sens(n, centers, sigma=n / 2)

    obj = np.zeros((n, n, n), dtype=complex)
    lo, hi = n // 4, 3 * n // 4
    obj[lo:hi, lo:hi, lo:hi] = 1.0

    img = obj[:, :, :, None] * sens
    axes = (0, 1, 2)
    ksp = np.fft.ifftshift(np.fft.fftn(np.fft.ifftshift(img, axes=axes), axes=axes), axes=axes)
    return ksp.astype(np.complex64)


def _cfg_paths_params(tmp_path, seqname, n, fov, n_epi=None, fov_epi=None):
    """n/fov are the deGRE grid; n_epi/fov_epi (defaulting to n/fov, i.e.
    deGRE grid == EPI grid) let a test distinguish which grid a given
    output actually landed on."""
    n_epi = n if n_epi is None else n_epi
    fov_epi = fov if fov_epi is None else fov_epi
    cfg = PreprocessingConfig(datdir=str(tmp_path), seqnames=[seqname], fn_gre='', crop=0.8)
    paths = SeqPaths(
        seqname=seqname, seqdir=str(tmp_path / 'seqs' / seqname),
        scan_info='', cal='', noise='', epi='',
        recon=str(tmp_path / 'recon' / f'{seqname}_epi_zf.h5'),
    )
    seq_params = SeqParams(
        Nx=n_epi, Ny=n_epi, Nz=n_epi, ETL=1, R=1, fov=fov_epi,
        volume_tr=1.0, discard_duration=0.0,
        Nx_degre=n, Ny_degre=n, Nz_degre=n, fov_degre=fov,
        n_echoes_degre=1, TE_degre=None,
    )
    return cfg, paths, seq_params


def _write_gre_cache(tmp_path, seqname, ksp_gre, ksp_gre_uncompressed=None):
    recon_dir = tmp_path / 'recon'
    recon_dir.mkdir(exist_ok=True)
    fn_gre = recon_dir / f'{seqname}_gre.h5'
    with h5py.File(fn_gre, 'w') as f:
        f.create_dataset('ksp_gre', data=ksp_gre)
        if ksp_gre_uncompressed is not None:
            f.create_dataset('ksp_gre_uncompressed', data=ksp_gre_uncompressed)
    return fn_gre


def test_load_smaps_fresh_estimation_includes_uncompressed_smaps(tmp_path):
    seqname = 'testseq'
    n, fov = 16, (0.1, 0.1, 0.1)
    nvcoils, ncoils = 4, 8

    ksp_gre = _synthetic_gre_ksp(n, nvcoils, seed=0)
    ksp_gre_uncompressed = _synthetic_gre_ksp(n, ncoils, seed=1)
    _write_gre_cache(tmp_path, seqname, ksp_gre, ksp_gre_uncompressed)

    cfg, paths, seq_params = _cfg_paths_params(tmp_path, seqname, n, fov)

    smaps, smaps_degre, emap_degre, nvcoils_out, smaps_degre_unc = load_smaps(
        cfg, paths, seq_params
    )

    assert nvcoils_out == nvcoils
    assert smaps.shape == (n, n, n, nvcoils)
    assert smaps_degre.shape == (n, n, n, nvcoils)
    assert smaps_degre_unc is not None
    assert smaps_degre_unc.shape == (n, n, n, ncoils)
    assert ncoils != nvcoils  # the whole point: a genuinely different coil count

    fn_smaps = tmp_path / 'recon' / f'smaps_{seqname}_sigpy.h5'
    with h5py.File(fn_smaps, 'r') as f:
        # smaps_raw_uncompressed/emap_uncompressed are at ESPIRiT's own
        # calibration resolution (estimate_smaps' cal_size, default 24),
        # not the deGRE grid -- same as the pre-existing smaps_raw/emap
        # datasets. Only smaps_degre_uncompressed is on the deGRE grid.
        assert f['smaps_raw_uncompressed'].shape[-1] == ncoils
        assert f['smaps_raw_uncompressed'].shape[:3] == f['emap_uncompressed'].shape
        assert f['smaps_degre_uncompressed'].shape == (n, n, n, ncoils)


def test_load_smaps_backfills_uncompressed_smaps_into_existing_cache(tmp_path):
    seqname = 'testseq'
    n, fov = 16, (0.1, 0.1, 0.1)
    nvcoils, ncoils = 3, 6

    ksp_gre = _synthetic_gre_ksp(n, nvcoils, seed=2)
    ksp_gre_uncompressed = _synthetic_gre_ksp(n, ncoils, seed=3)
    _write_gre_cache(tmp_path, seqname, ksp_gre, ksp_gre_uncompressed)

    cfg, paths, seq_params = _cfg_paths_params(tmp_path, seqname, n, fov)

    # Write a pre-existing cache in the *old* format: smaps_raw/emap/smaps/
    # smaps_degre/emap_degre already present, but no uncompressed set at
    # all -- exactly what a cache written before this feature looks like.
    smaps_raw = np.ones((n, n, n, nvcoils), dtype=complex)
    emap = np.ones((n, n, n))
    fn_smaps = tmp_path / 'recon' / f'smaps_{seqname}_sigpy.h5'
    with h5py.File(fn_smaps, 'w') as f:
        f.create_dataset('smaps_raw', data=smaps_raw)
        f.create_dataset('emap', data=emap)
        f.create_dataset('smaps', data=smaps_raw)
        f.create_dataset('smaps_degre', data=smaps_raw)
        f.create_dataset('emap_degre', data=emap)
        f.attrs['Nvcoils'] = nvcoils

    smaps, smaps_degre, emap_degre, nvcoils_out, smaps_degre_unc = load_smaps(
        cfg, paths, seq_params
    )

    assert nvcoils_out == nvcoils
    assert smaps_degre_unc is not None
    assert smaps_degre_unc.shape == (n, n, n, ncoils)

    with h5py.File(fn_smaps, 'r') as f:
        assert 'smaps_degre_uncompressed' in f
        assert f['smaps_degre_uncompressed'].shape == (n, n, n, ncoils)


def test_load_smaps_uncompressed_is_none_when_gre_cache_predates_it(tmp_path):
    seqname = 'testseq'
    n, fov = 16, (0.1, 0.1, 0.1)
    nvcoils = 4

    ksp_gre = _synthetic_gre_ksp(n, nvcoils, seed=4)
    # Legacy-format GRE cache: no 'ksp_gre_uncompressed' dataset at all.
    _write_gre_cache(tmp_path, seqname, ksp_gre, ksp_gre_uncompressed=None)

    cfg, paths, seq_params = _cfg_paths_params(tmp_path, seqname, n, fov)

    smaps, smaps_degre, emap_degre, nvcoils_out, smaps_degre_unc = load_smaps(
        cfg, paths, seq_params
    )

    assert nvcoils_out == nvcoils
    assert smaps_degre_unc is None

    fn_smaps = tmp_path / 'recon' / f'smaps_{seqname}_sigpy.h5'
    with h5py.File(fn_smaps, 'r') as f:
        assert 'smaps_degre_uncompressed' not in f


def test_load_smaps_uncompressed_backfill_is_none_without_gre_cache_at_all(tmp_path):
    seqname = 'testseq'
    n, fov = 16, (0.1, 0.1, 0.1)
    nvcoils = 3

    cfg, paths, seq_params = _cfg_paths_params(tmp_path, seqname, n, fov)

    # Existing smaps cache (old format), but the *_gre.h5 it would need to
    # backfill from has since been cleaned up / never existed on this path.
    smaps_raw = np.ones((n, n, n, nvcoils), dtype=complex)
    emap = np.ones((n, n, n))
    fn_smaps = tmp_path / 'recon' / f'smaps_{seqname}_sigpy.h5'
    (tmp_path / 'recon').mkdir(exist_ok=True)
    with h5py.File(fn_smaps, 'w') as f:
        f.create_dataset('smaps_raw', data=smaps_raw)
        f.create_dataset('emap', data=emap)
        f.create_dataset('smaps', data=smaps_raw)
        f.create_dataset('smaps_degre', data=smaps_raw)
        f.create_dataset('emap_degre', data=emap)
        f.attrs['Nvcoils'] = nvcoils

    smaps, smaps_degre, emap_degre, nvcoils_out, smaps_degre_unc = load_smaps(
        cfg, paths, seq_params
    )

    assert nvcoils_out == nvcoils
    assert smaps_degre_unc is None


def test_load_smaps_uncompressed_lands_on_degre_grid_not_epi_grid(tmp_path):
    # Distinguishes the deGRE-grid output from the EPI-grid output by
    # giving them different shapes/FOVs -- the task this whole feature
    # exists for ("at the deGRE resolution and FOV"), which a
    # deGRE-grid-equals-EPI-grid fixture (every other test here) can't
    # actually tell apart from an accidental fov/n_target swap.
    seqname = 'testseq'
    n_degre, fov_degre = 16, (0.1, 0.1, 0.12)
    n_epi, fov_epi = 12, (0.1, 0.1, 0.08)  # smaller z-FOV -> a real crop
    nvcoils, ncoils = 4, 6

    ksp_gre = _synthetic_gre_ksp(n_degre, nvcoils, seed=5)
    ksp_gre_uncompressed = _synthetic_gre_ksp(n_degre, ncoils, seed=6)
    _write_gre_cache(tmp_path, seqname, ksp_gre, ksp_gre_uncompressed)

    cfg, paths, seq_params = _cfg_paths_params(
        tmp_path, seqname, n_degre, fov_degre, n_epi=n_epi, fov_epi=fov_epi,
    )

    smaps, smaps_degre, emap_degre, nvcoils_out, smaps_degre_unc = load_smaps(
        cfg, paths, seq_params
    )

    assert smaps.shape[:3] == (n_epi, n_epi, n_epi)  # EPI grid
    assert smaps_degre.shape[:3] == (n_degre, n_degre, n_degre)  # deGRE grid
    assert smaps_degre_unc.shape[:3] == (n_degre, n_degre, n_degre)  # deGRE grid
    assert smaps_degre_unc.shape[-1] == ncoils


def test_load_smaps_repairs_stale_ncoils_in_cached_uncompressed_set(tmp_path):
    # A cached smaps_degre_uncompressed with a *different* Ncoils than the
    # current GRE cache's ksp_gre_uncompressed (e.g. a re-run archive, or a
    # different physical coil array) must be recomputed, not trusted --
    # Nvcoils matching (checked separately) says nothing about Ncoils.
    seqname = 'testseq'
    n, fov = 16, (0.1, 0.1, 0.1)
    nvcoils, ncoils_old, ncoils_new = 4, 5, 7

    ksp_gre = _synthetic_gre_ksp(n, nvcoils, seed=7)
    ksp_gre_uncompressed = _synthetic_gre_ksp(n, ncoils_new, seed=8)
    _write_gre_cache(tmp_path, seqname, ksp_gre, ksp_gre_uncompressed)

    cfg, paths, seq_params = _cfg_paths_params(tmp_path, seqname, n, fov)

    smaps_raw = np.ones((n, n, n, nvcoils), dtype=complex)
    emap = np.ones((n, n, n))
    stale_unc = np.ones((n, n, n, ncoils_old), dtype=complex)
    fn_smaps = tmp_path / 'recon' / f'smaps_{seqname}_sigpy.h5'
    with h5py.File(fn_smaps, 'w') as f:
        f.create_dataset('smaps_raw', data=smaps_raw)
        f.create_dataset('emap', data=emap)
        f.create_dataset('smaps', data=smaps_raw)
        f.create_dataset('smaps_degre', data=smaps_raw)
        f.create_dataset('emap_degre', data=emap)
        f.create_dataset('smaps_raw_uncompressed', data=stale_unc)
        f.create_dataset('emap_uncompressed', data=emap)
        f.create_dataset('smaps_degre_uncompressed', data=stale_unc)
        f.attrs['Nvcoils'] = nvcoils
        f.attrs['Ncoils'] = ncoils_old

    smaps, smaps_degre, emap_degre, nvcoils_out, smaps_degre_unc = load_smaps(
        cfg, paths, seq_params
    )

    assert smaps_degre_unc is not None
    assert smaps_degre_unc.shape[-1] == ncoils_new

    with h5py.File(fn_smaps, 'r') as f:
        assert int(f.attrs['Ncoils']) == ncoils_new
        assert f['smaps_degre_uncompressed'].shape[-1] == ncoils_new
