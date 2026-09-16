"""load_smaps: cache read/write, backfill, and the single-calibration
design -- one ESPIRiT run on the whitened, uncompressed (true
per-physical-coil) deGRE data, with the Nvcoils-compressed set derived by
linear projection through cc_matrix rather than a second, independent
calibration.
"""

import h5py
import numpy as np
import pytest

pytest.importorskip("sigpy")
pytest.importorskip("nibabel")

from preprocessing.coils import (  # noqa: E402
    apply_coil_compression,
    coil_compression_matrix,
    compute_coil_covariance,
)
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
    Gaussian-blob coil sensitivities -- real enough for ESPIRiT to fit a
    nontrivial calibration, small enough to run fast."""
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


def _make_cc_matrix(ksp_gre_uncompressed, nvcoils):
    """A realistic (not arbitrary) coil-compression matrix, the same way
    preprocess.py derives one -- PCA on the coil covariance."""
    cov = compute_coil_covariance(ksp_gre_uncompressed)
    return coil_compression_matrix(cov, nvcoils)


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


def _write_gre_cache(tmp_path, seqname, ksp_gre_uncompressed=None, cc_matrix=None, ksp_gre=None):
    recon_dir = tmp_path / 'recon'
    recon_dir.mkdir(exist_ok=True)
    fn_gre = recon_dir / f'{seqname}_gre.h5'
    with h5py.File(fn_gre, 'w') as f:
        if ksp_gre is not None:
            f.create_dataset('ksp_gre', data=ksp_gre)
        if ksp_gre_uncompressed is not None:
            f.create_dataset('ksp_gre_uncompressed', data=ksp_gre_uncompressed)
        if cc_matrix is not None:
            f.create_dataset('cc_matrix', data=cc_matrix)
    return fn_gre


def test_load_smaps_fresh_estimation_projects_compressed_from_single_calibration(tmp_path):
    seqname = 'testseq'
    n, fov = 16, (0.1, 0.1, 0.1)
    ncoils, nvcoils = 8, 4

    ksp_gre_uncompressed = _synthetic_gre_ksp(n, ncoils, seed=0)
    cc_matrix = _make_cc_matrix(ksp_gre_uncompressed, nvcoils)
    _write_gre_cache(
        tmp_path, seqname, ksp_gre_uncompressed=ksp_gre_uncompressed, cc_matrix=cc_matrix,
    )

    cfg, paths, seq_params = _cfg_paths_params(tmp_path, seqname, n, fov)

    smaps, smaps_degre, emap_degre, nvcoils_out, smaps_degre_unc = load_smaps(
        cfg, paths, seq_params
    )

    assert nvcoils_out == nvcoils
    assert smaps.shape == (n, n, n, nvcoils)
    assert smaps_degre.shape == (n, n, n, nvcoils)
    assert smaps_degre_unc is not None
    assert smaps_degre_unc.shape == (n, n, n, ncoils)

    fn_smaps = tmp_path / 'recon' / f'smaps_{seqname}_sigpy.h5'
    with h5py.File(fn_smaps, 'r') as f:
        # One shared calibration -- no independent 'emap_uncompressed'.
        assert 'emap_uncompressed' not in f
        smaps_raw = f['smaps_raw'][()]
        smaps_raw_unc = f['smaps_raw_uncompressed'][()]
        assert int(f.attrs['Ncoils']) == ncoils
        # The core claim of the swap: the compressed calibration-
        # resolution raw maps are an *exact* linear projection of the
        # uncompressed ones through cc_matrix, not independently
        # re-estimated -- verify numerically, not just check shapes.
        # rtol/atol sized for complex64 matmul error (sigpy's EspiritCalib
        # returns complex64; float32 summation over Ncoils terms is only
        # accurate to ~1e-7, not 1e-10), not tightened further.
        np.testing.assert_allclose(
            smaps_raw, apply_coil_compression(smaps_raw_unc, cc_matrix), rtol=1e-5, atol=1e-6,
        )


def test_load_smaps_recomputes_everything_via_projection_when_gre_cache_gains_support(tmp_path):
    # An "old-design" smaps cache (independently-estimated compressed set,
    # Nvcoils matches, but no 'Ncoils' attr at all -- written before this
    # gre cache had cc_matrix/ksp_gre_uncompressed to project from). Since
    # both coil counts now derive from one shared calibration, there's no
    # way to partially reuse the compressed set alone -- a missing/stale
    # Ncoils must invalidate the whole cache and trigger a full recompute.
    # This is a deliberate one-time opportunistic-upgrade cost (see
    # load_smaps' staleness-check comment), not a bug: it happens exactly
    # once per acquisition, the first load after its GRE cache gains
    # projection support.
    seqname = 'testseq'
    n, fov = 16, (0.1, 0.1, 0.1)
    ncoils, nvcoils = 6, 3

    ksp_gre_uncompressed = _synthetic_gre_ksp(n, ncoils, seed=1)
    cc_matrix = _make_cc_matrix(ksp_gre_uncompressed, nvcoils)
    _write_gre_cache(
        tmp_path, seqname, ksp_gre_uncompressed=ksp_gre_uncompressed, cc_matrix=cc_matrix,
    )

    cfg, paths, seq_params = _cfg_paths_params(tmp_path, seqname, n, fov)

    placeholder = np.ones((n, n, n, nvcoils), dtype=complex)
    emap_placeholder = np.ones((n, n, n))
    fn_smaps = tmp_path / 'recon' / f'smaps_{seqname}_sigpy.h5'
    (tmp_path / 'recon').mkdir(exist_ok=True)
    with h5py.File(fn_smaps, 'w') as f:
        f.create_dataset('smaps_raw', data=placeholder)
        f.create_dataset('emap', data=emap_placeholder)
        f.create_dataset('smaps', data=placeholder)
        f.create_dataset('smaps_degre', data=placeholder)
        f.create_dataset('emap_degre', data=emap_placeholder)
        f.attrs['Nvcoils'] = nvcoils

    smaps, smaps_degre, emap_degre, nvcoils_out, smaps_degre_unc = load_smaps(
        cfg, paths, seq_params
    )

    assert nvcoils_out == nvcoils
    assert smaps_degre_unc is not None
    assert smaps_degre_unc.shape == (n, n, n, ncoils)
    # A real ESPIRiT-derived, RSS-normalized map is never all-ones --
    # confirms the placeholder was actually recomputed, not reused.
    assert not np.allclose(smaps, placeholder)

    with h5py.File(fn_smaps, 'r') as f:
        assert int(f.attrs['Ncoils']) == ncoils


def test_load_smaps_falls_back_to_independent_calibration_without_cc_matrix(tmp_path):
    # Legacy <seqname>_gre.h5 (only 'ksp_gre', no cc_matrix/
    # ksp_gre_uncompressed -- a pre-this-feature preprocess() run):
    # load_smaps must still produce the compressed set via the original
    # design (an independent calibration directly on ksp_gre), and
    # gracefully return None for the uncompressed set.
    seqname = 'testseq'
    n, fov = 16, (0.1, 0.1, 0.1)
    nvcoils = 4

    ksp_gre = _synthetic_gre_ksp(n, nvcoils, seed=2)
    _write_gre_cache(tmp_path, seqname, ksp_gre=ksp_gre)

    cfg, paths, seq_params = _cfg_paths_params(tmp_path, seqname, n, fov)

    smaps, smaps_degre, emap_degre, nvcoils_out, smaps_degre_unc = load_smaps(
        cfg, paths, seq_params
    )

    assert nvcoils_out == nvcoils
    assert smaps.shape == (n, n, n, nvcoils)
    assert smaps_degre_unc is None

    fn_smaps = tmp_path / 'recon' / f'smaps_{seqname}_sigpy.h5'
    with h5py.File(fn_smaps, 'r') as f:
        assert 'smaps_degre_uncompressed' not in f
        assert 'Ncoils' not in f.attrs


def test_load_smaps_uncompressed_is_none_without_gre_cache_at_all(tmp_path):
    # Existing (valid) smaps cache, but the *_gre.h5 it could otherwise
    # check/project from has since been cleaned up / never existed.
    seqname = 'testseq'
    n, fov = 16, (0.1, 0.1, 0.1)
    nvcoils = 3

    cfg, paths, seq_params = _cfg_paths_params(tmp_path, seqname, n, fov)

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
    ncoils, nvcoils = 6, 4

    ksp_gre_uncompressed = _synthetic_gre_ksp(n_degre, ncoils, seed=5)
    cc_matrix = _make_cc_matrix(ksp_gre_uncompressed, nvcoils)
    _write_gre_cache(
        tmp_path, seqname, ksp_gre_uncompressed=ksp_gre_uncompressed, cc_matrix=cc_matrix,
    )

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


def test_load_smaps_repairs_stale_ncoils_via_full_recompute(tmp_path):
    # A cached smaps_degre_uncompressed with a *different* Ncoils than the
    # current GRE cache's ksp_gre_uncompressed (e.g. a re-run archive, or a
    # different physical coil array) must be recomputed, not trusted.
    seqname = 'testseq'
    n, fov = 16, (0.1, 0.1, 0.1)
    nvcoils, ncoils_old, ncoils_new = 4, 5, 7

    ksp_gre_uncompressed = _synthetic_gre_ksp(n, ncoils_new, seed=7)
    cc_matrix = _make_cc_matrix(ksp_gre_uncompressed, nvcoils)
    _write_gre_cache(
        tmp_path, seqname, ksp_gre_uncompressed=ksp_gre_uncompressed, cc_matrix=cc_matrix,
    )

    cfg, paths, seq_params = _cfg_paths_params(tmp_path, seqname, n, fov)

    placeholder_compressed = np.ones((n, n, n, nvcoils), dtype=complex)
    placeholder_unc = np.ones((n, n, n, ncoils_old), dtype=complex)
    emap_placeholder = np.ones((n, n, n))
    fn_smaps = tmp_path / 'recon' / f'smaps_{seqname}_sigpy.h5'
    (tmp_path / 'recon').mkdir(exist_ok=True)
    with h5py.File(fn_smaps, 'w') as f:
        f.create_dataset('smaps_raw', data=placeholder_compressed)
        f.create_dataset('emap', data=emap_placeholder)
        f.create_dataset('smaps', data=placeholder_compressed)
        f.create_dataset('smaps_degre', data=placeholder_compressed)
        f.create_dataset('emap_degre', data=emap_placeholder)
        f.create_dataset('smaps_raw_uncompressed', data=placeholder_unc)
        f.create_dataset('smaps_degre_uncompressed', data=placeholder_unc)
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
