"""End-to-end test of preprocessing/julia/b0map.jl via preprocessing/run_b0map.py.

Requires a `julia` executable on PATH with preprocessing/julia/'s
Project.toml/Manifest.toml already instantiated (`julia
--project=preprocessing/julia -e 'import Pkg; Pkg.instantiate()'`) -- skipped
entirely otherwise, the same tolerance this repo already extends to
MATLAB-based comparisons (see CLAUDE.md) and to raw_io.py's GERecon
dependency.
"""

import os
import shutil
import subprocess

import h5py
import nibabel as nib
import numpy as np
import pytest

from preprocessing.config import load_config
from preprocessing.run_b0map import run_b0map

pytestmark = pytest.mark.skipif(
    shutil.which('julia') is None, reason='julia executable not found on PATH'
)


def _write_synthetic_params_mat(seqdir, Nx, Ny, Nz, fov_degre):
    """Minimal params.mat -- just the fields load_seq_params reads -- in the
    same on-disk convention hdf5storage.savemat produces (scalars/vectors as
    plain arrays; load_seq_params's .item()/.ravel() tolerate any shape), so
    run_b0map's NIfTI export (which needs seq_params.fov_degre) has a real
    file to read rather than crashing on a fixture gap.
    """
    os.makedirs(seqdir, exist_ok=True)
    with h5py.File(os.path.join(seqdir, 'params.mat'), 'w') as f:
        f.create_dataset('Nx', data=Nx)
        f.create_dataset('Ny', data=Ny)
        f.create_dataset('Nz', data=Nz)
        f.create_dataset('ETL', data=1)
        f.create_dataset('R', data=1.0)
        f.create_dataset('fov', data=np.array([0.2, 0.2, 0.1]))
        f.create_dataset('volume_tr', data=1.0)
        f.create_dataset('discard_duration', data=0.0)
        f.create_dataset('Nx_degre', data=Nx)
        f.create_dataset('Ny_degre', data=Ny)
        f.create_dataset('Nz_degre', data=Nz)
        f.create_dataset('fov_degre', data=np.array(fov_degre))


def _fft3c(x: np.ndarray) -> np.ndarray:
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(x, axes=axes), axes=axes), axes=axes)


def _write_synthetic_gre_cache(path, rng, Nx=16, Ny=16, Nz=8, n_echoes=2, Ncoils=2):
    """Mirrors preprocess.py's `<seqname>_gre.h5` layout ('ksp_gre_echoes'
    dataset + 'TE_degre' attr, both plain numpy-order h5py -- see that
    module's STEP 2). Ground truth is a spatial ramp field map (Hz) baked
    into the inter-echo phase, recovered below via the real b0map.jl
    script and compared against this known value."""
    TE = np.array([0.0015, 0.0035])
    xx, yy, zz = np.meshgrid(
        np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny), np.linspace(-1, 1, Nz), indexing='ij'
    )
    mask = np.sqrt(xx**2 + yy**2 + zz**2) < 0.8
    f0_true = 60.0 * xx  # Hz

    coil_sens = (rng.normal(size=(Nx, Ny, Nz, Ncoils))
                 + 1j * rng.normal(size=(Nx, Ny, Nz, Ncoils))).astype(np.complex64)
    ksp = np.zeros((Nx, Ny, Nz, n_echoes, Ncoils), dtype=np.complex64)
    for e in range(n_echoes):
        base = mask.astype(np.complex64) * np.exp(1j * 2 * np.pi * f0_true * TE[e])
        for c in range(Ncoils):
            noise = 0.01 * (rng.normal(size=(Nx, Ny, Nz)) + 1j * rng.normal(size=(Nx, Ny, Nz)))
            ksp[..., e, c] = _fft3c(base * coil_sens[..., c] + noise)

    with h5py.File(path, 'w') as f:
        f.create_dataset('ksp_gre_echoes', data=ksp)
        f.attrs['TE_degre'] = TE

    return f0_true, mask


def test_run_b0map_recovers_known_field_map(tmp_path):
    rng = np.random.default_rng(0)
    seqname = 'synthtest'
    recon_dir = tmp_path / 'recon'
    recon_dir.mkdir()
    f0_true, mask = _write_synthetic_gre_cache(recon_dir / f'{seqname}_gre.h5', rng)
    fov_degre = (0.2, 0.2, 0.1)
    _write_synthetic_params_mat(
        tmp_path / 'seqs' / seqname, *f0_true.shape, fov_degre,
    )

    cfg = load_config(datdir=str(tmp_path), seqnames=[seqname])
    run_b0map(cfg)

    out_path = recon_dir / f'{seqname}_b0map.h5'
    assert out_path.exists()
    with h5py.File(out_path, 'r') as f:
        fhat = f['b0map_hz'][()]
        out_mask = f['mask'][()].astype(bool)
        assert f.attrs['mask_threshold'] == pytest.approx(0.1)

    # NIfTI export, on the deGRE grid's fov (not the EPI fov)
    nii = nib.load(recon_dir / f'{seqname}_b0map.nii.gz')
    np.testing.assert_allclose(nii.get_fdata(), fhat, atol=1e-3)
    expected_voxel_mm = [1000.0 * fov_degre[axis] / fhat.shape[axis] for axis in range(3)]
    np.testing.assert_allclose(np.diag(nii.affine)[:3], expected_voxel_mm)

    # (Nx, Ny, Nz) in numpy axis order, not the reversed HDF5.jl-native
    # layout -- this is the axis-order round trip this test is really for.
    assert fhat.shape == f0_true.shape

    diff = fhat[mask] - f0_true[mask]
    rmse = np.sqrt(np.mean(diff**2))
    assert rmse < 5.0  # true field map spans +-48 Hz within mask
    assert np.corrcoef(fhat[mask], f0_true[mask])[0, 1] > 0.98
    # the script's own magnitude mask should cover almost all of the true
    # (geometric) support -- boundary voxels can disagree by SNR alone
    assert out_mask[mask].mean() > 0.95


def _write_aliased_synthetic_gre_cache(path, rng, Nx=48, Ny=48, Nz=24, n_echoes=2, Ncoils=2):
    """Same idea as `_write_synthetic_gre_cache`, but with a field map (+-
    ~390 Hz) that exceeds the naive two-point `finit`'s +-1/(2 dTE) = +-250
    Hz unambiguous range for this dTE -- this is the case ROMEO unwrapping
    exists to fix (see b0map.jl's module docstring). Uses a smooth Gaussian
    magnitude taper rather than a hard-edged mask: a coarse discretized
    sphere boundary breaks ROMEO's local-smoothness assumption (adjacent
    voxels straddling the mask edge), which isn't representative of a real
    (smoothly coil-weighted) GRE image -- found empirically while
    validating this test."""
    TE = np.array([0.0015, 0.0035])
    xx, yy, zz = np.meshgrid(
        np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny), np.linspace(-1, 1, Nz), indexing='ij'
    )
    amp = np.exp(-3 * (xx**2 + yy**2 + zz**2))
    f0_true = 450.0 * xx  # Hz

    coil_sens = (rng.normal(size=(Nx, Ny, Nz, Ncoils))
                 + 1j * rng.normal(size=(Nx, Ny, Nz, Ncoils))).astype(np.complex64)
    ksp = np.zeros((Nx, Ny, Nz, n_echoes, Ncoils), dtype=np.complex64)
    for e in range(n_echoes):
        base = amp.astype(np.complex64) * np.exp(1j * 2 * np.pi * f0_true * TE[e])
        for c in range(Ncoils):
            noise = 0.01 * (rng.normal(size=(Nx, Ny, Nz)) + 1j * rng.normal(size=(Nx, Ny, Nz)))
            ksp[..., e, c] = _fft3c(base * coil_sens[..., c] + noise)

    with h5py.File(path, 'w') as f:
        f.create_dataset('ksp_gre_echoes', data=ksp)
        f.attrs['TE_degre'] = TE

    return f0_true, amp


def test_run_b0map_unwraps_a_field_map_beyond_the_naive_unambiguous_range(tmp_path):
    rng = np.random.default_rng(0)
    seqname = 'aliastest'
    recon_dir = tmp_path / 'recon'
    recon_dir.mkdir()
    f0_true, _amp = _write_aliased_synthetic_gre_cache(recon_dir / f'{seqname}_gre.h5', rng)
    _write_synthetic_params_mat(
        tmp_path / 'seqs' / seqname, *f0_true.shape, (0.2, 0.2, 0.1),
    )

    cfg = load_config(datdir=str(tmp_path), seqnames=[seqname])
    run_b0map(cfg)

    # Use the script's own mask (from the noisy reconstructed SOS), not a
    # ground-truth amplitude threshold recomputed here -- they can disagree
    # at individual voxels (random per-voxel coil sensitivities in this
    # synthetic model, not a smooth profile), and evaluating fhat outside
    # the mask MRIFieldmaps actually fit against picks up `embed()`'s
    # unfit-region fill value instead of a real field estimate.
    with h5py.File(recon_dir / f'{seqname}_b0map.h5', 'r') as f:
        fhat = f['b0map_hz'][()]
        finit = f['finit_hz'][()]
        mask = f['mask'][()].astype(bool)

    # the true field spans well beyond the naive two-point method's
    # +-1/(2*2ms) = +-250 Hz range -- confirm ROMEO's finit actually
    # extends past that cap (this is what would regress to ~250 Hz if
    # someone reverted b0map.jl to MRIFieldmaps' own wrapped default finit)
    assert np.abs(finit[mask]).max() > 300

    rmse = np.sqrt(np.mean((fhat[mask] - f0_true[mask]) ** 2))
    assert rmse < 60.0  # naive (no unwrap) finit gives ~207 Hz RMSE on this data
    assert np.corrcoef(fhat[mask], f0_true[mask])[0, 1] > 0.95


def test_run_b0map_reports_missing_gre_cache(tmp_path, capsys):
    cfg = load_config(datdir=str(tmp_path), seqnames=['nope'])
    run_b0map(cfg)  # shouldn't raise -- prints an error and continues, like the other batch drivers
    assert 'not found' in capsys.readouterr().out


def test_b0map_jl_errors_without_te_degre_attr(tmp_path):
    gre_path = tmp_path / 'no_te_gre.h5'
    with h5py.File(gre_path, 'w') as f:
        f.create_dataset('ksp_gre_echoes', data=np.zeros((4, 4, 4, 2, 1), dtype=np.complex64))

    julia_dir = os.path.join(os.path.dirname(__file__), '..', 'preprocessing', 'julia')
    script = os.path.join(julia_dir, 'b0map.jl')
    result = subprocess.run(
        ['julia', f'--project={julia_dir}', script, str(gre_path), str(tmp_path / 'out.h5')],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'TE_degre' in result.stderr
