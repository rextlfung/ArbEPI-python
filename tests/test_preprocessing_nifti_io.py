import json

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from preprocessing.nifti_io import save_recon_nifti  # noqa: E402


def test_save_recon_nifti_writes_magnitude_with_fov_derived_spacing(tmp_path):
    fn_base = str(tmp_path / 'seq_recon_rss')
    img = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(np.complex64) * (1 + 1j)
    fov = (0.2, 0.3, 0.4)  # m

    save_recon_nifti(fn_base, img, fov=fov, seqname='seq', runtime_s=1.5)

    nii = nib.load(f'{fn_base}.nii.gz')
    np.testing.assert_allclose(nii.get_fdata(), np.abs(img), atol=1e-3)

    expected_voxel_mm = [1000.0 * fov[axis] / img.shape[axis] for axis in range(3)]
    np.testing.assert_allclose(np.diag(nii.affine)[:3], expected_voxel_mm)


def test_save_recon_nifti_writes_json_sidecar_with_attrs(tmp_path):
    fn_base = str(tmp_path / 'seq_recon_rss')
    img = np.zeros((2, 2, 2, 1))

    save_recon_nifti(fn_base, img, fov=(0.1, 0.1, 0.1), seqname='seq', num_iter=50)

    with open(f'{fn_base}.json') as f:
        attrs = json.load(f)

    assert attrs['seqname'] == 'seq'
    assert attrs['num_iter'] == 50
