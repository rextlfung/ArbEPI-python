"""Write final reconstructed image volumes as NIfTI, for viewing in
ITK-SNAP/FSLeyes/3D Slicer/etc. instead of scrolling through raw HDF5 in
Python (see run_rss.py/run_recon_sigpy.py, the only
callers). This is a Python-viewing convenience only -- it does not replace
any of the .h5 files the pipeline reads/writes for its own use
(ksp_epi_zf, smaps, the GRE cache): those stay plain h5py (already
generic HDF5, readable from Julia via HDF5.jl) since they feed the
downstream Julia advanced-recon pipeline, which has no NIfTI reader in
this stack.

NIfTI has no complex dtype and no free-form attribute dict, so this saves
two files per recon: `<fn_base>.nii.gz` (magnitude-only, float32) and
`<fn_base>.json` (a sidecar with the recon parameters that used to be
HDF5 attrs -- seqname, num_iter, lamb_l1, runtime_s, etc.). This is the
common BIDS-style image+sidecar convention, not a repo-specific format.
"""

import json

import nibabel as nib
import numpy as np


def save_recon_nifti(fn_base: str, img: np.ndarray, **attrs) -> None:
    """img: [Nx, Ny, Nz, Nframes] (real or complex; complex is saved as
    magnitude, since NIfTI cannot store complex voxels). attrs: JSON-
    serializable recon metadata (seq_params fields plus each driver's own
    reconstruction parameters), written to `<fn_base>.json`.

    No patient-orientation information exists anywhere in this pipeline
    (unlike a scanner-produced DICOM/NIfTI), so the affine here is a plain
    diagonal voxel-size matrix -- correct spacing, but not necessarily
    radiological left/right or anterior/posterior orientation.
    """
    if np.iscomplexobj(img):
        img = np.abs(img)

    fov = attrs['fov']  # m
    voxel_size_mm = [1000.0 * fov[axis] / img.shape[axis] for axis in range(3)]
    affine = np.diag(voxel_size_mm + [1.0])

    nib.save(nib.Nifti1Image(img.astype(np.float32), affine), f'{fn_base}.nii.gz')
    with open(f'{fn_base}.json', 'w') as f:
        json.dump(attrs, f, indent=2)
