"""Shared helper for reading hdf5storage-written v7.3 .mat files via h5py.

hdf5storage stores arrays axis-reversed on disk (MATLAB's column-major
convention); h5py reads the raw on-disk layout, so anything with more than
one non-singleton axis needs a full transpose to recover the logical shape
hdf5storage.loadmat would give. Verified empirically against a real
output/samp_locs.mat: h5py's raw read of `schedules` comes back as
(2, 60, 20, 30), while the true logical shape (matching hdf5storage.loadmat,
and this repo's own documented Nframes x Nshots x ETL x 2 layout) is
(30, 20, 60, 2) -- exactly `raw.transpose()`. For vectors/scalars this is a
no-op on the values (only a singleton axis moves), so applying it
unconditionally, as read_mat_array does, is always correct.

Use this for every hdf5storage-written .mat this repo reads (never
scipy.io, per CLAUDE.md; never a bare h5py read without the transpose).
This is deliberately not used for anything preprocess.py itself writes
(recon output, GRE/smaps caches) -- those are plain numpy-order h5py with
no MATLAB consumer, see config.py's SeqPaths.recon docstring.
"""

import h5py
import numpy as np


def read_mat_array(f: h5py.File, name: str) -> np.ndarray:
    return f[name][()].transpose()


def read_mat(path: str, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read one or more top-level datasets from a hdf5storage-written .mat
    file. Reads every top-level dataset if `names` is not given."""
    with h5py.File(path, 'r') as f:
        keys = names if names is not None else list(f.keys())
        return {k: read_mat_array(f, k) for k in keys}
