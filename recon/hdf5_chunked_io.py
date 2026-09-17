"""Chunk-aware reads of this pipeline's own plain-numpy-order .h5 files
(ksp_epi_zf and friends -- chunked one frame per chunk along the last
axis; see preprocessing/preprocess.py's writer). No torch/mirtorch import
here, deliberately: recon/reconstruct.py's `_load_array` and
recon/lowres_calib_recon.py's (former) `_load_chunked` used to be two
independent copies of the same chunk-by-chunk-along-the-last-axis loop
(docs/review-findings.md item 200) specifically because
lowres_calib_recon.py runs in .venv-preprocessing (no torch) and importing
recon/reconstruct.py would pull in recon/operators.py's mirtorch/torch
dependency for no reason -- this module has neither, so both venvs can
share it.

`read_frames_cropped`'s `spatial_slices` parameter is the fix for
docs/review-findings.md item 204: one HDF5 chunk is one frame's *entire*
spatial+coil extent (Nx,Ny,Nz,Nc), so a calibration-region-only consumer
still has to decompress every frame's full chunk once -- chunk boundaries
can't be worked around -- but it does NOT have to hold every frame's full
decompressed volume in memory *simultaneously*. Cropping each frame down
to `spatial_slices` immediately after decompressing it, before moving to
the next frame, bounds peak memory to one frame's full volume instead of
every frame's: on this pipeline's real 0.8mm/R~93.5 dataset (Nx,Ny,Nz,Nc,
Nt = 270,270,180,32,60), that's the difference between ~201GB (the whole
array, previously required just to read out a few hundred calibration-
region k-space samples) and ~3.4GB (one frame) -- see that item for the
measurement.
"""

import h5py
import numpy as np


def read_frames_cropped(
    fn: str, key: str, spatial_slices: tuple[slice, slice, slice] | None = None,
) -> np.ndarray:
    """dataset shape (X, Y, Z, ..., T), chunked one frame per chunk along
    the last (T) axis -- ksp_epi_zf's own convention. spatial_slices, when
    given, is (x_slice, y_slice, z_slice) applied to the first three axes
    of every frame as it's decompressed (see module docstring for why this
    bounds peak memory to one frame instead of the whole dataset).
    spatial_slices=None reproduces the previous whole-array
    `_load_array`/`_load_chunked` behavior exactly (still frame-by-frame
    when chunked, to avoid the documented HDF5 chunk-cache pathology of a
    single `d[()]` call on a dataset this large -- a bare `d[()]` was
    measured at ~7 MB/s vs. ~500 MB/s reading one chunk at a time)."""
    with h5py.File(fn, 'r') as f:
        d = f[key]
        chunked_by_frame = d.chunks is not None and d.chunks[-1] < d.shape[-1]

        if not chunked_by_frame:
            full = np.asarray(d[()])
            return full if spatial_slices is None else full[(*spatial_slices,)]

        step = d.chunks[-1]
        if spatial_slices is None:
            out = np.empty(d.shape, dtype=d.dtype)
            for start in range(0, d.shape[-1], step):
                out[..., start : start + step] = d[..., start : start + step]
            return out

        out = None
        for start in range(0, d.shape[-1], step):
            frame = d[..., start : start + step]  # one HDF5 chunk, full spatial extent
            cropped = frame[(*spatial_slices,)]
            if out is None:
                out = np.empty(cropped.shape[:-1] + (d.shape[-1],), dtype=d.dtype)
            out[..., start : start + step] = cropped
        return out
