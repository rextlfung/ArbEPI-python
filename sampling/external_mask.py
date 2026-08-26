"""Load a precomputed 2D (ky, kz) or 3D (ky, kz, t) sampling mask from an
externally-provided .mat file, as an alternative to generating one via
gen_sampling_masks.py's caipi/ticaipi/pd/rand methods -- e.g. a mask
designed by an outside collaborator's own pipeline, or one derived from a
non-Cartesian/compressed-sensing sample-selection method this repo doesn't
itself implement.

Such a file is typically produced by an outside collaborator's own MATLAB
pipeline as a plain v5 .mat file -- unlike this repo's own
hdf5storage-written v7.3 `.mat` output (scan_info.mat), it must be read
with `scipy.io.loadmat`, not `hdf5storage.loadmat` (which cannot read v5 at
all).
"""

import numpy as np
import scipy.io as sio


def load_external_mask(path: str, Ny: int, Nz: int, key: str = 'samp') -> np.ndarray:
    """
    Parameters
    ----------
    path : path to a v5 .mat file holding a 0/1 array under `key`, either
        (Ny, Nz) -- a single static sampling pattern, reused by the caller
        across frames as needed -- or (Ny, Nz, Nt) -- an already
        time-resolved ky-kz(-t) mask, one pattern per frame/timepoint.
    Ny, Nz : expected mask shape (first two axes), checked against the
        loaded array so a geometry mismatch fails loudly instead of
        silently mis-registering ky/kz.
    key : variable name inside the .mat file.

    Returns
    -------
    mask : (Ny, Nz) or (Ny, Nz, Nt) boolean array, matching whatever
        dimensionality was stored in the file -- the caller decides how to
        turn this into an omegas array (e.g. `generate_arbepi`'s expected
        (Ny, Nz, Nframes) shape: pass a (Ny, Nz, Nt) mask through directly,
        or broadcast a (Ny, Nz) mask across frames).
    """
    data = sio.loadmat(path)
    if key not in data:
        found = sorted(k for k in data if not k.startswith('__'))
        raise KeyError(f'{path!r} has no variable {key!r}; found {found}')
    mask = np.asarray(data[key]).astype(bool)
    if mask.ndim not in (2, 3) or mask.shape[:2] != (Ny, Nz):
        raise ValueError(
            f'{path!r}[{key!r}] has shape {mask.shape}, expected '
            f'({Ny}, {Nz}) or ({Ny}, {Nz}, Nt)'
        )
    return mask
