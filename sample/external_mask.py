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


def resolve_custom_omegas(
    path: str, Ny: int, Nz: int, Nframes: int, ETL: int, key: str = 'samp',
) -> tuple[np.ndarray, int, float]:
    """Load and validate a custom sampling mask for params.py's
    `custom_mask_path` option -- see README.md's "Using custom ky-kz-t
    sampling masks" section for the user-facing walkthrough, and
    params.py's `custom_mask_path` field comment for how this fits into
    load_params().

    A static 2D (Ny, Nz) mask is broadcast across every one of `Nframes`
    frames; a time-resolved 3D (Ny, Nz, Nt) mask must have Nt == Nframes
    (Nframes here is already computed by the caller from
    duration/volume_tr/discard_duration, which this function doesn't know
    how to adjust on its own).

    Every frame must carry the same total sample count -- mask2epi_
    {laminar,radial} partition every frame with one shared Nshots, asserted
    as `Nshots * ETL == n_samples` in lib/mask2epi.py -- and that count must
    divide evenly by ETL.

    Returns
    -------
    omegas : (Ny, Nz, Nframes) boolean sampling mask.
    Nshots : samples_per_frame // ETL.
    R : effective acceleration factor (Ny*Nz / samples_per_frame) --
        informational only (scan_info.mat/plot-title bookkeeping), not a
        design input the way it is on the gen_sampling_masks path.
    """
    mask = load_external_mask(path, Ny, Nz, key=key)
    if mask.ndim == 2:
        omegas = np.repeat(mask[:, :, None], Nframes, axis=2)
    elif mask.shape[2] != Nframes:
        raise ValueError(
            f'{path!r} has {mask.shape[2]} time frames, but '
            f'duration/volume_tr/discard_duration compute Nframes={Nframes}; '
            'adjust one to match the other (e.g. set duration = '
            f'{mask.shape[2]} * volume_tr - discard_duration).'
        )
    else:
        omegas = mask

    frame_counts = omegas.sum(axis=(0, 1))
    n_samples = int(frame_counts[0])
    if not np.all(frame_counts == n_samples):
        raise ValueError(
            f'{path!r}: every frame must sample the same number of (ky, kz) '
            'locations (mask2epi partitions every frame with one shared '
            f'Nshots); got per-frame counts {sorted(set(frame_counts.tolist()))}'
        )
    if n_samples % ETL != 0:
        raise ValueError(
            f'{path!r}: {n_samples} samples/frame is not divisible by '
            f'ETL={ETL}; adjust the mask or ETL so that Nshots = '
            'samples/ETL is an integer (see lib/mask2epi.py).'
        )
    Nshots = n_samples // ETL
    R = Ny * Nz / n_samples
    return omegas, Nshots, R
