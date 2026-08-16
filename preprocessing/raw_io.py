"""Raw ScanArchive reading via GE's Orchestra Python SDK (GERecon).

Ports orc_read.m. This is the one module in preprocessing/ allowed to import
GERecon -- see CLAUDE.md for why: GE's raw k-space payload is an opaque
proprietary byte blob inside the HDF5 container (confirmed via `h5dump -H` --
H5T_STD_U8LE, no structured type info), and even the MRI community's own
ge_to_ismrmrd converter still links Orchestra's own Boost/HDF5 libraries to
decode it. GERecon is not pip-installable and its compiled extension is not
committed to this repo -- see the `preprocessing` extras group in
pyproject.toml for the required separate Python 3.10 / numpy<2.0 install.

GERecon.Archive exposes no reliable "how many shots does this file have"
property (Metadata()'s 'controlCount'/'frames' fields were checked against
real data and don't reliably match the number of successful NextFrame()
calls). Instead, mirror orc_read.m's own error-tolerant design: call
NextFrame() until the archive reports it's exhausted, rather than trusting a
count field up front. Verified empirically against real project data
(~/github/data/20251212ball/sa/cal.h5): exhaustion raises
`RuntimeError('...No next frame available in the archive.')`.
"""

import numpy as np
from GERecon import Archive

_EXHAUSTED_MARKER = 'No next frame available'


class ArchiveReader:
    """Thin wrapper around GERecon.Archive with a Python-iterator interface.

    Mirrors MATLAB's `archive = GERecon('Archive.Load', fn)` +
    `GERecon('Archive.Next', archive)` pattern, but raises StopIteration
    (rather than a bare RuntimeError) once the archive is exhausted, so
    callers can use the normal iterator protocol -- `for shot in
    ArchiveReader(fn): ...` -- or pull frames one at a time via
    `next_frame()` inside a frame-by-frame loop (see preprocess.py's EPI
    reader, which needs to bound memory by not holding a whole large
    acquisition at once, same reason preprocess.m calls
    GERecon('Archive.Next', ...) directly there instead of via orc_read.m).
    """

    def __init__(self, filename: str):
        self._archive = Archive(filename)

    def metadata(self) -> dict:
        return self._archive.Metadata()

    def next_frame(self) -> np.ndarray:
        """[Nfid, Ncoils] complex64 for the next shot.

        Raises StopIteration once the archive is exhausted. Any other
        RuntimeError from the SDK (a real read failure, not end-of-archive)
        propagates unchanged -- unlike orc_read.m, this does not swallow
        genuine errors with a printed warning and a hole in the output.
        """
        try:
            return self._archive.NextFrame()
        except RuntimeError as e:
            if _EXHAUSTED_MARKER in str(e):
                raise StopIteration from e
            raise

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_frame()


def read_archive(filename: str) -> np.ndarray:
    """Read every shot from a ScanArchive into [Nfid, Ncoils, Nacq] complex64.

    Equivalent to orc_read.m: ksp = orc_read(fn). Used for noise/GRE/cal
    scans, all small enough to load in full. The large EPI acquisition is
    instead streamed frame-by-frame via ArchiveReader directly (see
    preprocess.py).
    """
    shots = list(ArchiveReader(filename))
    if not shots:
        raise RuntimeError(f'read_archive: no frames found in {filename!r}')
    # Each shot is [Nfid, Ncoils]; stack along a new trailing axis to match
    # orc_read.m's [Nfid, Ncoils, Nacq].
    return np.stack(shots, axis=-1)
