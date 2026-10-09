"""I/O and small shared helpers for preprocess/.

- Raw GE ScanArchives: ArchiveReader, read_archive (GERecon, imported lazily)
- Simulated archives (simulate_fmri/): the same two, plus
  create_simulated_archive, write_simulated_archive, is_simulated_archive
- hdf5storage-written v7.3 .mat files: read_mat_array, read_mat
- scan_info.mat (written by sequences/ArbEPI.py): SeqParams, load_seq_params,
  load_kxoe, load_schedules, nominal_te_s
- NIfTI export: save_recon_nifti
- Numerics: matlab_round, ift3c, rss_echo_images
- QA figures: plot_gre_b0_diagnostics

Only numpy and h5py are imported at module level, so this module loads in both
.venv-preprocessing and .venv-recon. GERecon (GE's proprietary SDK, only in
.venv-preprocessing), nibabel and matplotlib are imported inside the functions
that need them.
"""

import json
import os
from dataclasses import dataclass

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Raw ScanArchives
# ---------------------------------------------------------------------------

_EXHAUSTED_MARKER = 'No next frame available'

# Simulated archives (simulate_fmri/): a plain HDF5 file holding the readouts
# in acquisition order, so preprocess() runs on a simulated session without
# GERecon. Real ScanArchives are HDF5 too, hence the explicit marker.
SIM_MARKER = 'arbepi_simulated_archive'  # file attribute
SIM_DATASET = 'readouts'  # [Nacq, Ncoils, Nfid] complex64


def is_simulated_archive(filename: str) -> bool:
    try:
        with h5py.File(filename, 'r') as f:
            return bool(f.attrs.get(SIM_MARKER, False)) and SIM_DATASET in f
    except OSError:
        return False


def create_simulated_archive(
    filename: str, n_acq: int, n_coils: int, n_fid: int, **attrs
) -> h5py.File:
    """A new simulated archive, open for writing, with an all-zero
    f['readouts'] of shape [n_acq, n_coils, n_fid] to fill in. attrs become
    file attributes (returned by ArchiveReader.metadata())."""
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    f = h5py.File(filename, 'w')
    f.attrs[SIM_MARKER] = True
    for k, v in attrs.items():
        f.attrs[k] = v
    f.create_dataset(SIM_DATASET, shape=(n_acq, n_coils, n_fid), dtype=np.complex64)
    return f


def write_simulated_archive(filename: str, readouts: np.ndarray, **attrs) -> None:
    """Write readouts [Nacq, Ncoils, Nfid] as a simulated archive."""
    with create_simulated_archive(filename, *readouts.shape, **attrs) as f:
        f[SIM_DATASET][...] = readouts


class _SimulatedArchive:
    """GERecon.Archive's two calls, for a simulated archive. Readouts are read
    in blocks, so streaming a long EPI run is one HDF5 read per `block`."""

    def __init__(self, filename: str, block: int = 4096):
        self._file = h5py.File(filename, 'r')
        self._data = self._file[SIM_DATASET]
        self._block, self._buffer, self._start, self._next = block, None, 0, 0

    def Metadata(self) -> dict:  # noqa: N802 -- GERecon's name
        return dict(self._file.attrs)

    def NextFrame(self) -> np.ndarray:  # noqa: N802
        if self._next >= self._data.shape[0]:
            raise RuntimeError(_EXHAUSTED_MARKER)
        if self._buffer is None or self._next >= self._start + len(self._buffer):
            self._start = self._next
            self._buffer = self._data[self._start : self._start + self._block]
        frame = self._buffer[self._next - self._start].T  # [Nfid, Ncoils]
        self._next += 1
        return np.ascontiguousarray(frame)


class ArchiveReader:
    """Iterator over the shots of a GE ScanArchive, via GERecon.Archive, or of
    a simulated archive (see SIM_MARKER above), which needs no GERecon.

    GERecon has no reliable "number of shots" field, so this reads until the
    archive reports it is exhausted (a RuntimeError containing
    'No next frame available'), which is turned into StopIteration. Any other
    RuntimeError from the SDK propagates.
    """

    def __init__(self, filename: str):
        if is_simulated_archive(filename):
            self._archive = _SimulatedArchive(filename)
            return
        from GERecon import Archive

        self._archive = Archive(filename)

    def metadata(self) -> dict:
        return self._archive.Metadata()

    def next_frame(self) -> np.ndarray:
        """[Nfid, Ncoils] complex64 for the next shot."""
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
    """Every shot of a ScanArchive as [Nfid, Ncoils, Nacq] complex64. For the
    small scans (noise, cal, deGRE); the EPI archive is streamed with
    ArchiveReader instead."""
    shots = list(ArchiveReader(filename))
    if not shots:
        raise RuntimeError(f'read_archive: no frames found in {filename!r}')
    return np.stack(shots, axis=-1)


# ---------------------------------------------------------------------------
# hdf5storage .mat files
# ---------------------------------------------------------------------------


def read_mat_array(f: h5py.File, name: str) -> np.ndarray:
    """hdf5storage stores arrays axis-reversed (MATLAB column-major); h5py reads
    the raw layout, so a full transpose recovers the logical shape. A no-op on
    the values of vectors and scalars."""
    return f[name][()].transpose()


def read_mat(path: str, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Top-level datasets of an hdf5storage-written .mat file (all of them if
    `names` is None)."""
    with h5py.File(path, 'r') as f:
        keys = names if names is not None else list(f.keys())
        return {k: read_mat_array(f, k) for k in keys}


# ---------------------------------------------------------------------------
# scan_info.mat
# ---------------------------------------------------------------------------


@dataclass
class SeqParams:
    """Per-acquisition scan scalars from scan_info.mat."""

    Nx: int
    Ny: int
    Nz: int
    ETL: int
    R: float
    fov: tuple[float, float, float]  # m
    volume_tr: float  # s
    discard_duration: float  # s

    Nx_degre: int
    Ny_degre: int
    Nz_degre: int
    fov_degre: tuple[float, float, float]  # m
    n_echoes_degre: int
    TE_degre: tuple[float, ...] | None  # s; None for pre-dual-echo snapshots


def load_seq_params(scan_info_path: str) -> SeqParams:
    """Read the scan-scalar snapshot sequences/ArbEPI.py writes into
    scan_info.mat. Scalars are stored as (1,1) arrays and 3-vectors as (3,1),
    hence .item()/.ravel(). Snapshots from before the dual-echo deGRE have no
    n_echoes_degre/TE_degre and were single-echo."""
    required = ['Nx', 'Ny', 'Nz', 'ETL', 'R', 'fov', 'volume_tr', 'discard_duration',
                'Nx_degre', 'Ny_degre', 'Nz_degre', 'fov_degre']
    with h5py.File(scan_info_path, 'r') as f:
        optional = [k for k in ('n_echoes_degre', 'TE_degre') if k in f]
    d = read_mat(scan_info_path, required + optional)

    def scalar(name):
        return d[name].item()

    def vec3(name):
        return tuple(d[name].ravel().tolist())

    return SeqParams(
        Nx=int(scalar('Nx')), Ny=int(scalar('Ny')), Nz=int(scalar('Nz')),
        ETL=int(scalar('ETL')), R=scalar('R'),
        fov=vec3('fov'),
        volume_tr=scalar('volume_tr'),
        discard_duration=scalar('discard_duration'),
        Nx_degre=int(scalar('Nx_degre')), Ny_degre=int(scalar('Ny_degre')),
        Nz_degre=int(scalar('Nz_degre')),
        fov_degre=vec3('fov_degre'),
        n_echoes_degre=int(scalar('n_echoes_degre')) if 'n_echoes_degre' in d else 1,
        TE_degre=tuple(d['TE_degre'].ravel().tolist()) if 'TE_degre' in d else None,
    )


def load_kxoe(scan_info_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Odd/even-echo readout trajectories kxo, kxe in cycles/cm."""
    d = read_mat(scan_info_path, ['kxo', 'kxe'])
    return d['kxo'].ravel() / 100, d['kxe'].ravel() / 100


def load_schedules(scan_info_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(schedules, echo_times).

    schedules: [Nframes, Nshots, ETL, 2] int, 0-based (ky, kz) (scan_info.mat
        stores them 1-based).
    echo_times: [Nframes, Nshots, ETL] float, seconds since RF excitation.
    """
    raw = read_mat(scan_info_path, ['schedules'])['schedules']
    return raw[..., :2].astype(np.int64) - 1, raw[..., 2]


def nominal_te_s(scan_info_path: str, etl: int) -> float:
    """Acquisition time of the nominal-TE echo (index (ETL-1)//2), which is
    the same for every frame and shot."""
    schedules = read_mat(scan_info_path, ['schedules'])['schedules']
    return float(schedules[0, 0, (etl - 1) // 2, 2])


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------


def save_recon_nifti(fn_base: str, img: np.ndarray, **attrs) -> None:
    """Write `<fn_base>.nii.gz` (float32 magnitude; NIfTI has no complex type)
    and a `<fn_base>.json` sidecar holding `attrs`. `attrs['fov']` (m) sets the
    voxel size. The affine is diagonal: spacing is right, but there is no
    patient orientation, so left/right is not guaranteed.

    img: [Nx, Ny, Nz] or [Nx, Ny, Nz, N] (frames, coils or echoes)."""
    import nibabel as nib

    if np.iscomplexobj(img):
        img = np.abs(img)
    fov = attrs['fov']
    voxel_size_mm = [1000.0 * fov[axis] / img.shape[axis] for axis in range(3)]
    affine = np.diag(voxel_size_mm + [1.0])
    nib.save(nib.Nifti1Image(img.astype(np.float32), affine), f'{fn_base}.nii.gz')
    with open(f'{fn_base}.json', 'w') as f:
        json.dump(attrs, f, indent=2, default=_json_default)


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f'not JSON serializable: {type(x)}')


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def matlab_round(x: float) -> int:
    """MATLAB's round(): halves round away from zero (Python/numpy round them
    to even). Used wherever a ported index boundary must match MATLAB."""
    return int(np.floor(x + 0.5)) if x >= 0 else int(np.ceil(x - 0.5))


def ift3c(d: np.ndarray) -> np.ndarray:
    """Centered inverse 3D FFT over the first three axes:
    fftshift(ifftn(ifftshift(d))). Trailing axes (coils, echoes) are batched."""
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(d, axes=axes), axes=axes), axes=axes)


def rss_echo_images(ksp_echoes: np.ndarray) -> np.ndarray:
    """[Nx, Ny, Nz, n_echoes, Nc] k-space -> [Nx, Ny, Nz, n_echoes]
    root-sum-of-squares magnitude image of each echo."""
    return np.sqrt(np.sum(np.abs(ift3c(ksp_echoes)) ** 2, axis=-1))


# ---------------------------------------------------------------------------
# QA figures
# ---------------------------------------------------------------------------


def plot_gre_b0_diagnostics(fn_output: str, out_dir: str | None = None) -> list[str]:
    """PNG snapshots of the deGRE-grid QA volumes in a preprocess() output
    file (its 'degre' group): both echo magnitudes, their ratio, the field-map
    initialization, the fitted field map and the fit mask, at the center slice
    and a quarter-way slice. For telling apart GRE-data problems from
    field-map-estimation problems. Returns the PNG paths."""
    from matplotlib.figure import Figure  # not pyplot: leaves the caller's backend alone

    with h5py.File(fn_output, 'r') as f:
        g = f['degre']
        img = g['img_echoes'][()]  # [Nx, Ny, Nz, n_echoes]
        te = np.asarray(f.attrs['TE_degre'])
        b0_key = 'b0_map' if 'b0_map' in g else 'b0map_hz'  # older files: b0map_hz
        b0 = g[b0_key][()] if b0_key in g else None
        finit = g['finit_hz'][()] if 'finit_hz' in g else None
        mask = g['mask'][()] if 'mask' in g else None
    out_dir = out_dir or os.path.dirname(fn_output)
    stem = os.path.splitext(os.path.basename(fn_output))[0]

    panels = [(f'|echo 1| (TE={te[0] * 1e3:.2f} ms)', img[..., 0], 'gray', None)]
    if img.shape[-1] > 1:
        ratio = img[..., 1] / (img[..., 0] + 1e-9)
        panels += [
            (f'|echo 2| (TE={te[1] * 1e3:.2f} ms)', img[..., 1], 'gray', None),
            ('echo 2 / echo 1', ratio, 'viridis', (0, 1.5)),
        ]
    for name, vol in [('finit (Hz)', finit), ('B0 map (Hz)', b0)]:
        if vol is not None:
            panels.append((name, vol, 'RdBu_r', (-350, 350)))
    if mask is not None:
        panels.append(('fit mask', mask.astype(float), 'gray', None))

    paths = []
    for iz in (img.shape[2] // 2, img.shape[2] // 4):
        fig = Figure(figsize=(3.2 * len(panels), 3.4))
        axes = fig.subplots(1, len(panels), squeeze=False)
        for ax, (title, vol, cmap, clim) in zip(axes[0], panels):
            kw = dict(vmin=clim[0], vmax=clim[1]) if clim else {}
            im = ax.imshow(vol[:, :, iz].T, origin='lower', cmap=cmap, **kw)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f'{stem}, deGRE z = {iz}')
        fig.tight_layout()
        path = os.path.join(out_dir, f'{stem}_gre_b0_z{iz}.png')
        fig.savefig(path, dpi=120)
        paths.append(path)
    return paths
