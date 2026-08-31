import h5py
import numpy as np
import pytest

# preprocess.py imports epi_gridding.py (sigpy) and nifti_io.py (nibabel) at
# module scope -- not used by these tests directly, but the whole module
# fails to import without both.
pytest.importorskip("sigpy")
pytest.importorskip("nibabel")

from preprocessing.preprocess import (  # noqa: E402
    _build_echo_times,
    _build_omegas,
    apply_delay,
    load_schedules,
    resume_start_frame,
    scatter_frame,
    unflatten_gre_echoes,
)


def test_unflatten_gre_echoes_places_data_at_correct_indices():
    """generate_degre's `c` (echo) loop is innermost, then iY, then iZ (see
    sequences/deGRE.py) -- build a raw archive where every acquisition
    encodes its own (echo, iY, iZ) and confirm unflatten_gre_echoes recovers
    exactly the right value at each output location for every echo (not
    just one -- both echoes must survive so an external B0-mapping
    consumer can use them), and drops the iZ=0 calibration block entirely.
    """
    Nx_degre, Ncoils = 2, 3
    Ny_degre, Nz_degre, n_echoes = 5, 4, 2

    def encode(c, iy, iz):
        return c + 1j * (iy * 100 + iz)

    calib = np.full((Nx_degre, Ncoils, Ny_degre * n_echoes), -999, dtype=complex)
    image = np.zeros((Nx_degre, Ncoils, Ny_degre * Nz_degre * n_echoes), dtype=complex)
    # Acquisition order: iZ outermost, iY middle, echo innermost.
    acq = 0
    for iz in range(Nz_degre):
        for iy in range(Ny_degre):
            for c in range(n_echoes):
                image[:, :, acq] = encode(c, iy, iz)
                acq += 1
    ksp_gre_raw = np.concatenate([calib, image], axis=-1)

    vol = unflatten_gre_echoes(ksp_gre_raw, Ny_degre, Nz_degre, n_echoes)
    assert vol.shape == (Nx_degre, Ny_degre, Nz_degre, n_echoes, Ncoils)
    for c in range(n_echoes):
        for iy in range(Ny_degre):
            for iz in range(Nz_degre):
                expected = encode(c, iy, iz)
                np.testing.assert_array_equal(vol[:, iy, iz, c, :], expected)


def test_scatter_frame_places_data_at_correct_indices():
    """The permute/reshape/flatten chain in scatter_frame (and the matching
    flatten of the schedule) is the single riskiest piece of preprocess.py
    -- a MATLAB column-major reshape chain translated to numpy. Build data
    where each (shot, echo) has a unique, decodable value, and confirm the
    scattered volume holds exactly the value the schedule says belongs at
    each (ky, kz) location -- this catches any flatten-order mismatch that
    a shape-only check would miss."""
    Nx, ETL, Nshots, Nvcoils = 4, 3, 5, 2
    Ny, Nz = 8, 6

    ksp = np.zeros((Nx, ETL, Nshots, Nvcoils), dtype=complex)
    for s in range(Nshots):
        for e in range(ETL):
            ksp[:, e, s, :] = s + 1j * e  # identity-encode (shot, echo)

    rng = np.random.default_rng(0)
    all_locs = [(iy, iz) for iy in range(Ny) for iz in range(Nz)]
    rng.shuffle(all_locs)
    chosen = all_locs[: Nshots * ETL]
    schedule_frame = np.array(chosen, dtype=int).reshape(Nshots, ETL, 2)

    zf = scatter_frame(ksp, schedule_frame, Ny, Nz)
    assert zf.shape == (Nx, Ny, Nz, Nvcoils)

    for s in range(Nshots):
        for e in range(ETL):
            iy, iz = schedule_frame[s, e]
            expected = s + 1j * e
            np.testing.assert_array_equal(zf[:, iy, iz, :], expected)

    # Every location not in the schedule must remain exactly zero.
    covered = {tuple(loc) for loc in chosen}
    for iy in range(Ny):
        for iz in range(Nz):
            if (iy, iz) not in covered:
                np.testing.assert_array_equal(zf[:, iy, iz, :], 0)


def test_scatter_frame_warns_on_duplicate_locations():
    Nx, ETL, Nshots, Nvcoils = 2, 2, 2, 1
    Ny, Nz = 4, 4
    ksp = np.ones((Nx, ETL, Nshots, Nvcoils), dtype=complex)
    schedule_frame = np.zeros((Nshots, ETL, 2), dtype=int)  # every (shot,echo) -> (0,0)

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scatter_frame(ksp, schedule_frame, Ny, Nz)
    assert any('duplicate' in str(w.message) for w in caught)


def test_apply_delay_extrapolates_linearly():
    # A perfectly linear base trajectory: any linear interpolation/
    # extrapolation scheme must reproduce it exactly, including beyond the
    # original domain -- this specifically checks true extrapolation
    # (np.interp would clamp instead, silently corrupting boundary samples).
    nfid = 20
    idx = np.arange(1, nfid + 1, dtype=float)
    slope, intercept = 0.37, -1.1
    kxo0 = slope * idx + intercept
    kxe0 = kxo0.copy()

    delay = 3.0  # large enough to push some shifted samples outside [1, nfid]
    kxo, kxe = apply_delay(kxo0, kxe0, nfid, delay)

    expected = slope * (idx - 0.5 - delay) + intercept
    np.testing.assert_allclose(kxo, expected, atol=1e-10)
    np.testing.assert_allclose(kxe, expected, atol=1e-10)


class _FakeArchiveReader:
    """Mimics ArchiveReader.next_frame(): each call returns the next shot's
    index (as a 1-element array standing in for [Nfid, Ncoils] data), so a
    test can decode exactly which shot a caller consumed."""

    def __init__(self):
        self._next = 0

    def next_frame(self) -> np.ndarray:
        shot = np.array([self._next])
        self._next += 1
        return shot


def test_resume_start_frame_fast_forwards_past_completed_shots(tmp_path):
    """Regression test: ArchiveReader has no seek, so resuming from a
    checkpoint must replay (and discard) every shot already consumed by the
    prior run -- otherwise frame `start_frame` would silently receive frame
    0's data instead of its own. Checkpoint at frame 2 (0-based) with 5
    shots/frame must skip shots 0-14, so the next shot read is shot 15."""
    shots_per_frame = 5
    with h5py.File(tmp_path / 'recon.h5', 'a') as mf:
        mf.attrs['last_completed_frame'] = 2

        reader = _FakeArchiveReader()
        start_frame = resume_start_frame(mf, reader, shots_per_frame)

        assert start_frame == 3
        next_shot = reader.next_frame()
        assert next_shot[0] == 3 * shots_per_frame


def test_resume_start_frame_no_checkpoint_does_not_advance_reader():
    with h5py.File('resume_start_frame_test', 'w', driver='core', backing_store=False) as mf:
        reader = _FakeArchiveReader()
        start_frame = resume_start_frame(mf, reader, shots_per_frame=5)

        assert start_frame == 0
        next_shot = reader.next_frame()
        assert next_shot[0] == 0


def _write_hdf5storage_style(path, arrays: dict):
    """See tests/test_preprocessing_matio.py's copy of this helper -- writes
    datasets axis-reversed, mimicking hdf5storage's on-disk convention,
    without depending on hdf5storage itself (not in the preprocessing
    venv)."""
    with h5py.File(path, 'w') as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=np.asarray(arr).transpose())


def test_load_schedules_splits_ky_kz_from_echo_time(tmp_path):
    """scan_info.mat's 'schedules' is (Nframes, Nshots, ETL, 3) --
    1-based (ky, kz) plus echo time in seconds (sequences/ArbEPI.py).
    load_schedules must convert the first two channels to 0-based ints and
    leave the 3rd (not an index) untouched."""
    Nframes, Nshots, ETL = 2, 3, 4
    ky_1based = np.arange(1, Nframes * Nshots * ETL + 1).reshape(Nframes, Nshots, ETL) % 5 + 1
    kz_1based = np.arange(1, Nframes * Nshots * ETL + 1).reshape(Nframes, Nshots, ETL) % 3 + 1
    te = 0.001 + 0.0001 * np.arange(ETL)[None, None, :] * np.ones((Nframes, Nshots, 1))
    raw = np.stack([ky_1based, kz_1based, te], axis=-1)

    path = tmp_path / 'scan_info.mat'
    _write_hdf5storage_style(path, {'schedules': raw})

    schedules, echo_times = load_schedules(str(path))
    assert schedules.shape == (Nframes, Nshots, ETL, 2)
    assert echo_times.shape == (Nframes, Nshots, ETL)
    np.testing.assert_array_equal(schedules[..., 0], ky_1based - 1)
    np.testing.assert_array_equal(schedules[..., 1], kz_1based - 1)
    np.testing.assert_allclose(echo_times, te)


def test_build_echo_times_places_values_at_scheduled_locations():
    # Real schedules never repeat a (ky, kz) location within a frame (see
    # CLAUDE.md: Nshots*ETL exactly equals the mask's sample count) --
    # sample without replacement so this fixture keeps that invariant and
    # each per-element assertion below is unambiguous.
    Nframes, Nshots, ETL = 2, 3, 4
    Ny, Nz = 6, 5
    rng = np.random.default_rng(2)
    grid = np.array([(y, z) for y in range(Ny) for z in range(Nz)])
    schedules = np.stack(
        [grid[rng.choice(len(grid), Nshots * ETL, replace=False)].reshape(Nshots, ETL, 2)
         for _ in range(Nframes)],
        axis=0,
    )
    echo_times = rng.uniform(0.01, 0.05, (Nframes, Nshots, ETL))

    t = _build_echo_times(schedules, echo_times, Ny, Nz)
    assert t.shape == (Ny, Nz, Nframes)

    omegas = _build_omegas(schedules, Ny, Nz)
    for frame in range(Nframes):
        for s in range(Nshots):
            for e in range(ETL):
                iy, iz = schedules[frame, s, e]
                assert t[iy, iz, frame] == pytest.approx(echo_times[frame, s, e])
    # Unsampled locations stay at 0, disambiguable via omegas.
    assert np.all(t[~omegas] == 0)


def test_build_omegas_marks_scheduled_locations():
    Nframes, Nshots, ETL = 2, 3, 4
    Ny, Nz = 6, 5
    rng = np.random.default_rng(1)
    schedules = np.stack(
        [rng.integers(0, Ny, (Nframes, Nshots, ETL)), rng.integers(0, Nz, (Nframes, Nshots, ETL))],
        axis=-1,
    )

    omegas = _build_omegas(schedules, Ny, Nz)
    assert omegas.shape == (Ny, Nz, Nframes)

    for frame in range(Nframes):
        for s in range(Nshots):
            for e in range(ETL):
                iy, iz = schedules[frame, s, e]
                assert omegas[iy, iz, frame]
