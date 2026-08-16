import numpy as np

from preprocessing.preprocess import _build_omegas, apply_delay, scatter_frame


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
