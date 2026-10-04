"""simulate_fmri's field (b0.py), coil (coils.py) and physiological-noise
(physio.py) models, each against something known independently."""

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('snake')

from simulate_fmri import b0, coils, physio  # noqa: E402
from simulate_fmri import phantom as phantoms  # noqa: E402
from simulate_fmri.session import gridding_noise_gain, slab_profile, steady_state  # noqa: E402

# ---------------------------------------------------------------------------
# field
# ---------------------------------------------------------------------------


def test_dipole_field_of_a_sphere_is_the_textbook_one():
    """A sphere of susceptibility chi in vacuum: no field shift inside (the
    Lorentz-corrected value), and outside the dipole field
    chi/3 (a/r)^3 (3 cos^2(theta) - 1)."""
    n, a, chi = 96, 10.0, 9.0
    ax = np.arange(n) - n // 2
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    r = np.sqrt(x**2 + y**2 + z**2)
    field = b0.dipole_field_ppm(np.where(r <= a, chi, 0.0), (1.0, 1.0, 1.0))
    field = field - field[r > 40].mean()  # the periodic box's constant
    inside = r < 0.6 * a
    assert np.abs(field[inside]).max() < 0.03 * chi
    shell = (r > 1.6 * a) & (r < 3 * a)
    cos2 = np.divide(z**2, r**2, out=np.zeros_like(r), where=r > 0)
    expected = chi / 3 * (a / np.maximum(r, 1e-9)) ** 3 * (3 * cos2 - 1)
    err = np.abs(field - expected)[shell]
    assert err.max() < 0.05 * np.abs(expected[shell]).max()
    # along B0 the field is raised, across it lowered
    assert field[n // 2, n // 2, n // 2 + 20] > 0 > field[n // 2 + 20, n // 2, n // 2]


def test_shim_removes_what_the_scanner_would():
    shape = (20, 18, 16)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = (-19, -17, -15)
    coords = b0.world_coords(shape, affine)
    mask = np.ones(shape, dtype=bool)
    rng = np.random.default_rng(0)
    linear = 3.0 + 0.2 * coords[0] - 0.1 * coords[1] + 0.05 * coords[2]
    quadratic = 1e-3 * (coords[2] ** 2 - (coords[0] ** 2 + coords[1] ** 2) / 2) + 2e-3 * coords[0] * coords[1]
    bump = rng.normal(size=shape)
    np.testing.assert_allclose(b0.shim(linear, mask, coords, 1), 0, atol=1e-9)
    assert np.abs(b0.shim(linear + quadratic, mask, coords, 1)).max() > 0.01
    np.testing.assert_allclose(b0.shim(linear + quadratic, mask, coords, 2), 0, atol=1e-9)
    # what the basis cannot represent stays, less its projection
    assert b0.shim(bump, mask, coords, 2).std() > 0.9 * bump.std()
    with pytest.raises(ValueError):
        b0.shim_basis(coords, 3)


def test_head_field_is_strongest_next_to_the_air_cavity():
    """The analytic head: one cavity below the front of the brain. After a
    linear shim the field is zero-mean over the brain, tens of Hz near the
    cavity and small far from it; without cavities only the head's outline
    shapes it."""
    anat = phantoms.ellipsoid_anatomy((48, 48, 36), 4.0)
    brain = anat.head_brain
    field = b0.head_field_hz(anat.head, brain, anat.head_affine, anat.cavities)
    assert field.shape == anat.head.shape and field.dtype == np.float32
    assert abs(field[brain].mean()) < 1e-3
    coords = b0.world_coords(anat.head.shape, anat.head_affine)
    cav = anat.cavities[0]
    dist = np.sqrt(sum((coords[i] - cav.center_mm[i]) ** 2 for i in range(3)))
    near, far = brain & (dist < 45), brain & (dist > 90)
    assert near.sum() > 50 and far.sum() > 500
    assert np.abs(field[near]).max() > 60
    assert np.abs(field[near]).mean() > 3 * np.abs(field[far]).mean()
    plain = b0.head_field_hz(anat.head, brain, anat.head_affine, ())
    assert np.abs(plain[near]).max() < 0.5 * np.abs(field[near]).max()
    # 3 T against 1.5 T: twice the field
    half = b0.head_field_hz(anat.head, brain, anat.head_affine, anat.cavities, field_strength_t=1.5)
    np.testing.assert_allclose(half, field / 2, atol=1e-3)


def test_resample_moves_a_ramp_between_grids_exactly():
    a = np.diag([2.0, 2.0, 2.0, 1.0])
    a[:3, 3] = (-10, -8, -6)
    coords = b0.world_coords((12, 10, 8), a)
    ramp = 1.0 * coords[0] - 2.0 * coords[1] + 0.5 * coords[2]
    b = np.diag([1.5, 1.5, 1.5, 1.0])
    b[:3, 3] = (-6, -5, -4)
    out = b0.resample(ramp, a, b, (6, 6, 5))
    cb = b0.world_coords((6, 6, 5), b)
    np.testing.assert_allclose(out, cb[0] - 2.0 * cb[1] + 0.5 * cb[2], atol=1e-9)


# ---------------------------------------------------------------------------
# coils
# ---------------------------------------------------------------------------


def test_coil_maps_are_functions_of_position_with_unit_rss():
    """The same point of space gets the same sensitivities on any grid: the
    deGRE and the EPI are simulated on different grids."""
    coarse = [np.arange(-40, 41, 8.0), np.arange(-32, 33, 8.0), np.arange(-24, 25, 8.0)]
    fine = [np.arange(-40, 41, 4.0), np.arange(-32, 33, 4.0), np.arange(-24, 25, 4.0)]
    sc, sf = coils.coil_maps(coarse, 12, coils_per_ring=4), coils.coil_maps(fine, 12, coils_per_ring=4)
    assert sc.shape == (12, 11, 9, 7) and sc.dtype == np.complex64
    np.testing.assert_allclose((np.abs(sf) ** 2).sum(0), 1, atol=1e-5)
    np.testing.assert_allclose(sc, sf[:, ::2, ::2, ::2], atol=1e-5)
    # each coil is strongest on its own side, and the rings differ along z
    pos, _ = coils.coil_centers(12, 4)
    for c in range(12):
        peak = np.unravel_index(np.argmax(np.abs(sf[c])), sf[c].shape)
        at = np.array([fine[0][peak[0]], fine[1][peak[1]], fine[2][peak[2]]])
        assert np.dot(at, pos[c]) > 0
    assert np.abs(np.abs(sf[0, :, :, 0]) - np.abs(sf[0, :, :, -1])).max() > 0.05
    assert np.array_equal(coils.coil_maps(coarse, 1), np.ones((1, 11, 9, 7), np.complex64))


def test_noise_covariance_is_positive_definite_with_the_requested_sizes():
    psi = coils.noise_covariance(32, np.random.default_rng(0))
    assert np.allclose(psi, psi.conj().T) and np.linalg.eigvalsh(psi).min() > 0
    std = np.sqrt(np.real(np.diag(psi)))
    assert np.mean(std**2) == pytest.approx(1.0)
    assert 1.03 < std.max() / std.min() < 1.2
    corr = np.abs(psi / np.outer(std, std))[~np.eye(32, dtype=bool)]
    assert 0.003 < corr.mean() < 0.03 and corr.max() < 0.08
    white = coils.noise_covariance(8, np.random.default_rng(0), 0.0, 0.0)
    np.testing.assert_allclose(white, np.eye(8), atol=1e-12)


# ---------------------------------------------------------------------------
# physiological noise
# ---------------------------------------------------------------------------


def physio_case(n_exc=6000, tr=0.05, **cfg_kw):
    shape = (24, 24, 16)
    gm = torch.zeros(shape)
    gm[4:20, 4:20, 3:13] = 1.0
    wm = torch.zeros(shape)
    csf = torch.zeros(shape)
    csf[10:14, 10:14, 6:10] = 1.0
    gm = gm - csf
    t = np.arange(n_exc) * tr
    z = (np.arange(16) - 8) * 5.0
    cfg = physio.PhysioConfig(**cfg_kw)
    modes, freq, courses = physio.physio({'gm': gm, 'wm': wm, 'csf': csf}, z, (5.0, 5.0, 5.0), t,
                                         ('wm', 'gm', 'csf'), seed=1, cfg=cfg)
    return modes, freq, courses, cfg, t, gm > 0.5, csf > 0.5


def fractional_std(modes, kind, mask, group, te=None):
    """rms over time of the summed modes of one kind, per voxel in mask."""
    total = 0
    for m in modes:
        if m.kind != kind:
            continue
        coef = 1.0 if m.groups is None else float(m.groups[group])
        total = total + coef * m.weight[mask].numpy()[:, None] * m.course[None, :]
    return (total * (1.0 if te is None else te)).std(axis=1)


def test_physio_sizes_match_the_calibration():
    """In gray matter: an R2* fluctuation of 0.27 1/s rms (the BOLD-like part,
    which at TE 45 ms and with the TE-independent part gives Bodurka's
    lambda = 0.0128); in CSF, a 2% pulsation."""
    modes, freq, courses, cfg, t, gm, csf = physio_case()
    r2s = fractional_std(modes, 'r2s', gm, group=1)
    assert np.sqrt(np.mean(r2s**2)) == pytest.approx(cfg.sigma_r2s_gm, rel=0.15)
    amp_gm = fractional_std(modes, 'amp', gm, group=1)
    lam = np.sqrt(np.mean((r2s * 0.045) ** 2 + amp_gm**2))
    assert lam == pytest.approx(0.0128, rel=0.2)
    amp_csf = fractional_std(modes, 'amp', csf, group=2)
    assert np.sqrt(np.mean(amp_csf**2)) == pytest.approx(0.021, rel=0.2)
    assert fractional_std(modes, 'r2s', csf, group=2).max() == 0  # CSF has no BOLD-like part
    # the field offset swings by resp_offset_hz, the gradient by 0.11 Hz/cm peak to peak
    assert np.abs(freq).max() == pytest.approx(cfg.resp_offset_hz, rel=0.05)
    grad = next(m for m in modes if m.kind == 'freq')
    span = (grad.weight[0, 0, 0] - grad.weight[0, 0, -1]).item() * np.abs(grad.course).max()
    assert span == pytest.approx(0.0055 * 75, rel=0.1)  # over 75 mm, larger toward -z


def test_physio_time_courses_live_at_their_frequencies():
    modes, freq, courses, cfg, t, _, _ = physio_case()
    f = np.fft.rfftfreq(len(t), t[1] - t[0])

    def peak(x):
        return f[np.argmax(np.abs(np.fft.rfft(x - x.mean())))]

    assert abs(peak(courses['cardiac']) - cfg.cardiac_hz) < 0.15
    assert abs(peak(courses['respiration']) - cfg.resp_hz) < 0.06
    spec = np.abs(np.fft.rfft(courses['bold_like_0'])) ** 2
    band = (f >= cfg.band_hz[0]) & (f <= cfg.band_hz[1])
    assert spec[band].sum() > 0.999 * spec.sum()
    assert abs(courses['drift'][-1] - courses['drift'][0]) == pytest.approx(cfg.drift, rel=1e-6)


def test_physio_scale_zero_is_off_and_seed_reproduces():
    modes, freq, courses, *_ = physio_case(n_exc=200, scale=0.0)
    assert modes == [] and not freq.any() and courses == {}
    a = physio_case(n_exc=200)
    b = physio_case(n_exc=200)
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[0][0].course, b[0][0].course)


# ---------------------------------------------------------------------------
# excitation and noise scaling
# ---------------------------------------------------------------------------


def test_slab_profile_and_steady_state():
    z = np.linspace(-100, 100, 2001)
    p = slab_profile(z, 129.6, 8.0)
    assert p[1000] == pytest.approx(1.0, abs=1e-6)
    half = z[np.argmin(np.abs(p[:1000] - 0.5))]
    assert half == pytest.approx(-64.8, abs=0.2)
    edge = z[np.argmin(np.abs(p[:1000] - 0.9))] - z[np.argmin(np.abs(p[:1000] - 0.1))]
    assert edge == pytest.approx(1.5 * 129.6 / 8, rel=0.03)
    # the Ernst angle maximizes the steady state
    tr, t1 = 0.0506, 1.331
    ernst = np.arccos(np.exp(-tr / t1))
    angles = torch.tensor([0.8 * ernst, ernst, 1.2 * ernst])
    s = steady_state(angles, tr, t1)
    assert s[1] > s[0] and s[1] > s[2]
    assert float(s[1]) == pytest.approx(np.sqrt((1 - np.exp(-tr / t1)) / (1 + np.exp(-tr / t1))), rel=1e-5)


def test_gridding_noise_gain_matches_preprocess_gridding():
    """The factor that turns raw-sample noise into Cartesian-sample noise,
    used to report the SNR: measured by pushing white noise through
    preprocess's own regridding."""
    pytest.importorskip('sigpy')
    from preprocess.epi_gridding import rampsamp2cart

    nx, fov, nfid = 32, 0.2, 40
    u = np.linspace(-1, 1, nfid)
    kx = nx / 2 / fov * (0.9 * u + 0.1 * u**3)
    rng = np.random.default_rng(0)
    noise = (rng.normal(size=(nfid, 4000)) + 1j * rng.normal(size=(nfid, 4000))) / np.sqrt(2)
    measured = np.mean(np.abs(rampsamp2cart(noise, kx / 100, nx, fov * 100)) ** 2)
    assert gridding_noise_gain(kx, nx, fov) == pytest.approx(measured, rel=0.05)
