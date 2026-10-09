"""Per-shot spoiler / gx_residual guard (docs/review-findings.md item 172).

ArbEPI and EPIcal draw a random spoiler strength per shot and scale the
post-readout spoiler so that the *total* gradient moment since the
excitation lands exactly on cycles/voxel / res on every axis (gx_residual
cancels the readout's net kx, the y/z terms cancel the blip train's net
ky/kz). Here the per-shot draws are recorded, the k-space position at the
end of each shot's spoiler block is integrated from the assembled
sequence's gradient waveforms (same convention as
Sequence.calculate_kspace: moment since the RF center), and compared with
draw / res. Even and odd ETL, so gx_residual's parity sign is covered.
"""

from dataclasses import replace

import numpy as np
import pytest

from params import load_params
from sample.gen_sampling_masks import resolve_omegas
from sequences.ArbEPI import generate_arbepi
from sequences.EPIcal import generate_epical

_REAL_DEFAULT_RNG = np.random.default_rng


class _RecordingRng:
    """Real Generator whose size=3 uniform() draws (the spoiler draws) are
    recorded."""

    def __init__(self, draws):
        self._rng = _REAL_DEFAULT_RNG(1234)
        self._draws = draws

    def uniform(self, *args, **kwargs):
        out = self._rng.uniform(*args, **kwargs)
        self._draws.append(np.array(out, dtype=float))
        return out

    def __getattr__(self, name):
        return getattr(self._rng, name)


def _params(tmp_path, ETL, Nshots, excitation):
    # Ny*Nz/R = 16*12/8 = 24 = Nshots*ETL
    return replace(
        load_params(),
        excitation=excitation,
        Ny=16, Nz=12, ETL=ETL, Nshots=Nshots, Nframes=1,
        sampling_method='caipi', R=8,
        output_dir=str(tmp_path),
    )


def _shot_end_kspace(seq, n_shots):
    """k (cycles/m, 3 x n_shots) at the end of each shot's spoiler block:
    gradient moment since that shot's RF center. The spoiler block is the
    block right after a shot's last ADC block."""
    gm = [g.antiderivative() for g in seq.get_gradients()]
    t_exc = np.asarray(seq.rf_times()[0])
    assert len(t_exc) == n_shots

    nums = sorted(seq.block_events)
    t_end, has_adc = {}, {}
    t = 0.0
    for n in nums:
        t += seq.block_durations[n]
        t_end[n] = t
        has_adc[n] = seq.block_events[n][5] != 0
    spoiler_ends = [
        t_end[nums[i + 1]] for i in range(len(nums) - 1)
        if has_adc[nums[i]] and not has_adc[nums[i + 1]]
    ]
    assert len(spoiler_ends) == n_shots
    return np.array([
        [float(gm[ax](te) - gm[ax](t0)) for te, t0 in zip(spoiler_ends, t_exc)]
        for ax in range(3)
    ])


def _check(seq, p, draws):
    assert len(draws) == p.Nshots
    expected = np.array(draws).T / p.res[:, None]  # cycles/voxel / voxel size
    got = _shot_end_kspace(seq, p.Nshots)
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3 * expected.max())


@pytest.mark.parametrize('excitation', ['fatsat', 'water'])
@pytest.mark.parametrize('ETL,Nshots', [(8, 3), (3, 8)], ids=['even-ETL', 'odd-ETL'])
def test_arbepi_spoiler_lands_on_intended_kspace(tmp_path, monkeypatch, ETL, Nshots, excitation):
    p = _params(tmp_path, ETL, Nshots, excitation)
    omegas = resolve_omegas(p)
    draws = []
    monkeypatch.setattr(np.random, 'default_rng', lambda *a, **k: _RecordingRng(draws))
    seq = generate_arbepi(omegas, p, seqname='spoil')
    _check(seq, p, draws)


@pytest.mark.parametrize('ETL,Nshots', [(8, 3), (3, 8)], ids=['even-ETL', 'odd-ETL'])
def test_epical_spoiler_lands_on_intended_kspace(tmp_path, monkeypatch, ETL, Nshots):
    p = _params(tmp_path, ETL, Nshots, 'fatsat')
    # EPIcal loads the schedule written by ArbEPI.
    generate_arbepi(resolve_omegas(p), p, seqname='ArbEPI')
    draws = []
    monkeypatch.setattr(np.random, 'default_rng', lambda *a, **k: _RecordingRng(draws))
    seq = generate_epical(p, seqname='spoil_cal')
    _check(seq, p, draws)
