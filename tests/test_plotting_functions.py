"""Smoke / regression tests for plot/plotting.py's five plotting functions
(review item 115), headless via matplotlib's Agg backend."""

from dataclasses import replace

import matplotlib

matplotlib.use('Agg')

import matplotlib.figure  # noqa: E402
import numpy as np  # noqa: E402
import pypulseq as pp  # noqa: E402
import pytest  # noqa: E402
from mpl_toolkits.mplot3d.axes3d import Axes3D  # noqa: E402

from params import load_params  # noqa: E402
from plot.plotting import (  # noqa: E402
    plot_one_tr,
    plot_pns_one_tr,
    plot_psf,
    plot_sampling_mask,
    plot_trajectory,
)


def _small_params():
    return replace(load_params(), Ny=12, Nz=8, Nframes=3)


def test_plot_psf_peaks_at_dc_location_for_all_ones_mask(monkeypatch):
    # Item 96's regression: a single fftshift (wrong convention) moved the
    # PSF peak off (Ny//2, Nz//2).
    p = _small_params()
    captured = {}
    orig = Axes3D.plot_surface

    def spy(self, X, Y, Z, *a, **k):
        captured['Z'] = np.asarray(Z)
        return orig(self, X, Y, Z, *a, **k)

    monkeypatch.setattr(Axes3D, 'plot_surface', spy)
    fig = plot_psf(np.ones((p.Ny, p.Nz), dtype=bool), p)
    assert isinstance(fig, matplotlib.figure.Figure)
    Z = captured['Z']
    assert Z.shape == (p.Ny, p.Nz)
    assert np.unravel_index(np.argmax(Z), Z.shape) == (p.Ny // 2, p.Nz // 2)
    assert np.sum(Z > 0.5 * Z.max()) == 1  # a delta, not a smear


def test_plot_psf_and_mask_frame_idx_selects_frame_and_title():
    p = _small_params()
    omegas = np.zeros((p.Ny, p.Nz, 3), dtype=bool)
    omegas[:, ::2, 0] = True
    omegas[::2, :, 1] = True
    omegas[::3, ::3, 2] = True

    fig = plot_sampling_mask(omegas, p, 2.0, frame_idx=1)
    ax = fig.axes[0]
    assert 'frame 2' in ax.get_title() and 'R = 2' in ax.get_title()
    red = [ln for ln in ax.lines if ln.get_color() == 'r']
    assert len(red[0].get_xdata()) == omegas[:, :, 1].sum()

    assert 'frame 3' in plot_psf(omegas, p, frame_idx=2).axes[0].get_title()
    # a 2D mask is accepted too
    assert 'frame 1' in plot_sampling_mask(omegas[:, :, 0], p, 2.0).axes[0].get_title()


@pytest.fixture(scope='module')
def seq_and_params(built_seq_dir):
    p = replace(load_params(output_dir=str(built_seq_dir)), Nframes=1, seed=0)
    seq = pp.Sequence(system=p.sys)
    seq.read(str(built_seq_dir / 'ArbEPI.seq'))
    return seq, p


def test_plot_trajectory_whole_sequence_and_single_frame(seq_and_params):
    seq, p = seq_and_params
    whole = plot_trajectory(seq, p, p.R)
    assert isinstance(whole, matplotlib.figure.Figure) and whole.axes[0].lines
    frame = plot_trajectory(seq, p, p.R, frame_idx=0)
    assert isinstance(frame, matplotlib.figure.Figure) and frame.axes[0].lines


def test_plot_one_tr_marks_te_on_every_row(seq_and_params):
    seq, p = seq_and_params
    fig = plot_one_tr(seq, p, shot_index=0)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 6
    for ax in fig.axes[:6]:
        assert any(ln.get_linestyle() == '--' for ln in ax.lines)


def test_plot_pns_one_tr_returns_figure_with_threshold_lines(seq_and_params):
    seq, p = seq_and_params
    fig = plot_pns_one_tr(seq, p, shot_index=0)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 3
