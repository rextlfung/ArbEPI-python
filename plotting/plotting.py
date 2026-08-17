"""Non-interactive matplotlib equivalents of ../ArbEPI/src/plot_epi.m
(cells 3 and 5) and the inline figures in ../ArbEPI/src/ArbEPI.m, plus a
thin wrapper around pypulseq's own Sequence.plot() for single-TR snippets.

Deliberately NOT ported: the interactive scroll/slider `plot_kt_mask.m`
viewer and the GE-specific `pge2.plotPGE2grads` cell — no algorithmic
content worth preserving in a headless port (see the port plan).
`plot_one_tr` below is not a port of anything MATLAB-side; it just calls
pypulseq's existing `Sequence.plot()` with a time window, no
pulse-diagram-rendering logic of our own.

The trajectory plot here is simplified relative to plot_epi.m's cell 3: the
MATLAB version segments the trajectory per-shot using `t_ktraj` (a
sample-time axis for the full trajectory) to color/break each shot
separately. pypulseq's `Sequence.calculate_kspace()` doesn't return that
axis, so this instead plots the full continuous trajectory as one line —
still useful as a diagnostic, without the (MATLAB-side, itself noted as
slow) nearest-timestamp segmentation loop.

`plot_sampling_mask`/`plot_psf`/`plot_trajectory` all accept an optional
`frame_idx` to select a single frame out of a multi-frame acquisition. For
`plot_trajectory` this is more involved than array slicing: pypulseq's
`Sequence.calculate_kspace()` only returns the trajectory for the *entire*
assembled sequence, with no per-frame breakdown. Two approaches were
considered:

1. Extract just one frame's blocks into a standalone sub-`Sequence` (via
   `get_block`/`add_block`, using the fact that every shot is assembled
   with the same fixed block count) and call `calculate_kspace()` on that.
   This was implemented and tested against the known-correct exact
   `k_traj_adc` slice (see point 2) as a check -- kz/ky matched almost
   exactly, but kx (the readout axis) showed a persistent, non-constant
   ~16-22 m^-1 offset with no evident cause after investigation, likely
   related to how `calculate_kspace()`'s per-excitation k-space reset
   interacts with the bipolar/blipped EPI readout across shot boundaries.
   Abandoned rather than shipped un-understood.
2. Exact-slice `k_traj_adc` (the ADC-sampled trajectory) directly: every
   shot contributes exactly `ETL * Nfid` samples in a fixed, predictable
   order, so frame `f`'s samples are
   `k_traj_adc[:, f*Nshots*ETL*Nfid : (f+1)*Nshots*ETL*Nfid]` (`Nfid`
   inferred from the array's total length). This is the same slicing
   logic `tests/test_trajectory_matches_schedule.py` already validates
   against the sampling schedule. Used below.

The tradeoff of (2): the fine continuous trajectory (`k_traj`, the smooth
line through the analog gradient ramps between echoes) isn't
frame-sliceable this way, so the per-frame plot instead draws a line
directly through the ADC-sampled points -- an honest simplification, not
a bug: it omits the sub-echo ramp shape but preserves the exact EPI
zigzag pattern and exact k-space locations.
"""

import matplotlib.figure
import numpy as np
import pypulseq as pp

from params import Params
from seq2ge.check import (
    PNS_FIRST_CONTROLLED_MODE_THRESHOLD,
    PNS_NORMAL_MODE_THRESHOLD,
    sample_gradients_tesla_per_m,
)
from seq2ge.pns import pns


def plot_sampling_mask(
    omega: np.ndarray, params: Params, R: float, frame_idx: int = 0
) -> matplotlib.figure.Figure:
    """2D sampling mask on the (ky, kz) grid. Ported from plot_epi.m's
    'Plot sampling mask on k-space grid' cell.

    `omega` may be a single frame's (Ny, Nz) mask, or the full
    (Ny, Nz, Nframes) array -- in the latter case `frame_idx` selects
    which frame to plot."""
    if omega.ndim == 3:
        omega = omega[:, :, frame_idx]

    Ny, Nz = params.Ny, params.Nz
    deltak_y = 1 / params.fov[1]
    deltak_z = 1 / params.fov[2]

    ky_grid, kz_grid = np.meshgrid(
        np.linspace(-Ny * deltak_y / 2, Ny * deltak_y / 2, Ny + 1),
        np.linspace(-Nz * deltak_z / 2, Nz * deltak_z / 2, Nz + 1),
    )

    ys, zs = np.nonzero(omega)
    ky_samp = (ys - Ny / 2) * deltak_y
    kz_samp = (zs - Nz / 2) * deltak_z

    fig = matplotlib.figure.Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.axhline(0, color='k', linewidth=1, zorder=0)
    ax.axvline(0, color='k', linewidth=1, zorder=0)
    ax.plot(ky_grid.ravel(), kz_grid.ravel(), '.', color=(0.7, 0.7, 0.7), markersize=3)
    ax.plot(ky_samp, kz_samp, 'r.', markersize=4)
    ax.set_aspect('equal')
    ax.set_title(f'2D sampling mask, frame {frame_idx}. R = {round(R)}')
    ax.set_xlabel('k_y (m$^{-1}$)')
    ax.set_ylabel('k_z (m$^{-1}$)')
    ax.set_xlim(-Ny * deltak_y / 2, Ny * deltak_y / 2)
    ax.set_ylim(-Nz * deltak_z / 2, Nz * deltak_z / 2)
    return fig


def plot_trajectory(
    seq: pp.Sequence, params: Params, R: float, frame_idx: int | None = None
) -> matplotlib.figure.Figure:
    """3D-EPI k-space trajectory (simplified — see module docstring).

    frame_idx : if None (default), plot the whole sequence's trajectory
        (continuous line + ADC-sampled points). If given, plot only that
        frame's exact ADC-sampled points, connected point-to-point (see
        module docstring for why the continuous line isn't available
        per-frame)."""
    Ny, Nz = params.Ny, params.Nz
    deltak_y = 1 / params.fov[1]
    deltak_z = 1 / params.fov[2]

    k_traj_adc, k_traj, _, _, _ = seq.calculate_kspace()

    fig = matplotlib.figure.Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    if frame_idx is None:
        ax.plot(k_traj[1, :], k_traj[2, :], 'b-', linewidth=1.0)
        ax.plot(k_traj_adc[1, :], k_traj_adc[2, :], 'r.', markersize=4)
        title = f'3D-EPI trajectory. R = {round(R)}'
    else:
        n_total_samples = k_traj_adc.shape[1]
        n_shot_samples = params.ETL * (n_total_samples // (params.Nframes * params.Nshots * params.ETL))
        frame_start = frame_idx * params.Nshots * n_shot_samples

        # One line per shot, each a distinct color -- not one continuous
        # line across all Nshots -- so consecutive shots' disjoint k-space
        # regions aren't visually joined by a spurious cross-shot segment,
        # and shots are easy to visually tell apart.
        cmap = matplotlib.colormaps['hsv']
        for shot in range(params.Nshots):
            shot_start = frame_start + shot * n_shot_samples
            shot_traj = k_traj_adc[:, shot_start : shot_start + n_shot_samples]
            color = cmap(shot / params.Nshots)
            ax.plot(shot_traj[1, :], shot_traj[2, :], '-', linewidth=0.7, color=color, label=f'shot {shot}')
            ax.plot(shot_traj[1, :], shot_traj[2, :], '.', markersize=4, color=color)
            # Mark each echo train's first sample distinctly from the rest
            # of its points, so where each shot begins (vs. wanders through
            # afterward) is visible at a glance -- most useful for
            # mask2epi_radial, where every shot starts at one end of its
            # spoke rather than at a shared corner of k-space.
            ax.plot(
                shot_traj[1, 0], shot_traj[2, 0], 'o', markersize=4,
                markerfacecolor=color, markeredgecolor='k', markeredgewidth=0.8, zorder=5,
            )
        ax.plot(
            [], [], 'o', markersize=4, markerfacecolor='none', markeredgecolor='k',
            markeredgewidth=0.8, label='echo train start',
        )
        ax.legend(fontsize=6, loc='upper right', ncol=2)
        title = f'3D-EPI trajectory, frame {frame_idx}. R = {round(R)}'

    ax.axhline(0, color='k', linewidth=1, zorder=0)
    ax.axvline(0, color='k', linewidth=1, zorder=0)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('k_y (m$^{-1}$)')
    ax.set_ylabel('k_z (m$^{-1}$)')
    ax.set_xlim(-Ny * deltak_y / 2, Ny * deltak_y / 2)
    ax.set_ylim(-Nz * deltak_z / 2, Nz * deltak_z / 2)
    return fig


def plot_psf(omega: np.ndarray, params: Params, frame_idx: int = 0) -> matplotlib.figure.Figure:
    """Point spread function in y-z space. Ported from plot_epi.m's PSF
    cell — rendered as a 2D magnitude heatmap rather than MATLAB's 3D
    `surf`.

    `omega` may be a single frame's (Ny, Nz) mask, or the full
    (Ny, Nz, Nframes) array -- in the latter case `frame_idx` selects
    which frame to plot."""
    if omega.ndim == 3:
        omega = omega[:, :, frame_idx]

    Ny, Nz = params.Ny, params.Nz
    res_y, res_z = params.res[1], params.res[2]

    psf = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(omega)))
    psf_mag = np.abs(psf)

    fig = matplotlib.figure.Figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    extent = [-Ny * res_y / 2, Ny * res_y / 2, -Nz * res_z / 2, Nz * res_z / 2]
    im = ax.imshow(psf_mag.T, origin='lower', extent=extent, aspect='equal', cmap='viridis')
    fig.colorbar(im, ax=ax, label='magnitude (a.u.)')
    ax.set_xlabel('y (m)')
    ax.set_ylabel('z (m)')
    ax.set_title(f'Point spread function in y-z space, frame {frame_idx}')
    return fig


def plot_one_tr(seq: pp.Sequence, params: Params, shot_index: int = 0) -> matplotlib.figure.Figure:
    """Single-TR snippet of an assembled sequence via pypulseq's own
    `Sequence.plot()`. Every shot loop iteration in generate_arbepi()
    (sequences/arbepi.py) is exactly one TR, so
    [shot_index*TR, (shot_index+1)*TR) covers one shot's worth of blocks —
    fat-sat, RF-spoiled excitation, and the full EPI readout.

    `Sequence.plot(stacked=True)` lays out its 6 rows (ADC, RF mag, RF/ADC
    phase, Gx, Gy, Gz) in a fixed-size figure meant for on-screen viewing,
    which is too short for 6 rows and leaves adjacent y-axis labels
    overlapping. Taller figure + tight_layout() fixes it without touching
    pypulseq's own plotting code."""
    t0 = shot_index * params.TR
    splot = seq.plot(time_range=(t0, t0 + params.TR), stacked=True, plot_now=False, time_disp='ms')
    fig = splot.fig1
    fig.set_size_inches(20, 10)
    fig.tight_layout()
    return fig


def plot_pns_one_tr(seq: pp.Sequence, params: Params, shot_index: int = 0) -> matplotlib.figure.Figure:
    """Gradient / slew / PNS diagnostic for one TR, mirroring PulCeq's own
    `pge2.pns(..., 'plt', true)` panel layout (../PulCeq/matlab/+pge2/pns.m)
    -- gradient waveforms, slew rate, and per-channel + total PNS (% of
    stimulation threshold), with the IEC 60601-2-33:2022 80%/100%
    operating-mode lines marked (see seq2ge/check.py's
    PNS_NORMAL_MODE_THRESHOLD/PNS_FIRST_CONTROLLED_MODE_THRESHOLD for what
    those mean and why only the 100% line blocks `--ge` export).

    Same one-TR window as `plot_one_tr` (shot_index selects
    [shot_index*TR, (shot_index+1)*TR)) and the same PNS model
    (seq2ge/pns.py) `--ge`'s own feasibility check uses, driven by
    `params.spec` (rheobase/alpha/chronaxie) and `params.PNSwt` -- this is
    a decomposition of the same peak number check_ge_feasibility reports,
    not an independent estimate."""
    t0 = shot_index * params.TR
    gw_tm, dt = sample_gradients_tesla_per_m(seq, time_range=(t0, t0 + params.TR))
    t_ms = (np.arange(gw_tm.shape[1]) + 0.5) * dt * 1e3

    s_min = params.spec.rheobase / params.spec.alpha
    pt, p = pns(s_min, params.spec.chronaxie, gw_tm, dt, wt=tuple(params.PNSwt))

    fig = matplotlib.figure.Figure(figsize=(12, 9))
    ax_grad, ax_slew, ax_pns = fig.subplots(3, 1, sharex=True)

    for ch, label in enumerate(('x', 'y', 'z')):
        ax_grad.plot(t_ms, gw_tm[ch] * 1e3, label=label)
    ax_grad.set_ylabel('gradient (mT/m)')
    ax_grad.legend(loc='upper right')
    ax_grad.grid(True)

    slew = np.diff(gw_tm, axis=1) / dt
    for ch, label in enumerate(('x', 'y', 'z')):
        ax_slew.plot(t_ms[:-1], slew[ch], label=label)
    ax_slew.set_ylabel('slew (T/m/s)')
    ax_slew.grid(True)

    for ch, label in enumerate(('x', 'y', 'z')):
        ax_pns.plot(t_ms, p[ch], '--', linewidth=1, label=f'{label} channel')
    ax_pns.plot(t_ms, pt, 'k', linewidth=1.5, label='total')
    ax_pns.axhline(
        PNS_NORMAL_MODE_THRESHOLD, color='tab:orange', linestyle=':',
        label=f'{PNS_NORMAL_MODE_THRESHOLD:.0f}% (normal mode)',
    )
    ax_pns.axhline(
        PNS_FIRST_CONTROLLED_MODE_THRESHOLD, color='tab:red', linestyle=':',
        label=f'{PNS_FIRST_CONTROLLED_MODE_THRESHOLD:.0f}% (first controlled mode)',
    )
    ax_pns.set_ylabel('PNS (% of threshold)')
    ax_pns.set_xlabel('time (ms)')
    ax_pns.legend(loc='upper right', fontsize=8)
    ax_pns.grid(True)

    fig.suptitle(f'PNS, one TR (shot {shot_index}), peak {pt.max():.1f}%')
    fig.tight_layout()
    return fig
