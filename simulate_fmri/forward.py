"""The raw-data signal model: what each ADC sample of each scan would measure.

The object lives on a grid `g` times finer than the acquisition's along every
axis (same field of view), so a voxel of the image holds g^3 spins with their
own field offset and tissue: intravoxel dephasing and partial volume come out
of the model, and the simulation does not share a voxel grid with the
reconstruction. For a spin at r, excited at time 0 of shot s,

    m(r, t; s) = [sum_g M_g(r) exp(-R_g t)] exp(i 2 pi df(r) t)        base
                 * (1 + sum_k a_k(s) w_k(r))                           'amp' modes
                 * exp(-t sum_k b_k(s) w_k(r))                         'r2s' modes
                 * exp(i 2 pi t sum_k c_k(s) w_k(r))                   'freq' modes

with M_g the transverse magnetization of tissue group g right after the pulse,
R_g its R2*, df the static field (Hz), and the modes small perturbations with a
spatial map w_k and one weight per excitation: BOLD activation and BOLD-like
physiological fluctuation are 'r2s' modes, pulsation and drift 'amp' modes, a
breathing-induced field gradient a 'freq' mode. Coil c then receives

    s_c(k, t) = scale * sum_r S_c(r) m(r, t; s) exp(-i 2 pi k . r).

The perturbations are kept to first order (exp(-t b w) = 1 - t b w, an error of
(t b w)^2 / 2: 5e-4 for a 3% BOLD change), which makes the signal linear in
them. Since an echo train visits (ky, kz) at fixed echo times, everything
spatial can then be computed once per echo index instead of once per shot: for
echo e, the base image and each mode image at t_e, times each coil, Fourier
transformed over (y, z), kept at the (ky, kz) that echo visits anywhere in the
run, and evaluated along the readout by an exact discrete Fourier sum at the
ADC's kx samples (ramp sampling, readout delay and odd/even shift included).
Each shot is then a weighted sum of those. The cost does not depend on the
length of the run.

The evolution during a readout, exp((i 2 pi df - R_g) tau) with tau the sample
time relative to the echo, is expanded to `order` terms in tau for the base
image (order 2: an error of 2% at 300 Hz at the ends of a 0.5 ms readout) and
dropped for the modes.

Everything is torch, on the GPU when there is one.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

MODE_KINDS = ('amp', 'r2s', 'freq')


def default_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class Grid:
    """An acquisition matrix and field of view, and the simulation grid that
    splits each of its voxels into g^3. Voxel i of the acquisition is centered
    at (i - N // 2) * fov / N, the convention of recon/operators.py's centered
    FFT."""

    shape: tuple[int, int, int]
    fov: tuple[float, float, float]  # m
    g: int = 1

    @property
    def fine_shape(self) -> tuple[int, int, int]:
        return tuple(n * self.g for n in self.shape)

    @property
    def res(self) -> tuple[float, float, float]:
        return tuple(f / n for f, n in zip(self.fov, self.shape))

    def coords(self, axis: int) -> NDArray[np.float64]:
        """Centers of the fine voxels along `axis`, m from the center of the
        field of view."""
        n, d = self.shape[axis], self.res[axis]
        m = np.arange(n * self.g)
        return ((m + 0.5) / self.g - 0.5 - n // 2) * d

    def fft_offset(self, axis: int) -> float:
        """coords(axis) minus the positions a centered FFT of the fine grid
        assumes, (m - M // 2) * d / g: a constant."""
        n, d = self.shape[axis], self.res[axis]
        return float(self.coords(axis)[0] + (n * self.g // 2) * d / self.g)

    def fine_affine(self, affine: NDArray) -> NDArray[np.float64]:
        """Voxel-to-world matrix of the fine grid, given the acquisition
        grid's (voxel i at affine @ i)."""
        a = np.asarray(affine, dtype=np.float64)
        out = a.copy()
        out[:3, :3] = a[:3, :3] / self.g
        out[:3, 3] = a[:3, 3] + a[:3, :3] @ np.full(3, 0.5 / self.g - 0.5)
        return out


@dataclass
class Spins:
    """The object on a fine grid (X, Y, Z).

    mag: (G, X, Y, Z) transverse magnetization of each tissue group right
        after excitation (steady state x tissue fraction x excitation profile).
    r2s: (G,) R2* of each group, 1/s.
    b0: (X, Y, Z) static field offset, Hz.
    smaps: (Nc, X, Y, Z) complex coil sensitivities.
    """

    mag: torch.Tensor
    r2s: torch.Tensor
    b0: torch.Tensor
    smaps: torch.Tensor

    def to(self, device) -> Spins:
        return Spins(*(v.to(device) for v in (self.mag, self.r2s, self.b0, self.smaps)))

    def image(self, t: float, order: int = 0) -> list[torch.Tensor]:
        """[sum_g M_g exp(-R_g t) psi_g^n / n! * exp(i 2 pi df t) for n <= order],
        psi_g = i 2 pi df - R_g: the image at time t and its Taylor coefficients
        in the time after t."""
        decay = torch.exp(-self.r2s * t)[:, None, None, None]
        phase = torch.exp(2j * math.pi * self.b0 * t)
        out = [(self.mag * decay).sum(0) * phase]
        if order > 0:
            psi = 2j * math.pi * self.b0[None] - self.r2s[:, None, None, None]
            term = (self.mag * decay).to(phase.dtype)
            for n in range(1, order + 1):
                term = term * psi / n
                out.append(term.sum(0) * phase)
        return out


@dataclass
class Mode:
    """A small perturbation of the signal: `kind` in MODE_KINDS, a spatial map
    `weight` (X, Y, Z), the tissue groups it acts on (`groups`, (G,), 1 for
    all), and `course`, its amplitude at every excitation of the run (1/s for
    'r2s', Hz for 'freq', a fraction for 'amp')."""

    kind: str
    weight: torch.Tensor
    course: NDArray[np.float64]
    groups: torch.Tensor | None = None
    name: str = ''

    def image(self, spins: Spins, t: float) -> torch.Tensor:
        decay = torch.exp(-spins.r2s * t)
        if self.groups is not None:
            decay = decay * self.groups.to(decay)
        base = (spins.mag * decay[:, None, None, None]).sum(0) * self.weight
        factor = {'amp': 1.0, 'r2s': -t, 'freq': 2j * math.pi * t}[self.kind]
        return base * factor * torch.exp(2j * math.pi * spins.b0 * t)


@dataclass
class EPIReadout:
    """The ADC samples of each echo of the train.

    kx: (ETL, Nfid) cycles/m, where each sample actually is.
    tau: (ETL, Nfid) s, each sample's time minus its echo time.
    phase: (ETL,) rad, a constant phase on each echo (the odd/even offset).
    echo_times: (ETL,) s since excitation.
    """

    kx: NDArray[np.float64]
    tau: NDArray[np.float64]
    phase: NDArray[np.float64]
    echo_times: NDArray[np.float64]

    @property
    def etl(self) -> int:
        return self.kx.shape[0]

    @property
    def nfid(self) -> int:
        return self.kx.shape[1]


def interp_extrap(x: NDArray, y: NDArray, xq: NDArray) -> NDArray:
    """Linear interpolation of y(x) at xq, extrapolating linearly beyond the
    ends (MATLAB's interp1(..., 'linear', 'extrap'); np.interp clamps)."""
    out = np.interp(xq, x, y)
    lo, hi = xq < x[0], xq > x[-1]
    out[lo] = y[0] + (xq[lo] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    out[hi] = y[-1] + (xq[hi] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return out


def epi_readout(
    kxo: NDArray,
    kxe: NDArray,
    echo_times: NDArray,
    dwell: float,
    fov_x: float,
    delay: float = 0.0,
    a0: NDArray | float = 0.0,
    a1: NDArray | float = 0.0,
) -> EPIReadout:
    """The readout of an ArbEPI echo train with its imperfections.

    kxo, kxe: (Nfid,) cycles/m, the nominal trajectories of the echoes at
        0-based even and odd positions of the train (scan_info.mat).
    dwell: ADC dwell, s.
    delay: readout delay in samples, in preprocess.apply_delay's convention:
        sample n (1-based) is at kx0(n - 0.5 - delay). preprocess calibrates
        this value from the EPIcal scan.
    a0, a1: odd/even phase of the echoes at odd positions, rad and rad/FOV,
        scalars or one per echo pair (ETL // 2): what oephase.epiphasecorrect
        removes, exp(i (a0 + a1 x)) with x = (ix - nx/2 + 0.5) / nx.
    """
    etl, nfid = len(echo_times), len(kxo)
    idx = np.arange(1, nfid + 1, dtype=float)
    n_pairs = etl // 2
    a0 = np.broadcast_to(np.asarray(a0, dtype=float), (n_pairs,)) if n_pairs else np.zeros(0)
    a1 = np.broadcast_to(np.asarray(a1, dtype=float), (n_pairs,)) if n_pairs else np.zeros(0)
    nx_times_res = fov_x  # a1 is per FOV: a k shift of a1 / (2 pi fov_x)

    kx = np.empty((etl, nfid))
    tau = np.empty((etl, nfid))
    phase = np.zeros(etl)
    for parity, k0 in enumerate((kxo, kxe)):
        true = interp_extrap(idx, np.asarray(k0, dtype=float), idx - 0.5 - delay)
        # fractional sample index at which this trajectory crosses kx = 0
        order = np.argsort(true)
        n0 = np.interp(0.0, true[order], idx[order])
        kx[parity::2] = true
        tau[parity::2] = (idx - n0) * dwell
    for j in range(n_pairs):
        e = 2 * j + 1
        kx[e] = kx[e] - a1[j] / (2 * math.pi * nx_times_res)
        phase[e] = a0[j]
    return EPIReadout(kx=kx, tau=tau, phase=phase, echo_times=np.asarray(echo_times, float))


def cartesian_readout(nx: int, fov_x: float, echo_times: NDArray) -> EPIReadout:
    """An ideal readout: nx samples on the Cartesian grid at the echo time, the
    same direction on every echo, no delay or phase error."""
    etl = len(echo_times)
    kx = np.tile((np.arange(nx) - nx // 2) / fov_x, (etl, 1))
    return EPIReadout(kx=kx, tau=np.zeros((etl, nx)), phase=np.zeros(etl),
                      echo_times=np.asarray(echo_times, float))


def _fft2c(x: torch.Tensor) -> torch.Tensor:
    dims = (-2, -1)
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=dims)), dim=dims)


def _fftnc(x: torch.Tensor) -> torch.Tensor:
    dims = (-3, -2, -1)
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=dims), dim=dims), dim=dims)


def _crop_index(n: int, g: int) -> slice:
    """Fine-FFT indices of the acquisition's n samples along one axis."""
    start = (n * g) // 2 - n // 2
    return slice(start, start + n)


def _scale(grid: Grid) -> float:
    """So that g = 1 is the orthonormal FFT of the image, and finer grids give
    the same value for the same object."""
    return 1.0 / (grid.g**3 * math.sqrt(math.prod(grid.shape)))


@dataclass
class _Locations:
    """The distinct (ky, kz) one echo index visits, and where each shot's is."""

    iy: torch.Tensor  # (L,) acquisition indices
    iz: torch.Tensor
    shot: torch.Tensor  # (n_excitations,) index into iy/iz
    shifts: torch.Tensor = field(default=None)  # (L,) phase for the fine grid's offset


def _locations(sched_e: NDArray, grid: Grid, extra: tuple[int, int], device) -> _Locations:
    """sched_e: (n_excitations, 2). `extra` (the k-space center, for the
    calibration scan) is always location 0."""
    ny, nz = grid.shape[1:]
    flat = np.concatenate([[extra[0] * nz + extra[1]], sched_e[:, 0] * nz + sched_e[:, 1]])
    uniq, inv = np.unique(flat, return_inverse=True)
    # put `extra` first
    first = inv[0]
    perm = np.concatenate([[first], np.delete(np.arange(len(uniq)), first)])
    rank = np.empty(len(uniq), dtype=np.int64)
    rank[perm] = np.arange(len(uniq))
    uniq, inv = uniq[perm], rank[inv]
    iy, iz = uniq // nz, uniq % nz
    ky = (iy - ny // 2) / grid.fov[1]
    kz = (iz - nz // 2) / grid.fov[2]
    shifts = np.exp(-2j * np.pi * (ky * grid.fft_offset(1) + kz * grid.fft_offset(2)))
    return _Locations(
        iy=torch.as_tensor(iy, device=device),
        iz=torch.as_tensor(iz, device=device),
        shot=torch.as_tensor(inv[1:], device=device),
        shifts=torch.as_tensor(shifts, dtype=torch.complex64, device=device),
    )


def _project(
    image: torch.Tensor, spins: Spins, grid: Grid, loc: _Locations, fx: torch.Tensor,
    coil_batch: int,
) -> torch.Tensor:
    """(X, Y, Z) image -> (L, Nc, Nfid): times each coil, Fourier transformed
    over (y, z) and kept at loc, then the exact Fourier sum along x at the kx
    samples in fx (Nfid, X)."""
    nc = spins.smaps.shape[0]
    g = grid.g
    cy, cz = _crop_index(grid.shape[1], g), _crop_index(grid.shape[2], g)
    out = torch.empty(len(loc.iy), nc, fx.shape[0], dtype=torch.complex64, device=image.device)
    for c0 in range(0, nc, coil_batch):
        v = _fft2c(spins.smaps[c0 : c0 + coil_batch] * image)  # (B, X, Yf, Zf)
        v = v[:, :, cy.start + loc.iy, cz.start + loc.iz] * loc.shifts  # (B, X, L)
        out[:, c0 : c0 + coil_batch] = torch.einsum('jx,bxl->lbj', fx, v)
    return out


def epi_readouts(
    spins: Spins,
    grid: Grid,
    readout: EPIReadout,
    schedules: NDArray,
    sink: Callable[[int, torch.Tensor], None],
    modes: list[Mode] | None = None,
    frequency: NDArray | None = None,
    order: int = 2,
    coil_batch: int = 8,
) -> torch.Tensor:
    """Noise-free readouts of an ArbEPI run, and of its blip-free calibration
    train.

    schedules: (Nframes, Nshots, ETL, 2) 0-based (ky, kz) of every echo.
    sink(e, data): called once per echo index with data (Nframes * Nshots, Nc,
        Nfid), the readouts of echo e of every shot, in acquisition order.
    modes: perturbations (see Mode); each course has Nframes * Nshots entries.
    frequency: (Nframes * Nshots,) Hz, a field offset common to all spins
        during each shot (breathing). Applied exactly.
    order: Taylor order of the evolution during a readout (see module docstring).

    Returns the calibration train, (ETL, Nc, Nfid): every echo at the center of
    k-space, with the same readout, unperturbed.
    """
    modes = modes or []
    device = spins.mag.device
    n_frames, n_shots, etl, _ = schedules.shape
    if etl != readout.etl:
        raise ValueError(f'schedule ETL {etl} != readout ETL {readout.etl}')
    sched = schedules.reshape(n_frames * n_shots, etl, 2)
    scale = _scale(grid)
    x = torch.as_tensor(grid.coords(0), device=device)
    center = (grid.shape[1] // 2, grid.shape[2] // 2)
    courses = [torch.as_tensor(m.course, dtype=torch.float32, device=device) for m in modes]
    freq = None if frequency is None else torch.as_tensor(frequency, device=device)
    cal = torch.empty(etl, spins.smaps.shape[0], readout.nfid, dtype=torch.complex64, device=device)

    for e in range(etl):
        t_e = float(readout.echo_times[e])
        kx = torch.as_tensor(readout.kx[e], device=device)
        tau = torch.as_tensor(readout.tau[e], dtype=torch.float32, device=device)
        fx = (torch.exp(-2j * math.pi * kx[:, None] * x[None, :]) * scale).to(torch.complex64)
        loc = _locations(sched[:, e], grid, center, device)

        base = None
        for n, image in enumerate(spins.image(t_e, order)):
            term = _project(image, spins, grid, loc, fx, coil_batch)
            base = term if base is None else base + term * tau**n
        data = base[loc.shot]  # (n_excitations, Nc, Nfid)
        for mode, course in zip(modes, courses):
            term = _project(mode.image(spins, t_e), spins, grid, loc, fx, coil_batch)
            data = data + course[:, None, None] * term[loc.shot]

        phasor = torch.exp(torch.tensor(1j * readout.phase[e], device=device)).to(torch.complex64)
        cal[e] = base[0] * phasor
        if freq is not None:
            t = (t_e + tau).to(torch.float64)
            data = data * torch.exp(2j * math.pi * freq[:, None] * t[None, :]).to(
                torch.complex64
            )[:, None, :]
        sink(e, data * phasor)
    return cal


def gre_kspace(spins: Spins, grid: Grid, echo_times: NDArray, coil_batch: int = 4) -> torch.Tensor:
    """Noise-free Cartesian k-space of a 3D multi-echo gradient echo:
    (Nx, Ny, Nz, n_echoes, Nc), index N // 2 at k = 0 along every axis. The
    evolution during the short readout is ignored."""
    device = spins.mag.device
    nc = spins.smaps.shape[0]
    crops = tuple(_crop_index(n, grid.g) for n in grid.shape)
    k = [(np.arange(n) - n // 2) / f for n, f in zip(grid.shape, grid.fov)]
    shift = [np.exp(-2j * np.pi * k[a] * grid.fft_offset(a)) for a in range(3)]
    shift = torch.as_tensor(
        shift[0][:, None, None] * shift[1][None, :, None] * shift[2][None, None, :],
        dtype=torch.complex64, device=device,
    )
    out = torch.empty(*grid.shape, len(echo_times), nc, dtype=torch.complex64, device=device)
    for i, te in enumerate(echo_times):
        (image,) = spins.image(float(te))
        for c0 in range(0, nc, coil_batch):
            v = _fftnc(spins.smaps[c0 : c0 + coil_batch] * image)[(slice(None), *crops)]
            out[..., i, c0 : c0 + coil_batch] = (v * shift * _scale(grid)).permute(1, 2, 3, 0)
    return out


def band_limited(image: torch.Tensor, grid: Grid) -> torch.Tensor:
    """A fine-grid image (X, Y, Z) at the acquisition's resolution: its Fourier
    transform cut to the acquired k-space and inverted on the acquisition
    grid. What a perfect reconstruction of fully sampled, noise-free data
    returns."""
    crops = tuple(_crop_index(n, grid.g) for n in grid.shape)
    k = [(np.arange(n) - n // 2) / f for n, f in zip(grid.shape, grid.fov)]
    shift = [np.exp(-2j * np.pi * k[a] * grid.fft_offset(a)) for a in range(3)]
    shift = torch.as_tensor(
        shift[0][:, None, None] * shift[1][None, :, None] * shift[2][None, None, :],
        dtype=torch.complex64, device=image.device,
    )
    ksp = _fftnc(image.to(torch.complex64))[crops] * shift / grid.g**3
    dims = (0, 1, 2)
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(ksp, dim=dims), dim=dims),
                              dim=dims)
