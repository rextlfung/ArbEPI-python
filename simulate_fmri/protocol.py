"""One ArbEPI scan session, as scan_info.mat records it."""

from __future__ import annotations

import math
from dataclasses import dataclass

import h5py
import numpy as np
from numpy.typing import NDArray

from preprocess import utils as scan_utils

T1_ERNST_S = 1.3  # params.py's T1, for scan_info.mat files that predate their flip angles
DEFAULT_DWELL_S = 4e-6  # GE's ADC raster, for files that predate 'adc_dwell'
DEFAULT_TR_DEGRE_S = 5.7e-3  # params.py's, for files that predate 'TR_degre'


def ernst_angle_deg(tr_s: float, t1_s: float = T1_ERNST_S) -> float:
    return math.degrees(math.acos(math.exp(-tr_s / t1_s)))


@dataclass
class DeGRE:
    """The dual-echo GRE reference scan (sequences/deGRE.py)."""

    shape: tuple[int, int, int]
    fov_mm: tuple[float, float, float]
    te_s: tuple[float, ...]
    tr_s: float
    fa_deg: float
    dwell_s: float

    @property
    def res_mm(self) -> tuple[float, float, float]:
        return tuple(f / n for f, n in zip(self.fov_mm, self.shape))


@dataclass
class Protocol:
    """One ArbEPI acquisition. The last four fields are only needed to
    simulate raw data (session.py); the ideal mode ignores them."""

    shape: tuple[int, int, int]  # (Nx, Ny, Nz)
    fov_mm: tuple[float, float, float]
    schedules: NDArray  # (Nframes, Nshots, ETL, 2) int, 0-based (ky, kz)
    echo_times_ms: NDArray  # (ETL,) ms since excitation
    volume_tr_s: float
    fa_deg: float
    kxo: NDArray | None = None  # (Nfid,) cycles/m, echoes at even positions (0-based)
    kxe: NDArray | None = None  # echoes at odd positions
    adc_dwell_s: float = DEFAULT_DWELL_S
    degre: DeGRE | None = None
    water_excitation: bool = True

    @property
    def n_frames(self) -> int:
        return self.schedules.shape[0]

    @property
    def n_shots(self) -> int:
        return self.schedules.shape[1]

    @property
    def etl(self) -> int:
        return self.schedules.shape[2]

    @property
    def res_mm(self) -> tuple[float, float, float]:
        return tuple(f / n for f, n in zip(self.fov_mm, self.shape))

    @property
    def tr_shot_ms(self) -> float:
        """Time between excitations."""
        return self.volume_tr_s * 1e3 / self.n_shots

    @property
    def echo_spacing_ms(self) -> float:
        return float(np.diff(self.echo_times_ms).mean()) if self.etl > 1 else 0.0

    @property
    def te_ms(self) -> float:
        """The nominal TE: echo-train position ETL/2 - 0.5 (sequences/ArbEPI.py),
        which for evenly spaced echoes is their mean."""
        return float(self.echo_times_ms.mean())

    @property
    def t_ref_s(self) -> float:
        """preprocess.utils.nominal_te_s: the time of echo (ETL - 1) // 2."""
        return float(self.echo_times_ms[(self.etl - 1) // 2]) / 1e3

    @property
    def acceleration(self) -> float:
        return self.shape[1] * self.shape[2] / (self.n_shots * self.etl)


def load_protocol(
    scan_info: str, fa_deg: float | None = None, frames: int | None = None
) -> Protocol:
    """The acquisition in a scan_info.mat.

    fa_deg: overrides the file's EPI flip angle.
    frames: keep only the first `frames` frames.

    Files written before sequences/ArbEPI.py recorded them get: the Ernst
    angle for T1 = 1.3 s for both flip angles (the rule and value params.py
    uses), a 4 us ADC dwell for both sequences, and a 5.7 ms deGRE TR.
    """
    sp = scan_utils.load_seq_params(scan_info)
    schedules, echo_times = scan_utils.load_schedules(scan_info)
    if not np.allclose(echo_times, echo_times[0, 0]):
        raise ValueError(f'{scan_info}: echo times differ between shots or frames')
    if frames is not None:
        schedules = schedules[:frames]
    n_shots = schedules.shape[1]
    with h5py.File(scan_info, 'r') as f:
        def scalar(name, default):
            return float(f[name][()].item()) if name in f else default

        saved_fa = scalar('fa', None)
        dwell = scalar('adc_dwell', DEFAULT_DWELL_S)
        tr_degre = scalar('TR_degre', DEFAULT_TR_DEGRE_S)
        alpha_degre = scalar('alpha_degre', ernst_angle_deg(tr_degre))
        dwell_degre = scalar('dwell_degre', dwell)
        water = bool(scalar('water_excitation', 1))
    if fa_deg is None:
        fa_deg = saved_fa
    if fa_deg is None:
        fa_deg = ernst_angle_deg(sp.volume_tr / n_shots)
        print(
            f"{scan_info} has no 'fa': using the Ernst angle for T1 = {T1_ERNST_S} s, "
            f'{fa_deg:.2f} degrees'
        )
    kxo, kxe = scan_utils.load_kxoe(scan_info)  # cycles/cm
    degre = None
    if sp.TE_degre is not None:
        degre = DeGRE(
            shape=(sp.Nx_degre, sp.Ny_degre, sp.Nz_degre),
            fov_mm=tuple(1e3 * v for v in sp.fov_degre),
            te_s=tuple(sp.TE_degre),
            tr_s=tr_degre,
            fa_deg=alpha_degre,
            dwell_s=dwell_degre,
        )
    return Protocol(
        shape=(sp.Nx, sp.Ny, sp.Nz),
        fov_mm=tuple(1e3 * f for f in sp.fov),
        schedules=schedules,
        echo_times_ms=echo_times[0, 0] * 1e3,
        volume_tr_s=sp.volume_tr,
        fa_deg=fa_deg,
        kxo=kxo * 100,
        kxe=kxe * 100,
        adc_dwell_s=dwell,
        degre=degre,
        water_excitation=water,
    )
