"""Automated k-space center delay tuning from odd/even ghost-correction
diagnostics. Ports calibrate_delay.m.

Automates the manual "adjust the delay until the odd/even phase-vs-x plot
stops wrapping" procedure: getoephase fits a constant + linear model to the
raw (unwrapped-along-x) odd/even phase difference and explicitly assumes no
phase wrap; once the k-space delay is far enough off that the true phase
ramp exceeds +-pi within the object support, that fit becomes unreliable.
This sweeps delay_range and, for each value, counts adjacent-pixel phase
jumps > wrap_thresh in the object support (a direct wrap detector,
mirroring visual inspection of phase(x)); among delays with zero detected
wraps, picks the one whose fitted linear term is closest to zero -- a
properly time-aligned trajectory should leave only a constant (coil/RF)
phase offset, no residual spatial ramp.

Only needs the noise and calibration scans (no GRE/smaps), so it's cheap to
sweep. Coil compression isn't used here: odd/even phase content survives
it, but it isn't needed for this diagnostic (mirrors calibrate_delay.m).
"""

import numpy as np

from preprocess.coils import compute_whitening_matrix
from preprocess.config import SeqPaths, load_seq_params
from preprocess.oephase import matlab_round
from preprocess.preprocess import apply_delay, compute_oephase, load_kxoe, prepare_cal_data


def select_best_delay(report: dict) -> float:
    """Among delays with zero detected phase wraps, pick the one whose
    fitted linear term a2 is closest to zero; if every candidate wrapped,
    fall back to the delay with the fewest wraps. Ports the selection logic
    at the end of calibrate_delay.m."""
    wrap_count = np.asarray(report['wrap_count'])
    a2 = np.asarray(report['a2'])
    delay = np.asarray(report['delay'])

    safe = wrap_count == 0
    if not np.any(safe):
        idx = int(np.argmin(wrap_count))
    else:
        candidates = np.flatnonzero(safe)
        idx = candidates[np.argmin(np.abs(a2[candidates]))]
    return float(delay[idx])


def calibrate_delay(
    paths: SeqPaths,
    delay_range: np.ndarray | None = None,
    wrap_thresh: float = np.pi,
) -> tuple[float, dict]:
    """Returns (best_delay, report); report has one entry per swept delay:
    {'delay': [...], 'a1': [...], 'a2': [...], 'wrap_count': [...]}.
    """
    from preprocess.raw_io import read_archive

    if delay_range is None:
        delay_range = np.arange(-6, 6 + 0.05, 0.05)

    seq_params = load_seq_params(paths)
    Nx, ETL, fov = seq_params.Nx, seq_params.ETL, seq_params.fov

    ksp_noise = read_archive(paths.noise)
    Nfid = ksp_noise.shape[0]
    W = compute_whitening_matrix(ksp_noise.transpose(0, 2, 1))

    ksp_cal_raw = read_archive(paths.cal)
    if ksp_cal_raw.shape[0] != Nfid:
        raise ValueError(
            f'calibrate_delay: Calibration Nfid ({ksp_cal_raw.shape[0]}) != '
            f'noise Nfid ({Nfid}) -- wrong noise file?'
        )
    ksp_cal = prepare_cal_data(ksp_cal_raw, W, None, ETL)  # [Nfid, ETL_even, N_shots, Ncoils]

    kxo0, kxe0 = load_kxoe(paths.scan_info)

    report: dict = {'delay': [], 'a1': [], 'a2': [], 'wrap_count': []}
    for d in delay_range:
        kxo, kxe = apply_delay(kxo0, kxe0, Nfid, d)
        a, th = compute_oephase(ksp_cal, kxo, kxe, Nx, fov[0] * 100)

        rows = slice(matlab_round(Nx / 4), matlab_round(3 * Nx / 4))
        cols = slice(th.shape[1] // 2, th.shape[1])
        d_th = np.diff(th[rows, cols], axis=0)
        wrap_count = int(np.sum(np.abs(d_th) > wrap_thresh))

        report['delay'].append(d)
        report['a1'].append(a[0])
        report['a2'].append(a[1])
        report['wrap_count'].append(wrap_count)

    for k in report:
        report[k] = np.array(report[k])

    return select_best_delay(report), report
