"""Tune the readout's k-space center offset (cfg.delay) from the calibration
scan. Ports calibrate_delay.m.

A wrong delay puts a linear phase ramp between odd and even echoes; once the
ramp passes +-pi inside the object, getoephase's no-wrap linear fit breaks.
For each candidate delay this counts adjacent-pixel jumps > wrap_thresh in
the odd/even phase over the object, and picks, among delays with no wraps,
the one whose fitted linear term is closest to zero (a correctly aligned
readout leaves only a constant offset). Needs only the noise and calibration
scans, so it is cheap.

    from preprocess.preprocess import PreprocessConfig, set_seq_paths
    best, report = calibrate_delay(set_seq_paths(PreprocessConfig(datdir), seqname))
"""

import numpy as np

from preprocess import utils
from preprocess.coils import compute_whitening_matrix
from preprocess.preprocess import SeqPaths, apply_delay, compute_oephase, prepare_cal_data
from preprocess.utils import load_kxoe, load_seq_params, matlab_round


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
    if delay_range is None:
        delay_range = np.arange(-6, 6 + 0.05, 0.05)

    seq_params = load_seq_params(paths.scan_info)
    Nx, ETL, fov = seq_params.Nx, seq_params.ETL, seq_params.fov

    ksp_noise = utils.read_archive(paths.noise)
    Nfid = ksp_noise.shape[0]
    W = compute_whitening_matrix(ksp_noise.transpose(0, 2, 1))

    ksp_cal_raw = utils.read_archive(paths.cal)
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
