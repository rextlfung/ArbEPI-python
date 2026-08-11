"""Python -> MATLAB bridge for GE `.pge` export.

The GE export toolchain (`seq2ceq`, `+pge2`, `check_grad_acoustics`) is a
~2450-line, hardware-safety-relevant MATLAB toolbox — see the port plan for
why this is not reimplemented in Python. Instead, this shells out to a
local MATLAB install to run matlab/write_to_ge_from_seq.m (adapted from
../ArbEPI/lib/write_to_ge.m) against a Python-generated .seq file.

Requires MATLAB, plus pulseq, toppe, and PulCeq on disk as sibling
directories to this repo (../pulseq, ../toppe, ../PulCeq), and ../ArbEPI
(for check_grad_acoustics.m). Adjust `_sibling_repo_paths` below if your
checkout layout differs.
"""

import glob
import shutil
import subprocess
from pathlib import Path

from params import Params

REPO_ROOT = Path(__file__).resolve().parent


def _find_matlab_binary() -> str | None:
    """Locate the `matlab` executable. Checks PATH first, then common
    macOS install locations (MATLAB.app is often not added to PATH by the
    installer)."""
    on_path = shutil.which('matlab')
    if on_path is not None:
        return on_path

    candidates = sorted(glob.glob('/Applications/MATLAB_*.app/bin/matlab'), reverse=True)
    return candidates[0] if candidates else None


def _sibling_repo_paths() -> list:
    base = REPO_ROOT.parent
    return [
        base / 'pulseq' / 'matlab',
        base / 'toppe',
        base / 'PulCeq' / 'matlab',
        base / 'ArbEPI' / 'lib',
        REPO_ROOT / 'matlab',
    ]


def export_to_ge(seq_path: str, out_path: str, params: Params) -> None:
    """
    Convert a Pulseq .seq file to a GE TOPPE .pge file via MATLAB.

    Parameters
    ----------
    seq_path : path to an existing .seq file (e.g. written by
        sequences.arbepi.generate_arbepi).
    out_path : output path *without* extension — write_to_ge_from_seq.m
        appends '.pge'.
    params : loaded Params (see params.load_params); supplies the GE
        hardware constants (psd_rf_wait, psd_grd_wait, b1_max, g_max,
        slew_max, PNSwt, pislquant).

    Raises
    ------
    RuntimeError
        If the `matlab` binary can't be found, or if any required sibling
        repo directory is missing, or if the MATLAB call fails.
    """
    matlab_bin = _find_matlab_binary()
    if matlab_bin is None:
        raise RuntimeError(
            'GE export requires a local MATLAB install (checked PATH and '
            '/Applications/MATLAB_*.app). Install MATLAB with the pulseq, toppe, and '
            'PulCeq toolboxes, or skip .pge export and use the .seq file directly.'
        )

    paths = _sibling_repo_paths()
    missing = [str(p) for p in paths if not p.is_dir()]
    if missing:
        raise RuntimeError(
            'GE export requires the following sibling directories, which were not found: '
            + ', '.join(missing)
            + '. Expected pulseq, toppe, PulCeq, and ArbEPI checked out alongside this repo.'
        )

    seq_path = str(Path(seq_path).resolve())
    out_path = str(Path(out_path).resolve())

    addpaths = '; '.join(f"addpath('{p}')" for p in paths)
    pns_wt = f'[{params.PNSwt[0]} {params.PNSwt[1]} {params.PNSwt[2]}]'

    matlab_cmd = (
        f"{addpaths}; "
        f"sysPGE2 = pge2.opts({params.psd_rf_wait}, {params.psd_grd_wait}, "
        f"{params.b1_max}, {params.g_max}, {params.slew_max}, 'xrm'); "
        f"write_to_ge_from_seq('{seq_path}', '{out_path}', sysPGE2, {pns_wt}, {params.pislquant});"
    )

    result = subprocess.run(
        [matlab_bin, '-batch', matlab_cmd],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f'MATLAB GE export failed (exit {result.returncode}):\n{result.stderr}')
