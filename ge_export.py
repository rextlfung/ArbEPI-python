"""Python -> MATLAB bridge for GE `.pge` export and feasibility checking.

The GE export toolchain (`seq2ceq`, `+pge2`, `check_grad_acoustics`) is a
~2450-line, hardware-safety-relevant MATLAB toolbox — see the port plan for
why this is not reimplemented in Python. Instead, this shells out to a
local MATLAB install to run matlab/write_to_ge_from_seq.m (adapted from
../ArbEPI/lib/write_to_ge.m) and matlab/ge_feasibility_check.m against a
Python-generated .seq file.

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


def _matlab_setup() -> tuple:
    """Locate MATLAB and the sibling toolbox paths, or raise.

    Returns (matlab_bin, addpaths_matlab_snippet).
    """
    matlab_bin = _find_matlab_binary()
    if matlab_bin is None:
        raise RuntimeError(
            'GE export/feasibility checking requires a local MATLAB install (checked PATH '
            'and /Applications/MATLAB_*.app). Install MATLAB with the pulseq, toppe, and '
            'PulCeq toolboxes, or skip this step and use the .seq file directly.'
        )

    paths = _sibling_repo_paths()
    missing = [str(p) for p in paths if not p.is_dir()]
    if missing:
        raise RuntimeError(
            'GE export/feasibility checking requires the following sibling directories, '
            'which were not found: ' + ', '.join(missing)
            + '. Expected pulseq, toppe, PulCeq, and ArbEPI checked out alongside this repo.'
        )

    addpaths = '; '.join(f"addpath('{p}')" for p in paths)
    return matlab_bin, addpaths


def _run_matlab(matlab_bin: str, matlab_cmd: str, error_prefix: str) -> None:
    result = subprocess.run(
        [matlab_bin, '-batch', matlab_cmd],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f'{error_prefix} (exit {result.returncode}):\n{result.stderr}')


def _sys_pge2_snippet(params: Params) -> tuple[str, str]:
    pns_wt = f'[{params.PNSwt[0]} {params.PNSwt[1]} {params.PNSwt[2]}]'
    sys_pge2 = (
        f"pge2.opts({params.psd_rf_wait}, {params.psd_grd_wait}, "
        f"{params.b1_max}, {params.g_max}, {params.slew_max}, '{params.ge_coil}')"
    )
    return sys_pge2, pns_wt


def check_ge_feasibility(seq_path: str, params: Params) -> None:
    """
    Run GE hardware/PNS/acoustic-resonance feasibility checks on a .seq
    file, without writing a .pge file — a fast pre-flight so infeasibility
    (wrong scanner limits, PNS over threshold, forbidden acoustic
    frequencies) surfaces in seconds right after .seq generation, instead
    of only being discovered at the end of a full `export_to_ge` call.

    Parameters
    ----------
    seq_path : path to an existing .seq file.
    params : loaded Params (see params.load_params); supplies the GE
        hardware constants (psd_rf_wait, psd_grd_wait, b1_max, g_max,
        slew_max, ge_coil, PNSwt) for the scanner the sequence was built for.

    Raises
    ------
    RuntimeError
        If the `matlab` binary can't be found, if any required sibling
        repo directory is missing, or if MATLAB reports the sequence is
        infeasible (or otherwise fails).
    """
    matlab_bin, addpaths = _matlab_setup()
    seq_path = str(Path(seq_path).resolve())
    sys_pge2, pns_wt = _sys_pge2_snippet(params)

    matlab_cmd = (
        f"{addpaths}; "
        f"sysPGE2 = {sys_pge2}; "
        f"ge_feasibility_check('{seq_path}', sysPGE2, {pns_wt}, '{params.ge_coil}');"
    )
    _run_matlab(matlab_bin, matlab_cmd, 'MATLAB GE feasibility check failed')


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
        slew_max, ge_coil, PNSwt, pislquant).

    Raises
    ------
    RuntimeError
        If the `matlab` binary can't be found, or if any required sibling
        repo directory is missing, or if the MATLAB call fails.
    """
    matlab_bin, addpaths = _matlab_setup()
    seq_path = str(Path(seq_path).resolve())
    out_path = str(Path(out_path).resolve())
    sys_pge2, pns_wt = _sys_pge2_snippet(params)

    matlab_cmd = (
        f"{addpaths}; "
        f"sysPGE2 = {sys_pge2}; "
        f"write_to_ge_from_seq('{seq_path}', '{out_path}', sysPGE2, {pns_wt}, "
        f"{params.pislquant}, '{params.ge_coil}');"
    )
    _run_matlab(matlab_bin, matlab_cmd, 'MATLAB GE export failed')
