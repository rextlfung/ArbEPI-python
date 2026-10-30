"""What every sequence generator does at its end, and its scan_info.mat access.

`finalize_sequence` (timing check, FOV/Name definitions, write the .seq) and
`patch_scan_info` (overwrite a few keys of sequences/ArbEPI.py's scan_info.mat)
were copied into each of ArbEPI/EPIcal/noise/deGRE/meGRE; `load_blip_steps`
is the schedule read EPIcal and noise start with.
"""

import os

import hdf5storage
import pypulseq as pp

from lib.mask2epi import max_blip_steps
from params import Params


def scan_info_path(params: Params) -> str:
    return os.path.join(params.output_dir, 'scan_info.mat')


def finalize_sequence(
    seq: pp.Sequence, params: Params, seqname: str, fov, write: bool = True, verbose: bool = True
) -> bool:
    """check_timing (printing the verdict), set the FOV and Name definitions and write
    `<output_dir>/<seqname>.seq`. Returns whether the timing check passed."""
    ok, error_report = seq.check_timing()
    if verbose:
        if ok:
            print('Timing check passed successfully')
        else:
            print('Timing check failed! Error listing follows:')
            print(error_report)
    seq.set_definition('FOV', fov)
    seq.set_definition('Name', seqname)
    if write:
        os.makedirs(params.output_dir, exist_ok=True)
        seq.write(os.path.join(params.output_dir, f'{seqname}.seq'))
    return ok


def patch_scan_info(params: Params, updates: dict, hint: str) -> bool:
    """Overwrite just the keys of `updates` in scan_info.mat (hdf5storage.savemat's
    default truncate_existing=False updates them in place, leaving what
    sequences/ArbEPI.py wrote untouched). main.py always runs generate_arbepi
    (which creates the file) first, but the GRE generators can also be called
    standalone: with no file the patch is skipped, not an error, and `hint`
    says what that costs. Returns whether it was written."""
    fn = scan_info_path(params)
    if not os.path.exists(fn):
        print(f"  '{fn}' not found -- {hint}")
        return False
    hdf5storage.savemat(fn, updates, fmt='7.3')
    return True


def load_blip_steps(params: Params):
    """(max_ky_step, max_kz_step) of the schedule sequences/ArbEPI.py saved, which
    size the readout gradients EPIcal and noise must match. scan_info.mat is
    MATLAB v7.3 (HDF5), which scipy.io.loadmat cannot read, so this is
    hdf5storage. Index base doesn't matter: only consecutive-echo differences
    are used."""
    return max_blip_steps(hdf5storage.loadmat(scan_info_path(params))['schedules'])
