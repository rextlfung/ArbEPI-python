"""GE `.pge` export and feasibility checking -- pure Python, no MATLAB
round trip. Thin wrappers around ge/seq2ceq.py + ge/writeceq.py (validated
against real MATLAB seq2ceq/writeceq to byte-identical/near-identical
output) and ge/check.py (validated against real MATLAB
pge2.check/pge2.pns/check_grad_acoustics output to float64/<0.04%
precision) -- see CLAUDE.md for the full validation record.
"""

from pathlib import Path

import pypulseq as pp

from seq2ge.check import FeasibilityReport
from seq2ge.check import check_ge_feasibility as _check_ge_feasibility
from seq2ge.seq2ceq import seq2ceq
from seq2ge.writeceq import write_ceq
from params import Params


def check_ge_feasibility(seq_path: str, params: Params) -> FeasibilityReport:
    """
    Run GE hardware/PNS/acoustic-resonance feasibility checks on a .seq
    file, without writing a .pge file — a fast pre-flight so infeasibility
    (wrong scanner limits, PNS over threshold, forbidden acoustic
    frequencies) surfaces in seconds right after .seq generation, instead
    of only being discovered at the end of a full `export_to_ge` call.

    Parameters
    ----------
    seq_path : path to an existing .seq file.
    params : loaded Params (see params.load_params); supplies params.spec
        (hardware/PNS constants for the scanner the sequence was built
        for) and params.PNSwt (phantom-vs-human PNS channel weights).

    Raises
    ------
    RuntimeError
        If any hardware, PNS, or acoustic-resonance limit is exceeded (see
        ge.check.FeasibilityReport.ok).
    """
    seq = pp.Sequence()
    seq.read(str(Path(seq_path).resolve()))
    report = _check_ge_feasibility(seq, params.spec, pns_wt=tuple(params.PNSwt))
    print(f'{Path(seq_path).name}:\n{report.summary()}')
    if not report.ok:
        raise RuntimeError(f'GE feasibility check failed for {seq_path}:\n{report.summary()}')
    return report


def export_to_ge(seq_path: str, out_path: str, params: Params) -> None:
    """
    Convert a Pulseq .seq file to a GE TOPPE .pge file.

    Parameters
    ----------
    seq_path : path to an existing .seq file (e.g. written by
        sequences.arbepi.generate_arbepi).
    out_path : output path *without* extension — '.pge' is appended.
    params : loaded Params (see params.load_params); supplies params.spec
        (pislquant and the hardware/PNS constants used by the feasibility
        check run before writing) and params.PNSwt.

    Raises
    ------
    RuntimeError
        If the feasibility check fails (see check_ge_feasibility) --
        mirrors write_to_ge_from_seq.m's own internal ge_feasibility_check
        call, so export never silently writes an infeasible .pge.
    """
    seq = pp.Sequence()
    seq.read(str(Path(seq_path).resolve()))
    report = _check_ge_feasibility(seq, params.spec, pns_wt=tuple(params.PNSwt))
    if not report.ok:
        raise RuntimeError(f'GE feasibility check failed for {seq_path}:\n{report.summary()}')

    ceq = seq2ceq(seq)
    out_file = str(Path(out_path).resolve()) + '.pge'
    write_ceq(ceq, out_file, pislquant=params.spec.pislquant)
    print(f'Wrote {out_file}')
