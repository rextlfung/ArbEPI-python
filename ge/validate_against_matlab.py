"""Validate ge/seq2ceq.py + ge/writeceq.py against real MATLAB output,
field by field. Not a pytest test -- like the rest of this repo's GE
toolchain, it depends on a local MATLAB install and reference files that
aren't committed (see CLAUDE.md: no MATLAB-based end-to-end suite exists;
this is the same kind of one-off cross-check, not a CI-run test).

Usage (from repo root), to check seq2ceq only:
    uv run python -m ge.validate_against_matlab output/noise.seq output/noise_ceq_reference.mat

...or also check writeceq's .pge output, if a MATLAB-written .pge for the
same .seq exists (e.g. from ge.ge_export.export_to_ge / main.py --ge):
    uv run python -m ge.validate_against_matlab output/noise.seq output/noise_ceq_reference.mat output/noise.pge 10

The reference .mat is produced by the (now removed, see git history)
matlab_reference/dump_ceq.m script, e.g.:
    matlab -batch "addpath('../pulseq/matlab'); addpath('../toppe'); \
        addpath('../PulCeq/matlab'); addpath('../ArbEPI/lib'); addpath('matlab'); \
        dump_ceq('<abs path to .seq>', '<abs path to reference .mat>');"
"""

import sys
import tempfile
from pathlib import Path

import hdf5storage
import numpy as np
import pypulseq as pp

from ge.read_pge import read_pge
from ge.seq2ceq import seq2ceq
from ge.writeceq import write_ceq


def _flatten(x):
    return np.asarray(x).reshape(-1)


def validate(seq_path: str, reference_mat_path: str):
    """Returns (ok, ceq) -- ceq so callers (e.g. __main__'s optional .pge
    check below) don't have to rebuild it."""
    ref = hdf5storage.loadmat(reference_mat_path)

    seq = pp.Sequence()
    seq.read(seq_path)
    ceq = seq2ceq(seq)

    ok = True

    def check(name, got, want, cmp=lambda a, b: a == b):
        nonlocal ok
        if cmp(got, want):
            print(f'OK   {name}: {got}')
        else:
            ok = False
            print(f'FAIL {name}: python={got} matlab={want}')

    check('nParentBlocks', ceq.nParentBlocks, int(ref['nParentBlocks'].item()))
    check('nSegments', ceq.nSegments, int(ref['nSegments'].item()))
    check('nMax', ceq.nMax, int(ref['nMax'].item()))
    check('nReadouts', ceq.nReadouts, int(ref['nReadouts'].item()))
    check('duration', ceq.duration, float(ref['duration'].item()),
          cmp=lambda a, b: np.isclose(a, b, atol=1e-6))

    ref_parent_duration = _flatten(ref['parentBlockDuration'])
    for pb in ceq.parentBlocks:
        check(f'parentBlocks[{pb.ID}].block_duration', pb.block.block_duration,
              float(ref_parent_duration[pb.ID - 1]), cmp=lambda a, b: np.isclose(a, b, atol=1e-9))

    ref_trid = _flatten(ref['segmentTRID'])
    ref_nblocks = _flatten(ref['segmentNBlocks'])
    ref_emax_n = _flatten(ref['segmentEmaxN'])
    ref_block_ids = ref['segmentBlockIDs']
    for seg in ceq.segments:
        i = seg.ID - 1
        check(f'segments[{seg.ID}].TRID', seg.TRID, int(ref_trid[i]))
        check(f'segments[{seg.ID}].nBlocksInSegment', seg.nBlocksInSegment, int(ref_nblocks[i]))
        check(f'segments[{seg.ID}].Emax_n', seg.Emax_n, int(ref_emax_n[i]))
        ref_ids = _flatten(ref_block_ids[i, 0]).astype(int)
        check(f'segments[{seg.ID}].blockIDs', tuple(int(x) for x in seg.blockIDs), tuple(ref_ids))

    ref_loop = np.asarray(ref['loop'])
    check('loop.shape', ceq.loop.shape, ref_loop.shape)
    if ceq.loop.shape == ref_loop.shape:
        close = np.isclose(ceq.loop, ref_loop, atol=1e-4, rtol=1e-4)
        if close.all():
            print('OK   loop: all', ceq.loop.size, 'values match')
        else:
            ok = False
            bad_rows = np.where(~close.all(axis=1))[0]
            print(f'FAIL loop: mismatch in {len(bad_rows)} of {ceq.loop.shape[0]} rows, e.g. row {bad_rows[0]}:')
            print('  python:', ceq.loop[bad_rows[0]])
            print('  matlab:', ref_loop[bad_rows[0]])

    return ok, ceq


def compare_pge(ceq, matlab_pge_path: str, pislquant: int) -> bool:
    """Write ceq via ge/writeceq.py and compare the result against a
    MATLAB-written .pge, field by field (loose tolerance on floats, since
    derived values like maxSlew can differ by ~1 float32 ulp depending on
    summation order -- see ge/writeceq.py's module docstring)."""
    with tempfile.TemporaryDirectory() as tmp:
        python_pge_path = str(Path(tmp) / 'out.pge')
        write_ceq(ceq, python_pge_path, pislquant=pislquant)

        py_bytes = Path(python_pge_path).read_bytes()
        mat_bytes = Path(matlab_pge_path).read_bytes()
        if py_bytes == mat_bytes:
            print(f'OK   .pge: byte-identical ({len(py_bytes)} bytes)')
            return True

        py, mat = read_pge(python_pge_path), read_pge(matlab_pge_path)
        ok = True
        for key in ('n_parent_blocks', 'n_segments', 'n_max', 'max_b1', 'max_grad',
                    'max_slew', 'duration', 'n_readouts', 'n_heat_check'):
            a, b = py[key], mat[key]
            close = np.isclose(a, b, rtol=1e-4, atol=1e-3) if isinstance(a, float) else a == b
            print(f'{"OK  " if close else "FAIL"} .pge {key}: python={a} matlab={b}')
            ok &= bool(close)

        loop_close = np.allclose(py['loop'], mat['loop'], rtol=1e-4, atol=1e-4)
        print(f'{"OK  " if loop_close else "FAIL"} .pge loop table')
        ok &= loop_close

        for i, (pb, mb) in enumerate(zip(py['parent_blocks'], mat['parent_blocks'])):
            for ax in ('gx', 'gy', 'gz'):
                pv, mv = pb[ax], mb[ax]
                if (pv is None) != (mv is None):
                    print(f'FAIL .pge parent_blocks[{i}].{ax} presence mismatch')
                    ok = False
                    continue
                if pv is None or pv.get('type') == 'trap':
                    continue
                if not np.allclose(pv['magnitude'], mv['magnitude'], rtol=1e-3, atol=1e-4):
                    print(f'FAIL .pge parent_blocks[{i}].{ax}.magnitude mismatch')
                    ok = False
        if ok:
            print('OK   .pge: all fields match within tolerance (not byte-identical)')
        return ok


if __name__ == '__main__':
    seq_path, reference_mat_path = sys.argv[1], sys.argv[2]
    passed, ceq = validate(seq_path, reference_mat_path)

    if len(sys.argv) > 3:
        matlab_pge_path, pislquant = sys.argv[3], int(sys.argv[4])
        passed &= compare_pge(ceq, matlab_pge_path, pislquant)

    print('\nPASSED' if passed else '\nFAILED')
    sys.exit(0 if passed else 1)
