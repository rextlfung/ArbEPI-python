"""Validate ge/pns.py against real MATLAB pge2.pns.m output. Not a pytest
test -- depends on a local MATLAB install and a reference .mat file that
isn't committed (see ge/validate_against_matlab.py's module docstring for
why this repo keeps MATLAB cross-checks out of the automated suite).

Usage (from repo root), after generating the reference with the (now
removed, see git history) matlab_reference/dump_pns_test.m script:
    uv run python -m ge.validate_pns output/pns_test_reference.mat
"""

import sys

import hdf5storage
import numpy as np

from ge.pns import pns


def validate(reference_mat_path: str) -> bool:
    ref = hdf5storage.loadmat(reference_mat_path)
    dt = float(ref['dt'].item())
    chronaxie = float(ref['chronaxie'].item())
    s_min = float(ref['s_min'].item())
    g = np.asarray(ref['g'])
    ref_pt = np.asarray(ref['pt']).reshape(-1)
    ref_p = np.asarray(ref['p'])

    pt, p = pns(s_min, chronaxie, g, dt)

    pt_ok = np.allclose(pt, ref_pt, rtol=1e-4, atol=1e-3)
    p_ok = np.allclose(p, ref_p, rtol=1e-4, atol=1e-3)
    print(f'{"OK  " if pt_ok else "FAIL"} pt: max python={pt.max():.6f} matlab={ref_pt.max():.6f}')
    print(f'{"OK  " if p_ok else "FAIL"} p: max abs diff={np.abs(p - ref_p).max():.2e}')
    return pt_ok and p_ok


if __name__ == '__main__':
    passed = validate(sys.argv[1])
    print('\nPASSED' if passed else '\nFAILED')
    sys.exit(0 if passed else 1)
