from dataclasses import replace

import numpy as np
import pypulseq as pp
import pytest

from lib.make_prephasers import make_prephasers
from params import load_params


def test_make_prephasers_shares_one_duration_across_axes():
    """Item 28: each axis used to get its own independently-computed
    duration, so a non-isotropic FOV (unequal per-axis areas) left the
    smaller-area axes idling for the rest of the block once the block's
    duration was set by the longest one anyway -- a pypulseq block's
    duration is always its longest gradient event, so a shorter
    axis-specific duration has no effect except hiding what the real
    playout time is. All three should share the longest-needed duration.
    """
    p = load_params()
    # Deliberately non-isotropic FOV/matrix so the three axes would size
    # to different natural durations under the old per-axis computation.
    p = replace(p, fov=np.array([0.216, 0.216, 0.09]), Nx=240, Ny=240, Nz=45)

    gx_pre, gy_pre, gz_pre = make_prephasers(p.Nx, p.Ny, p.Nz, p.fov, p.sys, p.crt)

    durations = [pp.calc_duration(g) for g in (gx_pre, gy_pre, gz_pre)]
    assert durations[0] == durations[1] == durations[2]

    # Areas must still hit each axis's own target exactly -- sharing a
    # duration must not change what each prephaser actually rewinds.
    deltak = [1 / f for f in p.fov]
    expected_areas = [-p.Nx / 2 * deltak[0], -p.Ny / 2 * deltak[1], -p.Nz / 2 * deltak[2]]
    for g, expected in zip((gx_pre, gy_pre, gz_pre), expected_areas):
        assert g.area == pytest.approx(expected, rel=1e-6)
