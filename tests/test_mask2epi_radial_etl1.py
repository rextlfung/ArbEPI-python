"""mask2epi_radial with ETL=1 (review item 119): used to crash on
.max() of an empty step array; mask2epi_laminar always handled it."""

import numpy as np

from lib.mask2epi import mask2epi_laminar, mask2epi_radial


def test_mask2epi_radial_etl1_schedules_every_sample_once():
    mask = np.zeros((8, 8), dtype=bool)
    pts = [(1, 2), (3, 3), (5, 6), (7, 0)]
    for y, z in pts:
        mask[y, z] = True

    sched, parts = mask2epi_radial(mask, ETL=1, Nshots=4)
    assert sched.shape == (4, 1, 2)
    assert {tuple(s) for s in sched[:, 0, :]} == set(pts)
    assert set(np.unique(parts[mask])) == {1, 2, 3, 4}
    assert (parts[~mask] == 0).all()

    lam, _ = mask2epi_laminar(mask, ETL=1, Nshots=4)
    assert {tuple(s) for s in lam[:, 0, :]} == set(pts)
