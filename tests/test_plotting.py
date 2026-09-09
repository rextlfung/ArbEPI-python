import numpy as np

from plotting.plotting import nominal_te_value


def test_nominal_te_value_even_etl_averages_middle_pair():
    # Regression test for docs/review-findings.md items 127/149: the true
    # continuous nominal-TE echo index is ETL/2 - 0.5 (lib/calc_te_tr_delays.py's
    # min_te definition). For even ETL that's the midpoint between the two
    # middle echoes, not either one alone.
    ETL = 6
    values = np.arange(ETL, dtype=float)  # [0, 1, 2, 3, 4, 5]
    assert nominal_te_value(values, ETL) == 2.5  # avg(values[2], values[3])


def test_nominal_te_value_odd_etl_uses_exact_middle_echo():
    ETL = 7
    values = np.arange(ETL, dtype=float)  # [0, 1, ..., 6]
    assert nominal_te_value(values, ETL) == 3.0  # values[ETL // 2]
