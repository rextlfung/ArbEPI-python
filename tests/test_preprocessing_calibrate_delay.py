from preprocessing.calibrate_delay import _matlab_round, select_best_delay


def test_select_best_delay_picks_zero_wrap_closest_to_zero_a2():
    report = {
        'delay': [-1.0, -0.5, 0.0, 0.5, 1.0],
        'a1': [0.0, 0.0, 0.0, 0.0, 0.0],
        'a2': [0.9, -0.2, 0.05, 0.3, 1.1],
        'wrap_count': [2, 0, 0, 0, 3],
    }
    best = select_best_delay(report)
    assert best == 0.0  # a2=0.05 is closest to zero among the zero-wrap candidates


def test_select_best_delay_falls_back_to_fewest_wraps_if_none_safe():
    report = {
        'delay': [-1.0, 0.0, 1.0],
        'a1': [0.0, 0.0, 0.0],
        'a2': [5.0, -5.0, 0.1],
        'wrap_count': [3, 1, 2],
    }
    best = select_best_delay(report)
    assert best == 0.0  # delay index 1 has the fewest wraps (1)


def test_matlab_round_rounds_half_away_from_zero():
    # Nx=90 is a real value used in this project (see README) and lands
    # exactly on a .5 tie for Nx/4 -- Python's banker's-rounding round()
    # would give 22 here, MATLAB's round() gives 23.
    assert _matlab_round(90 / 4) == 23
    assert _matlab_round(3 * 90 / 4) == 68
    assert _matlab_round(2.4) == 2
    assert _matlab_round(2.5) == 3
