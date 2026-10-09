from dataclasses import replace

from params import load_params


def test_discard_duration_is_derived_from_n_frames_discard():
    """params.py sets n_frames_discard; discard_duration = n_frames_discard *
    volume_tr, and Params.n_frames_discard reads it back (never stale)."""
    p = load_params()
    assert p.discard_duration == p.n_frames_discard * p.volume_tr
    warm = replace(p, discard_duration=3 * p.volume_tr)
    assert warm.n_frames_discard == 3
