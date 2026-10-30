from dataclasses import replace

from params import load_params


def test_discard_duration_is_derived_from_n_frames_discard():
    """params.py sets n_frames_discard; discard_duration = n_frames_discard *
    volume_tr, and Params.n_frames_discard reads it back (never stale)."""
    p = load_params()
    assert p.discard_duration == p.n_frames_discard * p.volume_tr
    warm = replace(p, discard_duration=3 * p.volume_tr)
    assert warm.n_frames_discard == 3


def test_n_dummy_shots_is_the_same_warmup_in_shots():
    """EPIcal plays the discard time as dummy shots; the count follows
    discard_duration (never stale) and is about n_frames_discard * Nshots."""
    p = load_params()
    assert p.n_dummy_shots == round(p.n_frames_discard * p.Nshots)
    assert replace(p, discard_duration=0.0).n_dummy_shots == 0
