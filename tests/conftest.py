import warnings
from dataclasses import replace

import numpy as np
import pytest

from params import load_params
from sampling.gen_sampling_masks import gen_sampling_masks
from sequences.ArbEPI import generate_arbepi
from sequences.noise import generate_noise


@pytest.fixture(scope='session')
def built_seq_dir(tmp_path_factory):
    """Builds ArbEPI.seq + noise.seq (default params, Nframes=1 for speed)
    into a session-scoped tmp directory, once. Tests that need a real
    on-disk .seq file used to read from output/ and pytest.skip if it
    wasn't there -- which is every fresh checkout/CI run, since output/ is
    gitignored (docs/review-findings.md item 46). Building here instead
    means those tests always actually run."""
    out_dir = tmp_path_factory.mktemp('built_seqs')
    p = replace(load_params(output_dir=str(out_dir)), Nframes=1, seed=0)
    omegas = gen_sampling_masks(p.R, p, rng=np.random.default_rng(p.seed))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        generate_arbepi(omegas, p, seqname='ArbEPI')
        generate_noise(p, seqname='noise')
    return out_dir
