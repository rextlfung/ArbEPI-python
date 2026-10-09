"""deGRE's ADC delay follows the gx actually played (review item 181), at
the default crt and at one where trap4ge rounds the ramp up."""

import warnings
from dataclasses import replace

import pytest

from params import load_params
from sequences.deGRE import generate_degre


@pytest.mark.parametrize('crt', [4e-6, 20e-6])
def test_degre_adc_starts_where_gx_flat_top_starts(tmp_path, crt):
    # TR_degre loosened: the coarser raster lengthens the minimum TR.
    p = replace(load_params(output_dir=str(tmp_path)), crt=crt, TR_degre=7e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        seq = generate_degre(p)

    checked = 0
    for n in sorted(seq.block_events):
        b = seq.get_block(n)
        if b.adc is not None and b.gx is not None:
            assert b.adc.delay == pytest.approx(b.gx.delay + b.gx.rise_time, abs=1e-9)
            checked += 1
            if checked >= 4:
                break
    assert checked
