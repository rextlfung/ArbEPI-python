"""The start of every shot, shared by sequences/ArbEPI.py and
sequences/EPIcal.py (docs/review-findings.md item 244): the per-shot random
spoiler draw, the fat-sat block and its crusher (absent for water
excitation), the quadratic RF-spoiling phase, and the slab-selective
excitation with its slice-select rephaser.
"""

from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pypulseq as pp

from params import Params


def draw_spoil_scales(params: Params, spoil_rng: np.random.Generator) -> Tuple[float, float, float]:
    """One shot's random spoiler cycles/voxel, independent per axis, as
    fractions of the maximum the spoiler trapezoids were built for
    (params.spoil_cycles_max). Used for both the pre-excitation crusher and
    the post-readout spoiler of the same shot."""
    cx, cy, cz = spoil_rng.uniform(params.spoil_cycles_min, params.spoil_cycles_max, size=3)
    return (
        cx / params.spoil_cycles_max,
        cy / params.spoil_cycles_max,
        cz / params.spoil_cycles_max,
    )


def add_fatsat_and_excitation(
    seq: pp.Sequence,
    params: Params,
    rf: SimpleNamespace,
    adc: SimpleNamespace,
    gz_ss: SimpleNamespace,
    gz_ssr: SimpleNamespace,
    rfsat: SimpleNamespace | None,
    gx_spoil: SimpleNamespace,
    gy_spoil: SimpleNamespace,
    gz_spoil: SimpleNamespace,
    scales: Tuple[float, float, float],
    trid_value: int,
    rf_count: int,
) -> int:
    """Add [fat-sat + crusher], excitation and slice-select rephaser blocks.

    The first block of the shot carries the GE TRID label: the fat-sat
    block, or the excitation when there is no fat-sat (rfsat None). Sets the
    RF/ADC phase offsets from the quadratic RF-spoiling schedule and returns
    the incremented rf_count."""
    trid = pp.make_label('TRID', 'SET', trid_value)
    if rfsat is not None:
        seq.add_block(
            rfsat if params.fatsat.enabled else pp.make_delay(pp.calc_duration(rfsat)),
            trid,
        )
        x_scale, y_scale, z_scale = scales
        seq.add_block(
            pp.scale_grad(gx_spoil, x_scale),
            pp.scale_grad(gy_spoil, y_scale),
            pp.scale_grad(gz_spoil, z_scale),
        )

    # RF spoiling (quadratic phase cycling)
    rf_phase = (0.5 * params.rf_phase_0 * rf_count**2) % 360.0
    rf.phase_offset = rf_phase / 180 * np.pi
    adc.phase_offset = rf_phase / 180 * np.pi

    # Slab-selective excitation + slice-select rephaser
    if rfsat is None:
        seq.add_block(rf, gz_ss, trid)
    else:
        seq.add_block(rf, gz_ss)
    seq.add_block(gz_ssr)
    return rf_count + 1
