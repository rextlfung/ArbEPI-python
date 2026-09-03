"""Static (single-segment) B0 off-resonance correction for recon/'s SENSE
encoding operator -- the first, cheapest stage of a staged plan for adding
B0 correction to GatheredSense (recon/operators.py): a single per-voxel
conjugate-phase demodulation baked into the coil sensitivity maps before the
encoding operator is built, at zero added per-iteration cost. This corrects
the dominant geometric-shift component of EPI off-resonance distortion; it
does not correct the residual blur/ghosting from differential phase accrual
across the echo train -- that needs the full time-segmented correction
stage (recon/operators_b0.py's GatheredSenseB0), and this module was
deliberately implemented first: cheap enough to validate the field map's
sign/scale conventions in isolation before building the more expensive
machinery on top of them.

Sign convention: the forward signal model (Sutton, Noll, Fessler, "Fast,
iterative image reconstruction for MRI in the presence of field
inhomogeneities," IEEE TMI 2003, DOI 10.1109/TSP.2005.853152 -- the same
reference mirtorch.linear.mri.Gmri's own docstring cites) is

    s(t) = integral m(r) exp(i 2*pi*Delta_f(r)*t) exp(-i 2*pi*k(t).r) dr

i.e. the forward operator needs a *positive* exp(+i 2*pi*f(r)*t) phase
multiplied into the image before the spatial-encoding FFT, to reproduce the
extra phase off-resonance actually adds to the acquired signal. Cross-
checked against mirtorch's own Gmri, not just re-derived from the paper:
its demo notebook (examples/demo_mri.ipynb, "Non-Cartesian MRI with B0
correction") passes `zmap=-b0` into Gmri, whose internal mri_exp_approx
builds the per-segment demodulation as exp(-i 2*pi*zmap(r)*t_l) -- compose
the two and that's exp(-i 2*pi*(-b0)*t) = exp(+i 2*pi*b0(r)*t), the same
positive sign used here (mirtorch's `zmap=-b0` negation exists purely to
cancel mri_exp_approx's own internal negative sign, not to flip the
physical convention).

This assumes preprocessing/run_b0map.py's b0map_hz (from MRIFieldmaps.jl,
Lin & Fessler -- the same Fessler lineage as the TSP reference above, so a
priori likely to share this same sign convention by construction) already
follows it. Verified against a real reconstruction, not just assumed:
recon/run_b0_recon.py's real runs (see CLAUDE.md's recon/ B0 subsection)
show the field map reducing, not worsening, distortion, and the
correction's expected geometric-sharpening effect is preserved after the
b0map.jl preconditioner fix (precon=:diag) that separately addressed
field-map-noise-induced speckle. Flip the sign of b0map_hz at the call
site (`demodulate_smaps(smaps, -b0map_hz, te_s)`) if a future dataset's
comparison ever shows this backwards.
"""

import math

import torch


def demodulate_smaps(smaps: torch.Tensor, b0map_hz: torch.Tensor, te_s: float) -> torch.Tensor:
    """smaps: (Nc, *N) complex. b0map_hz: (*N,) real, Hz -- same spatial grid
    as smaps (see preprocessing/run_b0map.py's EPI-grid resize, which makes
    this true for the first time -- b0map_hz used to live on a different,
    coarser deGRE grid). te_s: effective echo time (s) this single-segment
    correction is centered at.

    A single scalar te_s suffices for every frame: this repo's per-echo
    acquisition timing is identical across every shot/frame (see
    sequences/ArbEPI.py's echo_times computation and its uniformity check in
    tests/test_trajectory_matches_schedule.py's
    test_arbepi_schedule_echo_times), so there is exactly one meaningful
    "effective TE" for this whole acquisition, not a per-frame one -- pass
    the prescribed TE (params.TE), or equivalently the mean of scan_info.mat
    schedules[...,2]/preprocessing's echo_times, which are the same value
    here by symmetry (see that same test).

    Returns smaps pre-multiplied by exp(+i*2*pi*b0map_hz*te_s) -- feed this
    into recon/operators.py's build_encoding_operator in place of the raw
    smaps. No other code change is needed: this correction is static
    (time-invariant across frames), so it needs no per-frame handling and
    no change to GatheredSense/build_encoding_operator/reconstruct.py at
    all -- it's entirely absorbed into the smaps array those already take.
    """
    angle = (2 * math.pi * te_s) * b0map_hz.to(torch.float32)
    phasor = torch.exp(1j * angle).to(smaps.dtype)
    return smaps * phasor.unsqueeze(0)
