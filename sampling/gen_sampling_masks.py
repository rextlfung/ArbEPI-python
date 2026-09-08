"""Ported from ../ArbEPI/src/gen_sampling_masks.m.

Generates spatiotemporally incoherent 2D (ky, kz) sampling masks, one per
frame, dispatched on `params.sampling_method`.
"""

from typing import Optional

import numpy as np

from params import Params
from sampling.caipi_sample import caipi_sample
from sampling.gen_gaussian_pdf import gen_gaussian_pdf
from sampling.pd_sample import pd_sample
from sampling.rand_sample import rand_sample
from sampling.ticaipi_sample import ticaipi_sample


def gen_sampling_masks(
    R: float,
    params: Params,
    seed_per_frame: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    R : target acceleration factor.
    params : loaded Params (see params.load_params).
    seed_per_frame : if True, reseed a fresh `default_rng(frame_index + 1)`
        every frame, replicating MATLAB's `rng(frame)` (1-based) called
        unconditionally each iteration. If False (default), a single `rng`
        is used continuously across all frames instead. Only affects the
        'pd' and 'rand' sampling methods ('caipi'/'ticaipi' are
        deterministic and use no randomness).
    rng : numpy random Generator to use when `seed_per_frame` is False. If
        omitted, a fresh one is created.

    Returns
    -------
    omegas : (Ny, Nz, Nframes) boolean sampling mask.
    """
    Ny, Nz, Nframes = params.Ny, params.Nz, params.Nframes

    rand_gaussian_sigma = params.rand_gaussian_sigma
    if rand_gaussian_sigma is None:
        rand_gaussian_sigma = np.array([Ny, Nz]) / 6

    if rng is None:
        rng = np.random.default_rng()

    omegas = np.zeros((Ny, Nz, Nframes), dtype=bool)
    for frame in range(Nframes):
        frame_rng = np.random.default_rng(frame + 1) if seed_per_frame else rng

        method = params.sampling_method
        if method == 'caipi':
            omegas[:, :, frame] = caipi_sample([Ny, Nz], R)
        elif method == 'ticaipi':
            omegas[:, :, frame] = ticaipi_sample([Ny, Nz], R, frame)
        elif method == 'pd':
            omegas[:, :, frame] = pd_sample(
                [Ny, Nz],
                R,
                frame_rng,
                calib_frac=params.pd_calib_frac,
                crop_corner=params.pd_crop_corner,
                decay=params.pd_decay,
            )
        elif method == 'rand':
            weights = gen_gaussian_pdf([Ny, Nz], rand_gaussian_sigma)
            omegas[:, :, frame] = rand_sample([Ny, Nz], R, weights, frame_rng)
        else:
            raise ValueError(
                f"Unknown sampling_method {method!r}. Choose 'caipi', 'ticaipi', 'pd', or 'rand'."
            )

    return omegas


def resolve_omegas(
    params: Params,
    seed_per_frame: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Returns params.custom_omegas when a custom sampling mask was
    configured (params.py's custom_mask_path), else generates one via
    gen_sampling_masks using params.R/sampling_method/seed -- the single
    place every call site (main.py, tests, plotting/compare_readout_pns.py)
    should get omegas from. Calling gen_sampling_masks directly instead
    would break on a custom-mask params (sampling_method is None on that
    path -- see params.py's custom_mask_path field comment).
    """
    if params.custom_omegas is not None:
        return params.custom_omegas
    if rng is None:
        rng = np.random.default_rng(params.seed)
    return gen_sampling_masks(params.R, params, seed_per_frame=seed_per_frame, rng=rng)
