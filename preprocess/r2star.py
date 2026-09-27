"""R2* (= 1/T2*) map from the deGRE echoes, for recon/'s SENSE_B0_R2star model.

The fit is log-linear over every available echo: ln|S(TE)| = ln S0 - R2* TE,
least squares per voxel. With two echoes this is exactly the two-point
estimate R2* = ln(|S1|/|S2|) / (TE2 - TE1).

LIMITATION -- this map is a placeholder with the current dual-echo deGRE.
The deGRE echo spacing is chosen for B0 mapping (short, so the inter-echo
phase does not wrap), not for relaxometry. On 20260915ball/2_6x_2.4mm,
dTE = 2.24 ms against T2* ~ 47 ms: the signal decays only ~5% between the
echoes (median echo ratio 0.953), so the log-ratio is dominated by noise. The
precision of a two-point fit is sigma(R2*) ~ sqrt(2) / (SNR * dTE): ~6 /s at
SNR 100, i.e. ~30% of R2* here, versus ~0.3 /s if dTE were close to T2*. Any
R2* correction in recon can be no better than this map.

A usable R2* map needs a true multi-echo GRE scan. Guidance from the literature:
- 3D monopolar (flyback) multi-echo spoiled GRE, so every echo shares one
  readout polarity. The ISMRM QSM consensus (MRM 2024,
  doi:10.1002/mrm.30006) recommends monopolar 3D multi-echo GRE with at least
  three echoes, and the same scan then also serves B0 mapping.
- Roughly 4-8 echoes, the first TE as short as possible and the last spanning
  about 1-2 x the target T2* (about 50-90 ms for this phantom; similar for
  brain at 3 T).
- Keep the first echo spacing short enough to unwrap for B0, or unwrap across
  echoes (MRIFieldmaps.jl accepts more than two echoes).
- Keep through-plane voxels thin, or correct for macroscopic B0 gradients,
  which make intra-voxel decay non-exponential and bias R2* upward (Hernando
  et al., MRM 2012, doi:10.1002/mrm.23306; z-shim / voxel-spread-function
  methods).
- Fit with a weighted log-linear fit, ARLO (Pei et al., MRM 2015,
  doi:10.1002/mrm.25137; fast, needs no initial guess) or nonlinear least
  squares with a noise-floor term at low SNR.
This module already fits over all echoes, so such a scan needs no code change
here.
"""

import numpy as np

R2STAR_METHOD = 'log-linear least squares over all echoes (unweighted)'


def fit_r2star(
    img_echoes: np.ndarray, te: np.ndarray, mask: np.ndarray, r2star_max: float = 200.0
) -> np.ndarray:
    """R2* in 1/s from magnitude images [..., n_echoes] at echo times te [n_echoes]
    (s). Zero outside `mask` and wherever any echo is non-positive; clipped to
    [0, r2star_max] (negative R2* is noise, and 200 /s is T2* = 5 ms, far
    shorter than anything real here)."""
    te = np.asarray(te, dtype=np.float64)
    if te.size < 2:
        raise ValueError(f'fit_r2star: need at least 2 echoes, got {te.size}')
    valid = mask & np.all(img_echoes > np.finfo(np.float64).eps, axis=-1)
    log_s = np.log(img_echoes[valid].astype(np.float64))  # [Nvalid, n_echoes]
    tc = te - te.mean()
    slope = (log_s - log_s.mean(axis=-1, keepdims=True)) @ tc / np.sum(tc**2)
    r2star = np.zeros(img_echoes.shape[:-1], dtype=np.float32)
    r2star[valid] = np.clip(-slope, 0.0, r2star_max)
    return r2star
