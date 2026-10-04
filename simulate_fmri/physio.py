"""Physiological noise: signal fluctuations that scale with the signal.

Kruger & Glover (Magn Reson Med 2001;46:631) split the temporal noise of a
resting fMRI series into thermal noise, which does not depend on the signal S,
and physiological noise sigma_P = lambda * S, itself made of a BOLD-like part
that grows with echo time (fluctuations of R2*) and a part that does not
(pulsation, respiration, drift):

    sigma_P^2 = (sigma_R * TE * S)^2 + (lambda_NB * S)^2.

The sizes here are set from Bodurka et al. (NeuroImage 2007;34:542, 3 T,
TE 45 ms, 16-channel array), whose temporal-SNR ceilings of 78, 117 and 47 in
gray matter, white matter and CSF mean lambda = 0.0128, 0.0085 and 0.021.
How lambda divides between the two parts is not measured there: a
TE-independent lambda_NB of 0.004 in brain tissue is assumed, which leaves
sigma_R = 0.27 1/s in gray matter and 0.17 1/s in white matter, and CSF is
taken as all TE-independent (pulsation).

Every component is sampled once per excitation, not once per frame: the shots
of a multi-shot 3D-EPI frame see different states of the same fluctuation,
which is what makes it more than a scaling of each image.

- BOLD-like ('r2s' modes): a few smooth random spatial patterns over gray and
  white matter, each with a 1/f time course between 0.01 and 0.1 Hz.
- Cardiac ('amp' mode): a quasi-periodic waveform near 1.1 Hz with a second
  harmonic, strongest in CSF.
- Respiratory: a quasi-periodic waveform near 0.25 Hz that modulates the
  signal slightly ('amp' mode) and, mainly, shifts the field, because the
  susceptibility of the chest changes with breathing (Raj et al., Phys Med
  Biol 2001;46:3331). Van de Moortele et al. (Magn Reson Med 2002;47:888)
  measured 1.45-4 Hz in the brain at 7 T, larger toward the chest; scaled by
  field that is 0.6-1.7 Hz at 3 T. The defaults are a common offset of 1.25 Hz
  peak to peak plus a gradient along z of 0.11 Hz/cm peak to peak. The offset
  is applied exactly, as a phase on every sample; the gradient is a 'freq'
  mode.
- Drift ('amp' mode): a slow change of the whole signal, 0.3% over the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from .forward import Mode


@dataclass
class PhysioConfig:
    """Sizes of the components (see the module docstring). scale multiplies
    all of them; 0 turns physiological noise off."""

    scale: float = 1.0
    sigma_r2s_gm: float = 0.27  # 1/s, rms R2* fluctuation in gray matter
    wm_ratio: float = 0.62  # white-matter R2* fluctuation relative to gray
    n_patterns: int = 4  # spatial patterns of the BOLD-like part
    pattern_fwhm_mm: float = 30.0
    band_hz: tuple[float, float] = (0.01, 0.1)
    cardiac_csf: float = 0.02  # rms fractional fluctuation in CSF
    cardiac_gm: float = 0.003
    cardiac_wm: float = 0.002
    cardiac_hz: float = 1.1
    resp_hz: float = 0.25
    resp_amp: float = 0.002  # rms fractional fluctuation, all tissue
    resp_offset_hz: float = 0.625  # amplitude of the common field offset (1.25 Hz peak to peak)
    resp_gradient_hz_per_mm: float = 0.0055  # amplitude along z (0.11 Hz/cm peak to peak)
    drift: float = 0.003  # fractional change over the run


def quasi_periodic(t: NDArray, hz: float, rng: np.random.Generator, jitter: float = 0.06,
                   harmonic: float = 0.0) -> NDArray[np.float64]:
    """A unit-amplitude oscillation near `hz` whose rate wanders by `jitter`
    (relative, slowly), plus `harmonic` of its second harmonic."""
    dt = np.gradient(t) if len(t) > 1 else np.array([1.0])
    slow = np.cumsum(rng.normal(size=len(t)) * np.sqrt(dt))  # random walk
    slow = slow - np.linspace(slow[0], slow[-1], len(t))
    slow = slow / (np.abs(slow).max() + 1e-12)
    rate = hz * (1 + jitter * slow)
    phase = 2 * np.pi * np.cumsum(rate * dt) + rng.uniform(0, 2 * np.pi)
    return np.cos(phase) + harmonic * np.cos(2 * phase + 0.6)


def band_noise(t: NDArray, band_hz: tuple[float, float], rng: np.random.Generator,
               n: int) -> NDArray[np.float64]:
    """(n, len(t)) unit-variance noise with a 1/f power spectrum inside
    band_hz and nothing outside, on the uniform time axis t."""
    nt = len(t)
    dt = (t[-1] - t[0]) / max(nt - 1, 1) if nt > 1 else 1.0
    f = np.fft.rfftfreq(nt, dt)
    keep = (f >= band_hz[0]) & (f <= band_hz[1])
    if not keep.any():  # a run too short to hold the band: its lowest frequency
        keep = np.zeros_like(keep)
        keep[min(1, len(f) - 1)] = True
    amp = np.where(keep, 1 / np.sqrt(np.maximum(f, f[1] if len(f) > 1 else 1.0)), 0.0)
    spec = (rng.normal(size=(n, len(f))) + 1j * rng.normal(size=(n, len(f)))) * amp
    x = np.fft.irfft(spec, n=nt, axis=1)
    return x / np.maximum(x.std(axis=1, keepdims=True), 1e-30)


def smooth_patterns(shape: tuple[int, int, int], voxel_mm: tuple[float, float, float],
                    fwhm_mm: float, n: int, rng: np.random.Generator,
                    device) -> torch.Tensor:
    """(n, X, Y, Z) smooth random fields: white noise on a grid of spacing
    fwhm / 2, interpolated to `shape`."""
    coarse = [max(2, int(round(s * v / (fwhm_mm / 2))) + 1) for s, v in zip(shape, voxel_mm)]
    z = torch.as_tensor(rng.normal(size=(n, 1, *coarse)), dtype=torch.float32, device=device)
    out = torch.nn.functional.interpolate(z, size=shape, mode='trilinear', align_corners=True)
    return out[:, 0]


def physio(
    tissues: dict[str, torch.Tensor],
    z_mm: NDArray,
    voxel_mm: tuple[float, float, float],
    t: NDArray,
    group_names: tuple[str, ...],
    seed: int = 0,
    cfg: PhysioConfig | None = None,
) -> tuple[list[Mode], NDArray[np.float64], dict[str, NDArray]]:
    """Physiological-noise modes for forward.epi_readouts.

    tissues: fine-grid tissue fractions by name ('gm', 'wm', 'csf').
    z_mm: (Z,) fine-grid z coordinates, mm from isocenter (superior positive).
    t: (n_excitations,) time of each excitation, s, uniformly spaced.
    group_names: the order of the tissue groups in forward.Spins.

    Returns (modes, frequency, courses): the modes, the common field offset
    per excitation (Hz), and the named time courses, for the truth file.
    """
    cfg = cfg or PhysioConfig()
    rng = np.random.default_rng(seed)
    device = tissues['gm'].device
    shape = tuple(tissues['gm'].shape)
    n_exc = len(t)
    if cfg.scale == 0:
        return [], np.zeros(n_exc), {}

    def groups(**coef) -> torch.Tensor:
        return torch.tensor([coef.get(name, 0.0) for name in group_names], dtype=torch.float32)

    one = torch.ones(shape, dtype=torch.float32, device=device)
    brain = (tissues['gm'] + tissues['wm']) > 0.5
    modes: list[Mode] = []
    courses: dict[str, NDArray] = {}

    # BOLD-like: smooth patterns, unit rms over the brain when summed in quadrature
    patterns = smooth_patterns(shape, voxel_mm, cfg.pattern_fwhm_mm, cfg.n_patterns, rng, device)
    norm = torch.sqrt((patterns**2).sum(0)[brain].mean()) if brain.any() else torch.tensor(1.0)
    patterns = patterns / norm
    slow = band_noise(t, cfg.band_hz, rng, cfg.n_patterns) * cfg.sigma_r2s_gm * cfg.scale
    for k in range(cfg.n_patterns):
        modes.append(Mode('r2s', patterns[k], slow[k], groups(gm=1.0, wm=cfg.wm_ratio),
                          name=f'bold_like_{k}'))
        courses[f'bold_like_{k}'] = slow[k]

    cardiac = quasi_periodic(t, cfg.cardiac_hz, rng, harmonic=0.4)
    cardiac = cardiac / cardiac.std() * cfg.scale
    modes.append(Mode('amp', one, cardiac,
                      groups(csf=cfg.cardiac_csf, gm=cfg.cardiac_gm, wm=cfg.cardiac_wm),
                      name='cardiac'))
    courses['cardiac'] = cardiac

    resp = quasi_periodic(t, cfg.resp_hz, rng, jitter=0.1)
    modes.append(Mode('amp', one, resp / resp.std() * cfg.resp_amp * cfg.scale, name='respiration'))
    gradient = -torch.as_tensor(z_mm, dtype=torch.float32, device=device)[None, None, :] * one
    modes.append(Mode('freq', gradient, resp * cfg.resp_gradient_hz_per_mm * cfg.scale,
                      name='respiration_gradient'))
    frequency = resp * cfg.resp_offset_hz * cfg.scale
    courses['respiration'] = resp

    span = (t - t[0]) / max(t[-1] - t[0], 1e-9) - 0.5
    modes.append(Mode('amp', one, span * cfg.drift * cfg.scale, name='drift'))
    courses['drift'] = span * cfg.drift * cfg.scale
    return modes, frequency, courses
