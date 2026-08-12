"""Port of ../ArbEPI/lib/check_grad_acoustics.m -- checks a gradient
waveform's frequency content against a per-coil table of forbidden
acoustic-resonance bands.

Self-contained FFT/band-energy check; the forbidden-band tables below are
copied verbatim from check_grad_acoustics.m's `case` statements (the
.dat-file paths in its docstring are stale -- the bands are hardcoded in
the function itself, not loaded from a file).
"""

from dataclasses import dataclass

import numpy as np

MAXFREQ = 3e3  # Hz, plotting range only -- unused here (no plotting)
ZF_FAC = 7  # zero-filling factor
THRESHOLD = 0.3

# coil -> [x_bands, y_bands, z_bands], each a list of (low_us, high_us, _unused) echo-spacing bands
_ESP_BANDS_US = {
    'xrmw': [[], [], [(330, 460, 0.0)]],
    'vrmw': [[], [], [(330, 460, 0.0)]],
    'xrm': [[(410, 510, 0.0)], [(410, 510, 0.0)], [(360, 440, 0.0)]],
    'hrmw': [[(420, 459, 0.0), (816, 865, 1.6)], [(420, 459, 0.0), (816, 876, 1.6)], []],
    'hrmb': [[(393, 445, 0.0), (476, 528, 0.0), (952, 988, 0.0)],
             [(393, 445, 0.0), (476, 504, 0.0)],
             [(393, 445, 0.0), (476, 491, 0.0)]],
    'hrmbuhp': [[(350, 445, 0.0), (481, 488, 0.0)],
                [(350, 445, 0.0), (481, 488, 0.0)],
                [(418, 426, 0.0)]],
    'irmw': [[(827, 896, 0.0)], [(827, 896, 0.0)], []],
}


@dataclass
class AcousticsResult:
    g_fresp: np.ndarray  # frequency response, shape (nfft, n2, n3)
    hz: np.ndarray  # frequency axis, shape (nfft,)
    max_in_band: float  # peak |g_fresp| found in any forbidden band, across all axes
    over_threshold: bool


def check_grad_acoustics(grad: np.ndarray, coil: str, gdt: float = 4e-6) -> AcousticsResult:
    """
    grad : gradient waveform(s), shape (n1, n2, n3), T/m, at `gdt` raster.
        n1 = samples per interleave, n2 = interleaves, n3 = axes (2 for 2D,
        3 for 3D imaging) -- matches check_grad_acoustics.m's convention.
    coil : gradient coil code, e.g. 'xrm', 'hrmbuhp' (see scanners.py).
    gdt : gradient update time (s).
    """
    bands_us = _ESP_BANDS_US[coil.lower()]
    grad = grad * 1e3  # T/m -> mT/m

    n1, n2, n3 = grad.shape

    # forbidden bands, in Hz, derived from the echo-spacing bands above
    bands_hz = []
    for axis_bands in bands_us:
        axis_hz = []
        for lo_us, hi_us, _ in axis_bands:
            # esp.m: bands(:,2) = 1/(2e-6*esp(:,1)); bands(:,1) = 1/(2e-6*esp(:,2))
            lo_hz = 1 / (2e-6 * hi_us)
            hi_hz = 1 / (2e-6 * lo_us)
            axis_hz.append((lo_hz, hi_hz))
        bands_hz.append(axis_hz)

    # zero-fill for FFT (must decay to zero to avoid ringing)
    zf1 = ZF_FAC * n1
    if (n1 + zf1) % 2:
        zf1 += 1
    padded = np.zeros((n1 + zf1, n2, n3))
    padded[:n1] = grad

    g_fresp = np.fft.ifftshift(np.fft.ifft(padded, axis=0), axes=0)
    n1p = padded.shape[0]
    hz = (np.arange(-n1p // 2, n1p // 2)) / n1p / gdt

    max_in_band = 0.0
    for lg in range(n3):
        for axis_hz in bands_hz:
            for lo_hz, hi_hz in axis_hz:
                in_band = (hz >= lo_hz) & (hz <= hi_hz)
                if not in_band.any():
                    continue
                magb = np.abs(g_fresp[in_band, :, lg]).max()
                max_in_band = max(max_in_band, float(magb))

    return AcousticsResult(
        g_fresp=g_fresp, hz=hz,
        max_in_band=max_in_band, over_threshold=max_in_band > THRESHOLD,
    )
