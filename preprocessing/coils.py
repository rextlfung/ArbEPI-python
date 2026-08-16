"""Noise whitening and PCA coil compression, replacing BART's `whiten`/`cc`/
`ccapply` (see CLAUDE.md for why BART was dropped in favor of plain numpy +
sigpy). Ports the whitening/compression logic embedded in preprocess.m
(there is no separate MATLAB source file for this -- it's inline BART calls
plus a manual eigenvalue computation for Nvcoils selection).

Convention: every function takes `coil_axis` (default -1) rather than
assuming a fixed array layout, since preprocess.py applies these to GRE
[Nx,Ny,Nz,Ncoils], cal [Nfid,ETL,N,Ncoils], and per-frame EPI arrays with
different shapes but coils always last (matching the MATLAB permute(...,
[1 3 4 2])-to-coils-last convention already used throughout preprocess.m).
"""

import numpy as np


def compute_whitening_matrix(noise: np.ndarray) -> np.ndarray:
    """[Ncoils, Ncoils] whitening matrix from a noise-only acquisition.

    `noise` may have any shape ending in the coil axis (last axis); all
    other axes are treated as samples. Decorrelates channels and normalizes
    each to unit noise variance -- the standard MRI prewhitening procedure,
    equivalent in effect to BART's `whiten -n`.
    """
    x = noise.reshape(-1, noise.shape[-1])  # [Nsamples, Ncoils]
    psi = (x.conj().T @ x) / x.shape[0]  # [Ncoils, Ncoils] Hermitian covariance
    L = np.linalg.cholesky(psi)  # psi = L @ L^H
    return np.linalg.inv(L)  # W: Cov(W x) = I


def apply_whitening(data: np.ndarray, W: np.ndarray, coil_axis: int = -1) -> np.ndarray:
    """Apply a [Ncoils, Ncoils] whitening matrix along `coil_axis`.

    For a single sample (coil vector) x, the whitened sample is W @ x. Batched
    over rows (samples), that's `data @ W.conj().T` rather than `data @ W.T`
    -- W is applied to column vectors, and transposing that onto the *row*
    layout used here picks up a conjugate (W^H, not W^T); verified against
    `test_whitening_decorrelates_and_normalizes`.
    """
    data = np.moveaxis(data, coil_axis, -1)
    whitened = data @ W.conj().T
    return np.moveaxis(whitened, -1, coil_axis)


def compute_coil_covariance(data: np.ndarray, coil_axis: int = -1) -> np.ndarray:
    """[Ncoils, Ncoils] Hermitian sample covariance, coils pooled over every
    other axis. Ports preprocess.m's `C = (ksp_mat*ksp_mat')/size(ksp_mat,2)`."""
    x = np.moveaxis(data, coil_axis, 0).reshape(data.shape[coil_axis], -1)  # [Ncoils, Nsamples]
    return (x @ x.conj().T) / x.shape[1]


def select_nvcoils(cov: np.ndarray, energy_thresh: float, floor: int) -> tuple[int, int]:
    """Choose how many virtual coils to keep from a coil covariance matrix.

    Ports preprocess.m's Nvcoils selection: keep components until cumulative
    explained variance reaches `energy_thresh`, subject to a `floor` (2R)
    lower bound. Returns (Nvcoils, Nvcoils_energy) -- the second value is the
    pre-floor count, kept for logging/plotting parity with preprocess.m.
    """
    ncoils = cov.shape[0]
    evals = np.linalg.eigvalsh(cov)[::-1]  # descending; eigvalsh guarantees real
    cumvar = np.cumsum(evals) / np.sum(evals)
    nvcoils_e = int(np.searchsorted(cumvar, energy_thresh) + 1)
    nvcoils = min(max(nvcoils_e, floor), ncoils)
    return nvcoils, nvcoils_e


def coil_compression_matrix(cov: np.ndarray, nvcoils: int) -> np.ndarray:
    """[Nvcoils, Ncoils] PCA compression matrix: top-`nvcoils` eigenvectors of
    `cov` by descending eigenvalue. Ports BART's `cc -p Nvcoils -A -M`."""
    evals, evecs = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(evals)[::-1][:nvcoils]
    return evecs[:, order].conj().T  # [Nvcoils, Ncoils]


def apply_coil_compression(data: np.ndarray, V: np.ndarray, coil_axis: int = -1) -> np.ndarray:
    """Apply a [Nvcoils, Ncoils] compression matrix along `coil_axis`.
    Ports BART's `ccapply`."""
    data = np.moveaxis(data, coil_axis, -1)
    compressed = data @ V.T
    return np.moveaxis(compressed, -1, coil_axis)
