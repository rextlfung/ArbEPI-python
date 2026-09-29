"""Noise whitening and coil compression.

Whitening (compute_whitening_matrix/apply_whitening) decorrelates the receive
channels and scales each to unit noise variance, using a noise-only scan.

Coil compression is GCC, geometric-decomposition coil compression (Zhang,
Pauly, Vasanawala, Lustig, MRM 2013). The readout (kx) is fully sampled, so the
data can be inverse-Fourier-transformed along kx into hybrid (x, ky, kz) space
and compressed with a different [Nv, Nc] matrix at each x. Only a few coils see
any one x position, so far fewer virtual coils keep the same signal than with
one global (PCA) matrix. The per-x matrices are then rotated to vary smoothly
along x (the paper's alignment step), which keeps virtual-coil images and
k-space kernels sensible without changing the per-x subspaces.

The number of virtual coils comes from select_nvcoils: the smallest Nv whose
kept eigenvalue energy, summed over all x, reaches a fraction of the total.

Every function takes the coil axis last unless stated otherwise.
"""

import numpy as np


def compute_whitening_matrix(noise: np.ndarray) -> np.ndarray:
    """[Ncoils, Ncoils] whitening matrix from a noise-only acquisition.

    `noise` may have any shape ending in the coil axis; all other axes are
    samples. Equivalent in effect to BART's `whiten -n`.
    """
    x = noise.reshape(-1, noise.shape[-1])  # [Nsamples, Ncoils]
    # psi = E[conj(c_i) c_j] = conj(Psi), the transpose of the standard
    # covariance Psi = E[c c^H]; see apply_whitening for why W is applied as
    # conj(W).
    psi = (x.conj().T @ x) / x.shape[0]
    L = np.linalg.cholesky(psi)  # psi = L L^H, so conj(L) is Psi's Cholesky factor
    return np.linalg.inv(L)


def apply_whitening(data: np.ndarray, W: np.ndarray, coil_axis: int = -1) -> np.ndarray:
    """Apply a whitening matrix along `coil_axis`. Each coil vector x becomes
    conj(W) @ x, the whitener for the standard covariance (see
    compute_whitening_matrix); in row layout that is data @ W^H."""
    data = np.moveaxis(data, coil_axis, -1)
    whitened = data @ W.conj().T
    return np.moveaxis(whitened, -1, coil_axis)


def select_nvcoils(evals: np.ndarray, energy_thresh: float) -> int:
    """Smallest Nv whose top-Nv eigenvalues hold at least `energy_thresh` of the
    total energy. evals: [Nx, Nc] (one row per x) or [Nc], each row sorted in
    descending order. The energy is summed over x, so x positions with little
    signal count for little -- no object mask needed."""
    evals = np.clip(np.atleast_2d(evals), 0, None)
    kept = np.cumsum(evals.sum(axis=0))
    return int(np.searchsorted(kept / kept[-1], energy_thresh - 1e-12) + 1)


def _center_crop_pad(a: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    """Crop or zero-pad `a` to length n along `axis`, keeping index N//2 (the
    k = 0 sample of centered k-space) at index n//2."""
    a = np.moveaxis(a, axis, 0)
    N = a.shape[0]
    out = np.zeros((n,) + a.shape[1:], dtype=a.dtype)
    lo = min(N // 2, n // 2)
    hi = min(N - N // 2, n - n // 2)
    out[n // 2 - lo:n // 2 + hi] = a[N // 2 - lo:N // 2 + hi]
    return np.moveaxis(out, 0, axis)


def _ifftc_x(ksp: np.ndarray) -> np.ndarray:
    """Centered, unitary inverse FFT along axis 0 (kx -> x)."""
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ksp, axes=0), axis=0, norm='ortho'), axes=0)


def _fftc_x(img: np.ndarray) -> np.ndarray:
    """Centered, unitary forward FFT along axis 0 (x -> kx)."""
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=0), axis=0, norm='ortho'), axes=0)


def gcc_calibration(ksp: np.ndarray, nx: int, calib_size: int = 24) -> np.ndarray:
    """Hybrid-space calibration data for GCC.

    ksp: [Nx_src, Ny, Nz, Nc] fully sampled, centered k-space (the whitened
        deGRE), covering the same x FOV as the target grid.
    nx: readout length of the grid the compression will be applied on (EPI Nx).
        kx is cropped or zero-padded to nx, so after the inverse FFT along kx
        the x positions are exactly the target grid's.
    Returns [nx, M, Nc]: at each x, the central calib_size x calib_size (ky, kz)
    block (the paper uses the ACS region), flattened to M samples.
    """
    hyb = _ifftc_x(_center_crop_pad(ksp, nx, axis=0))
    _, Ny, Nz, Nc = hyb.shape
    cy, cz = min(calib_size, Ny), min(calib_size, Nz)
    blk = hyb[:, Ny // 2 - cy // 2:Ny // 2 - cy // 2 + cy, Nz // 2 - cz // 2:Nz // 2 - cz // 2 + cz]
    return blk.reshape(nx, -1, Nc)


def gcc_compression(calib: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(A0, evals) from gcc_calibration's [Nx, M, Nc] data.

    A0: [Nx, Nc, Nc]; A0[x, :Nv] is x's [Nv, Nc] compression matrix: rows are
        the conjugated top eigenvectors (u^H) of that x's coil covariance
        sum(c c^H), so a coil vector c compresses to A0[x, :Nv] @ c (BART's
        `cc -A -M` convention). Not yet aligned across x -- see align_gcc.
    evals: [Nx, Nc] eigenvalues, descending, for select_nvcoils.
    """
    # Coil covariance per x: sum over samples of c c^H.
    cov = np.einsum('xmc,xmd->xcd', calib, calib.conj()) / calib.shape[1]
    evals, evecs = np.linalg.eigh(cov)  # ascending, batched over x
    A0 = np.conj(np.swapaxes(evecs[..., ::-1], -1, -2))
    return A0, evals[:, ::-1]


def align_gcc(A: np.ndarray, energy: np.ndarray | None = None) -> np.ndarray:
    """Rotate each x's [Nv, Nc] compression matrix within its own subspace so
    the virtual coils change smoothly along x (Zhang et al. 2013, "virtual coil
    alignment"). Starts from the x with the most energy (or the center) and
    works outward: at each step A[x] <- P A[x], with P the unitary closest to
    mapping A[x] onto its already-aligned neighbor (orthogonal Procrustes).
    The row space of each A[x] -- and so the compression itself -- is unchanged.
    """
    A = A.copy()
    Nx = A.shape[0]
    x0 = int(np.argmax(energy)) if energy is not None else Nx // 2

    def step(x, ref):
        U, _, Vh = np.linalg.svd(A[ref] @ A[x].conj().T)
        A[x] = (U @ Vh) @ A[x]

    for x in range(x0 + 1, Nx):
        step(x, x - 1)
    for x in range(x0 - 1, -1, -1):
        step(x, x + 1)
    return A


def apply_gcc_kspace(ksp: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Compress k-space with per-x matrices A [Nx, Nv, Nc]: centered unitary
    inverse FFT along kx (axis 0), A[x] at each x, forward FFT back.
    ksp: [Nx, ..., Nc] -> [Nx, ..., Nv]. Exact for SENSE whenever the sampling
    pattern does not depend on kx: compressed data then equal the forward model
    of maps compressed with apply_gcc_image."""
    hyb = _ifftc_x(ksp)
    return _fftc_x(np.einsum('xvc,x...c->x...v', A, hyb))


def apply_gcc_image(img: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Compress image-domain data (e.g. sensitivity maps) [Nx, ..., Nc] with
    per-x matrices A [Nx, Nv, Nc] -> [Nx, ..., Nv]."""
    return np.einsum('xvc,x...c->x...v', A, img)
