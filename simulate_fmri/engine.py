"""SNAKE acquisition engine for ArbEPI shots.

A subclass of SNAKE's EPIAcquisitionEngine that replaces only the per-shot
signal model (`_job_model_simple`, `_job_model_T2s`). The driver (`__call__`:
MRD file creation, handlers, worker processes, k-space noise) is SNAKE's.

Three things differ from the stock kernels:

1. T2* decay follows the sequence's own echo times. The stock model spaces all
   samples of a shot one dwell time apart and puts TE at the sample nearest the
   k-space center of the first shot. Here every echo decays by
   exp(-(t - TE) / T2*) with t read from the acquisition header (the echo time
   sequences/ArbEPI.py saved in scan_info.mat, see sampler.py), plus the sample's
   offset within the readout. Tissues that share a T2* are summed before the
   FFT, so the activation ROI (a copy of gray matter) costs no extra transform.

2. The FFT is centered the way recon/operators.py's SENSE is,
   fftshift(fftn(ifftshift(x))): index N//2 is the image center and k = 0 on
   every axis. SNAKE's own helper swaps the two shifts, which is the same thing
   for even N but one sample off, in both domains, for odd N.

3. The phantom is resampled only when it is not already on the acquisition
   grid. SNAKE resamples unconditionally; without cupy that starts a process
   pool per shot (about 1 s) to copy an array that needs no change. A handler
   that moves the phantom (motion) changes its affine, so it is still
   resampled.

Everything is computed on the CPU with scipy.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import NamedTuple

import ismrmrd as mrd
import numpy as np
import scipy.fft
from numpy.typing import NDArray
from snake.core.engine import EPIAcquisitionEngine
from snake.core.phantom import DynamicData, Phantom, PropTissueEnum
from snake.core.simulation import SimConfig
from snake.mrd_utils import MRDLoader

from .sampler import ECHO_TIME_SLOT

_AXES = (-3, -2, -1)


class Shots(NamedTuple):
    """A chunk of shots, as read back from the MRD file."""

    traj: NDArray  # (n_shots, ETL, Nx, 3) k-space indices, in acquisition order
    echo_times_ms: NDArray  # (n_shots, ETL) ms since excitation


def fft3c(image: NDArray, workers: int = 1) -> NDArray:
    """Centered, orthonormal 3D FFT over the last three axes (image center and
    k = 0 both at index N//2)."""
    return scipy.fft.fftshift(
        scipy.fft.fftn(
            scipy.fft.ifftshift(image, axes=_AXES), axes=_AXES, norm='ortho', workers=workers
        ),
        axes=_AXES,
    )


def phantom_state(
    phantom: Phantom, dyn_datas: list[DynamicData], i: int, sim_conf: SimConfig
) -> Phantom:
    """The phantom at excitation `i` of the chunk, on the acquisition grid."""
    state = deepcopy(phantom)
    for dyn in dyn_datas:
        state = dyn.func(state, dyn.data, i)
    on_grid = tuple(state.anat_shape) == tuple(sim_conf.shape) and np.allclose(
        state.affine, sim_conf.fov.affine, atol=1e-4
    )
    if not on_grid:
        state = state.resample(new_affine=sim_conf.fov.affine, new_shape=sim_conf.shape)
    return state


def _sample(ksp: NDArray, shot: NDArray) -> NDArray:
    """(Nc, X, Y, Z) k-space at one shot's (ETL, Nx, 3) indices -> (Nc, ETL, Nx)."""
    return ksp[:, shot[..., 0], shot[..., 1], shot[..., 2]]


class ArbEPIAcquisitionEngine(EPIAcquisitionEngine):
    """Acquire ArbEPISampler shots.

    Parameters
    ----------
    model : 'T2s' (decay along the echo train, the default) or 'simple' (every
        sample sees the contrast at TE).
    snr : SNAKE's noise level: complex white noise is added to every coil's
        k-space with variance 2 * mean(image^2) / snr, image being the
        noise-free magnitude at TE over the whole FOV. np.inf adds none.
    """

    __engine_name__ = 'ArbEPI'
    __mp_mode__ = 'forkserver'
    model: str = 'T2s'
    snr: float = np.inf
    slice_2d: bool = False

    def _job_trajectories(
        self,
        data_loader: MRDLoader,
        hdr: mrd.xsd.ismrmrdHeader,
        sim_conf: SimConfig,
        chunk: int | Sequence[int],
    ) -> Shots:
        if not isinstance(chunk, Sequence):
            chunk = [chunk]
        limits = hdr.encoding[0].encodingLimits
        etl = limits.kspace_encoding_step_1.maximum
        nx = limits.kspace_encoding_step_0.maximum
        acq = data_loader._dataset['data'][chunk[0] * etl : (chunk[-1] + 1) * etl]
        traj = acq['traj'].reshape(len(chunk), etl, nx, 3).astype(np.intp)
        echo_times_ms = acq['head']['user_float'][:, ECHO_TIME_SLOT].reshape(len(chunk), etl)
        return Shots(traj, echo_times_ms.astype(np.float64))

    @staticmethod
    def _job_model_simple(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: Shots,
        slice_2d: bool = False,
        fft_workers: int = 1,
    ) -> NDArray[np.complex64]:
        """Every sample sees the contrast at TE (no decay)."""
        if slice_2d:
            raise NotImplementedError('ArbEPI is a 3D sequence: slice_2d is not supported')
        traj = trajectories.traj
        out = np.zeros((len(traj), sim_conf.hardware.n_coils, *traj.shape[1:3]), np.complex64)
        for i, shot in enumerate(traj):
            state = phantom_state(phantom, dyn_datas, i, sim_conf)
            image = state.contrast(sim_conf=sim_conf, resample=False, aggregate=True)[None]
            if state.smaps is not None:
                image = image * state.smaps
            out[i] = _sample(fft3c(image, fft_workers), shot)
        return out

    @staticmethod
    def _job_model_T2s(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: Shots,
        slice_2d: bool = False,
        fft_workers: int = 1,
    ) -> NDArray[np.complex64]:
        """Each tissue decays as exp(-(t - TE) / T2*) through the echo train."""
        if slice_2d:
            raise NotImplementedError('ArbEPI is a 3D sequence: slice_2d is not supported')
        traj = trajectories.traj
        n_shots, etl, nx, _ = traj.shape
        # Time of each stored sample: its echo time plus its place in the
        # readout (stored samples are in acquisition order for both polarities).
        within = (np.arange(nx) - (nx - 1) / 2) * sim_conf.hardware.dwell_time_ms
        t = trajectories.echo_times_ms[:, :, None] + within - sim_conf.seq.TE

        t2s = phantom.props[:, PropTissueEnum.T2s]
        groups = [(v, np.flatnonzero(t2s == v)) for v in np.unique(t2s)]

        out = np.zeros((n_shots, sim_conf.hardware.n_coils, etl, nx), np.complex64)
        for i, shot in enumerate(traj):
            state = phantom_state(phantom, dyn_datas, i, sim_conf)
            tissues = state.contrast(sim_conf=sim_conf, resample=False, aggregate=False)
            for value, members in groups:
                image = tissues[members].sum(axis=0)[None]
                if state.smaps is not None:
                    image = image * state.smaps
                decay = np.exp(-t[i] / value).astype(np.float32)
                out[i] += _sample(fft3c(image, fft_workers), shot) * decay
        return out
