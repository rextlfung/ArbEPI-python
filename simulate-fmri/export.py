"""SNAKE's .mrd -> the files recon/ and analyze.py read.

export() writes two files from a finished simulation:

<outdir>/recon/<name>_preprocessed.h5
    The layout preprocess/ produces for real scans, so `python -m recon.rss`
    and `python -m recon.sense` run on a simulation unchanged: ksp_epi_zf
    (Nx, Ny, Nz, Ncoils, Nframes) zero-filled k-space, omegas and echo_times
    (Ny, Nz, Nframes), smaps (Nx, Ny, Nz, Ncoils), and the attrs noise_var,
    whitened, t_ref_s, fov and seqname. The sensitivity maps are the ones the
    simulation used, not maps estimated from calibration data. There is no
    b0_map or r2star_map: SNAKE models no off-resonance.

<outdir>/<name>_truth.h5
    What the simulation knows and a scan would not: the activated region
    (roi), its BOLD time course per frame (bold) and per excitation
    (bold_shots), the noise-free image without activation (image_ref), the
    peak fractional signal change (psc_peak), the tissue maps and a brain mask.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
from numpy.typing import NDArray
from snake.mrd_utils import CartesianFrameDataLoader

ROI = 'ROI'  # the tissue SNAKE's activation handlers add


def schedule_maps(
    schedules: NDArray, echo_times_s: NDArray, ny: int, nz: int
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """(omegas, echo_times), both (Ny, Nz, Nframes): the sampling mask and each
    sampled location's time since excitation in s (0 where unsampled), as in
    preprocess/preprocess.py."""
    n_frames = schedules.shape[0]
    omegas = np.zeros((ny, nz, n_frames), dtype=bool)
    times = np.zeros((ny, nz, n_frames), dtype=np.float64)
    per_echo = np.broadcast_to(echo_times_s, schedules.shape[1:3]).ravel()
    for frame in range(n_frames):
        iy, iz = schedules[frame, :, :, 0].ravel(), schedules[frame, :, :, 1].ravel()
        omegas[iy, iz, frame] = True
        times[iy, iz, frame] = per_echo
    return omegas, times


def noise_variance(image: NDArray, snr: float) -> float | None:
    """E|n|^2 of the complex noise SNAKE's engine adds to each k-space sample:
    real and imaginary parts are drawn separately with variance
    mean(image^2) / snr. None when snr is infinite (no noise)."""
    if not np.isfinite(snr) or snr <= 0:
        return None
    return float(2 * np.mean(image**2) / snr)


def export(fn_mrd: str, protocol, outdir: str, name: str, snr: float) -> tuple[str, str]:
    """Write <outdir>/recon/<name>_preprocessed.h5 and <outdir>/<name>_truth.h5
    from the SNAKE file `fn_mrd`. protocol: simulate.Protocol. snr: the
    engine's. Returns the two paths."""
    nx, ny, nz = protocol.shape
    n_frames, n_shots = protocol.n_frames, protocol.n_shots
    omegas, echo_times = schedule_maps(protocol.schedules, protocol.echo_times_ms / 1e3, ny, nz)

    os.makedirs(os.path.join(outdir, 'recon'), exist_ok=True)
    fn_pre = os.path.join(outdir, 'recon', f'{name}_preprocessed.h5')
    fn_truth = os.path.join(outdir, f'{name}_truth.h5')

    with CartesianFrameDataLoader(fn_mrd, squeeze_dims=False) as loader:
        sim_conf = loader.get_sim_conf()
        phantom = loader.get_phantom()
        n_coils = loader.n_coils
        if loader.n_frames != n_frames or tuple(loader.shape) != (nx, ny, nz):
            raise ValueError(
                f'{fn_mrd} holds {loader.n_frames} frames of shape {loader.shape}, '
                f'the protocol {n_frames} of {(nx, ny, nz)}'
            )

        tissues = phantom.contrast(sim_conf=sim_conf, resample=False, aggregate=False)
        # The engine scales its noise by the phantom as stored, which counts
        # the ROI at full weight.
        noise_var = noise_variance(tissues.sum(axis=0), snr)
        if phantom.smaps is None:
            smaps = np.ones((nx, ny, nz, 1), dtype=np.complex64)
        else:
            smaps = np.moveaxis(phantom.smaps, 0, -1).astype(np.complex64)

        with h5py.File(fn_pre, 'w') as f:
            d = f.create_dataset(
                'ksp_epi_zf', shape=(nx, ny, nz, n_coils, n_frames), dtype=np.complex64,
                chunks=(nx, ny, nz, n_coils, 1), compression='gzip', compression_opts=4,
            )
            for frame in range(n_frames):
                mask, ksp = loader.get_kspace_frame(frame)
                if not np.array_equal(mask, np.broadcast_to(omegas[..., frame], mask.shape)):
                    raise ValueError(f'frame {frame}: sampled locations differ from the schedule')
                d[..., frame] = np.moveaxis(ksp, 0, -1)
            f.create_dataset('omegas', data=omegas)
            f.create_dataset('echo_times', data=echo_times)
            f.create_dataset('smaps', data=smaps)
            if noise_var is not None:
                f.attrs['noise_var'] = noise_var
            f.attrs['whitened'] = True  # the simulated noise is white by construction
            f.attrs['t_ref_s'] = protocol.t_ref_s
            f.attrs['fov'] = np.asarray(protocol.fov_mm) / 1e3
            f.attrs['seqname'] = name
            f.attrs['Ncoils'] = n_coils
            f.attrs['simulated'] = True

        labels = [str(label) for label in phantom.labels]
        is_roi = np.array([label == ROI for label in labels])
        image_ref = tissues[~is_roi].sum(axis=0)
        anatomy = phantom.masks[~is_roi]
        with h5py.File(fn_truth, 'w') as f:
            f.create_dataset('image_ref', data=image_ref.astype(np.float32))
            f.create_dataset('tissues', data=anatomy.astype(np.float32))
            f.create_dataset('brain_mask', data=anatomy.sum(axis=0) > 0.5)
            f.attrs['tissue_labels'] = [label for label in labels if label != ROI]
            f.attrs['volume_tr'] = protocol.volume_tr_s
            f.attrs['tr_shot'] = protocol.tr_shot_ms / 1e3
            f.attrs['TE'] = protocol.te_ms / 1e3
            f.attrs['fa'] = protocol.fa_deg
            f.attrs['snr'] = snr
            f.attrs['model'] = loader.engine_model
            if noise_var is not None:
                f.attrs['noise_var'] = noise_var

            activations = [d for d in loader.get_all_dynamic() if d.name.startswith('activation')]
            if is_roi.any() and activations:
                # channel 0: fractional signal change of the ROI at every
                # excitation; channel 1: the stimulus (0/1).
                per_shot = np.asarray(activations[0].data[:, : n_frames * n_shots], np.float64)
                bold = per_shot[0].reshape(n_frames, n_shots).mean(axis=1)
                roi = phantom.masks[is_roi][0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    psc_peak = np.where(
                        image_ref > 0, bold.max() * tissues[is_roi][0] / image_ref, 0
                    )
                f.create_dataset('roi', data=roi.astype(np.float32))
                f.create_dataset('bold', data=bold)
                f.create_dataset('bold_shots', data=per_shot[0])
                f.create_dataset('stimulus_shots', data=per_shot[1])
                f.create_dataset('psc_peak', data=psc_peak.astype(np.float32))
    return fn_pre, fn_truth
