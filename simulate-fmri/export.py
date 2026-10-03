"""SNAKE's .mrd -> the file recon/ reads and recon/testbed.py scores against.

export() writes <outdir>/recon/<name>_preprocessed.h5 from a finished
simulation, in the layout preprocess/ produces for real scans, so
`python -m recon.rss` and `python -m recon.sense` run on it unchanged:

    ksp_epi_zf   (Nx, Ny, Nz, Ncoils, Nframes) zero-filled k-space
    omegas       (Ny, Nz, Nframes) sampling mask
    echo_times   (Ny, Nz, Nframes) s since excitation of each sampled location
    smaps        (Nx, Ny, Nz, Ncoils) the sensitivities the simulation used
                 (not maps estimated from calibration data)
    attrs        noise_var, whitened, t_ref_s, fov, volume_tr, seqname

There is no b0_map or r2star_map: SNAKE models no off-resonance.

It also holds a `truth` group in recon/testbed.py's layout, so
`python -m recon.testbed score <this file> <recon.h5>` scores a reconstruction
of a simulation the way it scores the real-data testbed:

    x0          (Nx, Ny, Nz) the noise-free magnitude image at TE, time-averaged
    roi_masks   (1, Nx, Ny, Nz) the activated region
    waveforms   (1, Nframes) its time course, zero-mean with peak |w| = 1
    attrs       roi_names, roi_kinds, amp (the peak fractional change)

so that the true image of frame t is x0 * (1 + amp_map * w(t)). The activation
is a copy of gray matter added on top of the anatomy, so its fractional size
amp_map (also saved) varies from voxel to voxel with the tissue mix; roi_masks
keeps the voxels at half its maximum or more, and amp is their median, which
makes testbed.score's amp_ratio 1 for a perfect reconstruction. The rest of the
group is what a scan would not know either: image_rest (no activation),
bold_shots and stimulus_shots (per excitation), tissues and brain_mask.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
from numpy.typing import NDArray
from snake.mrd_utils import CartesianFrameDataLoader

ROI = 'ROI'  # the tissue SNAKE's activation handlers add
ROI_NAME = 'block_occipital'  # 'block...': testbed.score also reports its low-band t


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


def activation_truth(
    image_rest: NDArray, roi_image: NDArray, bold: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray, float]:
    """(x0, waveform, amp_map, roi_mask, amp) of a frame series
    image_rest + bold(t) * roi_image, rewritten as x0 * (1 + amp_map * w(t))
    with w zero-mean and peak |w| = 1 (see the module docstring)."""
    centered = bold - bold.mean()
    peak = np.abs(centered).max()
    waveform = centered / peak
    x0 = image_rest + bold.mean() * roi_image
    amp_map = np.divide(peak * roi_image, x0, out=np.zeros_like(x0), where=x0 > 0)
    roi_mask = amp_map >= 0.5 * amp_map.max()
    return x0, waveform, amp_map, roi_mask, float(np.median(amp_map[roi_mask]))


def export(fn_mrd: str, protocol, outdir: str, name: str, snr: float) -> str:
    """Write <outdir>/recon/<name>_preprocessed.h5 from the SNAKE file `fn_mrd`
    and return its path. protocol: simulate.Protocol. snr: the engine's."""
    nx, ny, nz = protocol.shape
    n_frames, n_shots = protocol.n_frames, protocol.n_shots
    omegas, echo_times = schedule_maps(protocol.schedules, protocol.echo_times_ms / 1e3, ny, nz)

    os.makedirs(os.path.join(outdir, 'recon'), exist_ok=True)
    fn_out = os.path.join(outdir, 'recon', f'{name}_preprocessed.h5')

    loader = CartesianFrameDataLoader(fn_mrd, squeeze_dims=False)
    with loader, h5py.File(fn_out, 'w') as f:
        sim_conf = loader.get_sim_conf()
        phantom = loader.get_phantom()
        n_coils = loader.n_coils
        if loader.n_frames != n_frames or tuple(loader.shape) != (nx, ny, nz):
            raise ValueError(
                f'{fn_mrd} holds {loader.n_frames} frames of shape {loader.shape}, '
                f'the protocol {n_frames} of {(nx, ny, nz)}'
            )

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
        if phantom.smaps is None:
            smaps = np.ones((nx, ny, nz, 1), dtype=np.complex64)
        else:
            smaps = np.moveaxis(phantom.smaps, 0, -1).astype(np.complex64)
        f.create_dataset('smaps', data=smaps)

        tissues = phantom.contrast(sim_conf=sim_conf, resample=False, aggregate=False)
        # The engine scales its noise by the phantom as stored, which counts
        # the ROI at full weight.
        noise_var = noise_variance(tissues.sum(axis=0), snr)
        if noise_var is not None:
            f.attrs['noise_var'] = noise_var
            # What that noise level means for an image: signal of a pure
            # gray-matter voxel over the noise std of one component of a fully
            # sampled reconstruction with these (unit root-sum-of-squares) coils.
            if 'gm' in phantom.labels_idx:
                gm_signal = float(tissues[phantom.labels_idx['gm']].max())
                f.attrs['sim_snr0_gm'] = gm_signal / np.sqrt(noise_var / 2)
        f.attrs['whitened'] = True  # the simulated noise is white by construction
        f.attrs['t_ref_s'] = protocol.t_ref_s
        f.attrs['fov'] = np.asarray(protocol.fov_mm) / 1e3
        f.attrs['volume_tr'] = protocol.volume_tr_s
        f.attrs['seqname'] = name
        f.attrs['Ncoils'] = n_coils
        f.attrs['simulated'] = True
        f.attrs['sim_model'] = loader.engine_model
        f.attrs['sim_snr'] = snr
        f.attrs['TE'] = protocol.te_ms / 1e3
        f.attrs['tr_shot'] = protocol.tr_shot_ms / 1e3
        f.attrs['fa'] = protocol.fa_deg

        labels = [str(label) for label in phantom.labels]
        is_roi = np.array([label == ROI for label in labels])
        image_rest = tissues[~is_roi].sum(axis=0).astype(np.float32)
        anatomy = phantom.masks[~is_roi]
        g = f.create_group('truth')
        g.create_dataset('image_rest', data=image_rest)
        g.create_dataset('tissues', data=anatomy.astype(np.float32))
        g.create_dataset('brain_mask', data=anatomy.sum(axis=0) > 0.5)
        g.attrs['tissue_labels'] = [label for label in labels if label != ROI]

        activations = [d for d in loader.get_all_dynamic() if d.name.startswith('activation')]
        if is_roi.any() and activations:
            # channel 0: fractional signal change of the ROI at every
            # excitation; channel 1: the stimulus (0/1).
            per_shot = np.asarray(activations[0].data[:, : n_frames * n_shots], np.float64)
            bold = per_shot[0].reshape(n_frames, n_shots).mean(axis=1)
            x0, waveform, amp_map, roi_mask, amp = activation_truth(
                image_rest, tissues[is_roi][0], bold
            )
            g.create_dataset('x0', data=x0.astype(np.float32))
            g.create_dataset('roi_masks', data=roi_mask[None])
            g.create_dataset('waveforms', data=waveform[None])
            g.create_dataset('amp_map', data=amp_map.astype(np.float32))
            g.create_dataset('bold_shots', data=per_shot[0])
            g.create_dataset('stimulus_shots', data=per_shot[1])
            g.attrs['roi_names'] = [ROI_NAME]
            g.attrs['roi_kinds'] = ['block']
            g.attrs['amp'] = amp
        else:  # a resting phantom: nothing to score an activation against
            g.create_dataset('x0', data=image_rest)
            g.create_dataset('roi_masks', data=np.zeros((0, nx, ny, nz), dtype=bool))
            g.create_dataset('waveforms', data=np.zeros((0, n_frames)))
            g.attrs['roi_names'] = []
            g.attrs['roi_kinds'] = []
            g.attrs['amp'] = 0.0
    return fn_out
