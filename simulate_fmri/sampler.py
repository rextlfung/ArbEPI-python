"""SNAKE sampler that plays an ArbEPI schedule.

SNAKE's stock EPI3dAcquisitionSampler acquires whole ky lines of one kz plane
per shot. ArbEPI instead visits an arbitrary list of (ky, kz) locations per
shot, different in every frame (sequences/ArbEPI.py's `schedules`, saved in
scan_info.mat). ArbEPISampler writes exactly that list into the MRD file:

- one MRD acquisition per echo: Nx samples at (kx, ky, kz) integer indices,
  with kx reversed on odd echoes, as the readout gradient alternates;
- one SNAKE "shot" (one excitation, one phantom state) per echo train;
- each acquisition's echo time (ms since excitation, scan_info.mat's third
  schedule channel) in its header's user_float[0], for the T2* model.

Array axes are ArbEPI's (x readout, y phase encode, z partition) in that
order, which is also the order of SNAKE's phantom arrays, so nothing is
transposed between the simulation and recon/.
"""

from __future__ import annotations

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray
from snake.core.sampling import BaseSampler
from snake.core.simulation import SimConfig
from snake.mrd_utils.utils import ACQ

ECHO_TIME_SLOT = 0  # index into the MRD acquisition header's user_float


class ArbEPISampler(BaseSampler):
    """ArbEPI's (ky, kz) schedule as a SNAKE sampler.

    Parameters
    ----------
    schedules : (Nframes, Nshots, ETL, 2) int, 0-based (ky, kz) of every echo
        (preprocess.utils.load_schedules' first return value).
    echo_times_ms : (ETL,) echo times in ms since excitation, the same for
        every shot and frame.
    """

    __sampler_name__ = 'arbepi'
    __engine__ = 'ArbEPI'

    schedules: NDArray
    echo_times_ms: NDArray
    constant: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.schedules = np.asarray(self.schedules)
        self.echo_times_ms = np.asarray(self.echo_times_ms, dtype=np.float32)
        if self.schedules.ndim != 4 or self.schedules.shape[-1] != 2:
            raise ValueError(
                f'schedules has shape {self.schedules.shape}, expected (Nframes, Nshots, ETL, 2)'
            )
        if self.echo_times_ms.shape != (self.etl,):
            raise ValueError(
                f'echo_times_ms has shape {self.echo_times_ms.shape}, expected ({self.etl},)'
            )
        self._next = 0

    @property
    def n_frames(self) -> int:
        return self.schedules.shape[0]

    @property
    def n_shots(self) -> int:
        """Echo trains (excitations) per frame."""
        return self.schedules.shape[1]

    @property
    def etl(self) -> int:
        return self.schedules.shape[2]

    def frame(self, sim_conf: SimConfig, idx: int) -> NDArray[np.uint32]:
        """(Nshots, ETL, Nx, 3) k-space indices of frame `idx`, in acquisition order."""
        nx = sim_conf.shape[0]
        sched = self.schedules[idx]
        traj = np.empty((self.n_shots, self.etl, nx, 3), dtype=np.uint32)
        kx = np.arange(nx, dtype=np.uint32)
        traj[:, 0::2, :, 0] = kx
        traj[:, 1::2, :, 0] = kx[::-1]
        traj[..., 1] = sched[..., 0, None]
        traj[..., 2] = sched[..., 1, None]
        return traj

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """The next frame, cycling through the schedule."""
        traj = self.frame(sim_conf, self._next % self.n_frames)
        self._next += 1
        return traj

    def TR_vol_ms(self, sim_conf: SimConfig) -> float:
        """Volume TR in ms: one excitation every sim_conf.seq.TR."""
        return sim_conf.seq.TR * self.n_shots

    def add_all_acq_mrd(self, dataset: mrd.Dataset, sim_conf: SimConfig) -> mrd.Dataset:
        """Write every acquisition's header and trajectory (the data stay zero
        until the engine fills them)."""
        nx, ny, nz = sim_conf.shape
        n_frames, n_shots, etl = self.n_frames, self.n_shots, self.etl
        n_coils = sim_conf.hardware.n_coils
        if self.schedules.min() < 0 or np.any(self.schedules.max(axis=(0, 1, 2)) >= (ny, nz)):
            raise ValueError(
                f'schedule indices reach {self.schedules.max(axis=(0, 1, 2))}, '
                f'outside the simulated (Ny, Nz) = ({ny}, {nz})'
            )
        n_dyn = sim_conf.max_n_shots
        if n_dyn < n_frames * n_shots:
            raise ValueError(
                f'max_sim_time={sim_conf.max_sim_time} s holds {n_dyn} excitations, but the '
                f'schedule has {n_frames * n_shots}'
            )
        self.log.info('%d frames of %d shots, %d echoes each', n_frames, n_shots, etl)
        self.log.info('volume TR: %.3f ms', self.TR_vol_ms(sim_conf))

        # The engine and CartesianFrameDataLoader take their reshapes from
        # these: step_0 = samples per acquisition, step_1 = acquisitions per
        # shot, repetition = frames.
        hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())
        hdr.encoding[0].encodingLimits = mrd.xsd.encodingLimitsType(
            kspace_encoding_step_0=mrd.xsd.limitType(0, nx, nx // 2),
            kspace_encoding_step_1=mrd.xsd.limitType(0, etl, etl // 2),
            slice=mrd.xsd.limitType(0, n_shots, 0),
            repetition=mrd.xsd.limitType(0, n_frames, 0),
        )
        dataset.write_xml_header(mrd.xsd.ToXML(hdr))

        acq_dtype = np.dtype(
            [
                ('head', mrd.hdf5.acquisition_header_dtype),
                ('data', np.float32, (n_coils * nx * 2,)),
                ('traj', np.uint32, (nx * 3,)),
            ]
        )
        per_frame = n_shots * etl
        # One HDF5 chunk per shot: the engine reads and writes whole shots.
        dset = dataset._dataset.create_dataset(
            'data', shape=(n_frames * per_frame,), dtype=acq_dtype, chunks=(etl,)
        )
        echo = np.arange(per_frame) % etl
        shot = np.arange(per_frame) // etl
        for i in range(n_frames):
            acq = np.zeros(per_frame, dtype=acq_dtype)
            head = acq['head']
            head['version'] = 1
            flags = np.zeros(per_frame, dtype=np.uint64)
            flags[echo == 0] |= np.uint64(ACQ.FIRST_IN_ENCODE_STEP1 | ACQ.FIRST_IN_SLICE)
            flags[echo == etl - 1] |= np.uint64(ACQ.LAST_IN_ENCODE_STEP1 | ACQ.LAST_IN_SLICE)
            flags[echo % 2 == 1] |= np.uint64(ACQ.IS_REVERSE)
            flags[0] |= np.uint64(ACQ.FIRST_IN_REPETITION)
            flags[-1] |= np.uint64(ACQ.LAST_IN_REPETITION)
            if i == n_frames - 1:
                flags[-1] |= np.uint64(ACQ.LAST_IN_MEASUREMENT)
            head['flags'] = flags
            head['scan_counter'] = i * per_frame + np.arange(per_frame)
            head['number_of_samples'] = nx
            head['available_channels'] = n_coils
            head['active_channels'] = n_coils
            head['center_sample'] = nx // 2
            head['trajectory_dimensions'] = 3
            head['sample_time_us'] = sim_conf.hardware.dwell_time_ms * 1000
            head['read_dir'] = (1, 0, 0)
            head['phase_dir'] = (0, 1, 0)
            head['slice_dir'] = (0, 0, 1)
            head['idx']['kspace_encode_step_1'] = self.schedules[i, :, :, 0].ravel()
            head['idx']['kspace_encode_step_2'] = self.schedules[i, :, :, 1].ravel()
            head['idx']['segment'] = shot
            head['idx']['repetition'] = i
            head['user_float'][:, ECHO_TIME_SLOT] = np.tile(self.echo_times_ms, n_shots)
            acq['traj'] = self.frame(sim_conf, i).reshape(per_frame, nx * 3)
            dset[i * per_frame : (i + 1) * per_frame] = acq
        dataset._file.flush()
        return dataset
