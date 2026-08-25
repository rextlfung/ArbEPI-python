"""Central configuration for the EPI preprocessing/reconstruction pipeline.

Ports ../epi-preprocessing/config.m + set_seq_paths.m. Mirrors params.py's
load_params() pattern: a dataclass returned by a factory function, passed
explicitly rather than injected into the caller's workspace (MATLAB's
run('./config.m') has no Python equivalent).

Per-sequence acquisition scalars (Nx, Ny, Nz, ETL, R, fov, volume_tr, ...) are
not part of this repo -- in the MATLAB pipeline they live in a per-acquisition
<datdir>/seqs/<seqname>/params.m, loaded at runtime via run(cfg.fn.params).
There is no Python equivalent of that "script injects into workspace" pattern,
and copying the whole params.py module would drag pypulseq/scanners.py into
this pipeline just to read a handful of scalars -- so instead
sequences/ArbEPI.py exports those scalars to a params.mat snapshot (same mechanism and
location as its existing samp_locs.mat write, via
hdf5storage.savemat(fmt='7.3')). This module only ever reads it back with
h5py, per this repo's rule of never using scipy.io on v7.3 .mat files.
"""

import os
from dataclasses import dataclass

import h5py


@dataclass
class PreprocessingConfig:
    """Session-level config, analogous to MATLAB's `cfg` struct."""

    datdir: str
    seqnames: list[str]
    fn_gre: str  # single shared GRE ScanArchive (not per-sequence)

    # Coil compression: Nvcoils is auto-selected from the whitened GRE
    # eigenvalue spectrum; components are kept until cumulative explained
    # variance reaches cc_energy_thresh, subject to a 2R floor.
    cc_energy_thresh: float = 0.9

    # k-space center offset (samples). Scalar (applied to every sequence) or
    # a dict keyed by seqname (MATLAB's containers.Map). Scanner/dwell-time
    # dependent -- tune with calibrate_delay.py.
    delay: float | dict[str, float] = -1.0
    show_epi_phase_diff: bool = True

    # Sensitivity map estimation. SENSEmethod is always 'sigpy' in this port
    # (BART's ecalib / MATLAB PISCO are not ported -- see smaps.py).
    threshold_mask: float = 0.2

    # Which of generate_degre's TE_degre echoes to use for coil-sensitivity
    # estimation (0 = TE1, the shorter echo -- less T2* decay, so higher
    # SNR; either echo works, see sequences/deGRE.py's module docstring).
    gre_echo_idx: int = 0

    # sigpy wavelet+TV regularized recon / CG-SENSE
    lamb_l1: float = 0.005
    lamb_tv: float = 0.005
    num_iter: int = 100
    Nframes: float = float('inf')

    # B0 field map estimation (run_b0map.py -> preprocessing/julia/b0map.jl).
    # Fraction of peak first-echo GRE magnitude below which a voxel is
    # excluded from the fit -- matches MRIFieldmaps.jl's own b0init default.
    b0map_mask_thresh: float = 0.1

    do_sense: bool = True
    use_parfor: bool = False
    interactive: bool = False


def load_config(datdir: str, seqnames: list[str], **overrides) -> PreprocessingConfig:
    cfg = PreprocessingConfig(
        datdir=datdir,
        seqnames=list(seqnames),
        fn_gre=os.path.join(datdir, 'scanarchives', 'gre.h5'),
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f'load_config: unknown override {key!r}')
        setattr(cfg, key, value)
    return cfg


def seq_delay(cfg: PreprocessingConfig, seqname: str) -> float:
    """Resolve cfg.delay (scalar or per-seqname dict) for one sequence."""
    if isinstance(cfg.delay, dict):
        return cfg.delay[seqname]
    return cfg.delay


@dataclass
class SeqPaths:
    """Per-sequence file paths, analogous to set_seq_paths.m's cfg.fn.*."""

    seqname: str
    seqdir: str
    params: str
    cal: str
    noise: str
    epi: str
    kxoe: str  # placeholder; resolved once Nx is known from params (see preprocess.py)
    samp_log: str
    recon: str


def set_seq_paths(cfg: PreprocessingConfig, seqname: str) -> SeqPaths:
    seqdir = os.path.join(cfg.datdir, 'seqs', seqname)
    scanarchives = os.path.join(cfg.datdir, 'scanarchives')
    return SeqPaths(
        seqname=seqname,
        seqdir=seqdir,
        params=os.path.join(seqdir, 'params.mat'),
        cal=os.path.join(scanarchives, f'{seqname}_cal.h5'),
        noise=os.path.join(scanarchives, f'{seqname}_noise.h5'),
        epi=os.path.join(scanarchives, f'{seqname}_epi.h5'),
        kxoe=os.path.join(seqdir, 'kxoe90.mat'),
        samp_log=os.path.join(seqdir, 'samp_locs.mat'),
        # .h5, not .mat: everything preprocess.py itself writes (this file,
        # the GRE/smaps caches) is plain numpy-order h5py, the opposite
        # on-disk axis convention from the hdf5storage-written .mat files
        # this pipeline reads (samp_locs.mat, kxoe*.mat, params.mat -- see
        # preprocessing/matio.py). A .mat extension here would silently
        # invite reading it with hdf5storage.loadmat, which would return
        # every multi-axis array transposed.
        recon=os.path.join(cfg.datdir, 'recon', f'{seqname}_epi_zf.h5'),
    )


@dataclass
class SeqParams:
    """Per-acquisition scan scalars, ported from what params.m dumps into
    preprocess.m's workspace. Only the fields preprocess.py/recon_frames.py
    actually consume. Field names match params.Params (snake_case), not
    MATLAB's camelCase -- this is a fresh Python-native artifact, not a
    format shared with any MATLAB consumer."""

    Nx: int
    Ny: int
    Nz: int
    ETL: int
    R: float
    fov: tuple[float, float, float]  # m
    volume_tr: float  # s
    discard_duration: float  # s

    Nx_degre: int
    Ny_degre: int
    Nz_degre: int
    fov_degre: tuple[float, float, float]  # m
    n_echoes_degre: int  # len(params.TE_degre) -- see sequences/deGRE.py's `c` loop
    TE_degre: tuple[float, ...] | None  # s; None for pre-dual-echo snapshots (no TE_degre key)


def load_seq_params(paths: SeqPaths) -> SeqParams:
    """Read the params.mat snapshot written by sequences/ArbEPI.py.

    hdf5storage.savemat stores each dict key as a top-level dataset: Python
    scalars come back as shape-(1,1) arrays, 3-vectors as shape-(3,1) --
    hence the .item()/.ravel() below rather than a bare [()].
    """
    with h5py.File(paths.params, 'r') as f:
        def scalar(name):
            return f[name][()].item()

        def vec3(name):
            return tuple(f[name][()].ravel().tolist())

        return SeqParams(
            Nx=int(scalar('Nx')), Ny=int(scalar('Ny')), Nz=int(scalar('Nz')),
            ETL=int(scalar('ETL')), R=scalar('R'),
            fov=vec3('fov'),
            volume_tr=scalar('volume_tr'),
            discard_duration=scalar('discard_duration'),
            Nx_degre=int(scalar('Nx_degre')), Ny_degre=int(scalar('Ny_degre')),
            Nz_degre=int(scalar('Nz_degre')),
            fov_degre=vec3('fov_degre'),
            # Missing in params.mat snapshots written before the dual-echo
            # deGRE upgrade -- those acquisitions genuinely were single-echo,
            # so default to 1 rather than raising KeyError on durable,
            # non-regeneratable per-acquisition data records.
            n_echoes_degre=int(scalar('n_echoes_degre')) if 'n_echoes_degre' in f else 1,
            TE_degre=tuple(f['TE_degre'][()].ravel().tolist()) if 'TE_degre' in f else None,
        )
