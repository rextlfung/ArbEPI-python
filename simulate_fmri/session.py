"""Simulate a scan session as raw data, for preprocess/ and recon/ to process.

simulate_session writes what a scan leaves behind, in the layout preprocess/
reads:

    <outdir>/scanarchives/<name>_noise.h5   noise scan
    <outdir>/scanarchives/<name>_cal.h5     EPIcal: blip-free echo trains
    <outdir>/scanarchives/<name>_epi.h5     ArbEPI: the fMRI run
    <outdir>/scanarchives/gre.h5            deGRE: dual-echo 3D GRE
    <outdir>/seqs/<name>/scan_info.mat      the sequence's own record
    <outdir>/<name>_truth.h5                what the simulation knows

The archives hold ADC readouts in acquisition order (preprocess.utils'
simulated-archive format, read by the same ArchiveReader as GE ScanArchives),
so

    python -m preprocess.batch_preprocess <outdir> <name>
    python -m recon.sense <outdir> <name> --reg ... --B0
    python -m recon.testbed score <outdir>/<name>_truth.h5 <recon>.h5

run the real pipeline end to end: noise whitening, readout-delay and odd/even
calibration, ramp-sample gridding, coil compression, ESPIRiT maps, the B0 and
R2* maps, and a B0-corrected reconstruction, each against a known answer.

What goes into the data (forward.py has the signal equation):

- the phantom on a grid finer than the acquisition's (grid_factor), with a
  susceptibility-derived field map (b0.py), T2* decay, and the slab profile;
- 32 loop coils (coils.py) and thermal noise with a coil covariance, at a
  level calibrated on a real head scan (THERMAL_NOISE);
- ramp-sampled readouts on scan_info.mat's kxo/kxe, played with a readout
  delay and an odd/even phase offset that drifts along the echo train;
- BOLD activation as a change of R2* (so it grows with echo time), and
  physiological noise (physio.py), both updated at every excitation.

SNAKE-fMRI supplies the phantom (BrainWeb tissue maps, resampling), the
steady-state contrast and the block-design BOLD regressor. Its acquisition
engine is not used here: it has no field map (by design, see SNAKE's FAQ), and
models one Cartesian sample grid. simulate.simulate keeps that engine for the
ideal mode.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.special import erf
from snake.core.handlers.activations.bold import block_design, get_bold
from snake.core.phantom import PropTissueEnum

from preprocess.utils import SIM_DATASET, create_simulated_archive, write_simulated_archive

from . import b0 as b0_model
from .coils import coil_maps, noise_covariance
from .forward import (
    Grid,
    Mode,
    Spins,
    band_limited,
    default_device,
    epi_readout,
    epi_readouts,
    gre_kspace,
)
from .handlers import ellipsoid_mask
from .phantom import Anatomy, brainweb_anatomy, ellipsoid_anatomy, place_fov
from .physio import PhysioConfig, physio
from .protocol import load_protocol

# Thermal noise: the standard deviation of a raw ADC sample of one coil is
#   noise * THERMAL_NOISE / (voxel volume [mm^3] * sqrt(N voxels * dwell [s]))
# in the units of forward.py's signal (magnetization per unit proton density,
# orthonormal FFT), which is how thermal noise scales with a protocol: SNR of a
# fully sampled volume goes as voxel volume * sqrt(total sampling time).
# Calibrated on a 3 T, 32-channel head session (20260922xiaokai): its deGRE's
# coil-combined image is 56 times its background noise at 2 mm (TE 3 ms),
# which by the same scaling is an SNR of about 115 for a fully sampled volume
# of the default 2.4 mm ArbEPI protocol (TE 30 ms). Good to perhaps 30%.
THERMAL_NOISE = 0.0125

FAT_HZ_PER_T = 3.5e-6 * b0_model.GAMMA_BAR_HZ_PER_T  # fat-water shift, for water excitation


@dataclass
class SessionConfig:
    """What to put into the simulated session. The defaults describe a
    realistic 3 T head scan; each effect can be switched off on its own."""

    grid_factor: int = 2  # spins per voxel per axis, EPI
    degre_grid_factor: int | None = None  # deGRE; None = spins about 1 mm apart
    n_coils: int = 32
    coils_per_ring: int = 8
    b0_scale: float = 1.0  # multiplies the field map; 0 = a uniform field
    shim_order: int = 1  # spherical-harmonic order removed over the brain
    delay: float = -0.3  # readout delay, samples (preprocess calibrates it)
    oe_phase: tuple[float, float] = (-0.25, -0.32)  # odd/even phase, first and last echo pair, rad
    oe_linear: float = 0.0  # odd/even linear phase, rad/FOV, on top of the delay's
    noise: float = 1.0  # multiplies THERMAL_NOISE; 0 = noise-free
    noise_gain_spread: float = 0.06
    noise_correlation: float = 0.02
    physio: PhysioConfig | None = field(default_factory=PhysioConfig)  # None = off
    activation: bool = True
    delta_r2s: float = -0.98  # 1/s, peak R2* change on activation (van der Zwaag 2009, 3 T)
    block_on: float = 10.0  # s
    block_off: float = 10.0
    slab_tbw: float = 8.0  # sets the width of the slab profile's edges
    readout_order: int = 2  # Taylor order of the evolution during a readout
    seed: int = 0


def steady_state(alpha: torch.Tensor, tr_s: float, t1_s: float) -> torch.Tensor:
    """Spoiled gradient-echo transverse magnetization right after the pulse."""
    e1 = math.exp(-tr_s / t1_s)
    return torch.sin(alpha) * (1 - e1) / (1 - torch.cos(alpha) * e1)


def slab_profile(z_mm: NDArray, thickness_mm: float, tbw: float) -> NDArray[np.float64]:
    """Relative flip angle across a slab of the given thickness: 1 inside,
    falling to 0 over an edge whose 10-90% width is 1.5 * thickness / tbw
    (a windowed sinc)."""
    w = 1.5 * thickness_mm / tbw / 1.812
    return 0.5 * (erf((z_mm + thickness_mm / 2) / w) - erf((z_mm - thickness_mm / 2) / w))


def gridding_noise_gain(kx: NDArray, nx: int, fov_x: float) -> float:
    """Variance of a gridded Cartesian kx sample per unit variance of the raw
    ramp samples, for preprocess/epi_gridding.py's density-compensated adjoint
    (dcf = |diff(kx)| / max, then / nx): the mean over output samples."""
    dcf = np.append(np.abs(np.diff(kx)), 0.0)
    dcf = dcf / dcf.max()
    x = (np.arange(nx) - nx // 2) * fov_x / nx
    k_out = (np.arange(nx) - nx // 2) / fov_x
    g = (np.exp(2j * np.pi * (kx[None, :, None] - k_out[:, None, None]) * x[None, None, :])
         .sum(-1) * dcf[None, :] / nx)
    return float((np.abs(g) ** 2).sum(axis=1).mean())


@dataclass
class _Scan:
    """One scan's spins and what was used to build them."""

    grid: Grid
    spins: Spins
    tissues: dict[str, torch.Tensor]
    affine: NDArray  # fine grid, voxel-to-world mm
    labels: tuple[str, ...]


def _build_scan(
    anatomy: Anatomy,
    b0_head: NDArray | None,
    grid: Grid,
    iso_mm: NDArray,
    flip_deg: float,
    tr_s: float,
    slab_mm: float,
    cfg: SessionConfig,
    device,
    k_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    water_excitation: bool = False,
    field_strength_t: float = 3.0,
) -> _Scan:
    """The phantom as one scan sees it, on that scan's fine grid."""
    res_mm = np.array(grid.res) * 1e3
    acq_affine = np.diag([*res_mm, 1.0])
    acq_affine[:3, 3] = iso_mm - (np.array(grid.shape) // 2) * res_mm
    affine = grid.fine_affine(acq_affine)
    fine = grid.fine_shape
    phantom = anatomy.phantom
    on_grid = phantom.resample(
        new_affine=affine.astype(np.float32), new_shape=fine, mode='constant',
        n_jobs=len(phantom.masks),
    )
    masks = torch.as_tensor(np.clip(on_grid.masks, 0, 1), dtype=torch.float32, device=device)
    labels = tuple(str(name) for name in phantom.labels)

    if b0_head is None:
        b0 = torch.zeros(fine, dtype=torch.float32, device=device)
    else:
        b0 = torch.as_tensor(
            b0_model.resample(b0_head, anatomy.head_affine, affine, fine), dtype=torch.float32,
            device=device,
        )

    coords_mm = [grid.coords(a) * 1e3 for a in range(3)]
    flip = math.radians(flip_deg) * torch.as_tensor(
        slab_profile(coords_mm[2], slab_mm, cfg.slab_tbw), dtype=torch.float32, device=device
    )[None, None, :] * torch.ones(fine, dtype=torch.float32, device=device)
    if water_excitation:
        # binomial 1-3-3-1: off-resonant water gets cos^3(pi f tau) of the flip
        tau = 1 / (2 * FAT_HZ_PER_T * field_strength_t)
        flip = flip * torch.cos(math.pi * b0 * tau) ** 3

    props = phantom.props
    mag = torch.stack([
        float(props[i, PropTissueEnum.rho])
        * steady_state(flip, tr_s, float(props[i, PropTissueEnum.T1]) / 1e3) * masks[i]
        for i in range(len(labels))
    ])
    r2s = torch.as_tensor(1e3 / props[:, PropTissueEnum.T2s], dtype=torch.float32, device=device)

    smaps = coil_maps(coords_mm, cfg.n_coils, cfg.coils_per_ring)
    for a, off in enumerate(k_offset):
        if off:
            ramp = np.exp(-2j * np.pi * off * grid.coords(a) / grid.fov[a]).astype(np.complex64)
            smaps = smaps * ramp.reshape([-1 if i == a else 1 for i in range(3)])[None]
    spins = Spins(mag=mag, r2s=r2s, b0=b0, smaps=torch.as_tensor(smaps, device=device))
    return _Scan(grid, spins, dict(zip(labels, masks)), affine, labels)


def _block_average(x: torch.Tensor, g: int) -> torch.Tensor:
    """Mean over each acquisition voxel's g^3 fine voxels (last three axes)."""
    if g == 1:
        return x
    return torch.nn.functional.avg_pool3d(x[None, None].float(), g)[0, 0]


def _measured_field(spins: Spins, g: int) -> torch.Tensor:
    """The field a map of this scan can measure: the magnetization-weighted
    mean over each acquisition voxel's spins (0 where there are none). The
    plain mean would count the field in the air around the object."""
    weight = spins.mag.sum(0)
    total = _block_average(weight, g)
    mean = _block_average(spins.b0 * weight, g) / total.clamp_min(1e-12)
    return torch.where(total > 1e-6 * total.max(), mean, torch.zeros_like(mean))


def _copy_scan_info(src: str, dst: str, n_frames: int) -> None:
    """Copy scan_info.mat, keeping only the first n_frames frames of the
    per-frame arrays (hdf5storage stores them axis-reversed)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    with h5py.File(dst, 'r+') as f:
        sched = f['schedules']
        if sched.shape[-1] == n_frames:
            return
        for key, axis in (('schedules', -1), ('parts', 0)):
            if key not in f:
                continue
            d = f[key]
            data = d[..., :n_frames] if axis == -1 else d[:n_frames]
            attrs = dict(d.attrs)
            del f[key]
            f.create_dataset(key, data=data)
            for k, v in attrs.items():
                f[key].attrs[k] = v


def simulate_session(
    scan_info: str,
    outdir: str,
    name: str = 'sim',
    *,
    anatomy: str | Anatomy = 'brainweb',
    cfg: SessionConfig | None = None,
    frames: int | None = None,
    fa_deg: float | None = None,
    device=None,
) -> dict[str, str]:
    """Simulate the session recorded in `scan_info` as raw data (see the
    module docstring). anatomy: 'brainweb', 'ellipsoid', or an Anatomy.
    frames: only the first N frames. Returns the paths written."""
    cfg = cfg or SessionConfig()
    device = torch.device(device) if device is not None else default_device()
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    protocol = load_protocol(scan_info, fa_deg=fa_deg, frames=frames)
    if protocol.kxo is None:
        raise ValueError('the raw-data simulation needs kxo/kxe from scan_info.mat')
    if isinstance(anatomy, str):
        makers = {'brainweb': brainweb_anatomy, 'ellipsoid': ellipsoid_anatomy}
        if anatomy not in makers:
            raise ValueError(f'anatomy={anatomy!r}, expected one of {sorted(makers)}')
        anatomy = makers[anatomy]()

    n_frames, n_shots, etl = protocol.n_frames, protocol.n_shots, protocol.etl
    n_exc = n_frames * n_shots
    nfid = len(protocol.kxo)
    shape = protocol.shape
    res_mm = np.array(protocol.res_mm)
    fov_m = tuple(f / 1e3 for f in protocol.fov_mm)
    te = protocol.te_ms / 1e3
    tr_shot = protocol.tr_shot_ms / 1e3
    slab_mm = 0.9 * protocol.fov_mm[2]  # lib/make_excitation_pulse.py, sequences/deGRE.py

    acq_affine = place_fov(anatomy.phantom, shape, tuple(res_mm)).affine.astype(np.float64)
    iso_mm = acq_affine[:3, 3] + (np.array(shape) // 2) * res_mm

    b0_head = None
    if cfg.b0_scale:
        b0_head = cfg.b0_scale * b0_model.head_field_hz(
            anatomy.head, anatomy.head_brain, anatomy.head_affine, anatomy.cavities,
            shim_order=cfg.shim_order,
        )

    def parity_offset(n: int) -> float:
        # the sequences step k as (i - N/2) dk; the DFT grid is (i - N//2) dk
        return n // 2 - n / 2

    # ---- the fMRI run and its calibration scan ----
    grid = Grid(shape, fov_m, cfg.grid_factor)
    scan = _build_scan(
        anatomy, b0_head, grid, iso_mm, protocol.fa_deg, tr_shot, slab_mm, cfg, device,
        k_offset=(0.0, parity_offset(shape[1]), parity_offset(shape[2])),
        water_excitation=protocol.water_excitation,
    )
    spins = scan.spins
    t_exc = np.arange(n_exc) * tr_shot

    modes: list[Mode] = []
    act_mode, act_course = None, np.zeros(n_exc)
    if cfg.activation and 'gm' in scan.labels:
        inside = ellipsoid_mask(grid.fine_shape, scan.affine, **anatomy.roi)
        events = block_design(cfg.block_on, cfg.block_off, n_frames * protocol.volume_tr_s)
        bold = get_bold(protocol.tr_shot_ms, (n_exc + 0.5) * tr_shot, events, 'glover', 1, -24.0,
                        1.0).ravel()[:n_exc]
        act_course = cfg.delta_r2s * bold
        act_mode = Mode(
            'r2s', torch.as_tensor(inside, dtype=torch.float32, device=device), act_course,
            torch.tensor([1.0 if lab == 'gm' else 0.0 for lab in scan.labels]),
            name='activation',
        )
        modes.append(act_mode)
    frequency, courses = None, {}
    if cfg.physio is not None and cfg.physio.scale:
        p_modes, frequency, courses = physio(
            scan.tissues, grid.coords(2) * 1e3, tuple(res_mm / grid.g), t_exc, scan.labels,
            seed=cfg.seed + 1, cfg=cfg.physio,
        )
        modes += p_modes

    n_pairs = etl // 2
    readout = epi_readout(
        protocol.kxo, protocol.kxe, protocol.echo_times_ms / 1e3, protocol.adc_dwell_s, fov_m[0],
        delay=cfg.delay, a0=np.linspace(cfg.oe_phase[0], cfg.oe_phase[1], n_pairs),
        a1=cfg.oe_linear,
    )

    psi = noise_covariance(cfg.n_coils, rng, cfg.noise_gain_spread, cfg.noise_correlation)
    chol = torch.as_tensor(np.linalg.cholesky(psi), dtype=torch.complex64, device=device)
    voxel_mm3 = float(np.prod(res_mm))
    sigma = cfg.noise * THERMAL_NOISE / (
        voxel_mm3 * math.sqrt(math.prod(shape) * protocol.adc_dwell_s)
    )

    def noise(n: int, n_samples: int, std: float) -> torch.Tensor:
        """(n, Nc, n_samples) complex noise with covariance psi * std^2."""
        if std == 0:
            return torch.zeros(n, cfg.n_coils, n_samples, dtype=torch.complex64, device=device)
        z = torch.randn(n, cfg.n_coils, n_samples, 2, generator=gen, device=device)
        z = torch.view_as_complex(z) * (std / math.sqrt(2))
        return torch.einsum('cd,ndj->ncj', chol, z)

    arch = os.path.join(outdir, 'scanarchives')
    paths = {
        'datdir': outdir,
        'epi': os.path.join(arch, f'{name}_epi.h5'),
        'cal': os.path.join(arch, f'{name}_cal.h5'),
        'noise': os.path.join(arch, f'{name}_noise.h5'),
        'gre': os.path.join(arch, 'gre.h5'),
        'scan_info': os.path.join(outdir, 'seqs', name, 'scan_info.mat'),
        'truth': os.path.join(outdir, f'{name}_truth.h5'),
    }
    print(
        f'{n_frames} frames x {n_shots} shots x {etl} echoes of {nfid} samples, '
        f'{shape} at {tuple(round(float(r), 2) for r in res_mm)} mm, '
        f'R = {protocol.acceleration:.1f}; '
        f'{cfg.n_coils} coils, spins on a {grid.fine_shape} grid, {device}'
    )
    with create_simulated_archive(paths['epi'], n_exc * etl, cfg.n_coils, nfid,
                                  scan='ArbEPI') as f:
        dset = f[SIM_DATASET]

        def sink(e: int, data: torch.Tensor) -> None:
            dset[e::etl] = (data + noise(n_exc, nfid, sigma)).cpu().numpy()

        cal = epi_readouts(spins, grid, readout, protocol.schedules, sink, modes=modes,
                           frequency=frequency, order=cfg.readout_order)
    cal = cal[None].expand(n_shots, -1, -1, -1).reshape(n_shots * etl, cfg.n_coils, nfid)
    write_simulated_archive(
        paths['cal'], (cal + noise(n_shots * etl, nfid, sigma)).cpu().numpy(), scan='EPIcal'
    )
    n_noise = math.ceil(20 * cfg.n_coils**2 / nfid)  # sequences/noise.py
    write_simulated_archive(
        paths['noise'], noise(n_noise, nfid, sigma if sigma else 1.0).cpu().numpy(), scan='noise'
    )

    # ---- truth, on the acquisition grid ----
    x0c = band_limited(spins.image(te)[0], grid)
    x0_abs = x0c.abs()
    tissues_acq = torch.stack([_block_average(scan.tissues[lab], grid.g) for lab in scan.labels])
    full = tissues_acq.sum(0) >= 0.9
    eta = gridding_noise_gain(protocol.kxo, shape[0], fov_m[0])
    brain_signal = float(x0_abs[full].median()) if full.any() else float('nan')
    snr0 = brain_signal / (math.sqrt(eta) * sigma) if sigma else float('inf')
    print(f'thermal noise: SNR of a fully sampled volume {snr0:.0f} (median over brain voxels)')

    with h5py.File(paths['truth'], 'w') as f:
        f.attrs['volume_tr'] = protocol.volume_tr_s
        f.attrs['simulated'] = True
        g = f.create_group('truth')
        g.create_dataset('image_rest', data=x0_abs.cpu().numpy())
        g.create_dataset('tissues', data=tissues_acq.cpu().numpy())
        g.create_dataset('brain_mask', data=(tissues_acq.sum(0) > 0.5).cpu().numpy())
        g.attrs['tissue_labels'] = list(scan.labels)
        g.create_dataset('b0_map', data=_measured_field(spins, grid.g).cpu().numpy())
        acq_coords = [(np.arange(n) - n // 2) * r for n, r in zip(shape, res_mm)]
        g.create_dataset(
            'smaps',
            data=coil_maps(acq_coords, cfg.n_coils, cfg.coils_per_ring).transpose(1, 2, 3, 0),
        )
        g.create_dataset('noise_covariance', data=psi)
        g.create_dataset('oe_phase', data=np.stack([readout.phase[1::2],
                                                    np.full(n_pairs, cfg.oe_linear)], axis=1))
        g.attrs.update(delay=cfg.delay, sigma_raw=sigma, snr0=snr0, gridding_noise_gain=eta,
                       TE=te, tr_shot=tr_shot, fa=protocol.fa_deg, grid_factor=grid.g,
                       b0_scale=cfg.b0_scale, isocenter_mm=iso_mm)
        for key, course in courses.items():
            g.create_dataset(f'physio/{key}', data=course)
        if frequency is not None:
            g.create_dataset('physio/frequency_hz', data=frequency)

        if act_mode is not None:
            dx = band_limited(act_mode.image(spins, te), grid)  # per unit R2* change
            frac = (x0c.conj() * dx).real / x0_abs.clamp_min(1e-12 * float(x0_abs.max())) ** 2
            per_frame = act_course.reshape(n_frames, n_shots).mean(axis=1)
            centered = per_frame - per_frame.mean()
            peak = float(np.abs(centered).max()) or 1.0
            x0 = x0_abs * (1 + per_frame.mean() * frac)
            amp_map = (peak * frac / (1 + per_frame.mean() * frac)) * (x0_abs > 0)
            cut = 0.5 * float(amp_map[full].max()) if full.any() else float('inf')
            roi_mask = full & (amp_map >= cut)
            amp = float(amp_map[roi_mask].median()) if roi_mask.any() else 0.0
            g.create_dataset('x0', data=x0.cpu().numpy())
            g.create_dataset('roi_masks', data=roi_mask.cpu().numpy()[None])
            g.create_dataset('waveforms', data=(centered / peak)[None])
            g.create_dataset('amp_map', data=amp_map.cpu().numpy())
            g.create_dataset('r2s_change', data=act_course)
            g.attrs['roi_names'] = ['block_occipital']
            g.attrs['roi_kinds'] = ['block']
            g.attrs['amp'] = amp
        else:
            g.create_dataset('x0', data=x0_abs.cpu().numpy())
            g.create_dataset('roi_masks', data=np.zeros((0, *shape), dtype=bool))
            g.create_dataset('waveforms', data=np.zeros((0, n_frames)))
            g.attrs['roi_names'] = []
            g.attrs['roi_kinds'] = []
            g.attrs['amp'] = 0.0

    del spins, scan, modes, act_mode
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ---- the deGRE ----
    d = protocol.degre
    if d is not None:
        g_d = cfg.degre_grid_factor or max(1, round(min(d.res_mm)))
        grid_d = Grid(d.shape, tuple(v / 1e3 for v in d.fov_mm), g_d)
        # k steps as (i - N/2) dk on y and z; the readout's samples sit half a
        # sample off the echo ((n + 0.5) dwell on a symmetric flat top)
        k_off = (0.5, parity_offset(d.shape[1]), parity_offset(d.shape[2]))
        scan_d = _build_scan(anatomy, b0_head, grid_d, iso_mm, d.fa_deg, d.tr_s, slab_mm, cfg,
                             device, k_offset=k_off)
        ksp = gre_kspace(scan_d.spins, grid_d, np.asarray(d.te_s))  # (Nx, Ny, Nz, E, Nc)
        nx_d, ny_d, nz_d = d.shape
        n_e = len(d.te_s)
        image_block = ksp.permute(2, 1, 3, 4, 0).reshape(nz_d * ny_d * n_e, cfg.n_coils, nx_d)
        # the receive-gain block: every readout at ky = kz = 0 (blips off)
        gain_block = ksp[:, ny_d // 2, nz_d // 2].permute(1, 2, 0)[None].expand(
            ny_d, -1, -1, -1).reshape(ny_d * n_e, cfg.n_coils, nx_d)
        raw = torch.cat([gain_block, image_block])
        sigma_d = cfg.noise * THERMAL_NOISE / (
            float(np.prod(d.res_mm)) * math.sqrt(math.prod(d.shape) * d.dwell_s)
        )
        raw = raw + noise(raw.shape[0], nx_d, sigma_d)
        write_simulated_archive(paths['gre'], raw.cpu().numpy(), scan='deGRE')
        with h5py.File(paths['truth'], 'r+') as f:
            g = f['truth'].create_group('degre')
            g.create_dataset('b0_map', data=_measured_field(scan_d.spins, g_d).cpu().numpy())
            g.attrs['sigma_raw'] = sigma_d
        del scan_d, ksp, raw
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    _copy_scan_info(scan_info, paths['scan_info'], n_frames)
    print('Wrote ' + '\n      '.join(paths[k] for k in ('epi', 'cal', 'noise', 'gre', 'scan_info',
                                                         'truth')))
    return paths
