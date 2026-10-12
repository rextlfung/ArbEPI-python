"""meGRE: 3D spoiled multi-echo GRE, monopolar readouts (PROTOTYPE).

A multi-echo successor to sequences/deGRE.py for the three maps the
reconstruction needs from the calibration scan, in priority order: B0
(phase difference between echoes), R2* (magnitude decay over echo time) and
coil sensitivities. Not wired into main.py / preprocess yet -- see
`docs/review-findings.md`-style notes in CLAUDE.md's deGRE paragraph for the
measured numbers.

Differences from deGRE.py:
  - One excitation reads out `n_echoes` echoes (deGRE: one excitation per
    echo, two echoes). Phase encodes are applied once, before the train, and
    rewound once after it.
  - Monopolar: every readout lobe has the same polarity and a negative x
    rewinder ("flyback") returns kx to -kmax between lobes, so there is no
    odd/even phase or kx-shift difference between echoes and nothing to
    correct.
  - Echo spacing is one fat/water period 1/f_fat (2.237 ms at 3 T, rounded to
    the 4 us raster), so fat is in phase with water at every echo and TE_n =
    n / f_fat. The spacing is long compared with a 90-sample readout, so the
    readout is stretched to fill `read_frac` of it (low receive bandwidth,
    low gradient amplitude, low PNS) instead of running at deGRE's 4 us dwell.
  - Resolution and FOV default to the EPI's (`params.res`, `params.fov`:
    2.4 mm, 216 x 216 x 144 mm in the current protocol), not
    `params.fov_degre`, so the maps land on the EPI grid with no resize.
  - 20 echoes by default (last TE 44.7 ms, 267 s) for the first human scan,
    to measure how many echoes are really needed: thermal-noise precision
    saturates by ~8 (`precision_estimate`), so the scan acquires the maximum
    that fits in 5 min and preprocessing's `gre_n_echoes` analyses the first K.
    Scan time is set by the last TE, not the echo count.
  - `generate_megre` patches `scan_info.mat` (when present) with this scan's
    geometry, echo times and flip -- Nx/Ny/Nz/fov/n_echoes/TE/TR/alpha/dwell_degre --
    so `preprocess/` reads it exactly as it reads a deGRE (main.py --gre megre).

Everything else (slab-selective sinc, Ernst flip for the realized TR, RF
spoiling, dummy shots, the iZ = 0 receive-gain block, TRID labels, prephaser
stretching) is deGRE.py's. The ADC order is echo-fastest, then iY, then iZ,
the layout `preprocess.preprocess`'s `unflatten_gre_echoes` already reads for
any `n_echoes_degre`.

`python -m sequences.meGRE --echoes 6 8 12 --pns` prints the timing/PNS/precision table.
"""

import argparse
import copy
import math
import os
from dataclasses import dataclass

import hdf5storage
import numpy as np
import pypulseq as pp
from tqdm import tqdm

from lib.trap4ge import trap4ge
from params import Params, load_params

# T2* used for the R2* precision figure of merit in `precision_table`: gray
# matter at 3 T, uncorrected (Wansapura 1999 / Peters 2006, see CLAUDE.md's
# simulate_fmri section).
T2STAR_S = 47e-3


@dataclass
class MEGREDesign:
    """Everything `generate_megre` assembles, plus the numbers worth reporting."""

    n_echoes: int
    N: tuple[int, int, int]
    fov: tuple[float, float, float]
    dwell: float
    t_read: float
    echo_spacing: float  # s, realized (raster multiple)
    TE: np.ndarray  # s, realized echo times from the RF center
    TR: float
    flip_deg: float
    n_shots: int  # excitations in the whole scan, dummies and gain-cal included
    g_read_mT_m: float
    sys: pp.Opts
    blocks: dict

    @property
    def scan_time(self) -> float:
        return self.n_shots * self.TR


def _spoiler_cycles_area(Nx: int, dkx: float, cycles: float) -> float:
    return Nx * dkx * cycles


def design_megre(
    params: Params,
    *,
    res=None,
    fov=None,
    n_echoes: int = 20,
    echo_spacing: float | None = None,
    read_frac: float = 0.5,
    n_cycles_spoil: float = 4,
    spoil_slew: float = 70.0,
    n_dummy_zloops: int = 2,
    TR: float | None = None,
) -> MEGREDesign:
    """Build every gradient/RF/ADC object and the timing. `echo_spacing`
    defaults to 1/f_fat; `TR` defaults to the minimum; `res`/`fov` default
    to the EPI's."""
    res = tuple(params.res if res is None else res)
    fov = tuple(params.fov if fov is None else fov)
    N = tuple(int(round(f / r)) for f, r in zip(fov, res))
    for f, r, n in zip(fov, res, N):
        if abs(n * r - f) > 1e-9:
            raise ValueError(f'fov {f} m is not a whole number of {r} m voxels')
    Nx, Ny, Nz = N
    sys = copy.deepcopy(params.sys)
    sys.max_slew = params.slew_degre * sys.gamma
    crt = params.crt
    raster = sys.grad_raster_time
    T1 = -params.TR_degre / math.log(math.cos(math.radians(params.alpha_degre)))  # params.py's T1

    def _up(t):
        return math.ceil(t / crt - 1e-9) * crt

    T_esp = round((echo_spacing or 1 / params.fat_offres_freq) / raster) * raster
    dk = [1 / f for f in fov]

    # Slab-selective excitation; the flip is set below once TR is known (it
    # does not change any duration), so build a provisional one first.
    def _rf(flip_deg):
        rf, gz_ss, gz_ssr = pp.make_sinc_pulse(
            flip_deg / 180 * math.pi,
            duration=params.rf_dur_degre,
            slice_thickness=0.9 * fov[2],
            time_bw_product=params.rf_tb,
            system=sys,
            use='excitation',
            return_gz=True,
        )
        gz_ss = trap4ge(gz_ss, crt, sys)
        assert gz_ss.rise_time <= rf.delay
        gz_ss.delay = rf.delay - gz_ss.rise_time
        return rf, gz_ss, trap4ge(gz_ssr, crt, sys)

    rf, gz_ss, gz_ssr = _rf(5.0)

    # Readout: flat-top only (no ramp sampling), the dwell chosen so the lobe
    # takes ~read_frac of the echo spacing. Nx * dwell must be a raster
    # multiple so gradient blocks stay on the GE raster.
    dwell = round(read_frac * T_esp / Nx / sys.adc_raster_time) * sys.adc_raster_time
    dwell_min = dk[0] / sys.max_grad  # flat-top amplitude <= max_grad
    dwell = max(dwell, math.ceil(dwell_min / sys.adc_raster_time) * sys.adc_raster_time)
    while abs(Nx * dwell / raster - round(Nx * dwell / raster)) > 1e-6:
        dwell += sys.adc_raster_time
    t_read = Nx * dwell
    g_read = Nx * dk[0] / t_read  # Hz/m
    # Rise/fall built directly as raster multiples (like make_readout_grads's
    # POPE trapezoid): trap4ge's area-preserving rescale would perturb the
    # flat-top amplitude, i.e. the k-space step.
    t_ramp = _up(g_read / sys.max_slew)
    adc = pp.make_adc(Nx, duration=t_read, delay=t_ramp, system=sys)
    gx = pp.make_trapezoid(
        'x', system=sys, amplitude=g_read, rise_time=t_ramp, fall_time=t_ramp,
        flat_time=_up(t_read + adc.dead_time),
    )
    t_gx = pp.calc_duration(gx, adc)

    # Flyback rewinder, as slow as the spacing allows (lowest PNS).
    t_rw = T_esp - t_gx
    if t_rw < 2 * raster:
        raise ValueError(
            f'readout lobe ({t_gx * 1e6:.0f} us) leaves no room for a rewinder in the '
            f'{T_esp * 1e6:.0f} us echo spacing -- lower read_frac'
        )
    gx_rw = trap4ge(pp.make_trapezoid('x', system=sys, area=-gx.area, duration=t_rw), crt, sys)
    while pp.calc_duration(gx_rw) > T_esp - t_gx + 1e-12:
        t_rw -= raster
        gx_rw = trap4ge(pp.make_trapezoid('x', system=sys, area=-gx.area, duration=t_rw), crt, sys)
    delay_rw = T_esp - t_gx - pp.calc_duration(gx_rw)

    # Spoiler after the last echo plus y/z rewinders.
    # (its own slew limit: the spoiler ramp right after the last readout lobe
    # is the PNS peak, as in deGRE.py)
    gx_spoil = trap4ge(
        pp.make_trapezoid(
            'x', system=sys, area=_spoiler_cycles_area(Nx, dk[0], n_cycles_spoil),
            max_slew=spoil_slew * sys.gamma,
        ),
        crt, sys,
    )
    t_spoil = pp.calc_duration(gx_spoil)

    # Prephasers, stretched to fill TE1's padding (deGRE.py's argument).
    gx_full_area = g_read * (t_read + t_ramp)  # trapezoid without the dead-time tail
    pre_areas = {'x': -gx_full_area / 2, 'y': Ny * dk[1] / 2, 'z': Nz * dk[2] / 2}

    def _pre(duration):
        return tuple(
            trap4ge(pp.make_trapezoid(ax, system=sys, area=a, duration=duration), crt, sys)
            for ax, a in pre_areas.items()
        )

    te_base = (
        max(pp.calc_duration(rf), pp.calc_duration(gz_ss)) - (rf.delay + pp.calc_rf_center(rf)[0])
        + pp.calc_duration(gz_ssr) + adc.delay + t_read / 2
    )
    t_pre_min = _up(max(
        pp.calc_duration(pp.make_trapezoid(ax, system=sys, area=a))
        for ax, a in pre_areas.items()
    ))
    t_pre = max(t_pre_min, math.floor((T_esp - te_base) / crt + 1e-9) * crt)
    gx_pre, gy_pre, gz_pre = _pre(t_pre)

    def _pre_dur():
        return max(pp.calc_duration(g) for g in (gx_pre, gy_pre, gz_pre))

    while t_pre > t_pre_min and te_base + _pre_dur() > T_esp + 1e-9:
        t_pre -= crt
        gx_pre, gy_pre, gz_pre = _pre(t_pre)
    te_min = te_base + _pre_dur()
    if te_min > T_esp + 1e-9:
        raise ValueError(
            f'TE1 minimum {te_min * 1e3:.3f} ms exceeds echo spacing {T_esp * 1e3:.3f} ms'
        )
    delay_te = math.ceil((T_esp - te_min) / raster - 1e-9) * raster
    te1 = te_min + delay_te
    TE = te1 + T_esp * np.arange(n_echoes)

    gy_rew, gz_rew = (
        trap4ge(pp.make_trapezoid(ax, system=sys, area=-pre_areas[ax], duration=t_spoil), crt, sys)
        for ax in ('y', 'z')
    )

    # TR: RF block + ssr + TE1 padding + prephasers + train + spoiler.
    tr_min = (
        max(pp.calc_duration(rf), pp.calc_duration(gz_ss)) + pp.calc_duration(gz_ssr)
        + delay_te + _pre_dur() + n_echoes * t_gx + (n_echoes - 1) * (T_esp - t_gx)
        + max(t_spoil, pp.calc_duration(gy_rew), pp.calc_duration(gz_rew))
    )
    tr = _up(tr_min) if TR is None else TR
    if tr < tr_min - 1e-9:
        raise ValueError(f'TR {tr * 1e3:.3f} ms is below the minimum {tr_min * 1e3:.3f} ms')
    delay_tr = round((tr - tr_min) / raster) * raster
    tr_real = tr_min + delay_tr
    flip = math.degrees(math.acos(math.exp(-tr_real / T1)))  # Ernst
    rf, gz_ss, gz_ssr = _rf(flip)

    n_shots = (n_dummy_zloops + 1 + Nz) * Ny
    blocks = dict(
        rf=rf, gz_ss=gz_ss, gz_ssr=gz_ssr, adc=adc, gx=gx, gx_rw=gx_rw, delay_rw=delay_rw,
        gx_pre=gx_pre, gy_pre=gy_pre, gz_pre=gz_pre, gx_spoil=gx_spoil,
        gy_rew=gy_rew, gz_rew=gz_rew, delay_te=delay_te, delay_tr=delay_tr,
        n_dummy_zloops=n_dummy_zloops,
    )
    return MEGREDesign(
        n_echoes=n_echoes, N=N, fov=tuple(fov), dwell=dwell, t_read=t_read,
        echo_spacing=T_esp, TE=TE, TR=tr_real, flip_deg=flip, n_shots=n_shots,
        g_read_mT_m=g_read / sys.gamma * 1e3, sys=sys, blocks=blocks,
    )


def iter_encodes(N, n_dummy_zloops):
    """(iZ, iY) in acquisition order: iZ < 0 dummies, iZ = 0 receive-gain
    block, then the image."""
    _, Ny, Nz = N
    for iZ in range(-n_dummy_zloops, Nz + 1):
        for iY in range(Ny):
            yield iZ, iY


def generate_megre(
    params: Params,
    *,
    seqname: str = 'meGRE',
    encodes=None,
    write: bool = True,
    verbose: bool = True,
    **design_kwargs,
) -> tuple[pp.Sequence, MEGREDesign]:
    """Assemble (and by default write) the sequence. `encodes` replaces the
    full (iZ, iY) loop with an explicit list -- used to build a short
    worst-case snippet for the PNS check."""
    d = design_megre(params, **design_kwargs)
    b = d.blocks
    _, Ny, Nz = d.N
    pe1 = (np.arange(Ny) - Ny / 2) / Ny * 2
    pe2 = (np.arange(Nz) - Nz / 2) / Nz * 2
    seq = pp.Sequence(system=d.sys)
    rf, adc = b['rf'], b['adc']
    if encodes is None:
        encodes = list(iter_encodes(d.N, b['n_dummy_zloops']))
    rf_count = 1
    it = tqdm(encodes, desc='meGRE shots') if verbose else encodes
    for iZ, iY in it:
        dummy = iZ < 0
        y = pe1[iY] if iZ > 0 else 0.0
        z = pe2[iZ - 1] if iZ > 0 else 0.0
        ph = (0.5 * params.rf_phase_0 * rf_count**2) % 360.0 / 180 * math.pi
        rf.phase_offset = ph
        adc.phase_offset = ph
        rf_count += 1
        seq.add_block(rf, b['gz_ss'], pp.make_label('TRID', 'SET', 1 if dummy else 2))
        seq.add_block(b['gz_ssr'])
        if b['delay_te'] > 0:
            seq.add_block(pp.make_delay(b['delay_te']))
        seq.add_block(b['gx_pre'], pp.scale_grad(b['gy_pre'], y), pp.scale_grad(b['gz_pre'], z))
        for e in range(d.n_echoes):
            seq.add_block(b['gx'], *([] if dummy else [adc]))
            if e < d.n_echoes - 1:
                seq.add_block(b['gx_rw'])
                if b['delay_rw'] > 0:
                    seq.add_block(pp.make_delay(b['delay_rw']))
        seq.add_block(b['gx_spoil'], pp.scale_grad(b['gy_rew'], y), pp.scale_grad(b['gz_rew'], z))
        if b['delay_tr'] > 0:
            seq.add_block(pp.make_delay(b['delay_tr']))
    ok, report = seq.check_timing()
    if verbose:
        print('Timing check passed' if ok else f'Timing check FAILED:\n{report}')
    seq.set_definition('FOV', list(d.fov))
    seq.set_definition('Name', seqname)
    if write:
        os.makedirs(params.output_dir, exist_ok=True)
        seq.write(os.path.join(params.output_dir, f'{seqname}.seq'))
        _patch_scan_info(params, d)
    return seq, d


def _patch_scan_info(params: Params, d: MEGREDesign) -> None:
    """Overwrite scan_info.mat's *_degre keys with this scan's (the realized
    TE/TR/flip, not the prescribed ones). Like sequences/deGRE.py, skipped
    (not an error) when generate_arbepi has not created the file yet.
    hdf5storage.savemat updates just these keys in place."""
    fn = os.path.join(params.output_dir, 'scan_info.mat')
    if not os.path.exists(fn):
        print(f"  '{fn}' not found -- run generate_arbepi first if preprocess/ needs it.")
        return
    hdf5storage.savemat(
        fn,
        {
            'Nx_degre': d.N[0], 'Ny_degre': d.N[1], 'Nz_degre': d.N[2],
            'fov_degre': np.asarray(d.fov), 'n_echoes_degre': d.n_echoes,
            'TE_degre': d.TE, 'TR_degre': d.TR, 'alpha_degre': d.flip_deg,
            'dwell_degre': d.dwell,
        },
        fmt='7.3',
    )


def worst_case_encodes(N, n_dummy_zloops=2):
    """A short list that plays every distinct gradient shape at its largest:
    a dummy shot, the gain-cal shot, and the phase-encode corners."""
    _, Ny, Nz = N
    one = [(-n_dummy_zloops, 0), (0, 0)]
    corners = [(1, 0), (1, Ny - 1), (Nz, 0), (Nz, Ny - 1), (Nz // 2, Ny // 2)]
    return [e for e in one + corners for _ in range(3)]  # repeated: PNS has memory


# SNR calibration shared with simulate_fmri/session.py's THERMAL_NOISE (read
# there for the derivation): a fully sampled volume of the default EPI
# protocol has an image SNR of about 115 at TE 30 ms on a 3 T, 32-channel head
# session. Good to perhaps 30%.
EPI_SNR_REF = 115.0
EPI_TE_REF = 30e-3


def precision_estimate(
    d: MEGREDesign, params: Params, snr_scale: float = 1.0, t2star: float = T2STAR_S, t1=1.3
) -> tuple[float, float, float]:
    """(SNR at TE = 0, sd of R2* in 1/s, sd of the off-resonance frequency in
    Hz) per voxel from thermal noise alone (Cramer-Rao bound of a
    mono-exponential magnitude / linear-phase fit over this design's echoes).
    SNR scales from the EPI calibration as sqrt(dwell) x the Ernst steady
    state, at equal voxel volume and matrix. `snr_scale` < 1 for low-
    sensitivity regions. A bound, not a prediction: it ignores intravoxel
    dephasing, non-exponential decay and physiological noise."""
    from lib.readout_from_params import find_min_feasible_dwell

    def s_ss(tr, flip_deg):
        a, e1 = math.radians(flip_deg), math.exp(-tr / t1)
        return math.sin(a) * (1 - e1) / (1 - math.cos(a) * e1)

    # blip steps >= 2 in any undersampled mask; a step of 1 would give 6 us
    dwell_epi = find_min_feasible_dwell(2, 2, params)
    snr0_epi = EPI_SNR_REF / math.exp(-EPI_TE_REF / t2star)
    snr0 = (
        snr0_epi * math.sqrt(d.dwell / dwell_epi)
        * s_ss(d.TR, d.flip_deg) / s_ss(params.TR, params.fa) * snr_scale
    )
    te = d.TE
    w = (snr0 * np.exp(-te / t2star)) ** 2  # per-echo SNR squared
    sw, swt, swt2 = w.sum(), (w * te).sum(), (w * te**2).sum()
    sd_r2star = math.sqrt(sw / (sw * swt2 - swt**2))  # Fisher information of [ln S0, R2*]
    sd_hz = 1 / (2 * math.pi * math.sqrt(swt2 - swt**2 / sw))  # phase = phi0 + 2 pi f TE
    return snr0, sd_r2star, sd_hz


def snr_vs_degre(d: MEGREDesign, params: Params, t1: float = 1.3) -> float:
    """Echo-1 image SNR of this design over params' current deGRE (same coil,
    one average): voxel volume x sqrt(readout time x Ny x Nz) x the Ernst
    steady state, sqrt(TR / 2 T1) for TR << T1. Ignores T2* decay before TE1
    (2.24 ms in both) and T1 weighting of the tissue itself."""
    vol_new = np.prod(d.fov) / np.prod(d.N)
    vol_old = np.prod(params.fov_degre) / (params.Nx_degre * params.Ny_degre * params.Nz_degre)
    sys = params.sys
    dwell_old = max(sys.adc_raster_time, 1 / params.fov_degre[0] / sys.max_grad)
    dwell_old = math.ceil(dwell_old / sys.adc_raster_time) * sys.adc_raster_time
    ernst = lambda tr: math.sqrt((1 - math.exp(-tr / t1)) / (1 + math.exp(-tr / t1)))  # noqa: E731
    new = vol_new * math.sqrt(d.t_read * d.N[1] * d.N[2]) * ernst(d.TR)
    old = (vol_old * math.sqrt(params.Nx_degre * dwell_old * params.Ny_degre * params.Nz_degre)
           * ernst(params.TR_degre))
    return new / old


def _summarize(d: MEGREDesign, rep=None, snr=None, prec=None) -> str:
    s = (
        f'{d.n_echoes:3d} echoes  TE {d.TE[0] * 1e3:5.2f}..{d.TE[-1] * 1e3:6.2f} ms  '
        f'TR {d.TR * 1e3:6.2f} ms  flip {d.flip_deg:5.2f}  dwell {d.dwell * 1e6:4.0f} us '
        f'({1 / d.t_read:5.0f} Hz/px)  Gro {d.g_read_mT_m:5.2f} mT/m  '
        f'scan {d.scan_time:6.1f} s'
    )
    if snr is not None:
        s += f'  SNR(TE1) {snr:4.2f}x deGRE'
    if prec is not None:
        s += f'  sd R2* {prec[1]:4.2f}/s  sd B0 {prec[2]:4.2f} Hz'
    if rep is not None:
        s += f'  PNS {rep.peak_pns_percent:5.1f}%  acoustics {rep.acoustics_max_in_band:.3f}'
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--echoes', type=int, nargs='+', default=[6, 8, 12, 16, 20])
    ap.add_argument('--read-frac', type=float, default=0.5)
    ap.add_argument('--spoil-cycles', type=float, default=4)
    ap.add_argument('--spoil-slew', type=float, default=70.0)
    ap.add_argument('--dummy-zloops', type=int, default=2)
    ap.add_argument('--pns', action='store_true', help='run the GE PNS/acoustics check')
    ap.add_argument('--build', action='store_true',
                    help='write the full .seq of the first --echoes')
    args = ap.parse_args()
    params = load_params()
    kw = dict(read_frac=args.read_frac, n_cycles_spoil=args.spoil_cycles,
              spoil_slew=args.spoil_slew, n_dummy_zloops=args.dummy_zloops)
    for n in args.echoes:
        rep = None
        d = design_megre(params, n_echoes=n, **kw)
        if args.pns:
            from ge.check import check_seq_feasibility

            snippet, _ = generate_megre(
                params, write=False, verbose=False, n_echoes=n,
                encodes=worst_case_encodes(d.N, args.dummy_zloops), **kw,
            )
            rep = check_seq_feasibility(snippet, params.spec, pns_wt=tuple(params.PNSwt))
        print(_summarize(d, rep, snr_vs_degre(d, params), precision_estimate(d, params)))
    print('(precision: thermal-noise bound at the calibrated SNR, see precision_estimate)')
    if args.build:
        seq, d = generate_megre(params, n_echoes=args.echoes[0], **kw)
        print(f'built: {len(seq.block_events)} blocks, duration {seq.duration()[0]:.1f} s')


if __name__ == '__main__':
    main()
