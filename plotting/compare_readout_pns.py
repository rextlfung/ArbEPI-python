"""Symmetric-vs-POPE readout comparison: `uv run python -m plotting.compare_readout_pns`.

Builds two full ArbEPI sequences from the SAME sampling masks (params.seed
must be set) and identical nominal parameters -- FOV, matrix, ETL, dwell,
prescribed TE/TR -- differing only in the readout-gradient slew design:

  symmetric : rise = fall = blip = params.slew_derate (the pre-POPE design)
  pope      : params defaults (ro_slew_rise / ro_slew_fall / blip_slew)

and reports, for each: the whole-sequence GE feasibility numbers
(ge/check.py, the same gate `main.py --ge` applies), a per-TR
gradient/slew/PNS figure (plotting.plot_pns_one_tr), and one combined
overlay figure of the total PNS and the gx waveform over a few echo
spacings. Note the symmetric variant realizes a LONGER TE than prescribed
whenever the shared prescription is only achievable by the POPE variant --
calc_te_tr_delays warns and falls back to that variant's own min TE, and
the printed table reports both realized TEs; that TE difference at equal
PNS is POPE's payoff.

Outputs under <output_dir>/compare_pope/:
  PNS_one_tr_symmetric.png, PNS_one_tr_pope.png, compare_pns.png
"""

import os
import warnings
from dataclasses import replace

import matplotlib.figure
import numpy as np

from ge.check import (
    PNS_FIRST_CONTROLLED_MODE_THRESHOLD,
    PNS_NORMAL_MODE_THRESHOLD,
    check_seq_feasibility,
    sample_gradients_tesla_per_m,
)
from ge.pns import pns
from lib.mask2epi import max_blip_steps
from lib.readout_from_params import make_readout_grads_from_params
from params import load_params
from plotting.plotting import plot_pns_one_tr
from sampling.gen_sampling_masks import gen_sampling_masks
from sequences.ArbEPI import _compute_schedules, generate_arbepi


def _build(name, p, omegas):
    outdir = os.path.join(p.output_dir, name)
    p = replace(p, output_dir=outdir)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        seq = generate_arbepi(omegas, p, seqname=f'ArbEPI_{name}')
    te_warned = any('TE' in str(w.message) for w in caught)

    schedules, _ = _compute_schedules(
        omegas, p.ETL, p.Nshots, p.epi_trajectory, deltak=(1 / p.fov[1], 1 / p.fov[2]),
    )
    rg = make_readout_grads_from_params(*max_blip_steps(schedules), p)

    report = check_seq_feasibility(seq, p.spec, pns_wt=tuple(p.PNSwt))

    # Realized TE = the nominal echo's saved acquisition time (equals the
    # prescription when achievable, this variant's own min TE otherwise).
    import hdf5storage
    et = hdf5storage.loadmat(os.path.join(outdir, 'scan_info.mat'))['schedules'][0, 0, :, 2]
    te_realized = 0.5 * (et[p.ETL // 2 - 1] + et[p.ETL // 2])

    return dict(name=name, params=p, seq=seq, rg=rg, report=report,
                te_realized=te_realized, te_warned=te_warned)


def _overlay_figure(variants, compare_dir):
    fig = matplotlib.figure.Figure(figsize=(20, 10))
    ax_pns, ax_gx = fig.subplots(2, 1)
    colors = {'symmetric': 'tab:blue', 'pope': 'tab:red'}

    for v in variants:
        p = v['params']
        gw, dt = sample_gradients_tesla_per_m(v['seq'], time_range=(0.0, p.TR))
        t_ms = (np.arange(gw.shape[1]) + 0.5) * dt * 1e3
        pt, _ = pns(p.spec.rheobase / p.spec.alpha, p.spec.chronaxie, gw, dt,
                    wt=tuple(p.PNSwt))
        label = (f"{v['name']} (rise/fall/blip {p.ro_slew_rise:.0f}/"
                 f"{p.ro_slew_fall:.0f}/{p.blip_slew:.0f} T/m/s, "
                 f"peak {v['report'].peak_pns_percent:.1f}%)")
        ax_pns.plot(t_ms, pt, color=colors[v['name']], linewidth=1, label=label)

        # gx zoom: ~3 echo spacings around this variant's nominal TE echo.
        D = float(np.round(v['rg'].Tread + v['rg'].blip_duration, 9))
        t_c = v['te_realized']
        i0 = max(int((t_c - 1.5 * D) / dt), 0)
        i1 = min(int((t_c + 1.5 * D) / dt), gw.shape[1])
        ax_gx.plot((t_ms[i0:i1] - t_c * 1e3), gw[0, i0:i1] * 1e3,
                   color=colors[v['name']],
                   label=f"{v['name']} (echo spacing {D * 1e6:.0f} us)")

    ax_pns.axhline(PNS_NORMAL_MODE_THRESHOLD, color='tab:orange', linestyle=':',
                   label=f'{PNS_NORMAL_MODE_THRESHOLD:.0f}% (normal mode)')
    ax_pns.axhline(PNS_FIRST_CONTROLLED_MODE_THRESHOLD, color='tab:red', linestyle=':',
                   label=f'{PNS_FIRST_CONTROLLED_MODE_THRESHOLD:.0f}% (first controlled mode)')
    ax_pns.set_xlabel('time in first TR (ms)')
    ax_pns.set_ylabel('total PNS (% of threshold)')
    ax_pns.legend(loc='upper right', fontsize=9)
    ax_pns.grid(True)
    ax_pns.set_title('Total PNS over one TR, symmetric vs POPE readout')

    ax_gx.set_xlabel('time relative to each variant\'s nominal TE echo (ms)')
    ax_gx.set_ylabel('gx (mT/m)')
    ax_gx.legend(loc='upper right', fontsize=9)
    ax_gx.grid(True)
    ax_gx.set_title('Readout gradient, ~3 echo spacings around the TE echo '
                    '(POPE: slow rise on the left of each lobe, fast fall on the right)')

    fig.tight_layout()
    fig.savefig(os.path.join(compare_dir, 'compare_pns.png'), dpi=300)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare symmetric vs POPE (asymmetric) EPI readout PNS/TE.'
    )
    parser.add_argument('--rise', type=float, default=None,
                        help='override ro_slew_rise (T/m/s) for the POPE variant')
    parser.add_argument('--fall', type=float, default=None,
                        help='override ro_slew_fall (T/m/s) for the POPE variant')
    parser.add_argument('--blip', type=float, default=None,
                        help='override blip_slew (T/m/s) for the POPE variant')
    parser.add_argument('--te', type=float, default=None,
                        help='override prescribed TE (s) for BOTH variants; prescribing an '
                             'unachievably short TE makes each variant fall back to (and so '
                             'report) its own minimum TE')
    args = parser.parse_args()

    p0 = load_params()
    assert p0.seed is not None, 'set params.seed so both variants share one mask'
    compare_dir = os.path.join(p0.output_dir, 'compare_pope')
    os.makedirs(compare_dir, exist_ok=True)
    p0 = replace(p0, output_dir=compare_dir)
    if args.rise is not None:
        p0 = replace(p0, ro_slew_rise=args.rise)
    if args.fall is not None:
        p0 = replace(p0, ro_slew_fall=args.fall)
    if args.blip is not None:
        p0 = replace(p0, blip_slew=args.blip)
    if args.te is not None:
        p0 = replace(p0, TE=args.te)

    print('generating sampling masks (shared by both variants)...')
    omegas = gen_sampling_masks(p0.R, p0, rng=np.random.default_rng(p0.seed))

    p_sym = replace(p0, ro_slew_rise=p0.slew_derate, ro_slew_fall=p0.slew_derate,
                    blip_slew=p0.slew_derate)
    variants = [
        _build('symmetric', p_sym, omegas),
        _build('pope', p0, omegas),
    ]

    for v in variants:
        plot_pns_one_tr(v['seq'], v['params']).savefig(
            os.path.join(compare_dir, f"PNS_one_tr_{v['name']}.png"), dpi=300
        )
    _overlay_figure(variants, compare_dir)

    print()
    print(f"{'':>22} {'symmetric':>12} {'pope':>12}")
    rows = [
        ('rise/fall/blip (T/m/s)',
         *(f"{v['params'].ro_slew_rise:.0f}/{v['params'].ro_slew_fall:.0f}/"
           f"{v['params'].blip_slew:.0f}" for v in variants)),
        ('peak PNS (%)', *(f"{v['report'].peak_pns_percent:.1f}" for v in variants)),
        ('echo spacing (us)',
         *(f"{(v['rg'].Tread + v['rg'].blip_duration) * 1e6:.0f}" for v in variants)),
        ('Tread (us)', *(f"{v['rg'].Tread * 1e6:.0f}" for v in variants)),
        ('Nfid', *(f"{v['rg'].Nfid}" for v in variants)),
        ('prescribed TE (ms)', *(f"{v['params'].TE * 1e3:.2f}" for v in variants)),
        ('realized TE (ms)',
         *(f"{v['te_realized'] * 1e3:.2f}{' (fell back)' if v['te_warned'] else ''}"
           for v in variants)),
        ('max slew (T/m/s)', *(f"{v['report'].max_slew_T_m_s:.0f}" for v in variants)),
    ]
    for label, a, b in rows:
        print(f'{label:>22} {a:>12} {b:>12}')
    print()
    for v in variants:
        status = 'OK' if v['report'].peak_pns_percent < PNS_NORMAL_MODE_THRESHOLD else 'OVER'
        print(f"{v['name']}: peak PNS {v['report'].peak_pns_percent:.2f}% "
              f'({status} vs {PNS_NORMAL_MODE_THRESHOLD:.0f}% normal mode)')
    print(f'\nWrote {compare_dir}/PNS_one_tr_symmetric.png, PNS_one_tr_pope.png, '
          f'compare_pns.png')


if __name__ == '__main__':
    main()
