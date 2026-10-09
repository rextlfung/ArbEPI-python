"""python -m analyze: activation maps of a reconstruction.

    python -m analyze --recon <name>_recon.h5 --tr 0.506 \\
        --onsets 0 40 80 120 --duration 20 --out maps/ --scales all

Onsets are seconds from the first frame kept (after --n-discard). One condition
from the command line; several conditions are the Python API's
(analyze.run.analyze_recon with an ExperimentParams).
"""

from __future__ import annotations

import argparse

from analyze.design import Condition, ExperimentParams
from analyze.run import analyze_recon


def _scale(s: str):
    return s if s in ("sum", "all") else int(s)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m analyze", description=__doc__.split("\n\n")[0])
    ap.add_argument("--recon", required=True, help="<name>_recon.h5 from recon.sense")
    ap.add_argument("--tr", type=float, required=True, help="volume TR (s)")
    ap.add_argument("--onsets", type=float, nargs="+", required=True, help="task block onsets (s)")
    ap.add_argument("--duration", type=float, nargs="+", required=True,
                    help="block duration (s), one value or one per onset")
    ap.add_argument("--n-discard", type=int, default=0, help="leading frames to drop")
    ap.add_argument("--drift-cutoff", type=float, default=128.0,
                    help="DCT drift cutoff period (s); 0 -> a linear trend instead")
    ap.add_argument("--no-ar1", action="store_true", help="ordinary least squares, no prewhitening")
    ap.add_argument("--scales", nargs="+", default=["sum"], type=_scale,
                    help="'sum', 'all', or component indices (mslr runs)")
    ap.add_argument("--mask", default="auto",
                    help="'auto' (FSL bet if on PATH, else brainextractor), 'fsl', "
                         "'python', 'threshold', or a NIfTI mask path")
    ap.add_argument("--bet-frac", type=float, default=0.5,
                    help="BET fractional intensity threshold")
    ap.add_argument("--voxel-size", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
                    help="mm; default from the recon's .json sidecar")
    ap.add_argument("--q", type=float, default=0.05, help="FDR level for the summary")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--tag", help="output file prefix (default: the recon's name)")
    a = ap.parse_args(argv)

    params = ExperimentParams(a.tr, [Condition("task", tuple(a.onsets), tuple(a.duration))],
                              n_discard=a.n_discard, drift_cutoff_s=a.drift_cutoff or None)
    maps = analyze_recon(a.recon, params, scales=tuple(a.scales), brain=a.mask, out_dir=a.out,
                         ar1=not a.no_ar1, q=a.q, tag=a.tag, voxel_size_mm=a.voxel_size,
                         bet_frac=a.bet_frac)
    for label, m in maps.items():
        print(f"{label}: dof {m.dof:.0f}, rho {m.rho:.3f}, "
              f"{m.n_fdr}/{m.n_voxels} voxels survive FDR q={a.q} (|t| >= {m.t_fdr:.2f})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
