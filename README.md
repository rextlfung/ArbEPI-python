# ArbEPI-python: Python/pypulseq port of ArbEPI

Python port of [ArbEPI](../ArbEPI) (MATLAB/Pulseq), using
[pypulseq](../pypulseq) as the Pulseq layer. Generates fast, vendor-agnostic
3D-EPI sequences from arbitrary 2D `(ky, kz)` sampling masks in the
phase-encode-partition plane.

## Scope

This port covers Pulseq `.seq` sequence generation only. A few things from
the original MATLAB repo are handled differently — see below.

- **GE `.pge` export**: `seq2ge/` is a pure-Python port of the PulCeq/pge2
  toolchain (`seq2ceq`, `writeceq`, and the `pge2.pns`/`check_grad_acoustics`
  feasibility checks), and is the operative path for `seq2ge/ge_export.py`/
  `main.py --ge` — no MATLAB round trip, no sibling `../pulseq`/`../toppe`/
  `../PulCeq`/`../ArbEPI` checkouts required. Validated field-by-field and,
  for two of the four default sequences, byte-for-byte against real MATLAB
  output, including end to end through `main.py --ge` itself (see `seq2ge/`'s
  module docstrings and `CLAUDE.md`). See [GE export](#ge-export-pge) below.
- **Fat-sat RF pulse**: the MATLAB original designs this via GE's
  `toppe.utils.rf.makeslr` (min-phase SLR), which has no Python equivalent.
  This port uses pypulseq's built-in `make_gauss_pulse` instead — a simpler
  design with a less sharp spectral profile.
- **Plotting** (`plotting/plotting.py`) covers the sampling mask, k-space trajectory,
  point-spread-function, and single-TR pulse-diagram plots. The interactive
  scroll/slider mask viewer from the MATLAB repo is not ported. The
  single-TR plot (`plot_one_tr`) isn't a custom pulse-diagram renderer —
  it's a thin wrapper around pypulseq's own `Sequence.plot()`. Mask/PSF/
  trajectory plots take an optional `frame_idx` to select one frame out of
  a multi-frame run; see `plotting/plotting.py`'s module docstring for why the
  per-frame trajectory plot draws through exact-sliced ADC samples rather
  than the fine continuous line pypulseq's `calculate_kspace()` returns
  for the whole sequence. `plotting/plot_last_run.py` drives all of this against
  the most recent `output/` run (`python main.py --plot`, or standalone via
  `python -m plotting.plot_last_run`).
- **Poisson-disc sampling** (`sampling/pd_sample.py`) is a local
  reimplementation, not a dependency on [SigPy](https://github.com/mikgroup/sigpy)
  (whose `sigpy.mri.poisson` both `../ArbEPI/lib/pd_sample.m` and this
  module's algorithm are based on). SigPy was tried directly and rejected:
  its own `poisson()` has an unbounded `while slope_min < slope_max`
  binary-search loop with no iteration cap, which hangs forever on
  small/coarse grids where no achievable density slope lands within `tol`
  of the target acceleration (reproduced independently of any code in this
  repo — confirmed via `sigpy.mri.poisson` alone). Separately, both
  `../ArbEPI/lib/pd_sample.m` and this module's own first version had a
  *different* bug (not reseeding the point-placement RNG identically on
  every binary-search iteration, unlike real SigPy), which made the search
  non-convergent and slow rather than truly infinite. A third bug (a
  missing `nx*ny` active-list cap that real SigPy has) let the
  point-placement active list grow unboundedly in the same
  radius-floor/dense-center regime. `pd_sample.py`'s docstring has the
  full writeup of all three; the local implementation fixes all of them
  and adds a bounded outer-search iteration cap (`max_search_iters`) that
  SigPy itself lacks. The point-placement core is still JIT-compiled with
  [numba](https://numba.pydata.org/) -- a narrow, single-function
  dependency (unlike depending on the `sigpy` package wholesale) -- since
  even with all three fixes it's an inherently sequential loop that can
  run hundreds of thousands of iterations for worst-case seeds, and pure
  Python can't get there without JIT (measured ~1-12s/frame in pure
  Python vs. ~0.02-0.2s/frame JIT-compiled, at production scale).

## Requirements

Managed with [uv](https://docs.astral.sh/uv/):

```
uv sync --extra test
```

Depends on `pypulseq` (from PyPI), numpy, scipy, matplotlib, hdf5storage,
and numba (see `pyproject.toml`; versions pinned in `uv.lock`).

## Getting started

1. Edit `params.py` (`load_params()`) to configure the experiment — scan
   geometry, timing, sampling method (`sampling_method`: `'caipi'`,
   `'ticaipi'`, `'pd'`, or `'rand'`), echo-train ordering (`epi_trajectory`:
   `'laminar'` or `'radial'`, see [Demo](#demo) below), and `seed` (an int
   for a reproducible sampling mask across runs, or `None` for a fresh one
   each time).
2. Run `main.py` to generate all four sequences:
   ```
   uv run python main.py
   ```
   Or step by step:
   ```python
   from params import load_params
   from sampling.gen_sampling_masks import gen_sampling_masks
   from sequences.arbepi import generate_arbepi
   from sequences.epical import generate_epical
   from sequences.gre import generate_gre
   from sequences.noise import generate_noise

   params = load_params()
   omegas = gen_sampling_masks(params.R, params)
   generate_arbepi(omegas, params)   # writes output/ArbEPI.seq, output/samp_locs.mat
   generate_epical(params)           # writes output/EPIcal.seq, output/kxoe<Nx>.mat
   generate_gre(params)              # writes output/GRE.seq
   generate_noise(params)            # writes output/noise.seq
   ```
   `generate_epical` and `generate_noise` must run after `generate_arbepi` —
   they load `output/samp_locs.mat`. All outputs go to `params.output_dir`
   (default `output/`, gitignored).
3. Add `--plot` to also write diagnostic plots (`mask.png`, `psf.png`,
   `trajectory.png`, `one_tr.png`) via `plotting/plot_last_run.py`, and/or `--ge` to
   also export each sequence to GE `.pge` (see [GE export](#ge-export-pge)
   below):
   ```
   uv run python main.py --plot --ge
   ```

There is no automated end-to-end pytest suite against MATLAB for `.seq`
generation itself (no MATLAB install was available during that initial
port — see `tests/` for unit tests on algorithm invariants instead,
including an independent check that reads k-space back out of the
assembled sequence and confirms it matches the sampling schedule).
`sampling_method='caipi'` is deterministic (no RNG) and is the easiest
configuration to sanity-check by hand. (The separate GE `.pge` export path
below *was* validated against real MATLAB output, once a MATLAB install
became available — see the GE export section and `CLAUDE.md`.)

## Demo

Diagnostic plots from a default-params run (`main.py --plot`; see
`plotting/plot_last_run.py`), R = 9, `sampling_method='pd'`:

| | |
|---|---|
| ![Sampling mask](docs/demo/mask.png) | ![Point spread function](docs/demo/psf.png) |
| 2D Poisson-disc `(ky, kz)` sampling mask, one frame | Point spread function for that mask |

`epi_trajectory` selects how `mask2epi_*` partitions that mask into each
shot's echo train — `'laminar'` (ky non-decreasing rows) vs. `'radial'`
(every shot a spoke through k-space center). Everything below is from the
same `params.seed` (so the same sampling mask), one run per
`epi_trajectory` value; `main.py --plot` always writes `trajectory.png`/
`one_tr.png` for whichever setting was active, renamed here for
side-by-side comparison:

| `epi_trajectory = 'laminar'` | `epi_trajectory = 'radial'` |
|---|---|
| ![Laminar trajectory](docs/demo/trajectory_laminar.png) | ![Radial trajectory](docs/demo/trajectory_radial.png) |

And the single-TR pulse diagram (`plot_one_tr`, a thin wrapper around
pypulseq's own `Sequence.plot()`) for one shot under each ordering — note
`radial`'s larger, single-echo Gy blip spike (~t=38ms) vs. `laminar`'s
comparatively even blip sizes throughout the train:

| `epi_trajectory = 'laminar'` | `epi_trajectory = 'radial'` |
|---|---|
| ![Laminar single-TR pulse diagram](docs/demo/one_tr_laminar.png) | ![Radial single-TR pulse diagram](docs/demo/one_tr_radial.png) |

## GE export (`.pge`)

Pure Python, no MATLAB install or sibling-repo checkouts required:

```python
from seq2ge.ge_export import export_to_ge
export_to_ge('output/ArbEPI.seq', 'output/ArbEPI', params)
```

`export_to_ge` runs a feasibility check (hardware limits, PNS,
acoustic-resonance — see below) and raises `RuntimeError` if it fails,
then writes the `.pge` via `seq2ge/seq2ceq.py` + `seq2ge/writeceq.py`. Verified
end-to-end on a small test sequence and on a full-scale default-params run
(`main.py`'s output): the resulting `.pge` files match freshly-regenerated
real MATLAB output (`seq2ceq`/`pge2.writeceq`) byte-for-byte for two of the
four default sequences and to within a single float32 ULP on a derived
header field for the other two — see `CLAUDE.md` for the full record.

**Hardware limits are keyed off `--scanner`** (`GE_MR750` or `GE_UHP`, see
`scanners.py`): `load_params(scanner=...)` builds `params.spec` (a
`ScannerSpec`) once and derives `sys.max_grad`/`sys.max_slew` for `.seq`
generation from the same instance that `seq2ge/check.py`/`seq2ge/writeceq.py` read
directly, so they can't drift out of sync with each other. `main.py --ge`
also calls `seq2ge.ge_export.check_ge_feasibility()` on all four sequences —
running the hardware/PNS/acoustics check without writing a `.pge` —
*before* exporting any of them, so an infeasibility surfaces immediately
instead of after several full exports. PNS is a physiological safety
limit, not a hardware one — `PNSwt` (a separate `Params` field, not part
of `ScannerSpec`, since it's phantom-vs-human scan context) defaults to
the IEC 60601-2-33:2022-recommended `[0.8, 1.0, 0.7]`; `[0, 0, 0]` disables
the PNS check entirely and is only appropriate for phantom/non-human
scanning. Acoustic-resonance is checked but never blocks export — it's a
`WARN` in the report, matching MATLAB's own `check_grad_acoustics.m`,
which only ever calls `warning(...)`, never `error(...)`, when over
threshold.

**`main.py --ge` now fails by default on three of the four sequences —
this is a real finding, not a bug.** `PNSwt` was `[0, 0, 0]` for the
entire lifetime of this port until now, so PNS was never actually
evaluated in any `--ge` run to date (weight zero makes the per-channel
contribution zero regardless of the real waveform). With the current
default weights (validated against MATLAB's real per-instance pipeline to
~0.02 percentage points via `seq2ge/pns.py`/`matlab_reference/dump_pns_peak.m`),
`EPIcal`/`ArbEPI`/`GRE` all *exceed* MATLAB's own PNS throw threshold
(>80% "exceeds normal mode", >100% "exceeds first controlled mode") at
default params — see `CLAUDE.md` for the exact numbers. `seq2ge/check.py`
matches MATLAB's throw condition exactly, so this now correctly blocks
`main.py --ge` for these three sequences. This needs a resolution (lower
slew rate or lengthen blip rise times) before scanning a human on these
default sequences.

## Architecture

Only entry points and tightly-coupled global config sit at the repo root —
mirroring `../ArbEPI` having `params.m`/`main.m` directly at its repo root
— everything else lives in a subpackage grouped by role:

```
params.py                   Params dataclass + load_params() (replaces params.m)
scanners.py                 ScannerSpec hardware profiles (GE_MR750, GE_UHP)
main.py                     Generate all 4 sequences, mirrors ArbEPI/main.m; --scanner/--plot/--ge flags
sampling/                   Sampling mask generators (caipi, ticaipi, pd, rand)
lib/
  mask2epi.py                Core algorithm: partitions a 2D mask into EPI trajectories
                              (mask2epi_laminar / mask2epi_radial, selected by params.epi_trajectory)
  trap4ge.py                 GE-raster trapezoid rounding (from ../PulCeq)
  (RF/gradient helpers, TE/TR delay calculation)
sequences/                  ArbEPI, EPIcal, GRE, noise sequence assembly
plotting/
  plotting.py                Diagnostic plots (mask/PSF/trajectory/single-TR)
  plot_last_run.py           Drives plotting.py against the most recent output/ run
seq2ge/                      Pure-Python GE .pge toolchain (Pulseq -> GE), wired into main.py --ge
  ge_export.py                 GE feasibility checking / .pge export entry points
  ceq.py                       Ceq/ParentBlock/Segment data contract
  blocks.py                    Block type/dynamics/comparison (port of PulCeq's compareblocks.m etc.)
  seq2ceq.py                   Pulseq Sequence -> Ceq (port of seq2ceq.m)
  writeceq.py                  Ceq -> binary .pge (port of writeceq.m)
  read_pge.py                  .pge binary reader, used only for validation
  pns.py                       PNS check (port of pge2.pns.m)
  acoustics.py                 Acoustic-resonance check (port of check_grad_acoustics.m)
  check.py                     Combines hardware/PNS/acoustics into one FeasibilityReport
  validate_against_matlab.py   Field-by-field / byte-for-byte comparison vs. MATLAB output
matlab_reference/            One-off scripts for re-validating seq2ge/ against a real MATLAB install
                              (not called by any Python code; MATLAB is not needed to use this repo)
  dump_ceq.m                 Dumps seq2ceq.m output for seq2ge/validate_against_matlab.py
  dump_pns_test.m            Dumps pge2.pns.m's synthetic sub_test() reference for seq2ge/validate_pns.py
  dump_pns_peak.m            Dumps peak PNS%% across every segment instance of a real sequence
  dump_acoustics_test.m      Dumps a check_grad_acoustics.m reference for seq2ge/acoustics.py's validation
  dump_acoustics_blockrange.m  Dumps the former MATLAB export path's exact acoustics blockRange check
tests/                       Unit tests (pytest)
docs/demo/                   Static images embedded in this README's Demo section
```

Index convention: internal computation is 0-based throughout (mask2epi's
`schedule`, sampling masks, etc.). `output/samp_locs.mat` is written with
`schedules` converted to 1-based (matching what MATLAB-side reconstruction
code expects) — `parts` is already a 1-based shot label with 0 = unsampled,
so it needs no conversion.

`.mat` files (`samp_locs.mat`, `kxoe<Nx>.mat`) are written via `hdf5storage`
in MATLAB v7.3 format (HDF5-based), matching the original MATLAB code's
`save(..., '-v7.3')`. `scipy.io.savemat`/`loadmat` cannot write or read
v7.3 at all — use `hdf5storage.loadmat` (or `h5py` directly) to read these
files from Python, not `scipy.io.loadmat`.

See [../ArbEPI/README.md](../ArbEPI/README.md) for background on the
sampling methods and the `mask2epi` algorithm design.
