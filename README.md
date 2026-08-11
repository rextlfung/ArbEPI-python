# ArbEPI-python: Python/pypulseq port of ArbEPI

Python port of [ArbEPI](../ArbEPI) (MATLAB/Pulseq), using
[pypulseq](../pypulseq) as the Pulseq layer. Generates fast, vendor-agnostic
3D-EPI sequences from arbitrary 2D `(ky, kz)` sampling masks in the
phase-encode-partition plane.

## Scope

This port covers Pulseq `.seq` sequence generation only. A few things from
the original MATLAB repo are handled differently — see below.

- **GE `.pge` export** is not reimplemented in Python. `seq2ceq` (from
  [PulCeq](../PulCeq)) accepts a `.seq` filename directly, so
  `matlab/write_to_ge_from_seq.m` (adapted from `../ArbEPI/lib/write_to_ge.m`)
  reads a Python-generated `.seq` file and runs the existing, unmodified GE
  export logic. `ge_export.py` calls this via `matlab -batch` — see
  [GE export](#ge-export-pge) below.
- **Fat-sat RF pulse**: the MATLAB original designs this via GE's
  `toppe.utils.rf.makeslr` (min-phase SLR), which has no Python equivalent.
  This port uses pypulseq's built-in `make_gauss_pulse` instead — a simpler
  design with a less sharp spectral profile.
- **Plotting** (`plotting.py`) covers the sampling mask, k-space trajectory,
  point-spread-function, and single-TR pulse-diagram plots. The interactive
  scroll/slider mask viewer from the MATLAB repo is not ported. The
  single-TR plot (`plot_one_tr`) isn't a custom pulse-diagram renderer —
  it's a thin wrapper around pypulseq's own `Sequence.plot()`. Mask/PSF/
  trajectory plots take an optional `frame_idx` to select one frame out of
  a multi-frame run; see `plotting.py`'s module docstring for why the
  per-frame trajectory plot draws through exact-sliced ADC samples rather
  than the fine continuous line pypulseq's `calculate_kspace()` returns
  for the whole sequence. `plot_last_run.py` drives all of this against
  the most recent `output/` run (`python main.py --plot`, or standalone).
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
   `'ticaipi'`, `'pd'`, or `'rand'`).
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
   `trajectory.png`, `one_tr.png`) via `plot_last_run.py`, and/or `--ge` to
   also export each sequence to GE `.pge` (see [GE export](#ge-export-pge)
   below):
   ```
   uv run python main.py --plot --ge
   ```

There is no automated end-to-end test suite against MATLAB (no MATLAB
install was available while porting — see `tests/` for unit tests on
algorithm invariants instead, including an independent check that reads
k-space back out of the assembled sequence and confirms it matches the
sampling schedule). `sampling_method='caipi'` is deterministic (no RNG) and
is the easiest configuration to sanity-check by hand.

## GE export (`.pge`)

Requires a local MATLAB install with `pulseq`, `toppe`, and `PulCeq` checked
out as sibling directories to this repo (`../pulseq`, `../toppe`,
`../PulCeq`), plus `../ArbEPI` (for `check_grad_acoustics.m`):

```python
from ge_export import export_to_ge
export_to_ge('output/ArbEPI.seq', 'output/ArbEPI', params)
```

`export_to_ge` locates MATLAB via `PATH` or, on macOS, by globbing
`/Applications/MATLAB_*.app/bin/matlab` if not on `PATH`. Verified
end-to-end on a small test sequence and on a full-scale default-params run
(`main.py`'s output), both producing a valid, non-trivial `.pge` file via
the real `seq2ceq`/`pge2.check`/`pge2.writeceq` toolchain — no MATLAB-side
errors on either.

**If you change `params.py`'s `sys.max_grad`/`sys.max_slew`** (used for
`.seq` generation), also check `g_max`/`slew_max`/`PNSwt` (separate fields,
used only here, by MATLAB's `pge2.check`) — they don't auto-sync, and note
the unit difference: `g_max` is in G/cm, `sys.max_grad` in mT/m
(`1 G/cm = 10 mT/m`). A `.seq` file can build and pass
`seq.check_timing()` cleanly and still fail `pge2.check` at export time
with a hardware-limit or PNS error if these have drifted out of sync with
each other. PNS is a physiological safety limit, not a hardware one —
`PNSwt = [0, 0, 0]` disables it and is only appropriate for phantom/
non-human scanning.

## Architecture

Mirrors the original MATLAB repo's structure — no top-level package
directory, matching `../ArbEPI` having `params.m`/`main.m` directly at its
repo root alongside `src/` and `lib/`:

```
params.py                Params dataclass + load_params() (replaces params.m)
trap4ge.py                GE-raster trapezoid rounding (from ../PulCeq)
mask2epi.py                Core algorithm: partitions a 2D mask into EPI trajectories
sampling/                   Sampling mask generators (caipi, ticaipi, pd, rand)
lib/                        RF/gradient helpers, TE/TR delay calculation
sequences/                  ArbEPI, EPIcal, GRE, noise sequence assembly
plotting.py                 Diagnostic plots (mask/PSF/trajectory/single-TR)
plot_last_run.py            Drives plotting.py against the most recent output/ run
ge_export.py                Python -> MATLAB bridge for .pge export
matlab/
  write_to_ge_from_seq.m    MATLAB-side GE export, called by ge_export.py
main.py                     Generate all 4 sequences, mirrors ArbEPI/main.m; --plot/--ge flags
tests/                       Unit tests (pytest)
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
