# Start here

ArbEPI-python takes you from a sampling pattern to an activation map. The work
happens in six steps, each with its own page and (for the later ones) its own
Python environment.

## The steps, in order

| # | Step | You do | You get | Page |
|---|---|---|---|---|
| 1 | **Configure** | edit `params.py`'s `load_params()` (resolution, matrix size, `TE`, `volume_tr`, `R`, `ETL`, sampling method) | | {doc}`generate` |
| 2 | **Generate sequences** | `uv run python main.py --plot` | four Pulseq files `noise.seq`, `EPIcal.seq`, `deGRE.seq`, `ArbEPI.seq`, plus `scan_info.mat` and diagnostic plots in `output/` | {doc}`generate`, {doc}`mask2epi` |
| 3 | **Export and scan** | `uv run python main.py --ge`, copy the `.pge` files to the scanner (`ge.coppe`), run all four scans | raw ScanArchives | {doc}`ge_export`, {doc}`ge` |
| 4 | **Preprocess** | `python -m preprocess.batch_preprocess <datdir> <seq> ...` | one `<seq>_preprocessed.h5` per sequence: zero-filled k-space, sensitivity and B0 maps | {doc}`preprocess` |
| 5 | **Reconstruct** | `python -m recon.rss ...` or `python -m recon.sense ...` | image time series (`.h5`, `.nii.gz`) | {doc}`recon` |
| 6 | **Analyze** | `python -m analyze ...` | t- and z-score maps | {doc}`analyze` |

Without a scanner, {doc}`simulate_fmri` replaces step 3 by simulating the raw
data of all four scans, so steps 4 to 6 run unchanged on it.

## Why four sequences

Each scan has one job, and all four are needed to reconstruct the fifth:

| Sequence | Purpose |
|---|---|
| `noise` | receive-chain noise only: sets the whitening matrix $W$ |
| `EPIcal` | the EPI readout with the blips off: calibrates the readout delay and the odd/even echo phase |
| `deGRE` | a dual-echo gradient echo: coil compression, sensitivity maps, the B0 field map and R2\* |
| `ArbEPI` | the accelerated 3D-EPI time series itself |

The sampling schedule never goes through the scanner. It travels in
`scan_info.mat`, which preprocessing needs to put each echo back at its
`(ky, kz)` location, so keep that file with each session's data
(`<datdir>/seqs/<seq>/scan_info.mat`).

## Environments

The stages need conflicting dependencies, so they use separate virtual
environments, all created from the same `uv.lock`.

| Environment | Used for | Created with |
|---|---|---|
| `.venv` | sequence generation, GE export, plotting, `analyze/` | `uv sync --extra test` (add `--extra analyze`) |
| `.venv-preprocessing` | `preprocess/` on real scanner data (needs GE's proprietary `GERecon`, Python 3.10, `numpy<2`) | see {doc}`preprocess` |
| `.venv-recon` | `recon/` (needs PyTorch) | `UV_PROJECT_ENVIRONMENT=.venv-recon uv sync --extra recon --extra test` |
| `.venv-simulate` | `simulate_fmri/` (also runs preprocessing and recon on the simulated session, no `GERecon` needed) | `UV_PROJECT_ENVIRONMENT=.venv-simulate uv sync --extra simulate --extra preprocessing --extra recon --extra test` |

## Things that trip people up

- **Order matters when generating.** `EPIcal` and `noise` read
  `scan_info.mat`, which `ArbEPI` writes, so generate `ArbEPI` first
  (`main.py` does).
- **`scan_info.mat` is HDF5 (v7.3).** Read it with `hdf5storage.loadmat`, not
  `scipy.io.loadmat`.
- **`TE` and `TR` are warnings, not errors.** If the prescribed `TE` is
  unreachable the sequence still builds, with the minimum achievable value.
  Check the printed output.
- **PNS limits are a safety gate.** `--ge` refuses a sequence over 80% of the
  normal-mode PNS limit. Lower the slew settings rather than the PNS weights.
- **`uv run pytest` needs the `test` extra** (`uv sync --extra test`), or it
  fails to import `numpy`.

## Where the reasoning lives

The user guides say *what to run*. For *why it is built that way*, with the
measurements behind each default, read the design notes:
{doc}`design_sequences`, {doc}`design_preprocess`, {doc}`design_recon`,
{doc}`design_analysis`.
