# ArbEPI-python

Fast, vendor-agnostic 3D-EPI MRI sequences from arbitrary 2D `(ky, kz)`
sampling masks, a pure-Python GE `.pge` exporter, and the preprocessing,
reconstruction and analysis stages that turn the scanner's data back into
images and activation maps. A Python/[pypulseq](https://github.com/imr-framework/pypulseq)
port of the MATLAB ArbEPI.

Where to start:

| I want to... | Read |
|---|---|
| generate sequences, pick parameters, use my own sampling mask | {doc}`overview` |
| export to a GE scanner or copy `.pge` files over | {doc}`ge` |
| turn raw ScanArchives into reconstruction-ready files | {doc}`preprocess` |
| reconstruct images (RSS, SENSE, low-rank, wavelet-TV, B0) | {doc}`recon` |
| compute task activation maps | {doc}`analyze` |
| simulate a whole session without a scanner | {doc}`simulate_fmri` |
| look up a function, class or module | {doc}`api/index` |

```{toctree}
:caption: User guide
:maxdepth: 2

overview
ge
preprocess
recon
analyze
simulate_fmri
notebooks
```

```{toctree}
:caption: Reference
:maxdepth: 2

api/index
```
