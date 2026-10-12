# ArbEPI-python

Fast, vendor-agnostic 3D-EPI MRI sequences from arbitrary 2D `(ky, kz)`
sampling masks, a pure-Python GE `.pge` exporter, and the preprocessing,
reconstruction and analysis stages that turn the scanner's data back into
images and activation maps. A Python/[pypulseq](https://github.com/imr-framework/pypulseq)
port of the MATLAB ArbEPI.

New to the project? Read {doc}`start_here`: the steps in order, what each
needs, and the mistakes people make first.

| I want to... | Read |
|---|---|
| see the whole pipeline and what runs where | {doc}`start_here`, {doc}`overview` |
| generate sequences, pick parameters | {doc}`generate` |
| use my own sampling mask | {doc}`custom_masks` |
| understand how echo trains are ordered | {doc}`mask2epi` |
| export to a GE scanner or copy `.pge` files over | {doc}`ge_export`, {doc}`ge` |
| turn raw ScanArchives into reconstruction-ready files | {doc}`preprocess` |
| reconstruct images, and see the cost functions | {doc}`recon` |
| compute task activation maps | {doc}`analyze` |
| simulate a whole session without a scanner | {doc}`simulate_fmri` |
| know why a default is what it is | {doc}`design_sequences` and the other design notes |
| look up a term | {doc}`glossary` |
| look up a function, class or module | {doc}`api/index` |

```{toctree}
:caption: Start
:maxdepth: 1

start_here
overview
glossary
```

```{toctree}
:caption: Generate sequences
:maxdepth: 1

generate
custom_masks
mask2epi
ge_export
ge
```

```{toctree}
:caption: Process data
:maxdepth: 1

preprocess
recon
analyze
simulate_fmri
notebooks
```

```{toctree}
:caption: Design notes
:maxdepth: 1

design_sequences
design_preprocess
design_recon
design_analysis
```

```{toctree}
:caption: Reference
:maxdepth: 1

background
api/index
```
