"""fMRI experiments simulated with ArbEPI's own sampling schedules, on SNAKE-fMRI.

SNAKE (https://github.com/mind-inria/snake-fmri) supplies the phantom, the
BOLD activation and noise models, the MRD file and the parallel acquisition
driver. This package adds the ArbEPI-specific pieces through SNAKE's extension
points:

- sampler.py   ArbEPISampler: plays scan_info.mat's (ky, kz, echo time) schedule
- engine.py    ArbEPIAcquisitionEngine: the per-shot signal model
- handlers.py  EllipsoidActivationHandler: a block-design ROI placed in world mm
- phantom.py   BrainWeb at 3 T, an offline analytic phantom, coil sensitivities
- simulate.py  scan_info.mat -> <name>.mrd -> recon/<name>_preprocessed.h5
- analyze.py   reconstruction -> activation z-map, ROC against the true ROI, tSNR

The folder name has a hyphen, so it cannot be named in an `import` statement.
Run the modules with `python -m simulate-fmri.simulate`, and load them from
other code with `importlib.import_module('simulate-fmri.simulate')`. See
README.md in this folder.
"""
