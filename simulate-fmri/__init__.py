"""fMRI experiments simulated with ArbEPI's own sampling schedules, on SNAKE-fMRI.

SNAKE (https://github.com/mind-inria/snake-fmri) supplies the phantom, the
BOLD activation and noise models, the MRD file and the parallel acquisition
driver. This package adds the ArbEPI-specific pieces through SNAKE's extension
points:

- sampler.py   ArbEPISampler: plays scan_info.mat's (ky, kz, echo time) schedule
- engine.py    ArbEPIAcquisitionEngine: the per-shot signal model
- handlers.py  EllipsoidActivationHandler: a block-design ROI placed in world mm
- phantom.py   BrainWeb at 3 T, an offline analytic phantom, coil sensitivities
- export.py    SNAKE's .mrd -> recon/<name>_preprocessed.h5 with a `truth` group
- simulate.py  scan_info.mat -> all of the above; the command line

A simulation is reconstructed by recon/ and scored by recon/testbed.py like the
real-data testbed.

The folder name has a hyphen, so it cannot be named in an `import` statement.
Run the modules with `python -m simulate-fmri.simulate`, and load them from
other code with `importlib.import_module('simulate-fmri.simulate')`. See
README.md in this folder.
"""

import warnings

import ismrmrd  # noqa: F401 -- importing it resets the warning filters, so it goes first

# xsdata, once per MRD header read (so in every worker and every loader):
# SNAKE stores its handlers' time courses under waveform ids outside the MRD
# schema's enumeration. The ids round-trip regardless.
warnings.filterwarnings('ignore', message='Failed to convert value')
