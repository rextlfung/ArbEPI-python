"""fMRI scans simulated with ArbEPI's own sampling schedules.

Two levels of fidelity, both starting from the scan_info.mat a sequence wrote:

session.py   Raw data of a whole scan session (noise scan, EPIcal, deGRE,
             ArbEPI) for the real preprocess/ and recon/ to process: field
             map, ramp sampling, readout delay and odd/even phase, coil noise,
             physiological noise, BOLD as an R2* change.
               forward.py   the signal equation, evaluated per echo index
               b0.py        field map from the head's susceptibility
               coils.py     receive array and noise covariance
               physio.py    physiological noise
ideal.py     SNAKE-fMRI's acquisition engine on the Cartesian grid, written
             straight to recon's input format: sampling, T2* decay and thermal
             noise only.
               sampler.py, engine.py, handlers.py, export.py

from_real.py is a third source of truth files that simulates no scanner: it
injects known activations into a real, fully sampled run (a real object, coils
and field; needs only the recon extra).

protocol.py reads scan_info.mat; phantom.py holds the BrainWeb and analytic
phantoms (SNAKE's Phantom). Every mode's output carries its ground truth in
analyze/score.py's layout, which scores a reconstruction against it. See
README.md in this folder.
"""

import warnings

try:
    import ismrmrd  # noqa: F401 -- importing it resets the warning filters, so it goes first
except ImportError:  # from_real.py needs only the recon extra (the SNAKE modules import it)
    pass

# xsdata, once per MRD header read (so in every worker and every loader):
# SNAKE stores its handlers' time courses under waveform ids outside the MRD
# schema's enumeration. The ids round-trip regardless.
warnings.filterwarnings('ignore', message='Failed to convert value')
