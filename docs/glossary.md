# Glossary

Terms used throughout the guides, grouped by where they appear.

## Acquisition

```{glossary}
shot
  One RF excitation followed by one echo train. A volume is built from
  `Nshots` shots, and the per-shot repetition time is `TR = volume_tr / Nshots`.

echo train, ETL
  The readout lines acquired after one excitation. `ETL` (echo train length)
  is how many; each echo sits at its own `(ky, kz)` location, reached by small
  phase-encode gradient pulses between readouts.

blip
  The small phase-encode gradient pulse between two consecutive readout lines
  that steps `(ky, kz)` to the next sample of the echo train.

frame, volume
  One complete acquisition of all `Nshots`. A scan is `Nframes` frames, each
  with its own sampling mask, taking `volume_tr` seconds.

sampling mask, omegas
  The `(Ny, Nz)` boolean pattern of phase-encode locations acquired in a frame.
  `omegas` is the stack over frames, `(Ny, Nz, Nframes)`.

R, acceleration
  The factor by which the `(ky, kz)` plane is undersampled. `Nshots =
  ceil(Ny*Nz / R / ETL)`.

schedule
  For every frame, shot and echo, the `(ky, kz)` index pair and echo time.
  Saved in `scan_info.mat` as `schedules`, shape `Nframes x Nshots x ETL x 3`.

laminar, radial
  The two ways `lib/mask2epi.py` partitions a mask into echo trains. Laminar
  keeps `ky` non-decreasing along a train; radial gives every shot a spoke
  through k-space center. See {doc}`mask2epi`.

TE (nominal), echo offset
  The prescribed echo time. It is anchored at the readout's true `kx = 0`
  crossing, which for the radial ordering is echo index `(ETL - 1) // 2`.

fat-sat, water excitation
  Two fat-suppression modes (`params.excitation`): a fat-saturation pulse and
  crusher before a slab-selective excitation, or a binomial 1-3-3-1 water
  excitation that replaces both.
```

## The four sequences

```{glossary}
deGRE
  The dual-echo gradient-echo scan (`deGRE.seq`) that provides coil
  compression, sensitivity maps, the B0 field map and R2\*.

EPIcal
  The calibration scan (`EPIcal.seq`): the EPI readout with all blips off, so it
  acquires unencoded lines at k-space center for ghost correction.

noise scan
  The receive-only scan (`noise.seq`) whose covariance gives the whitening
  matrix.

scan_info.mat
  The HDF5 (v7.3) file written by `ArbEPI` holding the schedule, the odd/even
  k-space trajectories (`kxo`, `kxe`) and the scan scalars that preprocessing
  needs.

.seq, .pge
  `.seq` is the vendor-neutral Pulseq sequence file. `.pge` is the file the GE
  scanner runs, produced from it by `ge/`.
```

## Hardware limits

```{glossary}
PNS
  Peripheral nerve stimulation: the physiological limit on how fast and how
  often the gradients may switch. `ge/pns.py` implements GE's IEC 60601-2-33
  model; the export gate is 80% (normal mode).

POPE
  "PNS Optimized Pulses for EPI": an asymmetric readout trapezoid with a
  throttled rising slew and a faster falling slew, because predicted PNS peaks
  at the end of each sustained slew event. See {doc}`design_sequences`.

acoustics
  The gradient waveform's energy in the scanner's mechanically resonant
  frequency bands. Over its threshold it warns but does not block an export.
```

## Preprocessing and reconstruction

```{glossary}
whitening
  Multiplying coil data by $W$ so receive noise is unit-variance and
  uncorrelated across coils. Mandatory whenever a noise scan exists.

odd/even (Nyquist ghost) correction
  A phase applied to alternate echoes to remove the ghost caused by the
  readout's alternating polarity, estimated from `EPIcal`.

ramp sampling, gridding
  The ADC samples while the readout gradient ramps, so samples are not
  uniform in `kx`; gridding resamples them onto a Cartesian grid.

GCC, virtual coils
  Geometric coil compression: a per-position linear combination of the
  physical coils into fewer virtual coils, with a controlled signal loss.

sensitivity maps (smaps)
  Each coil's spatial sensitivity, estimated by ESPIRiT from the deGRE.

B0 map, off-resonance
  The static field deviation $\Delta f(r)$ in Hz, from the deGRE's two echoes.
  It shifts and blurs EPI, so reconstruction models it.

R2\*
  The transverse relaxation rate, as a magnitude decay along the echo train.

time segmentation
  Approximating the per-sample off-resonance phase with $L$ phased copies of
  the image, combined with interpolation weights. See {doc}`recon`.

MSLR
  Multi-scale low-rank reconstruction (`--reg mslr`): the image is a sum of
  components, each low rank over patches of one size.

PDHG, POGM, CG
  The solvers: primal-dual hybrid gradient (wavelet + TV), proximal optimized
  gradient method (low rank), and conjugate gradient (no regularizer).
```
