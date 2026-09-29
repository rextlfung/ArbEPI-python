# TODO

- **A real multi-echo GRE for R2\*.** `preprocess/r2star.py` fits R2\* on the
  dual-echo deGRE, whose echo spacing is set for B0 mapping (2.24 ms on
  `20260915ball/2_6x_2.4mm` against T2\* ≈ 47 ms, so the echoes differ by only
  ~5%). That map is noise-dominated and `SENSE_B0_R2star` can be no better. Add a
  3D monopolar multi-echo spoiled GRE: ~4–8 echoes, first TE as short as possible,
  last TE ~1–2 × the target T2\*, first echo spacing still short enough to unwrap
  for B0, thin through-plane voxels (or a macroscopic-gradient correction). See the
  module docstring for the references. The fit already uses every echo, so only the
  sequence and `unflatten_gre_echoes`' echo count need to change.
- **Temporal stability: fix the fat-sat pulse, then re-test.** The `20260924ball`
  A/B experiment (fully sampled, static 5.4 mm ball phantom, one change per run)
  found that the fat-sat pulse is the main source of the ~3% frame-to-frame
  fluctuation in static scans. Turning it off cut the median voxel CV from 3.20%
  to 0.58% and raised signal 56%. The pulse tips on-resonance water by 26° every
  shot, so its coherence refocuses somewhere different each frame under random
  spoiling. A wider random spoiler (3–8 cycles/voxel) reached 1.83%. The readout
  itself is stable: an unencoded EPIcal time series gave 0.015° odd/even phase std.
  Details in `docs/review-findings.md` item 255 and
  `/StorageRAID/rexfung/20260924ball/analysis/README.md`. Next:
  1. ~~Replace the pulse with one that leaves water alone.~~ Done 2026-09-29:
     min-phase SLR, TBW 2, 6 ms (item 255 has the Bloch numbers). Check any
     further change with `lib/make_fatsat_rf.py`'s `flip_profile`.
  2. Scan the new pulse against fat-sat off and the current baseline, on the
     same protocol in one session. Repeat the baseline last as a drift control;
     that repeat wasn't acquired in `20260924ball`. Also keep the 3–8
     cycles/voxel spoiler as a factor, since its remaining 1.83% was a per-shot
     signal level, not refocused echoes. Timing on `20260924ball`'s 5.4 mm
     protocol with the new 6 ms pulse: per-shot min TR is 57.48 ms at 3–4
     cycles/voxel (fits the 1.05 s volume TR's 58.33 ms), but 58.88 ms at
     3–8, so a spoiler arm needs volume TR ≥ 1.06 s for every arm.
     The new pulse also trades fat suppression at the band edges for water
     preservation: 60° at fat ±100 Hz vs the Gaussian's 84° (81° vs 88° at
     ±50 Hz). The ball phantom has no fat, so this scan can't see that side;
     check it on something with fat (an oil phantom, or in vivo with fat-sat
     on/off) before relying on the default for human work, and retune
     `FatsatParams.tbw`/`dur` with `flip_profile` if needed.
  3. Look into the ~1.4° per-shot phase jitter that makes up most of the 0.58%
     left with fat-sat off: frequency drift or receiver phase, or correct it
     per shot from the data. Then check the result against the ~3% BOLD target.
  4. Until then, for phantom stability tests, run with `FatsatParams.enabled=False`.
     Discard at least 3 s of leading frames (measured T1 approach ~2.3 s to
     within 0.1%); the `discard_duration` default is still 0.
