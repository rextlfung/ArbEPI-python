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
