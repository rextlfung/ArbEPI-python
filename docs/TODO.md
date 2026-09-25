# TODO

- **Low-res calibration-region k-space in preprocessing.** `recon/lowres_calib.py`
  (removed in the recon restructure) reconstructed the fully sampled (ky, kz)
  calibration region at recon time, cropping it out of `ksp_epi_zf` on every run.
  Isolate that region during preprocessing instead: have `preprocessing/preprocess.py`
  also write a small low-res calibration k-space `.h5` (plus its sampling mask and
  echo times), so a quick look or a calibration-based recon doesn't need to read the
  full zero-filled volume. The old implementation, including its B0 / B0+R2*
  adjoint-only variants and the temporal-stability check, is in git history (see
  `docs/recon-notes.md` for its module docstrings). The stability check itself now
  lives in `recon/utils.py`'s `tsnr_report`.
- **recon/ documentation pass.** Trim the long function docstrings carried over
  from the old modules, and distill `docs/recon-notes.md` into human-readable
  documentation.
