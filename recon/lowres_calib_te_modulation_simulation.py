"""Forward-simulates the deterministic TE-modulation mechanism (off-resonance
dephasing + T2* decay, driven purely by the real per-(ky,kz)-sample echo time
that each frame's shot order assigns -- see
preprocessing/lowres_calib_te_distribution_check.py, and the field-map-based
correction in recon/lowres_calib_recon_b0complex.py) against a *static*
ground-truth object, to measure how much temporal "shading" this single,
already-quantified mechanism produces on its own -- for direct, controlled
comparison against how much shading the real acquired data actually shows.

Experiment design
------------------
1. Ground truth I_gt(r): the time-average (post-warm-up) of
   recon/lowres_calib_recon_b0complex.py's own complex, TE-nominal-referenced
   reconstruction -- the best available static estimate of the object,
   consistent with the same reference time (TE_nominal) the forward model
   below is written against.
2. For every frame t and every calibration-region (ky,kz) sample, look up
   that sample's *real* acquisition time from the schedule (echo_times),
   relative to TE_nominal (dTE). Modulate the *entire spatial* ground-truth
   volume by the exact (not time-segmented/approximated) physical forward
   model exp(i*2*pi*Δf(r)*dTE) * exp(-R2*(r)*dTE), multiply by each coil's
   real sensitivity map, forward-FFT to native-grid k-space, and keep only
   the (ky,kz) slice actually sampled at that dTE. Samples sharing the same
   dTE within a frame reuse one modulated volume/FFT (there are at most ETL
   distinct dTE values per frame, not one per calibration sample).
3. Reconstruct the resulting synthetic per-coil k-space with the *exact,
   unmodified* preprocessing.lowres_calib_recon.lowres_calib_recon function
   -- the same code path used for the real, uncorrected reconstruction here
   -- so any difference in measured temporal fluctuation reflects the
   forward model, not a difference in reconstruction code.
4. Compare simulated vs. real-uncorrected temporal fluctuation (percent
   fluctuation, not tSNR -- the simulation is noiseless, so its tSNR is
   unbounded and not a meaningful number on its own) and the *spatial
   pattern* of the dominant (SVD PC1) fluctuation mode, since the user's
   complaint is spatially-structured shading, not just a scalar wobble.

Discriminating check: if recon/lowres_calib_recon_b0complex.py's correction
removes exactly this mechanism and nothing else, then (treating the
TE-modulation-induced and residual/other fluctuation as independent)
   TE_component = sqrt(X^2 - Y^2)
where X = real uncorrected percent fluctuation, Y = real b0complex-corrected
percent fluctuation. The simulated fluctuation Z should be comparable to
that. Z >> sqrt(X^2-Y^2) would mean the real correction underperforms its
own predicted size (e.g. its L=32 time-segmented approximation loses
accuracy); Z << sqrt(X^2-Y^2) would mean this simulation is missing a
contributor the real correction is nonetheless removing.

Smaps normalization is deliberately consistent across every reconstruction
in this script (X, Y, Z all use the same RSS-renormalized smaps
recon/lowres_calib_recon_b0complex.py already uses internally) -- mixing a
"sum_c|s_c|^2<=1" convention with a "==1" one between the ground-truth
source and the forward model would introduce a spurious, spatially-varying
rescaling of the simulated reconstruction relative to I_gt, contaminating
exactly the spatial pattern this script is trying to measure.

Usage (.venv-recon):
    .venv-recon/bin/python -m recon.lowres_calib_te_modulation_simulation <datdir> [--seqname ArbEPI] [--device cuda]
"""

import argparse
import gc
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from preprocessing.config import load_config, load_seq_params, set_seq_paths
from preprocessing.grid_resize import resize_to_epi_grid
from preprocessing.lowres_calib_recon import (
    _load_chunked,
    compute_calib_mask,
    lowres_calib_recon,
    native_calib_grid,
)
from preprocessing.lowres_temporal_stability import object_mask, temporal_stability
from preprocessing.r2star_map import estimate_r2star_map_epi_grid
from recon.lowres_calib_recon_b0complex import nominal_te_s, run_b0complex_corrected_calib_recon
from recon.reconstruct import _load_array


def _ft3(d: np.ndarray) -> np.ndarray:
    """Forward transform matching lowres_calib_recon._ift3's convention
    (fftshift(ifftn(fftshift(.)))) -- its exact inverse."""
    axes = (0, 1, 2)
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(d, axes=axes), axes=axes), axes=axes)


def simulate_te_modulated_ksp(
    I_gt: np.ndarray,
    smaps_native: np.ndarray,
    b0map_hz_native: np.ndarray,
    r2star_native: np.ndarray,
    echo_times_crop_raw: np.ndarray,
    calib_mask_crop: np.ndarray,
    te_nominal_s: float,
) -> np.ndarray:
    """I_gt: (Nx,Ny,Nz) complex, static ground-truth image (spatial domain,
    native/low-res grid). smaps_native: (Nx,Ny,Nz,Nc) complex. b0map_hz_native,
    r2star_native: (Nx,Ny,Nz) float (Hz, 1/s), spatial domain. echo_times_crop_raw:
    (Ny,Nz,Nt) float s, per-(ky,kz)-*sample* acquisition time (k-space-index
    domain, pre-TE-shift). calib_mask_crop: (Ny,Nz) bool, k-space-index domain.

    Returns (Nx,Ny,Nz,Nc,Nt) complex64 synthetic per-coil k-space, zero
    outside the calibration (ky,kz) locations (matching the real "_zf"
    zero-filled convention). One modulated-volume FFT per distinct dTE value
    per frame, not one per calibration sample -- there are at most ETL
    distinct echo times per frame.

    Physical forward model, exact (not time-segmented): for a sample
    acquired dTE seconds after TE_nominal, the object's contribution to that
    sample is I_gt(r) * exp(i*2*pi*Δf(r)*dTE) * exp(-R2*(r)*dTE) -- note the
    MINUS sign on R2*: this is a plain forward simulation (no adjoint
    involved), unlike recon/lowres_calib_recon_b0complex.py's psi_recon,
    whose plus sign is specifically an artifact of GatheredSenseB0's adjoint
    always conjugating (see that module's docstring) and does not apply
    here.
    """
    Nx, Ny, Nz = I_gt.shape
    Nc = smaps_native.shape[-1]
    Nt = echo_times_crop_raw.shape[-1]
    ksp_synth = np.zeros((Nx, Ny, Nz, Nc, Nt), dtype=np.complex64)

    ys, zs = np.nonzero(calib_mask_crop)
    for t in range(Nt):
        frame_view = ksp_synth[:, :, :, :, t]  # basic-indexed view (plain slice on last axis)
        dte = echo_times_crop_raw[ys, zs, t] - te_nominal_s  # (Ncalib,) s
        uniq_dte, inv = np.unique(dte, return_inverse=True)
        for gi, dte_val in enumerate(uniq_dte):
            sel = inv == gi
            y_sel, z_sel = ys[sel], zs[sel]
            mod = np.exp(1j * 2 * np.pi * b0map_hz_native * dte_val) * np.exp(-r2star_native * dte_val)
            I_mod = I_gt * mod  # (Nx,Ny,Nz) complex
            S = smaps_native * I_mod[..., None]  # (Nx,Ny,Nz,Nc)
            K = _ft3(S)  # (Nx,Ny,Nz,Nc)
            # Mixing a trailing scalar frame index with the fancy y/z indices
            # in one expression (ksp_synth[:, y_sel, z_sel, :, t]) triggers
            # numpy's "advanced indices separated by a slice/scalar" rule
            # (a plain integer index counts as advanced for this purpose --
            # see numpy's advanced-indexing docs), which reorders axes to
            # (combined, Nx, Nc) instead of the expected (Nx, combined, Nc)
            # -- confirmed empirically. Slicing out the frame first (a plain
            # view, basic indexing only) keeps the assignment below to just
            # the two adjacent advanced indices (y_sel, z_sel), which stay
            # in place as expected.
            frame_view[:, y_sel, z_sel, :] = K[:, y_sel, z_sel, :]
    return ksp_synth


def spatial_svd_pc1(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """img: [Nx,Ny,Nz,Nframes] complex, mask: [Nx,Ny,Nz] bool. Returns
    (u1, var_explained): u1 is the dominant complex-SVD left singular vector
    (spatial pattern of the largest mean-centered temporal fluctuation mode),
    restricted to mask and unit-norm -- see lowres_calib_b1_drift_check.py's
    complex_gain_drift_decomposition for the same construction (this is the
    spatial-pattern half of it, needed here for cross-series correlation,
    which that function doesn't expose)."""
    X = img[mask]
    Xc = X - X.mean(axis=1, keepdims=True)
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    var_explained = float(S[0] ** 2 / np.sum(S**2))
    return U[:, 0], var_explained


def pattern_correlation(u1_a: np.ndarray, mask_a: np.ndarray, u1_b: np.ndarray, mask_b: np.ndarray) -> float:
    """Complex correlation magnitude between two PC1 spatial patterns
    defined on (possibly different) masks, restricted to their common
    support so voxels present in only one are excluded."""
    common = mask_a & mask_b
    ia = np.zeros(mask_a.shape, dtype=bool)
    idx_a = np.flatnonzero(mask_a)
    keep_a = common[mask_a]
    idx_b = np.flatnonzero(mask_b)
    keep_b = common[mask_b]
    va = u1_a[keep_a]
    vb = u1_b[keep_b]
    eps = np.finfo(np.float64).eps
    return float(np.abs(np.vdot(va, vb)) / (np.linalg.norm(va) * np.linalg.norm(vb) + eps))


def run_one(
    datdir: str, seqname: str = 'ArbEPI', L: int = 32, nbins: int = 128, device: str = 'cuda',
    skip_frames: int | None = None, gt_source: str = 'b0complex',
) -> dict:
    label = os.path.basename(os.path.normpath(datdir))
    print(f'\n{"=" * 70}\n{label}\n{"=" * 70}')

    cfg = load_config(datdir=datdir, seqnames=[seqname])
    paths = set_seq_paths(cfg, seqname)
    seq_params = load_seq_params(paths)
    fov, fov_degre = seq_params.fov, seq_params.fov_degre

    if skip_frames is None:
        # scan_info.mat's discard_duration is 0 on this acquisition (it
        # tracks a different, sequence-level dead time, not the
        # magnetization-steady-state warm-up the user described
        # conversationally as "roughly the first 10 seconds") -- so this
        # fallback under-drops frames unless overridden explicitly. Callers
        # who care about that warm-up should pass skip_frames=round(10/TR).
        skip_frames = max(1, round(seq_params.discard_duration / seq_params.volume_tr))
    print(f'TR = {seq_params.volume_tr:.3f} s, discard_duration = {seq_params.discard_duration:.1f} s '
          f'-> skip_frames = {skip_frames}')

    fn_ksp = paths.recon
    fn_smaps = os.path.join(datdir, 'recon', f'smaps_{seqname}_sigpy.h5')
    fn_b0map = os.path.join(datdir, 'recon', f'{seqname}_b0map.h5')

    print('Loading smaps / B0 field map / R2* map (EPI grid)...')
    smaps_raw = _load_array(fn_smaps, 'smaps').astype(np.complex64)  # (Nx,Ny,Nz,Nc)
    smaps_rss = np.sqrt(np.sum(np.abs(smaps_raw) ** 2, axis=-1, keepdims=True))
    smaps = (smaps_raw / (smaps_rss + np.finfo(np.float32).eps)).astype(np.complex64)  # sum_c|.|^2==1
    Nx, Ny, Nz, Nc = smaps.shape

    b0map_hz = _load_array(fn_b0map, 'b0map_hz').astype(np.float32)
    r2star_hz = estimate_r2star_map_epi_grid(datdir, seqname, fov_degre, fov, (Nx, Ny, Nz))

    with h5py.File(fn_ksp, 'r') as f:
        omegas = f['omegas'][()]  # (Ny,Nz,Nt)
    echo_times_2d = _load_array(fn_ksp, 'echo_times').astype(np.float64)  # (Ny,Nz,Nt) s
    calib_mask = compute_calib_mask(omegas)
    Nt = omegas.shape[-1]
    n_calib = int(calib_mask.sum())
    print(f'Calibration region: {n_calib} / {calib_mask.size} (ky,kz) locations')

    te_nominal_s = nominal_te_s(paths.scan_info, seq_params.ETL)
    print(f'TE_nominal = {te_nominal_s * 1000:.3f} ms')

    grid = native_calib_grid(calib_mask, fov, Nx)
    xs, ys_sl, zs_sl = grid['x_slice'], grid['y_slice'], grid['z_slice']
    Nx_eff, Ny_eff, Nz_eff = grid['Nx_eff'], grid['Ny_eff'], grid['Nz_eff']
    print(f'Native grid: ({Nx_eff}, {Ny_eff}, {Nz_eff})')
    n_target = (Nx_eff, Ny_eff, Nz_eff)

    smaps_native = resize_to_epi_grid(smaps, fov, fov, n_target, order=3).astype(np.complex64)
    b0map_hz_native = resize_to_epi_grid(b0map_hz, fov, fov, n_target, order=3).astype(np.float32)

    ratios = [s / t for s, t in zip(r2star_hz.shape, n_target)]
    sigmas = [max(r / 2, 0.0) for r in ratios]
    r2star_pref = gaussian_filter(r2star_hz, sigma=sigmas)
    r2star_native = np.clip(
        resize_to_epi_grid(r2star_pref, fov, fov, n_target, order=3), 0.0, None
    ).astype(np.float32)

    echo_times_crop_raw = echo_times_2d[ys_sl, zs_sl, :]  # (Ny_eff,Nz_eff,Nt)
    calib_mask_crop = calib_mask[ys_sl, zs_sl]  # (Ny_eff,Nz_eff)

    # --- Real, uncorrected reconstruction (same code/smaps as simulated) ---
    # Computed before the ground truth / forward simulation below since
    # gt_source='uncorrected' needs it as an alternate ground-truth source
    # (see that option's docstring note on the b0complex-recon leakage
    # caveat), and X needs it regardless.
    print(f'\nLoading real k-space ({fn_ksp}) and reconstructing (uncorrected)...')
    with h5py.File(fn_ksp, 'r') as f:
        ksp_real_full = _load_chunked(f, 'ksp_epi_zf')  # (Nx,Ny,Nz,Nc,Nt)
    img_real, grid_real = lowres_calib_recon(ksp_real_full, calib_mask, smaps, fov)
    del ksp_real_full
    gc.collect()
    assert (grid_real['Nx_eff'], grid_real['Ny_eff'], grid_real['Nz_eff']) == (Nx_eff, Ny_eff, Nz_eff)

    mask_real = object_mask(np.abs(img_real))
    stats_x = temporal_stability(np.abs(img_real)[..., skip_frames:], mask_real, seq_params.volume_tr)
    u1_real, var_real = spatial_svd_pc1(img_real[..., skip_frames:], mask_real)
    X = stats_x['percent_fluctuation']
    print(f'  X (real, uncorrected) percent fluctuation: {X:.3f}%')

    # --- Ground truth: b0complex-corrected time average (default) ---
    print('\nComputing b0complex-corrected reconstruction (ground truth source)...')
    gt_result = run_b0complex_corrected_calib_recon(datdir, seqname, L, nbins, device)
    img_gt_series = gt_result['img_np']  # (Nx_eff,Ny_eff,Nz_eff,Nt) complex64
    assert img_gt_series.shape[:3] == (Nx_eff, Ny_eff, Nz_eff), (
        f'ground-truth grid {img_gt_series.shape[:3]} != {(Nx_eff, Ny_eff, Nz_eff)}'
    )

    mask_gt = object_mask(np.abs(img_gt_series))
    stats_y = temporal_stability(np.abs(img_gt_series)[..., skip_frames:], mask_gt, seq_params.volume_tr)
    u1_gt, var_gt = spatial_svd_pc1(img_gt_series[..., skip_frames:], mask_gt)
    Y = stats_y['percent_fluctuation']
    print(f'  Y (real, b0complex-corrected) percent fluctuation: {Y:.3f}%')

    # gt_source='uncorrected': use the plain uncorrected time-average
    # instead. Ground-truth-leakage check -- I_gt from the b0complex recon
    # is not perfectly static (its own residual fluctuation is Y, not 0),
    # which biases Z upward by however much of that residual is coherent
    # with the modulation this script applies. If Z barely moves under
    # this alternate source, that leakage is negligible.
    if gt_source == 'b0complex':
        I_gt = img_gt_series[..., skip_frames:].mean(axis=-1)  # (Nx_eff,Ny_eff,Nz_eff) complex
    elif gt_source == 'uncorrected':
        I_gt = img_real[..., skip_frames:].mean(axis=-1)
    else:
        raise ValueError(f'unknown gt_source: {gt_source!r}')

    # --- Forward simulation ---
    print('\nForward-simulating TE-modulated k-space from the static ground truth...')
    ksp_synth_native = simulate_te_modulated_ksp(
        I_gt, smaps_native, b0map_hz_native, r2star_native,
        echo_times_crop_raw, calib_mask_crop, te_nominal_s,
    )  # (Nx_eff,Ny_eff,Nz_eff,Nc,Nt)

    print('Reconstructing simulated k-space with the unmodified real-data recon code...')
    ksp_embed = np.zeros((Nx_eff, Ny, Nz, Nc, Nt), dtype=np.complex64)
    ksp_embed[:, ys_sl, zs_sl, :, :] = ksp_synth_native
    del ksp_synth_native
    img_sim, grid_sim = lowres_calib_recon(ksp_embed, calib_mask, smaps, fov)
    assert (grid_sim['Nx_eff'], grid_sim['Ny_eff'], grid_sim['Nz_eff']) == (Nx_eff, Ny_eff, Nz_eff)
    del ksp_embed
    gc.collect()

    mask_sim = object_mask(np.abs(img_sim))
    stats_z = temporal_stability(np.abs(img_sim)[..., skip_frames:], mask_sim, seq_params.volume_tr)
    u1_sim, var_sim = spatial_svd_pc1(img_sim[..., skip_frames:], mask_sim)
    Z = stats_z['percent_fluctuation']
    print(f'  Z (simulated) percent fluctuation: {Z:.3f}%')

    # --- Discriminating comparison ---
    # sqrt(X^2-Y^2) is only meaningful when the correction actually reduced
    # fluctuation (X>Y); Y>=X means the correction added net fluctuation
    # rather than removing this mechanism's share of it, which is itself a
    # real (and separate) finding -- not a domain error to paper over with
    # a lopsided ratio.
    predictor_defined = X > Y
    te_component_predicted = float(np.sqrt(max(X**2 - Y**2, 0.0))) if predictor_defined else None
    pattern_corr_sim_vs_real = pattern_correlation(u1_sim, mask_sim, u1_real, mask_real)
    pattern_corr_gt_vs_real = pattern_correlation(u1_gt, mask_gt, u1_real, mask_real)

    print(f'\n--- Summary: {label} (gt_source={gt_source}) ---')
    print(f'  X  = real uncorrected fluctuation:        {X:.3f}%  (PC1 var {100 * var_real:.1f}%)')
    print(f'  Y  = real b0complex-corrected fluctuation: {Y:.3f}%  (PC1 var {100 * var_gt:.1f}%)')
    print(f'  Z  = simulated (TE-modulation only):       {Z:.3f}%  (PC1 var {100 * var_sim:.1f}%)  Z/X={Z / X:.2f}')
    if predictor_defined:
        print(f'  sqrt(X^2 - Y^2) (predicted TE-mod component): {te_component_predicted:.3f}%  '
              f'(Z/predicted = {Z / max(te_component_predicted, 1e-9):.2f})')
    else:
        print('  sqrt(X^2 - Y^2): undefined (Y >= X -- correction added net fluctuation rather than '
              'removing this mechanism\'s share of it; see Z/X and the pattern correlations instead)')
    print(f'  PC1 spatial pattern correlation, simulated vs. real-uncorrected: {pattern_corr_sim_vs_real:.3f}')
    print(f'  PC1 spatial pattern correlation, b0complex-corrected vs. real-uncorrected: {pattern_corr_gt_vs_real:.3f}')

    return dict(
        label=label, skip_frames=skip_frames, tr_s=seq_params.volume_tr, gt_source=gt_source,
        X=X, Y=Y, Z=Z, te_component_predicted=te_component_predicted,
        var_real=var_real, var_gt=var_gt, var_sim=var_sim,
        pattern_corr_sim_vs_real=pattern_corr_sim_vs_real,
        pattern_corr_gt_vs_real=pattern_corr_gt_vs_real,
        stats_real=stats_x, stats_gt=stats_y, stats_sim=stats_z,
        img_real=img_real, img_sim=img_sim, img_gt_series=img_gt_series,
        mask_real=mask_real, mask_sim=mask_sim,
    )


def main(
    datdirs: list[str], seqname: str = 'ArbEPI', L: int = 32, nbins: int = 128, device: str = 'cuda',
    skip_frames: int | None = None, gt_source: str = 'b0complex',
) -> None:
    results = {}
    for datdir in datdirs:
        result = run_one(datdir, seqname, L, nbins, device, skip_frames, gt_source)
        results[datdir] = result

        out_dir = os.path.join(datdir, 'recon', 'basic')
        os.makedirs(out_dir, exist_ok=True)

        stats_x, stats_y, stats_z = result['stats_real'], result['stats_gt'], result['stats_sim']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(stats_x['time_s'], stats_x['roi_signal'] / stats_x['roi_signal'].mean(),
                     'o-', label=f'real uncorrected (X={result["X"]:.2f}%)')
        axes[0].plot(stats_y['time_s'], stats_y['roi_signal'] / stats_y['roi_signal'].mean(),
                     's-', label=f'real b0complex-corrected (Y={result["Y"]:.2f}%)')
        axes[0].plot(stats_z['time_s'], stats_z['roi_signal'] / stats_z['roi_signal'].mean(),
                     '^-', label=f'simulated TE-modulation only (Z={result["Z"]:.2f}%)')
        axes[0].set_xlabel('time (s)')
        axes[0].set_ylabel('ROI-mean signal, normalized')
        axes[0].set_title(f'{result["label"]}: ROI-mean signal')
        axes[0].legend(fontsize=8)

        iz = stats_x['tsnr_map'].shape[2] // 2
        Nx_eff, Ny_eff, _ = stats_x['tsnr_map'].shape
        diff_real = np.abs(result['img_real'][:, :, iz, -1]) - np.abs(result['img_real'][:, :, iz, result['skip_frames']])
        diff_sim = np.abs(result['img_sim'][:, :, iz, -1]) - np.abs(result['img_sim'][:, :, iz, result['skip_frames']])
        vmax = max(np.percentile(np.abs(diff_real), 99), np.percentile(np.abs(diff_sim), 99), 1e-9)
        # diff_real/diff_sim are (Nx_eff,Ny_eff); concatenating on axis=0
        # (the Nx axis) before .T puts them side by side horizontally in
        # the final (Ny_eff, 2*Nx_eff) image -- concatenating on axis=1
        # instead stacks them vertically after the transpose (a bug an
        # earlier version of this script had: both panels rendered, just
        # one above the other instead of side by side, easy to mistake for
        # only one panel existing).
        im = axes[1].imshow(np.concatenate([diff_real, diff_sim], axis=0).T, origin='lower',
                             cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1].axvline(Nx_eff, color='k', lw=1)
        axes[1].set_title('last-minus-first-kept-frame: real (left) | simulated (right)')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        plt.tight_layout()
        fn_out = os.path.join(out_dir, 'lowres_calib_te_modulation_simulation.png')
        plt.savefig(fn_out, dpi=130)
        plt.close(fig)
        print(f'Wrote {fn_out}')

        # Free the large per-frame arrays before moving to the next dataset.
        for k in ('img_real', 'img_sim', 'img_gt_series'):
            result[k] = None
        gc.collect()

    print(f'\n{"=" * 70}\nOverall summary\n{"=" * 70}')
    for datdir, r in results.items():
        predicted_str = f'{r["te_component_predicted"]:.2f}%' if r['te_component_predicted'] is not None else 'undefined (Y>=X)'
        print(f'{r["label"]}: X={r["X"]:.2f}%  Y={r["Y"]:.2f}%  Z={r["Z"]:.2f}%  Z/X={r["Z"] / r["X"]:.2f}  '
              f'sqrt(X^2-Y^2)={predicted_str}  '
              f'pattern_corr(sim,real)={r["pattern_corr_sim_vs_real"]:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datdirs', nargs='+')
    parser.add_argument('--seqname', default='ArbEPI')
    parser.add_argument('--L', type=int, default=32)
    parser.add_argument('--nbins', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip-frames', type=int, default=None)
    parser.add_argument('--gt-source', default='b0complex', choices=['b0complex', 'uncorrected'])
    args = parser.parse_args()
    main(args.datdirs, args.seqname, args.L, args.nbins, args.device, args.skip_frames, args.gt_source)
