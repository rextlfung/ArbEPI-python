"""One reconstruction -> activation maps: the pipeline behind `python -m analyze`."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyze import glm, mask
from analyze.design import ExperimentParams, build_design_matrix, contrast_vector
from analyze.io import read_recon, save_stat_nifti


@dataclass
class Maps:
    """Volumes (Nx, Ny, Nz; zero outside the mask) of one analyzed series."""

    label: str
    t: NDArray
    z: NDArray
    psc: NDArray  # contrast effect in percent of the voxel's mean signal
    dof: float
    rho: float
    t_fdr: float  # |t| surviving FDR at q (inf if none)
    n_fdr: int
    n_voxels: int


def analyze_recon(path: str, params: ExperimentParams, scales: tuple = ("sum",),
                  brain: NDArray | str = "auto", out_dir: str | None = None,
                  ar1: bool = True, q: float = 0.05, tag: str | None = None,
                  voxel_size_mm=None, bet_frac: float = 0.5) -> dict[str, Maps]:
    """Fit the GLM to a reconstruction file and (with out_dir) write NIfTIs.

    brain: a boolean (Nx, Ny, Nz) mask, a NIfTI path, or a method for
    analyze.mask.brain_mask ('auto', 'fsl', 'python', 'threshold'); computed
    from the summed image's temporal mean and shared by every scale.
    ar1: SPM-style AR(1) prewhitening (False: ordinary least squares)."""
    rec = read_recon(path, scales=scales, n_discard=params.n_discard, voxel_size_mm=voxel_size_mm)
    shape = rec.mean_signal.shape
    nt = next(iter(rec.series.values())).shape[-1]

    if isinstance(brain, str) and os.path.exists(brain):
        brain = mask.load_mask(brain)
    elif isinstance(brain, str):
        brain = mask.brain_mask(rec.mean_signal, rec.voxel_size_mm, method=brain, frac=bet_frac)
    brain = np.asarray(brain, dtype=bool)
    if brain.shape != shape:
        raise ValueError(f"mask {brain.shape} does not match the image {shape}")
    base = np.maximum(rec.mean_signal[brain], 1e-30)
    keep = base > 0.01 * base.max()  # a voxel with no signal has no percent change
    idx = np.flatnonzero(brain.ravel())[keep]
    sel = np.zeros(brain.size, bool)
    sel[idx] = True
    brain = sel.reshape(shape)
    base = rec.mean_signal[brain]

    X, names = build_design_matrix(params, nt)
    c = contrast_vector(params, names)
    tag = tag or os.path.splitext(os.path.basename(path))[0]
    out: dict[str, Maps] = {}
    summary = {"file": path, "tr": params.tr, "n_frames": nt, "n_discard": params.n_discard,
               "design": names, "contrast": c.tolist(), "ar1": ar1,
               "n_mask_voxels": int(brain.sum()),
               "voxel_size_mm": list(rec.voxel_size_mm), "scales": {}}
    for label, series in rec.series.items():
        Y = series[brain]
        r = glm.fit_glm(Y, X, c, ar1=ar1)
        p = glm.t_to_p(r.t, r.dof)
        p_thr = glm.fdr_threshold(p, q)
        surv = p <= p_thr if p_thr > 0 else np.zeros_like(p, bool)
        t_fdr = float(np.abs(r.t[surv]).min()) if surv.any() else float("inf")
        m = Maps(label=label, t=glm.to_volume(r.t, brain), z=glm.to_volume(r.z, brain),
                 psc=glm.to_volume(100 * r.contrast_beta / base, brain), dof=r.dof, rho=r.rho,
                 t_fdr=t_fdr, n_fdr=int(surv.sum()), n_voxels=int(brain.sum()))
        out[label] = m
        summary["scales"][label] = dict(dof=m.dof, rho=m.rho, t_fdr=m.t_fdr, n_fdr=m.n_fdr,
                                        t_p001=glm.t_threshold(m.dof, 0.001),
                                        t_max=float(r.t.max()), t_min=float(r.t.min()))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            for kind in ("t", "z", "psc"):
                save_stat_nifti(os.path.join(out_dir, f"{tag}_{label}_{kind}.nii.gz"),
                                getattr(m, kind), rec.voxel_size_mm, label=label, dof=m.dof,
                                rho=m.rho, ar1=ar1, contrast=c.tolist(), design=names)
    if out_dir:
        save_stat_nifti(os.path.join(out_dir, f"{tag}_mask.nii.gz"), brain.astype(np.uint8),
                        rec.voxel_size_mm)
        with open(os.path.join(out_dir, f"{tag}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return out
