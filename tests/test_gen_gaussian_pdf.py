import math

import numpy as np
import pytest

from sampling.gen_gaussian_pdf import gen_gaussian_pdf


@pytest.mark.parametrize('dims', [(16, 16), (32, 24), (15, 21)])
def test_gen_gaussian_pdf_range_and_peak(dims):
    pdf = gen_gaussian_pdf(dims, 4)
    assert pdf.shape == dims
    assert pdf.max() == pytest.approx(1.0)
    assert (pdf >= 0).all() and (pdf <= 1.0 + 1e-12).all()


def test_gen_gaussian_pdf_peak_at_matlab_center():
    Nx, Ny = 16, 16
    pdf = gen_gaussian_pdf((Nx, Ny), 4)
    cx = math.ceil(Nx / 2)
    cy = math.ceil(Ny / 2)
    # cx, cy are 1-based MATLAB centers; 0-based peak index is (cx-1, cy-1).
    peak_idx = np.unravel_index(np.argmax(pdf), pdf.shape)
    assert peak_idx == (cx - 1, cy - 1)


def _weighted_std(pdf, axis):
    n = pdf.shape[axis]
    coords = np.arange(n)
    marginal = pdf.sum(axis=1 - axis)
    mean = np.average(coords, weights=marginal)
    return math.sqrt(np.average((coords - mean) ** 2, weights=marginal))


def test_gen_gaussian_pdf_anisotropic_sigma_elongates_along_axis():
    pdf_iso = gen_gaussian_pdf((32, 32), 4)
    pdf_aniso = gen_gaussian_pdf((32, 32), (4, 12))
    # sig_y=12 (axis 1) should spread the anisotropic pdf's mass further
    # along axis 1 than the isotropic (sigma=4) pdf, while axis 0 (sig_x=4
    # in both) stays about the same.
    assert _weighted_std(pdf_aniso, axis=1) > _weighted_std(pdf_iso, axis=1) * 2
    assert _weighted_std(pdf_aniso, axis=0) == pytest.approx(_weighted_std(pdf_iso, axis=0), rel=0.05)
