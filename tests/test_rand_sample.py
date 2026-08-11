import numpy as np

from sampling.gen_gaussian_pdf import gen_gaussian_pdf
from sampling.rand_sample import rand_sample


def test_rand_sample_exact_count_uniform():
    rng = np.random.default_rng(0)
    mask = rand_sample([20, 30], 5, None, rng)
    assert mask.shape == (20, 30)
    assert mask.sum() == round(20 * 30 / 5)


def test_rand_sample_exact_count_weighted():
    rng = np.random.default_rng(1)
    weights = gen_gaussian_pdf((24, 24), 6)
    mask = rand_sample([24, 24], 4, weights, rng)
    assert mask.sum() == round(24 * 24 / 4)


def test_rand_sample_biased_toward_high_weight_region():
    rng = np.random.default_rng(2)
    Nx, Ny = 40, 40
    weights = gen_gaussian_pdf((Nx, Ny), 4)  # peak near center
    # High acceleration exaggerates the density bias so the statistical
    # check isn't flaky.
    mask = rand_sample([Nx, Ny], 8, weights, rng)

    cx, cy = Nx // 2, Ny // 2
    center_region = mask[cx - 8 : cx + 8, cy - 8 : cy + 8]
    center_density = center_region.mean()
    overall_density = mask.mean()
    assert center_density > overall_density
