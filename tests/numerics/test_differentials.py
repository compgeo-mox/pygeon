import numpy as np

import pygeon as pg

"""
Module contains tests to validate the differential operators.
"""

# It's not straightforward to concatenate parameterized fixtures, so we define a test
# for each fixture.


def check_cochain_property(sd):
    grad = pg.grad(sd)
    curl = pg.curl(sd)
    div = pg.div(sd)

    curl_grad = curl @ grad
    div_curl = div @ curl

    assert curl_grad.nnz == 0
    assert div_curl.nnz == 0


def test_0d(ref_sd_0d):
    check_cochain_property(ref_sd_0d)


def test_cochain_unit_sd(unit_sd):
    check_cochain_property(unit_sd)

    # Make sure the zero maps have the appropriate shapes
    zero = pg.numerics.differentials.exterior_derivative(unit_sd, 4)
    assert np.allclose(zero.shape, (unit_sd.num_peaks, 0))

    zero = pg.numerics.differentials.exterior_derivative(unit_sd, 0)
    assert np.allclose(zero.shape, (0, unit_sd.num_cells))


def test_cochain_unit_cart(unit_cart_sd):
    check_cochain_property(unit_cart_sd)


def test_cochain_mdg(mdg):
    check_cochain_property(mdg)
