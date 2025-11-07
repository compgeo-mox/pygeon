import pygeon as pg

"""
Module contains tests to validate the differential operators.
"""

# It's not straightforward to concatenate parameterized fixtures, so we define a test
# for each fixture.


def test_cochain_unit_sd(unit_sd):
    check_cochain_property(unit_sd)


def test_cochain_unit_cart(unit_cart_sd):
    check_cochain_property(unit_cart_sd)


def test_cochain_mdg(mdg):
    check_cochain_property(mdg)


def check_cochain_property(sd):
    grad = pg.grad(sd)
    curl = pg.curl(sd)
    div = pg.div(sd)

    curl_grad = curl @ grad
    div_curl = div @ curl

    assert curl_grad.nnz == 0
    assert div_curl.nnz == 0
