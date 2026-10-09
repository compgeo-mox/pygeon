"""Module contains general tests for all vector H(div) discretizations."""

import pytest

import pygeon as pg


@pytest.fixture(
    params=[
        pg.VecRT0,
        pg.VecBDM1,
        pg.VecRT1,
    ]
)
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


def test_range(discr, ref_sd):
    if ref_sd.dim == 1:
        return
    known_range = pg.get_PwPolynomials(discr.poly_order - 1, 1)
    assert discr.get_range_discr_class(ref_sd.dim) is known_range
