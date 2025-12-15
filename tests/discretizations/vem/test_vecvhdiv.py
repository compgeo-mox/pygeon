"""Module contains general tests for all vector H(div) discretizations."""

import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VecVRT0("test")


def test_range(discr, ref_sd):
    known_range = pg.get_PwPolynomials(discr.poly_order - 1, discr.tensor_order - 1)
    assert discr.get_range_discr_class(ref_sd.dim) is known_range
