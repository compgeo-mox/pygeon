"""Module contains vector Lagrangean2 fem tests."""

import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VecLagrange2("test")


def test_ndof(discr, unit_sd):
    scalar_discr = pg.Lagrange2(discr.keyword)
    assert discr.ndof(unit_sd) == unit_sd.dim * scalar_discr.ndof(unit_sd)


def test_range_disc(discr):
    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(2)
