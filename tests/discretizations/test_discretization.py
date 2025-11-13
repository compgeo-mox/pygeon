"""Module contains general tests for all discretizations."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture(
    params=[
        pg.Lagrange1,
        pg.Lagrange2,
        pg.RT0,
        pg.BDM1,
        pg.RT1,
        pg.Nedelec0,
        pg.Nedelec1,
    ]
)
def discr(request):
    return request.param("test")


def test_string_repr():
    discr = pg.PwConstants("test")
    repr = str(discr)
    known = "Discretization of type PwConstants with keyword test"

    assert repr == known


def test_cochain_property(discr, unit_sd):
    unit_sd.compute_geometry()

    Diff = discr.assemble_diff_matrix(unit_sd)
    range_discr = discr.get_range_discr_class(unit_sd.dim)(discr.keyword)
    range_Diff = range_discr.assemble_diff_matrix(unit_sd)

    prod = range_Diff @ Diff
    assert np.allclose(prod.data, 0)
