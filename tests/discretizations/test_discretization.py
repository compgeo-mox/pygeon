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
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


def test_string_repr():
    discr = pg.PwConstants("test")
    repr = str(discr)
    known = "Discretization of type PwConstants with keyword test"

    assert repr == known


def test_cochain_property(discr, unit_sd):
    # Skip the Nedelec spaces in 1D
    if isinstance(discr, (pg.Nedelec0, pg.Nedelec1)) and unit_sd.dim == 1:
        return

    unit_sd.compute_geometry()

    Diff = discr.assemble_diff_matrix(unit_sd)
    range_discr = discr.get_range_discr_class(unit_sd.dim)(discr.keyword)
    range_Diff = range_discr.assemble_diff_matrix(unit_sd)

    prod = range_Diff @ Diff
    assert np.allclose(prod.data, 0)


def test_eval_at_cc(discr, unit_sd):
    Pi_child = discr.eval_at_cell_centers(unit_sd)
    Pi_super = pg.Discretization.eval_at_cell_centers(discr, unit_sd)

    assert np.allclose((Pi_child - Pi_super).data, 0)
