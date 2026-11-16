"""Module contains specific tests for the piecewise monomials."""

import numpy as np
import pytest

import pygeon as pg


def test_linears(unit_poly_sd):
    # A linear function is interpolated exactly: its value at the cell centers and its
    # integral over the cells, the projection to the constants, are exact
    sd = unit_poly_sd
    discr = pg.PwMLinears()
    fun = lambda x: 1 + 2 * x[0] - x[1]
    interp = discr.interpolate(sd, fun)

    at_centers = discr.eval_at_cell_centers(sd) @ interp
    known = fun(sd.cell_centers)
    assert np.allclose(at_centers, known), f"Values {at_centers} != {known}"

    integrals = discr.proj_to_lower_PwPolynomials(sd) @ interp
    known = known * sd.cell_volumes
    assert np.allclose(integrals, known), f"Integrals {integrals} != {known}"


def test_vector_linears(unit_poly_sd):
    discr = pg.VecPwMLinears()

    assert isinstance(discr.base_discr, pg.PwMLinears), (
        f"Base discretization is {type(discr.base_discr).__name__}, not PwMLinears"
    )
    ndof, known = discr.ndof(unit_poly_sd), 2 * pg.PwMLinears().ndof(unit_poly_sd)
    assert ndof == known, f"{ndof} dofs, expected {known}"
