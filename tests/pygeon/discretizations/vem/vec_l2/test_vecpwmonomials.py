"""Module contains specific tests for the piecewise monomials."""

import pygeon as pg


def test_vector_linears(unit_poly_sd):
    discr = pg.VecPwMLinears()

    assert isinstance(discr.base_discr, pg.PwMLinears), (
        f"Base discretization is {type(discr.base_discr).__name__}, not PwMLinears"
    )
    ndof, known = discr.ndof(unit_poly_sd), 2 * pg.PwMLinears().ndof(unit_poly_sd)
    assert ndof == known, f"{ndof} dofs, expected {known}"
