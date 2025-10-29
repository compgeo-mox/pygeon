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
        pg.PwConstants,
        pg.PwLinears,
        pg.PwQuadratics,
    ]
)
def discr(request):
    return request.param("test")


def test_mass_matrix(discr, unit_sd):
    poly_order = discr.poly_order
    tensor_order = discr.tensor_order

    mass = discr.assemble_mass_matrix(unit_sd)
    pi = pg.proj_to_PwPolynomials(discr, unit_sd, poly_order)

    poly = pg.get_PwPolynomials(poly_order, tensor_order)()
    poly_mass = poly.assemble_mass_matrix(unit_sd)

    diff = pi.T @ poly_mass @ pi - mass

    assert np.allclose(diff.data, 0)


def test_lumped_matrix(discr, unit_sd):
    if isinstance(discr, pg.RT0):
        # The RT0 lumped matrix does not coincide with the
        # one from the piecewise polynomial interpretation
        # so this test is skipped
        return

    poly_order = discr.poly_order
    tensor_order = discr.tensor_order

    lumped = discr.assemble_lumped_matrix(unit_sd)
    pi = pg.proj_to_PwPolynomials(discr, unit_sd, poly_order)

    poly = pg.get_PwPolynomials(poly_order, tensor_order)()
    poly_lumped = poly.assemble_lumped_matrix(unit_sd)

    diff = pi.T @ poly_lumped @ pi - lumped

    assert np.allclose(diff.data, 0)
