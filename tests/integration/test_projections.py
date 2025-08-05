import numpy as np
import pytest

import pygeon as pg


@pytest.fixture(params=[1, 2, 3])
def sd(request):
    dim = request.param
    sd = pg.unit_grid(dim, 0.25, as_mdg=False)
    sd.compute_geometry()

    return sd


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
def disc(request):
    return request.param("test")


def test_mass_matrix(sd, disc):
    poly_order = disc.poly_order
    tensor_order = disc.tensor_order

    mass = disc.assemble_mass_matrix(sd)
    pi = pg.proj_to_PwPolynomials(disc, sd, poly_order)

    poly = pg.get_PwPolynomials(poly_order, tensor_order)()
    poly_mass = poly.assemble_mass_matrix(sd)

    diff = pi.T @ poly_mass @ pi - mass

    assert np.allclose(diff.data, 0)


def test_lumped_matrix(sd, disc):
    if isinstance(disc, pg.RT0):
        # The RT0 lumped matrix does not coincide with the
        # one from the piecewise polynomial interpretation
        # so this test is skipped
        return

    poly_order = disc.poly_order
    tensor_order = disc.tensor_order

    lumped = disc.assemble_lumped_matrix(sd)
    pi = pg.proj_to_PwPolynomials(disc, sd, poly_order)

    poly = pg.get_PwPolynomials(poly_order, tensor_order)()
    poly_lumped = poly.assemble_lumped_matrix(sd)

    diff = pi.T @ poly_lumped @ pi - lumped

    assert np.allclose(diff.data, 0)


if __name__ == "__main__":
    pytest.main([__file__])
