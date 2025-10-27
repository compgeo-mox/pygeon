import numpy as np
import pytest

import pygeon as pg


@pytest.fixture(
    params=[
        pg.Lagrange1,
        pg.Lagrange2,
    ]
)
def discr(request):
    return request.param("test")


def test_interpolate_and_evaluate(discr: pg.Discretization, unit_sd: pg.Grid):
    func = lambda x: x[0] ** discr.poly_order
    known_vals = func(unit_sd.cell_centers)

    interp = discr.interpolate(unit_sd, func)
    proj = discr.eval_at_cell_centers(unit_sd)

    assert np.allclose(proj @ interp, known_vals)


def test_lumped_consistency(discr, unit_sd):
    M_lumped = discr.assemble_lumped_matrix(unit_sd)
    M_full = discr.assemble_mass_matrix(unit_sd)

    func = lambda x: x[0]
    func_interp = discr.interpolate(unit_sd, func)
    one_interp = discr.interpolate(unit_sd, lambda _: 1)

    integral_L = one_interp @ M_lumped @ func_interp
    integral_M = one_interp @ M_full @ func_interp

    assert np.isclose(integral_L, integral_M)
