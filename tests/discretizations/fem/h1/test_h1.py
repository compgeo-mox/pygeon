"""Module contains general tests for all H1 discretizations."""

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


def test_interpolate_and_evaluate(discr, unit_sd):
    func = lambda x: x[0] ** discr.poly_order
    known_vals = func(unit_sd.cell_centers)

    interp = discr.interpolate(unit_sd, func)
    proj = discr.eval_at_cell_centers(unit_sd)

    assert np.allclose(proj @ interp, known_vals)


def test_lumped_consistency(discr, unit_sd):
    M_lumped = discr.assemble_lumped_matrix(unit_sd)
    M_full = discr.assemble_mass_matrix(unit_sd)

    one_interp = discr.interpolate(unit_sd, lambda _: 1)

    integral_L = M_lumped @ one_interp
    integral_M = M_full @ one_interp

    assert np.allclose(integral_L, integral_M)


def test_stiffness_consistency(discr, unit_sd):
    """Compare the implemented stiffness matrix
    to the one obtained by mapping to the range discretization"""

    if isinstance(discr, pg.Lagrange2) and unit_sd.dim == 3:
        with pytest.raises(NotImplementedError):
            Stiff_2 = pg.Discretization.assemble_stiff_matrix(discr, unit_sd)
        return

    Stiff_1 = discr.assemble_stiff_matrix(unit_sd)
    Stiff_2 = pg.Discretization.assemble_stiff_matrix(discr, unit_sd)

    diff = Stiff_1 - Stiff_2
    assert np.allclose(diff.data, 0)
