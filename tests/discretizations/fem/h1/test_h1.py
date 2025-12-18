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
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
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


def test_point_grid(discr, ref_sd_0d):
    """Tests with a point grid"""

    assert discr.ndof(ref_sd_0d) == 0
    assert discr.assemble_diff_matrix(ref_sd_0d).nnz == 0
    assert discr.eval_at_cell_centers(ref_sd_0d).nnz == 0
    assert discr.interpolate(ref_sd_0d, lambda x: x).size == 0

    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(ref_sd_0d.dim)

    with pytest.raises(ValueError):
        sd_copy = ref_sd_0d.copy()
        sd_copy.dim = -1
        discr.assemble_diff_matrix(sd_copy)
