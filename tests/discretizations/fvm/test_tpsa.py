import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.TPSA:
    return pg.TPSA("test")


def test_ndof(discr, ref_sd_3d):
    assert discr.ndof(ref_sd_3d) == 7


def test_ndof_2D(discr, ref_square):
    assert discr.ndof(ref_square) == 4


def test_body_force(discr, unit_sd_3d):
    gravity = lambda _: np.array([0, 0, -1])
    bf = discr.assemble_body_force(unit_sd_3d, gravity)

    known_bf = np.zeros(discr.ndof(unit_sd_3d))
    known_bf[2 * unit_sd_3d.num_cells : 3 * unit_sd_3d.num_cells] = (
        unit_sd_3d.cell_volumes
    )

    assert np.allclose(bf, known_bf)


def test_1D(discr, unit_sd_1d):
    data = pp.initialize_data({}, discr.keyword)
    with pytest.raises(ValueError):
        discr.assemble_elasticity_matrix(unit_sd_1d, data)


def test_assemble_rhs_first(discr, unit_sd_3d):
    data = pp.initialize_data({}, discr.keyword)
    with pytest.raises(AssertionError):
        discr.assemble_rhs_boundary_terms(unit_sd_3d, data)


def test_without_bcs(discr, unit_sd_3d):
    data = pp.initialize_data({}, discr.keyword)
    discr.assemble_elasticity_matrix(unit_sd_3d, data)
    rhs = discr.assemble_rhs_boundary_terms(unit_sd_3d, data)

    assert np.allclose(rhs, 0)


def test_without_indices(discr, unit_sd_3d):
    data = pp.initialize_data({}, discr.keyword)
    bcs = pg.ElasticityBC(unit_sd_3d, data, discr.keyword)
    bcs.set_displacement_bcs()

    discr.assemble_elasticity_matrix(unit_sd_3d, data)
    rhs = discr.assemble_rhs_boundary_terms(unit_sd_3d, data)

    assert np.allclose(rhs, 0)


def test_external_cell_center(ref_sd_3d, discr):
    sd = ref_sd_3d.copy()
    sd.cell_centers = sd.cell_centers + 100

    with pytest.warns(UserWarning):
        discr.fvm_precomputations(sd, np.ones(1))
