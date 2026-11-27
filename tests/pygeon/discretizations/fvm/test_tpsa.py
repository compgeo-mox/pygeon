"""Unit tests for the TPSA finite-volume discretization."""

import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.TPSA:
    """Create a TPSA discretization with a test keyword."""
    return pg.TPSA("test")


def test_ndof(discr, ref_sd_3d):
    """Check number of TPSA degrees of freedom per 3D cell."""
    assert discr.ndof(ref_sd_3d) == 7


def test_ndof_2D(discr, ref_square):
    """Check number of TPSA degrees of freedom per 2D cell."""
    assert discr.ndof(ref_square) == 4


def test_body_force(discr, unit_sd_3d):
    """Check body-force assembly for constant gravity in 3D."""
    gravity = lambda _: np.array([0, 0, -1])
    bf = discr.assemble_body_force(unit_sd_3d, gravity)

    known_bf = np.zeros(discr.ndof(unit_sd_3d))
    known_bf[2 * unit_sd_3d.num_cells : 3 * unit_sd_3d.num_cells] = (
        unit_sd_3d.cell_volumes
    )

    assert np.allclose(bf, known_bf)


def test_1D(discr: pg.TPSA, unit_sd_1d):
    """Check that TPSA rejects one-dimensional grids."""
    data = pp.initialize_data({}, discr.keyword)
    with pytest.raises(ValueError):
        discr.assemble_system_matrix(unit_sd_1d, data)


def test_without_data(discr: pg.TPSA, unit_sd_3d):
    """Check that the boundary contribution is zero when data is absent."""
    discr.assemble_system_matrix(unit_sd_3d)
    rhs = discr.assemble_rhs_boundary_vector(unit_sd_3d)

    assert np.allclose(rhs, 0)


def test_without_bcs(discr: pg.TPSA, unit_sd_3d):
    """Check that the boundary contribution is zero when BCs are absent."""
    data = pp.initialize_data({}, discr.keyword)
    discr.assemble_system_matrix(unit_sd_3d, data)
    rhs = discr.assemble_rhs_boundary_vector(unit_sd_3d, data)

    assert np.allclose(rhs, 0)


def test_without_indices(discr: pg.TPSA, unit_sd_3d):
    """Check BC setters default to no-op when no indices are provided."""
    data = pp.initialize_data({}, discr.keyword)
    bcs = pg.ElasticityBC(unit_sd_3d, data, discr.keyword)
    bcs.set_displacement_bcs()

    discr.assemble_system_matrix(unit_sd_3d, data)
    rhs = discr.assemble_rhs_boundary_vector(unit_sd_3d, data)

    assert np.allclose(rhs, 0)


def test_external_cell_center(ref_sd_3d, discr):
    """Check warning when weighted distances indicate an external cell center."""
    sd = ref_sd_3d.copy()
    sd.cell_centers = sd.cell_centers + 100
    data = pp.initialize_data({}, discr.keyword)

    with pytest.warns(UserWarning):
        discr.precompute_arrays(sd, data)
