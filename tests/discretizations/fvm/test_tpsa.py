import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture
def sd_and_sol():
    sd = pp.CartGrid([5, 5, 5], [1, 1, 1])
    sd = pg.convert_from_pp(sd)
    sd.compute_geometry()

    u_known = np.zeros_like(sd.cell_centers)
    u_known[0] = sd.cell_centers[-1]

    r_known = np.zeros_like(sd.cell_centers)
    r_known[1, :] = -1

    p_known = np.zeros(sd.num_cells)
    x_known = np.hstack((u_known.ravel(), r_known.ravel(), p_known))
    return sd, x_known


def test_displacement_bcs(sd_and_sol):
    sd, x_known = sd_and_sol

    tpsa = pg.TPSA("test")

    data = pp.initialize_data({}, "test", {pg.LAME_LAMBDA: 1, pg.LAME_MU: 1})
    bcs = pg.ElasticityBC(sd, data, "test")

    bdry_faces = sd.tags["domain_boundary_faces"]
    u_0 = np.zeros_like(sd.face_centers)
    u_0[0] = sd.face_centers[-1]
    bcs.set_displacement_bcs(bdry_faces, u_0)

    M = tpsa.assemble_elasticity_matrix(sd, data)
    rhs = tpsa.assemble_rhs_boundary_terms(sd, data)

    assert np.allclose(M @ x_known, rhs)


def test_traction_bcs(sd_and_sol):
    sd, x_known = sd_and_sol

    tpsa = pg.TPSA("test")

    data = pp.initialize_data({}, "test", {pg.LAME_LAMBDA: 1, pg.LAME_MU: 1})
    bcs = pg.ElasticityBC(sd, data, "test")

    bdry_faces = sd.tags["domain_boundary_faces"]
    bottom = np.isclose(sd.face_centers[-1], 0)
    bcs.set_displacement_bcs(bottom)

    sig_0 = np.zeros((3, 3))
    sig_0[0, 2] = 1
    sig_0[2, 0] = 1
    sig_0 = sig_0 @ sd.face_normals / sd.face_areas

    tract_faces = np.logical_xor(bottom, bdry_faces)
    bcs.set_traction_bcs(tract_faces, sig_0)

    M = tpsa.assemble_elasticity_matrix(sd, data)
    rhs = tpsa.assemble_rhs_boundary_terms(sd, data)

    assert np.allclose(M @ x_known, rhs)
