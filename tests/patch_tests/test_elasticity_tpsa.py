import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

import pygeon as pg


@pytest.fixture
def setup():
    sd = pp.CartGrid([5, 5, 5], [1, 1, 1])
    sd = pg.convert_from_pp(sd)
    sd.compute_geometry()

    u_known = np.zeros_like(sd.cell_centers[: sd.dim])
    u_known[0] = sd.cell_centers[sd.dim - 1]

    r_known = np.zeros_like(sd.cell_centers)
    r_known[1, :] = -1

    p_known = np.zeros(sd.num_cells)
    x_known = np.hstack((u_known.ravel(), r_known.ravel(), p_known))

    data = pp.initialize_data({}, "test", {pg.LAME_LAMBDA: 1, pg.LAME_MU: 1})
    bcs = pg.ElasticityBC(sd, data, "test")

    return sd, x_known, data, bcs


def check_residual(sd, data, x_known):
    tpsa = pg.TPSA("test")
    M = tpsa.assemble_elasticity_matrix(sd, data)
    rhs = tpsa.assemble_rhs_boundary_terms(sd, data)

    assert np.allclose(M @ x_known, rhs)


def test_displacement_bcs(setup):
    sd, x_known, data, bcs = setup

    bdry_faces = sd.tags["domain_boundary_faces"]
    u_0 = np.zeros_like(sd.face_centers)
    u_0[0] = sd.face_centers[sd.dim - 1]
    bcs.set_displacement_bcs(bdry_faces, u_0)

    check_residual(sd, data, x_known)


def test_traction_bcs(setup):
    sd, x_known, data, bcs = setup

    bdry_faces = sd.tags["domain_boundary_faces"]
    bottom = np.isclose(sd.face_centers[sd.dim - 1], 0)
    bcs.set_displacement_bcs(bottom)

    sig_0 = np.zeros((3, 3))
    sig_0[0, 2] = 1
    sig_0[2, 0] = 1
    sig_0 = sig_0 @ sd.face_normals / sd.face_areas

    tract_faces = np.logical_xor(bottom, bdry_faces)
    bcs.set_traction_bcs(tract_faces, sig_0)

    check_residual(sd, data, x_known)


def test_sliding_bcs(setup):
    sd, x_known, data, bcs = setup

    u_0 = np.zeros_like(sd.face_centers)
    u_0[0] = sd.face_centers[sd.dim - 1]

    bdry_faces = sd.tags["domain_boundary_faces"]
    bottom = np.isclose(sd.face_centers[sd.dim - 1], 0)
    top = np.isclose(sd.face_centers[sd.dim - 1], 1)
    bottom_and_top = np.logical_or(bottom, top)

    disp_faces = np.tile(bdry_faces, (3, 1))
    disp_faces[0] = bottom_and_top
    bcs.set_displacement_bcs(disp_faces, u_0)

    tract_faces = np.logical_xor(disp_faces, bdry_faces)
    bcs.set_traction_bcs(tract_faces)

    check_residual(sd, data, x_known)


def test_spring_bcs(setup):
    sd, _, data, bcs = setup

    spring_faces = sd.tags["domain_boundary_faces"]
    bcs.set_spring_bcs(np.ones_like(bcs.weighted_dists), spring_faces)

    tpsa = pg.TPSA("test")

    M = tpsa.assemble_elasticity_matrix(sd, data)
    rhs = tpsa.assemble_rhs_boundary_terms(sd, data)
    rhs += tpsa.assemble_body_force(sd, lambda _: np.array([0, 0, -1]))

    sol = sps.linalg.spsolve(M, rhs)
    u, r, p = tpsa.split_solution(sd, sol)

    assert np.all(u[-sd.num_cells :] < 0)


def test_displacement_bcs_2D():
    sd = pp.CartGrid([5, 5], [1, 1])
    sd = pg.convert_from_pp(sd)
    sd.compute_geometry()

    u_known = np.zeros_like(sd.cell_centers[: sd.dim])
    u_known[0] = sd.cell_centers[sd.dim - 1]

    r_known = np.ones(sd.num_cells)

    p_known = np.zeros(sd.num_cells)
    x_known = np.hstack((u_known.ravel(), r_known.ravel(), p_known))

    data = pp.initialize_data({}, "test", {pg.LAME_LAMBDA: 1, pg.LAME_MU: 1})
    bcs = pg.ElasticityBC(sd, data, "test")

    bdry_faces = sd.tags["domain_boundary_faces"]
    u_0 = np.zeros_like(sd.face_centers)
    u_0[0] = sd.face_centers[sd.dim - 1]
    bcs.set_displacement_bcs(bdry_faces, u_0)

    tpsa = pg.TPSA("test")
    M = tpsa.assemble_elasticity_matrix(sd, data)
    rhs = tpsa.assemble_rhs_boundary_terms(sd, data)

    assert np.allclose(M @ x_known - rhs, 0)
