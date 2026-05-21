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

    displacement = lambda x: np.array([x[sd.dim - 1], 0, 0])
    rotation = lambda _: np.array([0, -1, 0])
    solid_pressure = lambda _: 0

    x_known = pg.TPSA().interpolate(sd, displacement, rotation, solid_pressure)

    data = pp.initialize_data({}, "test", {pg.LAME_LAMBDA: 1, pg.LAME_MU: 1})
    bcs = pg.ElasticityBC(sd, data, "test")

    return sd, x_known, data, bcs


def sig_fun(_):
    sigma = np.zeros((3, 3))
    sigma[0, 2] = 1
    sigma[2, 0] = 1
    return sigma


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

    sig_0 = pg.VecRT0().interpolate(sd, sig_fun).reshape((3, -1))

    tract_faces = np.logical_xor(bottom, bdry_faces)
    bcs.set_traction_bcs(tract_faces, sig_0)

    check_residual(sd, data, x_known)


def test_sliding_bcs(setup):
    sd, x_known, data, bcs = setup

    u_0 = np.zeros_like(sd.face_centers)
    u_0[0] = sd.face_centers[sd.dim - 1]

    sig_0 = pg.VecRT0().interpolate(sd, sig_fun).reshape((3, -1))

    bdry_faces = sd.tags["domain_boundary_faces"]
    bottom = np.isclose(sd.face_centers[sd.dim - 1], 0)

    disp_faces = np.tile(bottom, (3, 1))
    disp_faces[sd.dim - 1] = bdry_faces
    bcs.set_displacement_bcs(disp_faces, u_0)

    tract_faces = np.logical_xor(disp_faces, bdry_faces)
    bcs.set_traction_bcs(tract_faces, sig_0)

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


def test_displacement_bcs_2D(cart_sd_2d):
    sd = cart_sd_2d
    tpsa = pg.TPSA("test")

    displacement = lambda x: np.array([x[sd.dim - 1], 0, 0])
    rotation = lambda _: 1
    solid_pressure = lambda _: 0

    x_known = tpsa.interpolate(sd, displacement, rotation, solid_pressure)

    data = pp.initialize_data({}, "test", {pg.LAME_LAMBDA: 1, pg.LAME_MU: 1})
    bcs = pg.ElasticityBC(sd, data, "test")

    bdry_faces = sd.tags["domain_boundary_faces"]
    u_0 = np.zeros_like(sd.face_centers)
    u_0[0] = sd.face_centers[sd.dim - 1]
    bcs.set_displacement_bcs(bdry_faces, u_0)

    M = tpsa.assemble_elasticity_matrix(sd, data)
    rhs = tpsa.assemble_rhs_boundary_terms(sd, data)

    assert np.allclose(M @ x_known - rhs, 0)
