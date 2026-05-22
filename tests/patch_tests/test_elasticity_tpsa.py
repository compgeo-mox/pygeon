import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

import pygeon as pg


@pytest.fixture(params=[2, 3], ids=["2D", "3D"])
def setup(request):
    """
    Leaning tower setup inspired by Porepy test_tpsa.py in which the displacement is
    given by [z, 0, 0] and we test several choices of boundary conditions.

    The 2D analogue has displacement u = [y, 0].
    """
    dim = request.param
    sd = pp.CartGrid([5] * dim, [1] * dim)
    sd = pg.convert_from_pp(sd)
    sd.compute_geometry()

    tpsa = pg.TPSA("test")

    # Compute a known solution
    displacement = lambda x: np.array([x[dim - 1], 0, 0])
    if dim == 2:
        rotation = lambda _: 1
    else:
        rotation = lambda _: np.array([0, -1, 0])
    solid_pressure = lambda _: 0
    x_known = tpsa.interpolate(sd, displacement, rotation, solid_pressure)

    # Generate data and a boundary condition object
    data = pp.initialize_data({}, "test", {pg.LAME_LAMBDA: 1, pg.LAME_MU: 1})
    bcs = pg.ElasticityBC(sd, data, "test")
    bdry_faces = sd.tags["domain_boundary_faces"]

    def sig_fun(_):
        sigma = np.zeros((3, 3))
        sigma[0, dim - 1] = 1
        sigma[dim - 1, 0] = 1
        return sigma

    u_0 = np.array([displacement(x) for x in sd.face_centers.T]).T[:dim]
    sig_0 = pg.VecRT0().interpolate(sd, sig_fun).reshape((dim, -1))

    return tpsa, sd, x_known, data, bcs, bdry_faces, u_0, sig_0


def check_residual(tpsa, sd, data, x_known):
    M = tpsa.assemble_elasticity_matrix(sd, data)
    rhs = tpsa.assemble_rhs_boundary_vector(sd, data)

    assert np.allclose(M @ x_known, rhs)


def test_displacement_bcs(setup):
    """
    Displacement boundary conditions on the entire boundary
    """
    tpsa, sd, x_known, data, bcs, bdry_faces, u_0, sig_0 = setup
    bcs.set_displacement_bcs(bdry_faces, u_0)

    check_residual(tpsa, sd, data, x_known)


def test_traction_bcs(setup):
    """
    Traction boundary conditions on the entire boundary. The solution is unique up to
    rigid body motions.
    """
    tpsa, sd, x_known, data, bcs, bdry_faces, u_0, sig_0 = setup
    bcs.set_traction_bcs(bdry_faces, sig_0)

    check_residual(tpsa, sd, data, x_known)


def test_traction_and_disp_bcs(setup):
    """
    Clamped bottom, zero traction on the rest
    """
    tpsa, sd, x_known, data, bcs, bdry_faces, u_0, sig_0 = setup

    bottom = np.isclose(sd.face_centers[sd.dim - 1], 0)
    disp_faces = np.tile(bottom, (sd.dim, 1))
    bcs.set_displacement_bcs(disp_faces, u_0)

    trac_faces = np.logical_xor(disp_faces, bdry_faces)
    bcs.set_traction_bcs(trac_faces, sig_0)

    check_residual(tpsa, sd, data, x_known)


def test_sliding_bcs(setup):
    """
    Bottom clamped and sliding conditions on the remaining boundaries. Zero traction in
    the x and y directions and zero z-displacement. (zero y-displacement in 2D)
    """
    tpsa, sd, x_known, data, bcs, bdry_faces, u_0, sig_0 = setup

    bottom = np.isclose(sd.face_centers[sd.dim - 1], 0)
    disp_faces = np.tile(bottom, (sd.dim, 1))
    disp_faces[sd.dim - 1] = bdry_faces
    bcs.set_displacement_bcs(disp_faces, u_0)

    tract_faces = np.logical_xor(disp_faces, bdry_faces)
    bcs.set_traction_bcs(tract_faces, sig_0)

    check_residual(tpsa, sd, data, x_known)


def test_spring_bcs(setup):
    """
    Sanity test to ensure that a body force produces a displacement in the same
    direction
    """
    tpsa, sd, x_known, data, bcs, bdry_faces, u_0, sig_0 = setup

    bcs.set_spring_bcs(np.ones_like(bcs.weighted_dists), bdry_faces)
    tpsa = pg.TPSA("test")

    M = tpsa.assemble_elasticity_matrix(sd, data)
    rhs = tpsa.assemble_rhs_boundary_vector(sd, data)
    rhs += tpsa.assemble_body_force(sd, lambda _: np.array([1, 0, 0]))

    sol = sps.linalg.spsolve(M, rhs)
    u, r, p = tpsa.split_solution(sd, sol)

    assert np.all(u[: sd.num_cells] > 0)
