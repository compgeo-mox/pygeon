"""Module to test the primal formulations of elasticity."""

import numpy as np
import porepy as pp

import pygeon as pg


def test_rigid_body_motion_Lagrange1(unit_sd):
    discr = pg.VecLagrange1("test")
    data = {pp.PARAMETERS: {discr.keyword: {pg.LAME_LAMBDA: 1, pg.LAME_MU: 0.5}}}
    A = discr.assemble_stiff_matrix(unit_sd, data)

    ess_dofs = np.hstack([unit_sd.tags["domain_boundary_nodes"]] * unit_sd.dim)

    bc_fun = lambda x: np.array([0.5 - x[1], 0.5 + x[0] - x[2], 0.5 + x[1]])
    u_ex = discr.interpolate(unit_sd, bc_fun)

    ls = pg.LinearSystem(A)
    ls.flag_ess_bc(ess_dofs, u_ex)
    u = ls.solve()

    assert np.allclose(u, u_ex)

    sigma = discr.compute_stress(unit_sd, u, data)
    assert np.allclose(sigma, 0)


def test_footing_problem(unit_sd):
    # TODO: in 1D, map_geometry reverts the ordering, causing an opposite sign in the
    # stress.
    if unit_sd.dim == 1:
        return

    discr = pg.VecLagrange1("test")
    data = {pp.PARAMETERS: {discr.keyword: {pg.LAME_LAMBDA: 1, pg.LAME_MU: 0.5}}}
    A = discr.assemble_stiff_matrix(unit_sd, data)

    bottom = np.hstack([np.isclose(unit_sd.nodes[unit_sd.dim - 1, :], 0)] * unit_sd.dim)
    top = np.isclose(unit_sd.face_centers[unit_sd.dim - 1, :], 1)

    vec = np.zeros(3)
    vec[unit_sd.dim - 1] = -1
    fun = lambda _: vec

    b = discr.assemble_nat_bc(unit_sd, fun, top)

    ls = pg.LinearSystem(A, b)
    ls.flag_ess_bc(bottom, np.zeros(discr.ndof(unit_sd)))
    u = ls.solve()
    sigma = discr.compute_stress(unit_sd, u, data)

    assert np.all(np.trace(sigma, axis1=1, axis2=2) <= 0)
