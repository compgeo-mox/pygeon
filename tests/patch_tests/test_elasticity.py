"""Module to test the primal formulations of elasticity."""

import numpy as np
import porepy as pp
import pytest
import sympy as sp

import pygeon as pg


def test_rigid_body_motion_Lagrange1(unit_sd):
    discr = pg.VecLagrange1("test")

    data = pp.initialize_data({}, discr.keyword, {pg.LAME_MU: 0.5, pg.LAME_LAMBDA: 1})
    A = discr.assemble_stiff_matrix_elasticity(unit_sd, data)

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
    discr = pg.VecLagrange1("test")
    data = pp.initialize_data({}, discr.keyword, {pg.LAME_MU: 0.5, pg.LAME_LAMBDA: 1})
    A = discr.assemble_stiff_matrix_elasticity(unit_sd, data)

    bottom = np.hstack([np.isclose(unit_sd.nodes[unit_sd.dim - 1, :], 0)] * unit_sd.dim)
    top = np.isclose(unit_sd.face_centers[unit_sd.dim - 1, :], 1)

    vec = np.zeros(pg.AMBIENT_DIM)
    vec[unit_sd.dim - 1] = -1
    fun = lambda _: vec

    b = discr.assemble_nat_bc(unit_sd, fun, top)

    ls = pg.LinearSystem(A, b)
    ls.flag_ess_bc(bottom, np.zeros(discr.ndof(unit_sd)))
    u = ls.solve()
    sigma = discr.compute_stress(unit_sd, u, data)

    assert np.all(np.trace(sigma, axis1=0, axis2=1) <= 0)


def test_quadratic_displacement_Lagrange2(unit_sd):
    if unit_sd.dim == 1:
        pytest.skip("No elasticity in 1D")

    discr = pg.VecLagrange2("test")
    data = pp.initialize_data({}, discr.keyword, {pg.LAME_MU: 0.5, pg.LAME_LAMBDA: 1})

    # A quadratic displacement is represented exactly, with its constant body force
    x, y, z = pg.exact.coordinates()
    u = sp.Matrix([x**2 + x * y, y**2 - x * z, z**2 + x * y])
    eps = pg.exact.symmetric_gradient(u)
    sigma = 2 * 0.5 * eps + eps.trace() * pg.exact.identity()
    f = -pg.exact.matrix_divergence(sigma)

    A = discr.assemble_stiff_matrix_elasticity(unit_sd, data)
    b = discr.assemble_mass_matrix(unit_sd) @ discr.interpolate(
        unit_sd, pg.exact.to_callable(f)
    )

    # Essential conditions at the boundary nodes and edges
    edges = "domain_boundary_faces" if unit_sd.dim == 2 else "domain_boundary_ridges"
    bd = np.hstack((unit_sd.tags["domain_boundary_nodes"], unit_sd.tags[edges]))
    ess_dofs = np.tile(bd, unit_sd.dim)
    u_ex = discr.interpolate(unit_sd, pg.exact.to_callable(u))

    ls = pg.LinearSystem(A, b)
    ls.flag_ess_bc(ess_dofs, u_ex)
    u_h = ls.solve()

    assert np.allclose(u_h, u_ex)

    # The stress is linear, so exact at the cell centers in the dim x dim block
    dim = unit_sd.dim
    sigma_h = discr.compute_stress(unit_sd, u_h, data)
    sigma_ex = pg.exact.to_callable(sigma)(unit_sd.cell_centers)
    assert np.allclose(sigma_h[:dim, :dim], sigma_ex[:dim, :dim])
