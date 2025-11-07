import pytest

import numpy as np
import scipy.sparse as sps

import pygeon as pg

"""
Module contains tests to validate the consistency between H(div) discretizations.
"""


@pytest.fixture(
    params=[
        (pg.RT0, pg.PwConstants),
        (pg.BDM1, pg.PwConstants),
        (pg.RT1, pg.PwLinears),
    ],
    ids=["RT0", "BDM1", "RT1"],
)
def discr(request):
    Disc_q, Disc_p = request.param
    return Disc_q("test"), Disc_p("test")


@pytest.mark.parametrize("use_lumped", [False, True])
def test_linear_distribution(discr, unit_sd, use_lumped):
    discr_q, discr_p = discr

    if isinstance(discr_q, pg.RT0) and use_lumped:
        # The lumped version of RT0 is not consistent
        return

    # Provide the solution
    def q_func(_):
        return np.array([-1, 2, 1])

    def p_func(x):
        return -x @ q_func(x)

    # assemble the saddle point problem
    if use_lumped:
        face_mass = discr_q.assemble_lumped_matrix(unit_sd)
    else:
        face_mass = discr_q.assemble_mass_matrix(unit_sd)
    cell_mass = discr_p.assemble_mass_matrix(unit_sd, None)
    div = cell_mass @ discr_q.assemble_diff_matrix(unit_sd)

    spp = sps.block_array([[face_mass, -div.T], [div, None]]).tocsc()

    # set the boundary conditions
    b_faces = unit_sd.tags["domain_boundary_faces"]
    bc_val = -discr_q.assemble_nat_bc(unit_sd, p_func, b_faces)

    rhs = np.zeros(spp.shape[0])
    rhs[: bc_val.size] += bc_val

    # solve the problem
    ls = pg.LinearSystem(spp, rhs)
    x = ls.solve()

    q = x[: bc_val.size]
    p = x[-discr_p.ndof(unit_sd) :]

    known_q = discr_q.interpolate(unit_sd, q_func)
    known_p = discr_p.interpolate(unit_sd, p_func)

    assert np.allclose(p, known_p)
    assert np.allclose(q, known_q)


def test_convergence_2D(discr):
    # Provide the solution
    def q_func(x):
        return np.array([x[1] * x[0], 2 * x[0], 0])

    def p_func(x):
        return x[0]

    def g_func(x):
        return q_func(x) + np.array([1, 0, 0])

    def div_func(x):
        return x[1]

    discr_q, discr_p = discr

    h_list = np.array([1 / 6, 1 / 12])

    error_p = np.zeros_like(h_list)
    error_q = np.zeros_like(h_list)

    for ind, h in enumerate(h_list):
        sd = pg.unit_grid(2, h, as_mdg=False)
        sd.compute_geometry()
        h_list[ind] = sd.mesh_size

        # assemble the saddle point problem
        face_mass = discr_q.assemble_mass_matrix(sd)
        cell_mass = discr_p.assemble_mass_matrix(sd, None)
        div = cell_mass @ discr_q.assemble_diff_matrix(sd)

        spp = sps.bmat([[face_mass, -div.T], [div, None]], format="csc")

        # set the boundary conditions
        b_faces = sd.tags["domain_boundary_faces"]
        bc_val = -discr_q.assemble_nat_bc(sd, p_func, b_faces)

        rhs = np.zeros(spp.shape[0])
        rhs[bc_val.size :] += discr_p.source_term(sd, div_func)
        rhs[: bc_val.size] += bc_val
        rhs[: bc_val.size] += discr_q.source_term(sd, g_func)

        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        x = ls.solve()

        q = x[: bc_val.size]
        p = x[-discr_p.ndof(sd) :]

        error_p[ind] = discr_p.error_l2(sd, p, p_func)
        error_q[ind] = discr_q.error_l2(sd, q, q_func)

    rates_p = np.log(error_p[1:] / error_p[:-1]) / np.log(h_list[1:] / h_list[:-1])
    rates_q = np.log(error_q[1:] / error_q[:-1]) / np.log(h_list[1:] / h_list[:-1])

    match type(discr_q):
        case pg.RT0:
            rates = [1, 1]
        case pg.BDM1:
            rates = [2, 1]
        case pg.RT1:
            rates = [2, 2]

    assert np.all(rates_q >= rates[0] - 0.1)
    assert np.all(rates_p >= rates[1] - 0.1)
