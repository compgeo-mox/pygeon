import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


def setup(sd):
    keyword = "test"
    tpfa = pg.TPFA(keyword)
    data = pp.initialize_data({}, keyword)
    bcs = pg.FlowBC(sd, data, keyword)

    bdry_faces = sd.tags["domain_boundary_faces"]
    p_known = tpfa.interpolate(sd, lambda x: x[0])

    return tpfa, data, bcs, bdry_faces, p_known


def check_residual(tpfa, sd, data, x_known):
    M = tpfa.assemble_flow_matrix(sd, data)
    rhs = tpfa.assemble_rhs_boundary_vector(sd, data)

    assert np.allclose(M @ x_known, rhs)


def test_pressure_bcs(unit_cart_sd):
    """
    Pressure conditions on all boundaries.
    """
    tpfa, data, bcs, bdry_faces, p_known = setup(unit_cart_sd)

    bcs.set_pressure_bcs(bdry_faces, unit_cart_sd.face_centers[0])

    check_residual(tpfa, unit_cart_sd, data, p_known)


def test_flux_bcs(unit_cart_sd):
    """
    Flux boundary conditions on all boundaries. The solution is defined up to a
    constant.
    """
    tpfa, data, bcs, bdry_faces, p_known = setup(unit_cart_sd)

    q_known = pg.VRT0().interpolate(unit_cart_sd, lambda _: np.array([-1, 0, 0]))
    bcs.set_flux_bcs(bdry_faces, q_known)

    check_residual(tpfa, unit_cart_sd, data, p_known)
    check_residual(tpfa, unit_cart_sd, data, 1 + p_known)


def test_robin_bcs(unit_cart_sd):
    """
    Sanity test that a positive source term leads to a positive pressure.
    """
    tpfa, data, bcs, bdry_faces, p_known = setup(unit_cart_sd)

    bcs.set_robin_bcs(1, bdry_faces)

    rhs = tpfa.assemble_source(unit_cart_sd, lambda _: 1)
    M = tpfa.assemble_flow_matrix(unit_cart_sd, data)

    sol = sps.linalg.spsolve(M, rhs)

    assert np.all(sol > 0)
