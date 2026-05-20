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

    return tpfa, data, bcs, bdry_faces


def test_pressure_bcs(unit_cart_sd):
    tpfa, data, bcs, bdry_faces = setup(unit_cart_sd)
    p_known = unit_cart_sd.cell_centers[0]

    bcs.set_pressure_bcs(bdry_faces, unit_cart_sd.face_centers[0])

    M = tpfa.assemble_flow_matrix(unit_cart_sd, data)
    rhs = tpfa.assemble_rhs_bdry_terms(unit_cart_sd, data)

    assert np.allclose(M @ p_known, rhs)


def test_flux_bcs(unit_cart_sd):
    tpfa, data, bcs, bdry_faces = setup(unit_cart_sd)
    p_known = unit_cart_sd.cell_centers[0]

    bottom = np.isclose(unit_cart_sd.face_centers[0], 0)
    bcs.set_pressure_bcs(bottom, unit_cart_sd.face_centers[0])

    q_known = (
        -unit_cart_sd.face_centers[0]
        * unit_cart_sd.face_normals[0]
        / unit_cart_sd.face_areas
    )

    flux_faces = np.logical_xor(bottom, bdry_faces)
    bcs.set_flux_bcs(flux_faces, q_known)

    M = tpfa.assemble_flow_matrix(unit_cart_sd, data)
    rhs = tpfa.assemble_rhs_bdry_terms(unit_cart_sd, data)

    assert np.allclose(M @ p_known, rhs)


def test_robin_bcs(unit_cart_sd):
    tpfa, data, bcs, bdry_faces = setup(unit_cart_sd)

    bcs.set_robin_bcs(1, bdry_faces)

    rhs = tpfa.assemble_source(unit_cart_sd, lambda _: 1)
    M = tpfa.assemble_flow_matrix(unit_cart_sd, data)

    sol = sps.linalg.spsolve(M, rhs)

    assert np.all(sol > 0)
