import pytest

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg

""" 
Module to test the mixed formulations of elasticity. Instead of solving the full saddle
point problems, we check if a known distribution is a solution.
"""


@pytest.fixture
def _spaces_dict():
    key = "test"

    space_dict = {}
    Sigma_h = pg.VecBDM1(key)
    U_h = pg.VecPwConstants(key)
    R_h = [pg.PwConstants(key), pg.VecPwConstants(key)]

    space_dict["BDM1-P0"] = Sigma_h, U_h, R_h

    Sigma_h = pg.VecBDM1(key)
    U_h = pg.VecPwConstants(key)
    R_h = [pg.Lagrange1(key), pg.VecLagrange1(key)]

    space_dict["BDM1-L1"] = Sigma_h, U_h, R_h

    Sigma_h = pg.VecRT1(key)
    U_h = pg.VecPwLinears(key)
    R_h = [pg.Lagrange1(key), pg.VecLagrange1(key)]

    space_dict["RT1-L1"] = Sigma_h, U_h, R_h

    return space_dict


@pytest.fixture(params=["BDM1-P0", "BDM1-L1", "RT1-L1"])
def spaces(_spaces_dict: dict, request: pytest.FixtureRequest):
    return _spaces_dict[request.param]


@pytest.fixture(params=[False, True], ids=["Full", "Lumped"])
def use_lumped(request: pytest.FixtureRequest):
    return request.param


def setup_elasticity_natural_bcs(Sigma_h, U_h, R_h, sd, u_boundary, use_lumped):
    data = {pp.PARAMETERS: {Sigma_h.keyword: {"mu": 0.5, "lambda": 0.5}}}

    if use_lumped:
        Ms = Sigma_h.assemble_lumped_matrix(sd, data)
    else:
        Ms = Sigma_h.assemble_mass_matrix(sd, data)
    Mu = U_h.assemble_mass_matrix(sd)

    Pi = pg.proj_to_PwPolynomials(R_h, sd, Sigma_h.poly_order)
    P = pg.get_PwPolynomials(Sigma_h.poly_order, R_h.tensor_order)(R_h.keyword)
    Mr = Pi.T @ P.assemble_mass_matrix(sd)

    div = Mu @ Sigma_h.assemble_diff_matrix(sd)
    asym = Mr @ Sigma_h.assemble_asym_matrix(sd)

    spp = sps.block_array(
        [
            [Ms, div.T, -asym.T],
            [-div, None, None],
            [asym, None, None],
        ],
        format="csc",
    )

    b_faces = sd.tags["domain_boundary_faces"]
    bc = Sigma_h.assemble_nat_bc(sd, u_boundary, b_faces)

    rhs = np.zeros(spp.shape[0])
    rhs[: Sigma_h.ndof(sd)] = bc

    return spp, rhs


def test_elasticity_rbm(unit_sd, spaces, use_lumped):
    if unit_sd.dim == 1:
        return

    S_h, U_h, R_h = spaces
    R_h = R_h[unit_sd.dim - 2]  # Select the scalar or vector variant

    u_rbm = lambda x: np.array([-0.5 - x[1], -0.5 + x[0] - x[2], -0.5 + x[1]])
    r_rbm = lambda _: np.array([-1, 0, -1]) if unit_sd.dim == 3 else -1
    s_rbm = lambda _: np.zeros((unit_sd.dim, 3))

    u_known = U_h.interpolate(unit_sd, u_rbm)
    r_known = R_h.interpolate(unit_sd, r_rbm)
    s_known = S_h.interpolate(unit_sd, s_rbm)

    x_known = np.hstack((s_known, u_known, r_known))

    spp, rhs = setup_elasticity_natural_bcs(S_h, U_h, R_h, unit_sd, u_rbm, use_lumped)

    assert np.allclose(spp @ x_known, rhs)


def test_elasticity_stretching(unit_sd, spaces, use_lumped):
    if unit_sd.dim == 1:
        return

    u_stretch = lambda x: np.array([x[0], 0, 0])
    r_stretch = lambda _: np.zeros(3) if unit_sd.dim == 3 else 0
    s_stretch = lambda _: np.array([[1.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])

    S_h, U_h, R_h = spaces
    R_h = R_h[unit_sd.dim - 2]  # Select the scalar or vector variant

    u_known = U_h.interpolate(unit_sd, u_stretch)
    r_known = R_h.interpolate(unit_sd, r_stretch)
    s_known = S_h.interpolate(unit_sd, s_stretch)

    x_known = np.hstack((s_known, u_known, r_known))

    spp, rhs = setup_elasticity_natural_bcs(
        S_h, U_h, R_h, unit_sd, u_stretch, use_lumped
    )

    assert np.allclose(spp @ x_known, rhs)
