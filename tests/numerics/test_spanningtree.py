"""Module contains Spanning Tree tests."""

import numpy as np
import pytest
import scipy.sparse as sps

import pygeon as pg

spt_names = ["first_bdry", "all_bdry", "bottom", "weighted"]


@pytest.fixture(scope="session", params=spt_names)
def sd_and_sptr(unit_sd, request):
    mdg = pg.as_mdg(unit_sd)
    return create_pair(mdg, pg.SpanningTree, request.param)


@pytest.fixture(scope="session", params=spt_names)
def mdg_and_sptr(mdg, request):
    return create_pair(mdg, pg.SpanningTree, request.param)


def create_pair(mdg, tree_type, key):
    match key:
        case "first_bdry":
            return (mdg, tree_type(mdg, "first_bdry"))
        case "all_bdry":
            return (mdg, tree_type(mdg, "all_bdry"))
        case "bottom":
            sd = mdg.subdomains(dim=mdg.dim_max())[0]
            bottom = np.isclose(
                sd.face_centers[sd.dim - 1, :], sd.face_centers[sd.dim - 1, :].min()
            )
            return (mdg, tree_type(mdg, bottom))
        case "weighted":
            return (
                mdg,
                pg.SpanningWeightedTrees(mdg, tree_type, [0.25, 0.5, 0.25]),
            )


def test_spt_flow_sd(sd_and_sptr):
    check_flow(*sd_and_sptr)


def test_spt_flow_mdg(mdg_and_sptr):
    check_flow(*mdg_and_sptr)


def check_flow(mdg, sptr):
    """
    Check whether the constructed flux balances the given mass-source
    """

    source_known = np.random.rand(mdg.num_subdomain_cells())
    p_known = np.random.rand(source_known.size)
    div = pg.cell_mass(mdg) @ pg.div(mdg)

    q_f = sptr.solve(source_known)
    p_sptr = sptr.solve_transpose(div.T @ p_known)

    assert np.allclose(div @ q_f, source_known)
    assert np.allclose(p_known, p_sptr)


def test_assemble_SI(sd_and_sptr):
    mdg, sptr = sd_and_sptr
    SI = sptr.assemble_SI()
    div = pg.cell_mass(mdg) @ pg.div(mdg)

    check = sps.eye_array(div.shape[0]) - div @ SI
    assert np.allclose(check.data, 0)


def test_for_errors_in_string(unit_sd_2d):
    mdg = pg.as_mdg(unit_sd_2d)
    with pytest.raises(KeyError):
        pg.SpanningTree(mdg, "error_str")


# ---------------------------------- Elasticity ----------------------------------


@pytest.fixture(scope="session", params=spt_names)
def sd_and_sptr_elas(unit_sd, request):
    mdg = pg.as_mdg(unit_sd)

    if mdg.dim_max() == 1:
        return mdg, None

    return create_pair(mdg, pg.SpanningTreeElasticity, request.param)


def assemble_B(mdg):
    sd = mdg.subdomains(dim=mdg.dim_max())[0]

    vec_bdm1 = pg.VecBDM1()
    vec_p0 = pg.VecPwConstants()

    M_div = vec_p0.assemble_mass_matrix(sd)
    if sd.dim == 2:
        p0 = pg.PwConstants()
        M_asym = p0.assemble_mass_matrix(sd)
    else:
        M_asym = M_div

    div = M_div @ vec_bdm1.assemble_diff_matrix(sd)
    asym = M_asym @ vec_bdm1.assemble_asym_matrix(sd, True)

    return sps.vstack((-div, -asym))


def test_elasticity_spanningtree_solve(sd_and_sptr_elas):
    mdg, sptr = sd_and_sptr_elas

    if mdg.dim_max() == 1:
        return

    B = assemble_B(mdg)

    f = np.random.rand(B.shape[0])
    ur = np.random.rand(B.shape[0])

    s_f = sptr.solve(f)
    ur_sptr = sptr.solve_transpose(B.T @ ur)

    assert np.allclose(B @ s_f, f)
    assert np.allclose(ur, ur_sptr)


def test_for_error_in_1d(unit_sd_1d):
    mdg = pg.as_mdg(unit_sd_1d)

    with pytest.raises(NotImplementedError):
        pg.SpanningTreeElasticity(mdg)


# ---------------------------------- Cosserat ----------------------------------


def test_cosserat(unit_sd_3d):
    mdg = pg.as_mdg(unit_sd_3d)

    B = assemble_B_cosserat(mdg)
    f = np.random.rand(B.shape[0])

    sptr = pg.SpanningTreeCosserat(mdg)

    s_f = sptr.solve(f)
    assert np.allclose(B @ s_f, f)


def assemble_B_cosserat(mdg):
    sd = mdg.subdomains(dim=mdg.dim_max())[0]

    key = "tree"
    vec_rt0 = pg.VecRT0(key)
    vec_p0 = pg.VecPwConstants(key)

    M = vec_p0.assemble_mass_matrix(sd)

    div = M @ vec_rt0.assemble_diff_matrix(sd)
    asym = M @ vec_rt0.assemble_asym_matrix(sd, True)

    return sps.block_array([[-div, None], [-asym, -div]]).tocsc()
