"""Module contains tests to validate the Poincare operator."""

from pathlib import Path

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

import pygeon as pg


@pytest.fixture(scope="session")
def poin(unit_sd: pg.Grid) -> pg.Poincare:
    mdg = pg.as_mdg(unit_sd)
    return pg.Poincare(mdg)


def poin_from_geo(filename: str, dim: int) -> pg.Poincare:
    dirname = Path(__file__).parents[1]
    geo_file = dirname / "geo_files" / filename

    mdg = pp.fracs.fracture_importer.dfm_from_gmsh(geo_file, dim)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    return pg.Poincare(mdg)


@pytest.mark.parametrize("k", range(1, 4))
def test_chain_property_sd(poin, k):
    """
    Check the chain property, i.e. whether pp=0
    """
    if k > poin.dim:
        return

    f = np.random.rand(poin.bar_spaces[k].size)
    pf = poin.apply(k, f)
    ppf = poin.apply(k - 1, pf)

    assert np.allclose(ppf, 0)


@pytest.mark.parametrize("k", range(0, 4))
def test_decomposition(poin, k):
    """
    For given f, check whether the decomposition (pd + pd + q) f = f holds
    """
    if k > poin.dim:
        return

    f = np.random.rand(poin.bar_spaces[k].size)
    pdf, dpf, qf = poin.decompose(k, f)
    assert np.allclose(f, pdf + dpf + qf)


@pytest.mark.parametrize("k", range(0, 4))
def test_solve_subproblem(poin, k):
    if k > poin.dim:
        return

    ndof = poin.bar_spaces[k].size
    system = sps.eye(ndof)

    sol = poin.solve_subproblem(k, system, np.zeros(ndof))

    assert np.allclose(sol, 0)


def test_euler_char(poin):
    poin_euler = poin.compute_euler_char()
    mdg_euler = poin.mdg.compute_euler_char()

    assert poin_euler == mdg_euler


def test_missing_donut():
    poin = poin_from_geo("missing_donut.geo", 3)
    betti = [1, 1, 1, 0]

    for k in range(poin.dim + 1):
        assert poin.hom_basis[k].shape[1] == betti[k]
        test_decomposition(poin, k)
    test_euler_char(poin)


def test_two_holes_2D():
    poin = poin_from_geo("two_holes_2D.geo", 2)
    betti = [1, 2, 0]

    for k in range(poin.dim + 1):
        assert poin.hom_basis[k].shape[1] == betti[k]
        test_decomposition(poin, k)
    test_euler_char(poin)


def test_one_cell_1D():
    mdg = pg.unit_grid(1, 1, as_mdg=True)
    mdg.compute_geometry()

    with pytest.warns():
        pg.Poincare(mdg)


def test_harmonic_form_computation():
    poin = poin_from_geo("missing_donut.geo", 3)

    basis_0 = poin.compute_basis_harmonic_forms(0)
    assert np.allclose(basis_0, 1)

    basis_1 = poin.compute_basis_harmonic_forms(1)
    assert np.any(basis_1)
    assert np.allclose(pg.curl(poin.mdg) @ basis_1, 0)
    assert np.allclose(pg.grad(poin.mdg).T @ pg.ridge_mass(poin.mdg) @ basis_1, 0)


def test_linking_number_failure():
    poin = poin_from_geo("two_holes_2D.geo", 2)
    U = np.eye(3, 4)
    U = np.hstack((U, U[:, 0][:, None]))
    P = U + 1e-10

    with pytest.raises(RuntimeError):
        poin.compute_linking_number(U, P, 2)
