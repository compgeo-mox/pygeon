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


@pytest.fixture(scope="session")
def poin_donut() -> pg.Poincare:
    dirname = Path(__file__).parents[1]
    geo_file = dirname / "geo_files" / "missing_donut.geo"

    mdg = pp.fracs.fracture_importer.dfm_from_gmsh(geo_file, 3)
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


def test_missing_donut(poin_donut: pg.Poincare):
    poin = poin_donut
    betti = [1, 1, 1, 0]

    for k in range(4):
        assert poin.hom_basis[k].shape[1] == betti[k]
