import numpy as np
import pytest

import pygeon as pg


@pytest.fixture(scope="session")
def poin(unit_sd):
    mdg = pg.as_mdg(unit_sd)
    return pg.Poincare(mdg)


@pytest.mark.parametrize("k", range(1, 4))
def test_chain_property_sd(poin, k):
    """
    Check the chain property, i.e. whether pp=0
    """
    if k > poin.mdg.dim_max():
        return

    f = np.random.rand(poin.bar_spaces[k].size)
    pf = poin.apply(k, f)
    ppf = poin.apply(k - 1, pf)

    assert np.allclose(ppf, 0)


@pytest.mark.parametrize("k", range(0, 4))
def test_decomposition(poin, k):
    """
    For given f, check whether the decomposition
    (pd + pd) f = f
    holds
    """

    if k > poin.mdg.dim_max():
        return

    f = np.random.rand(poin.bar_spaces[k].size)
    pdf, dpf = poin.decompose(k, f)
    assert np.allclose(f, pdf + dpf)
