import pytest

import numpy as np
import porepy as pp

import pygeon as pg

""" 
Module contains tests to validate the inner products.
"""


def test_cell_mass_cart(unit_cart_sd):
    mdg = pg.as_mdg(unit_cart_sd)
    cell_mass = pg.cell_mass(mdg)

    assert np.allclose(cell_mass.data, cell_mass.data[0])


def test_cell_mass_simplices(unit_sd):
    mdg = pg.as_mdg(unit_sd)
    cell_mass = pg.cell_mass(mdg)
    interp_one = unit_sd.cell_volumes
    assert np.allclose(cell_mass @ interp_one, 1)


@pytest.mark.parametrize("n_minus_k", range(3))
def test_mass_and_stiffness(mdg, n_minus_k):
    if n_minus_k > mdg.dim_max():
        return

    mass = pg.numerics.innerproducts.mass_matrix(mdg, n_minus_k, None)
    stiff = pg.numerics.stiffness.stiff_matrix(mdg, n_minus_k, None)

    assert np.allclose((mass - mass.T).data, 0)
    assert np.allclose(mass.shape, stiff.shape)
