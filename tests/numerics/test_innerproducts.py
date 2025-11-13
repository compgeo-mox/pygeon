"""Module contains tests to validate the inner products."""

import numpy as np
import pytest

import pygeon as pg


def test_cell_mass_cart(unit_cart_sd):
    mdg = pg.as_mdg(unit_cart_sd)
    cell_mass = pg.cell_mass(mdg)

    assert np.allclose(cell_mass.data, cell_mass.data[0])


def test_cell_mass_simplices(unit_sd):
    mdg = pg.as_mdg(unit_sd)
    cell_mass = pg.cell_mass(mdg)
    interp_one = unit_sd.cell_volumes
    assert np.allclose(cell_mass @ interp_one, 1)


@pytest.mark.parametrize("n_minus_k", range(0, 4))
def test_matrix_aliases(mdg, n_minus_k):
    if n_minus_k > mdg.dim_max():
        return

    match n_minus_k:
        case 0:
            M = pg.cell_mass(mdg)
            L = pg.lumped_cell_mass(mdg)
            n = mdg.num_subdomain_cells()
        case 1:
            M = pg.face_mass(mdg)
            L = pg.lumped_face_mass(mdg)
            n = mdg.num_subdomain_faces()
        case 2:
            M = pg.ridge_mass(mdg)
            L = pg.lumped_ridge_mass(mdg)
            n = mdg.num_subdomain_ridges()
        case 3:
            M = pg.peak_mass(mdg)
            L = pg.lumped_peak_mass(mdg)
            n = mdg.num_subdomain_peaks()

    # Symmetry and shape checks
    assert np.allclose((M - M.T).data, 0)
    assert np.allclose((L - L.T).data, 0)

    assert M.shape == (n, n)
    assert M.shape == L.shape


def test_wrong_value(unit_sd_3d):
    with pytest.raises(ValueError):
        pg.numerics.innerproducts.default_discr(unit_sd_3d, -1)
