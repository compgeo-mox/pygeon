import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.PwConstants("test")


def test_ndof(discr, unit_sd):
    assert discr.ndof(unit_sd) == unit_sd.num_cells


def test_assemble_mass_matrix(discr, ref_sd):
    M = discr.assemble_mass_matrix(ref_sd)
    L = discr.assemble_lumped_matrix(ref_sd)
    P = discr.eval_at_cell_centers(ref_sd)

    assert np.allclose(M.data * ref_sd.cell_volumes, 1)
    assert np.all(M.indptr[:-1] == M.indices)
    assert np.all(M.indices == np.arange(ref_sd.num_cells))
    assert np.allclose((M - L).data, 0)
    assert np.allclose((M - P).data, 0)


def test_interpolate(discr, unit_sd_2d):
    func = lambda x: np.sin(x[0])  # Example function
    vals = discr.interpolate(unit_sd_2d, func)
    vals_known = func(unit_sd_2d.cell_centers) * unit_sd_2d.cell_volumes

    assert np.allclose(vals, vals_known)


def test_source(discr, unit_sd):
    func = lambda _: 2
    source = discr.source_term(unit_sd, func)

    assert np.allclose(source, 2)


def test_proj_to_lower_PwPolynomials(discr, unit_sd_2d):
    with pytest.raises(NotImplementedError):
        discr.proj_to_lower_PwPolynomials(unit_sd_2d)


def test_error_l2(discr, unit_cart_sd):
    def fun(x):
        return x[0] + 2 * x[1]

    int_sol = discr.interpolate(unit_cart_sd, fun)

    err = discr.error_l2(unit_cart_sd, np.zeros_like(int_sol), fun)
    assert np.isclose(err, 1)

    err = discr.error_l2(unit_cart_sd, int_sol, fun, etype="standard")
    assert np.isclose(err, 0)

    err = discr.error_l2(unit_cart_sd, int_sol, fun, etype="specific")
    assert not np.isclose(err, 0)
