import pytest

import numpy as np
import scipy.sparse as sps

import pygeon as pg


@pytest.fixture
def discr():
    return pg.PwConstants("test")


def test_ndof(discr, unit_sd):
    assert discr.ndof(unit_sd) == unit_sd.num_cells


def test_assemble_mass_matrix(discr, unit_sd):
    M = discr.assemble_mass_matrix(unit_sd)
    L = discr.assemble_lumped_matrix(unit_sd)
    P = discr.eval_at_cell_centers(unit_sd)

    assert np.allclose(M.data * unit_sd.cell_volumes, 1)
    assert np.all(M.indptr[:-1] == M.indices)
    assert np.all(M.indices == np.arange(unit_sd.num_cells))
    assert np.allclose((M - L).data, 0)
    assert np.allclose((M - P).data, 0)


def test_assemble_diff_matrix(discr, unit_sd_2d):
    D = discr.assemble_diff_matrix(unit_sd_2d)
    D_known = sps.csc_array((0, discr.ndof(unit_sd_2d)))

    assert np.allclose((D - D_known).data, 0)


def test_assemble_stiff_matrix(discr, unit_sd_2d):
    D = discr.assemble_stiff_matrix(unit_sd_2d)
    D_known = sps.csc_array((discr.ndof(unit_sd_2d), discr.ndof(unit_sd_2d)))

    assert np.allclose((D - D_known).data, 0)


def test_interpolate(discr, unit_sd_2d):
    func = lambda x: np.sin(x[0])  # Example function
    vals = discr.interpolate(unit_sd_2d, func)
    vals_known = func(unit_sd_2d.cell_centers) * unit_sd_2d.cell_volumes

    assert np.allclose(vals, vals_known)


def test_assemble_nat_bc(discr, unit_sd_2d):
    b_faces = np.array([0, 1, 3])  # Example boundary faces
    func = lambda x: np.sin(x[0])  # Example function

    vals = discr.assemble_nat_bc(unit_sd_2d, func, b_faces)
    vals_known = np.zeros(unit_sd_2d.num_cells)

    assert np.allclose(vals, vals_known)


def test_proj_to_higherPwPolynomials(discr, unit_sd):
    proj_p0 = discr.proj_to_higher_PwPolynomials(unit_sd)
    mass_p0 = discr.assemble_mass_matrix(unit_sd)

    p1 = pg.PwLinears()
    mass_p1 = p1.assemble_mass_matrix(unit_sd)

    diff = proj_p0.T @ mass_p1 @ proj_p0 - mass_p0

    assert np.allclose(diff.data, 0.0)


def test_source(discr, unit_sd):
    func = lambda _: 2
    source = discr.source_term(unit_sd, func)

    assert np.allclose(source, 2)


if __name__ == "__main__":
    pytest.main([__file__])
