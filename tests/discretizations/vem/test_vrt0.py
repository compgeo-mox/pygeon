"""Module contains specific tests for the virtual RT0 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VRT0("test")


def test_ndof(discr, pentagon_sd):
    assert discr.ndof(pentagon_sd) == 5


def test_mass_matrix(discr, ref_square):
    M = discr.assemble_mass_matrix(ref_square)

    M_known = (
        np.array(
            [
                [3, -1, 0, 0],
                [-1, 3, 0, 0],
                [0, 0, 3, -1],
                [0, 0, -1, 3],
            ]
        )
        / 4
    )

    assert np.allclose(M.todense(), M_known)


def test_lumped_matrix(discr, ref_square):
    L = discr.assemble_lumped_matrix(ref_square)
    L_known = np.eye(4) / 2

    assert np.allclose(L.todense(), L_known)


def test_assemble_nat_bc(discr, pentagon_sd):
    fun = lambda x: x[0] + x[1]
    faces = pentagon_sd.tags["domain_boundary_faces"]

    vals = discr.assemble_nat_bc(pentagon_sd, fun, faces.nonzero()[0])
    vals_from_bool = discr.assemble_nat_bc(pentagon_sd, fun, faces)

    vals_known = np.array([1.5, 4.0, 5.25, 4.75, 2.0])

    assert np.allclose(vals, vals_known)
    assert np.allclose(vals_from_bool, vals_known)


def test_interp_and_eval(discr, ref_octagon):
    func = lambda x: x
    interp_func = discr.interpolate(ref_octagon, func)
    eval_interp = discr.eval_at_cell_centers(ref_octagon) @ interp_func

    known_vals = np.hstack([func(c) for c in ref_octagon.cell_centers.T])
    # TODO: Change the eval_at_cc of VRT0 from pp to pg ordering

    assert np.allclose(eval_interp, known_vals)


def test_range_disc(discr):
    assert discr.get_range_discr_class(2) is pg.PwConstants


def test_diff_matrix(discr, ref_square):
    D = discr.assemble_diff_matrix(ref_square)
    D_known = np.array([[-1, 1, -1, 1]])

    assert np.allclose(D.todense(), D_known)
