"""Module contains specific tests for the virtual Lagrangean L1 discretization."""

import numpy as np
import pytest
import scipy.sparse as sps

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VLagrange1("test")


def test_ndof(discr, pentagon_sd):
    assert discr.ndof(pentagon_sd) == 5


def test_ndof_octagon(discr, ref_octagon):
    assert discr.ndof(ref_octagon) == 12


def test_on_pentagon(discr, pentagon_sd):
    # Test the three matrices from Hitchhikers sec 4.2

    diam = pentagon_sd.cell_diameters()[0]
    loc_nodes = np.arange(5)

    B = discr.assemble_loc_L2proj_rhs(pentagon_sd, 0, diam, loc_nodes)
    B_known = (
        np.array(
            [
                [4.0, 4.0, 4.0, 4.0, 4.0],
                [-8.0, 4.0, 8.0, 4.0, -8.0],
                [-6.0, -6.0, 3.0, 6.0, 3.0],
            ]
        )
        / 20
    )

    D = discr.assemble_loc_dofs_of_monomials(pentagon_sd, 0, diam, loc_nodes)
    D_known = (
        np.array(
            [
                [1470.0, -399.0, -532.0],
                [1470.0, 483.0, -532.0],
                [1470.0, 483.0, 56.0],
                [1470.0, 42.0, 644.0],
                [1470.0, -399.0, 644.0],
            ]
        )
        / 1470
    )

    G = discr.assemble_loc_L2proj_lhs(pentagon_sd, 0, diam, loc_nodes)
    G_known = (
        np.array([[1050.0, 30.0, 40.0], [0.0, 441.0, 0.0], [0.0, 0.0, 441.0]]) / 1050
    )

    assert np.allclose(B, B_known)
    assert np.allclose(D, D_known)
    assert np.allclose(G, G_known)


def test_diff_matrix(discr, pentagon_sd):
    D = discr.assemble_diff_matrix(pentagon_sd)

    D_known_data = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    D_known_indices = np.array([0, 4, 0, 1, 1, 2, 2, 3, 3, 4])
    D_known_indptr = np.array([0, 2, 4, 6, 8, 10])
    D_known = sps.csc_array((D_known_data, D_known_indices, D_known_indptr))

    assert np.allclose((D - D_known).data, 0)


def test_eval_at_cc(discr, pentagon_sd):
    P = discr.eval_at_cell_centers(pentagon_sd)
    P_known = np.full(5, 0.2)
    assert np.allclose(P.todense(), P_known)


def test_interpolate(discr, pentagon_sd):
    fun = lambda x: x[0] + x[1]
    vals = discr.interpolate(pentagon_sd, fun)

    vals_known = np.array([0.0, 3.0, 5.0, 5.5, 4.0])

    assert np.allclose(vals, vals_known)


def test_assemble_nat_bc(discr, pentagon_sd):
    b_nodes = pentagon_sd.tags["domain_boundary_nodes"]
    vals = discr.assemble_nat_bc(pentagon_sd, lambda _: np.ones(1), b_nodes)

    vals_known = np.array([3.5, 2.5, 2.25, 2.0, 2.75])

    assert np.allclose(vals, vals_known)


def test_range_disc(discr):
    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(2)


def test_mass_oct_grid(discr, ref_octagon):
    # Compute \int (x + y)^2
    M = discr.assemble_mass_matrix(ref_octagon)
    fun = lambda x: x[0] + x[1]

    interp = discr.interpolate(ref_octagon, fun)

    assert np.isclose(interp @ M @ interp, 7 / 6)


def test_stiff_oct_grid(discr, ref_octagon):
    A = discr.assemble_stiff_matrix(ref_octagon)
    fun = lambda x: x[0] + x[1]

    interp = discr.interpolate(ref_octagon, fun)

    assert np.isclose(interp @ A @ interp, 2)


def test_eval_and_interp(discr, ref_octagon):
    # In the special case of the regular octagon grid, we can interpolate and evaluate
    # linears

    P = discr.eval_at_cell_centers(ref_octagon)
    fun = lambda x: x[0] + x[1]

    interp = discr.interpolate(ref_octagon, fun)
    known_vals = np.array([fun(c) for c in ref_octagon.cell_centers.T])
    assert np.allclose(P @ interp, known_vals)
