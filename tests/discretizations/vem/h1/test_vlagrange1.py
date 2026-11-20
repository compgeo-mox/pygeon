"""Module contains specific tests for the virtual Lagrangean L1 discretization."""

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps
from porepy.grids.grid_extrusion import extrude_grid

import pygeon as pg
from tests.helpers import matrix_equals


@pytest.fixture
def discr():
    return pg.VLagrange1("test")


def test_ndof(discr, pentagon_sd):
    assert discr.ndof(pentagon_sd) == 5


def test_ndof_octagon(discr, ref_octagon):
    assert discr.ndof(ref_octagon) == 12


def test_on_pentagon(discr, pentagon_sd):
    # Test the three matrices from Hitchhikers sec 4.2

    B = discr.assemble_proj_rhs(pentagon_sd)
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

    D = discr.assemble_dofs_of_monomials(pentagon_sd)
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

    G = discr.assemble_proj_lhs(pentagon_sd)
    G_known = (
        np.array([[1050.0, 30.0, 40.0], [0.0, 441.0, 0.0], [0.0, 0.0, 441.0]]) / 1050
    )

    assert matrix_equals(B.todense(), B_known)
    assert matrix_equals(D.todense(), D_known)
    assert matrix_equals(G.todense(), G_known)


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
    vals = discr.assemble_nat_bc(pentagon_sd, lambda _: 1, b_nodes)

    vals_known = np.array([3.5, 2.5, 2.25, 2.0, 2.75])

    assert np.allclose(vals, vals_known)


def test_range_discr(discr: pg.VLagrange1):
    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(0)

    assert discr.get_range_discr_class(1) is pg.PwMConstants
    assert discr.get_range_discr_class(2) is pg.VRT0

    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(3)


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


def test_lumped_matrix(discr, unit_poly_sd):
    L = discr.assemble_lumped_matrix(unit_poly_sd)
    one = np.ones(discr.ndof(unit_poly_sd))

    area = unit_poly_sd.cell_volumes.sum()
    off_diag = L - sps.diags_array(L.diagonal())
    assert np.allclose(off_diag.data, 0), "The lumped matrix should be diagonal"
    assert np.isclose(one @ L @ one, area), f"1^T L 1 = {one @ L @ one} != {area}"


def test_lumped_matrix_on_simplices(discr, unit_sd):
    L = discr.assemble_lumped_matrix(unit_sd)
    L_known = pg.Lagrange1().assemble_lumped_matrix(unit_sd)

    diff = abs(L - L_known).max()
    assert np.isclose(diff, 0), f"Lumped matrices differ from Lagrange1 by {diff}"


def test_adv_matrix(discr, unit_poly_sd):
    # With a constant field beta, 1^T A x_i is the integral of beta_i
    sd = unit_poly_sd
    beta = np.array([1.0, 2.0, 3.0])
    field = np.tile(beta.reshape((-1, 1)), sd.num_cells)
    data = pp.initialize_data({}, "test", {pg.VECTOR_FIELD: field})

    A = discr.assemble_adv_matrix(sd, data)
    one = np.ones(discr.ndof(sd))
    computed = [one @ A @ discr.interpolate(sd, lambda x: x[i]) for i in range(2)]
    known = beta[:2] * sd.cell_volumes.sum()

    assert np.allclose(computed, known), f"1^T A x_i = {computed} != {known}"


def test_scaling_with_data(discr, unit_poly_sd):
    # The consistency and the stabilization scale together with the data
    sd = unit_poly_sd
    w = 3.7
    weight = pp.initialize_data({}, "test", {pg.WEIGHT: np.full(sd.num_cells, w)})
    tensor = pp.SecondOrderTensor(np.full(sd.num_cells, w))
    perm = pp.initialize_data({}, "test", {pg.SECOND_ORDER_TENSOR: tensor})

    M = discr.assemble_mass_matrix(sd)
    A = discr.assemble_stiff_matrix(sd)
    scaled = {
        "M_w - w M": discr.assemble_mass_matrix(sd, weight) - w * M,
        "L_w - w L": discr.assemble_lumped_matrix(sd, weight)
        - w * discr.assemble_lumped_matrix(sd),
        "A_w - w A": discr.assemble_stiff_matrix(sd, weight) - w * A,
        "A_K - w A": discr.assemble_stiff_matrix(sd, perm) - w * A,
    }

    for name, diff in scaled.items():
        assert np.allclose(diff.data, 0), f"{name} = {abs(diff).max()}, not 0"


def test_mass_matrix(discr, unit_cart_sd):
    func = lambda x: x[0]
    interp = discr.interpolate(unit_cart_sd, func)

    known = 1 / 3
    M = discr.assemble_mass_matrix(unit_cart_sd)
    integral = interp @ M @ interp

    assert np.isclose(integral, known)


def test_proj_on_tilted_prism(discr, pentagon_sd):
    # A prism on the pentagon with a tilted top face: on the pentagonal faces, the mean
    # of the nodal values of a linear function is not its mean on the face
    sd, _, _ = extrude_grid(pentagon_sd, np.array([0.0, 1.0]))
    top = np.isclose(sd.nodes[2], 1)
    sd.nodes[2, top] += 0.3 * sd.nodes[0, top]
    sd.history = []
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    # The projection of a linear function is exact
    grad = np.array([1.0, 2.0, 3.0])
    interp = discr.interpolate(sd, lambda x: grad @ x)
    proj = discr.proj_to_PwPolynomials(sd) @ interp
    known = np.hstack((grad @ sd.cell_centers, sd.cell_diameters() * grad))
    assert np.allclose(proj, known), f"Projection error {abs(proj - known).max()}"
