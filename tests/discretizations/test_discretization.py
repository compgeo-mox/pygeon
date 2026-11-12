"""Module contains general tests for all discretizations."""

import numpy as np
import porepy as pp
import pytest
import sympy as sp

import pygeon as pg


@pytest.fixture(
    params=[
        pg.Lagrange1,
        pg.Lagrange2,
        pg.RT0,
        pg.BDM1,
        pg.RT1,
        pg.NedelecR0,
        pg.NedelecF1,
    ]
)
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


def test_string_repr():
    discr = pg.PwConstants("test")
    repr = str(discr)
    known = "Discretization of type PwConstants with keyword test"

    assert repr == known


def test_cochain_property(discr, unit_sd):
    if isinstance(discr, (pg.NedelecR0, pg.NedelecF1)) and unit_sd.dim == 1:
        pytest.skip("No Nedelec spaces in 1D")

    unit_sd.compute_geometry()

    Diff = discr.assemble_diff_matrix(unit_sd)
    range_discr = discr.get_range_discr_class(unit_sd.dim)(discr.keyword)
    range_Diff = range_discr.assemble_diff_matrix(unit_sd)

    prod = range_Diff @ Diff
    assert np.allclose(prod.data, 0)


def test_eval_at_cc(discr, unit_sd):
    Pi_child = discr.eval_at_cell_centers(unit_sd)
    Pi_super = pg.Discretization.eval_at_cell_centers(discr, unit_sd)

    assert np.allclose((Pi_child - Pi_super).data, 0)


# Coverage test
def test_unvectorized_interpolation(unit_sd_2d):
    discr = pg.Lagrange1()
    func = lambda _: np.arange(15)

    with pytest.raises(RuntimeError):
        discr.interpolate(unit_sd_2d, func)


@pytest.mark.parametrize(
    "discr_class, dim, known",
    [
        (pg.Lagrange1, 1, [1, 0, 0, 0]),
        (pg.Lagrange1, 3, [1, 0, 0, 0]),
        (pg.Lagrange2, 2, [1, 1, 0, 0]),
        (pg.NedelecR0, 3, [0, 1, 0, 0]),
        (pg.NedelecF1, 3, [0, 2, 0, 0]),
        (pg.RT0, 1, [1, 0, 0, 0]),
        (pg.RT0, 2, [0, 1, 0, 0]),
        (pg.RT0, 3, [0, 0, 1, 0]),
        (pg.BDM1, 2, [0, 2, 0, 0]),
        (pg.BDM1, 3, [0, 0, 3, 0]),
        (pg.RT1, 2, [0, 2, 2, 0]),
        (pg.RT1, 3, [0, 0, 3, 3]),
        (pg.PwConstants, 2, [0, 0, 1, 0]),
        (pg.PwConstants, 3, [0, 0, 0, 1]),
        (pg.PwLinears, 2, [0, 0, 3, 0]),
        (pg.PwQuadratics, 3, [0, 0, 0, 10]),
        (pg.VecLagrange1, 3, [3, 0, 0, 0]),
        (pg.VecRT0, 2, [0, 2, 0, 0]),
        (pg.SymMatPwConstants, 2, [0, 0, 3, 0]),
        (pg.TPFA, 2, [0, 0, 1, 0]),
        (pg.TPSA, 2, [0, 0, 4, 0]),
    ],
)
def test_ndof_per_entity(discr_class, dim, known):
    assert np.array_equal(discr_class("test").ndof_per_entity(dim), known)


@pytest.mark.parametrize(
    "discr_class",
    [
        pg.Lagrange1,
        pg.Lagrange2,
        pg.NedelecR0,
        pg.NedelecF1,
        pg.RT0,
        pg.BDM1,
        pg.RT1,
        pg.PwConstants,
        pg.PwLinears,
        pg.PwQuadratics,
        pg.VecLagrange1,
        pg.VecRT0,
        pg.SymMatPwLinears,
        pg.TPFA,
        pg.TPSA,
    ],
)
def test_ndof_per_element_on_reference_element(discr_class, ref_sd):
    discr = discr_class("test")
    dofs = discr.ndof_per_entity(ref_sd.dim)

    assert dofs.size == 4
    assert np.all(dofs >= 0)
    # a single element carries all the degrees of freedom of the grid
    assert discr.ndof_per_element(ref_sd.dim) == discr.ndof(ref_sd)


def test_ndof_per_entity_of_point_grid(ref_sd_0d):
    # a point grid consists of a single cell, on which the piecewise constants
    # place their only degree of freedom
    p0 = pg.PwConstants("test")
    assert np.array_equal(p0.ndof_per_entity(0), [1, 0, 0, 0])
    assert p0.ndof_per_element(0) == 1
    assert p0.ndof(ref_sd_0d) == 1

    # such a grid has neither nodes nor faces, so the other spaces have no dofs
    assert pg.Lagrange1("test").ndof(ref_sd_0d) == 0
    assert pg.RT0("test").ndof(ref_sd_0d) == 0


def test_ndof_per_entity_on_polygonal_cells(unit_poly_sd):
    # the dofs per entity do not depend on the grid, so they also describe the
    # elements of a virtual element space, whose cells are general polygons
    sd = unit_poly_sd
    num_entities = np.vstack(
        (
            np.diff(sd.cell_nodes().indptr),
            np.diff(sd.cell_faces.indptr),
            np.ones(sd.num_cells, dtype=int),
            np.zeros(sd.num_cells, dtype=int),
        )
    )

    for discr, known in [
        (pg.VLagrange1("test"), np.diff(sd.cell_nodes().indptr)),
        (pg.VRT0("test"), np.diff(sd.cell_faces.indptr)),
    ]:
        dofs = discr.ndof_per_entity(sd.dim) @ num_entities
        assert np.array_equal(dofs, known)


@pytest.mark.parametrize(
    "dim, known",
    [(1, [2, 1, 0, 0]), (2, [3, 3, 1, 0]), (3, [4, 6, 4, 1])],
)
def test_num_entities_of_reference_element(_ref_elements_dict, dim, known):
    assert np.array_equal(_ref_elements_dict[dim].num_entities(), known)


def test_num_entities(unit_sd):
    num_entities = unit_sd.num_entities()

    assert num_entities[0] == unit_sd.num_nodes
    assert num_entities[1] == unit_sd.num_edges
    assert num_entities[unit_sd.dim] == unit_sd.num_cells
    assert np.all(num_entities[unit_sd.dim + 1 :] == 0)


def test_num_entities_of_point_grid(ref_sd_0d):
    # a point grid has a single cell and no nodes
    assert np.array_equal(ref_sd_0d.num_entities(), [1, 0, 0, 0])


@pytest.mark.parametrize(
    "discr_class, known",
    [
        (pg.Lagrange1, pg.VecPwConstants),
        (pg.Lagrange2, pg.VecPwLinears),
        (pg.PwQuadratics, pg.VecPwLinears),
        (pg.RT1, pg.MatPwLinears),
        (pg.VecLagrange2, pg.MatPwLinears),
    ],
)
def test_get_broken_grad_discr(discr_class, known):
    assert discr_class("test").get_broken_grad_discr() is known


@pytest.mark.parametrize(
    "discr_class",
    [
        pg.Lagrange1,
        pg.Lagrange2,
        pg.PwLinears,
        pg.PwQuadratics,
        pg.VecPwLinears,
        pg.VecPwQuadratics,
        pg.RT0,
        pg.BDM1,
        pg.VecLagrange1,
        pg.VecLagrange2,
    ],
)
def test_broken_directional_derivative(discr_class, tilted_sd):
    # With u in the space and beta constant, both in the ambient coordinates,
    # beta . grad u on the grid is (P beta) . grad u, with P the projection on the
    # plane of the grid, for each component of u in the plane: a polynomial of one
    # order less, interpolated exactly by the piecewise polynomials of that order
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix
    discr = discr_class("test")
    k = discr.poly_order

    x, y, z = pg.exact.coordinates()
    if discr_class is pg.RT0:
        u = [0.3 + 1.5 * x, -0.7 + 1.5 * y, 1.1 + 1.5 * z]
    else:
        u = [(1 + x + 2 * y + z) ** k, (2 - x + y - z) ** k, (x - y + 2 * z) ** k]
    beta = np.array([1.0, 2.0, 3.0])
    f = [sum(b * g for b, g in zip(P @ beta, pg.exact.gradient(u_i))) for u_i in u]
    if discr.tensor_order == pg.SCALAR:
        u_ex, f_ex = u[0], f[0]
    else:
        u_ex, f_ex = sp.Matrix(u), sp.Matrix(f)

    low = pg.get_PwPolynomials(k - 1, discr.tensor_order)("test")
    u_h = discr.interpolate(sd, pg.exact.to_callable(u_ex))
    f_h = low.interpolate(sd, pg.exact.to_callable(f_ex))
    vector_field = np.tile(beta.reshape((-1, 1)), sd.num_cells)

    dir_der = discr.assemble_broken_directional_derivative(sd, vector_field)
    assert np.allclose(dir_der @ u_h, f_h)


@pytest.mark.parametrize(
    "discr_class",
    [
        pg.RT0,
        pg.BDM1,
        pg.VecLagrange1,
        pg.VecLagrange2,
        pg.VecPwConstants,
        pg.VecPwLinears,
        pg.VecPwQuadratics,
        pg.VecVLagrange1,
        pg.MatLagrange1,
        pg.MatPwConstants,
        pg.MatPwLinears,
        pg.MatPwQuadratics,
        pg.VecRT0,
        pg.VecBDM1,
    ],
)
def test_interpolate_and_eval_are_physical(discr_class, tilted_sd):
    # The functions are given in the ambient coordinates, and their evaluation at the
    # cell centers is in the ambient coordinates as well, projected on the plane of
    # the grid: P f for a vector and P F P for a matrix. The fields are linear, and of
    # the form a + b x for each vector of RT0, so that every space reproduces them
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix
    discr = discr_class("test")

    x = sp.Matrix(pg.exact.coordinates())
    a = sp.Matrix([0.3, -0.7, 1.1])
    if discr_class in (pg.RT0, pg.VecRT0):
        vector = a + 1.5 * x
        matrix = a * sp.ones(1, 3) + sp.Matrix([1.0, -0.5, 2.0]) * x.T
    else:
        A = sp.Matrix([[1.0, 2.0, -1.0], [0.5, -1.0, 2.0], [3.0, 1.0, 1.0]])
        vector = a + A * x
        matrix = A + (x[0] + 2 * x[1] + 3 * x[2]) * A.T

    if discr.tensor_order == pg.VECTOR:
        func = pg.exact.to_callable(vector)
        known = P @ func(sd.cell_centers)
    else:
        func = pg.exact.to_callable(matrix)
        known = np.einsum("ij,jkc,kl->ilc", P, func(sd.cell_centers), P)

    eval_h = discr.eval_at_cell_centers(sd) @ discr.interpolate(sd, func)
    assert np.allclose(eval_h, known.reshape((-1, sd.num_cells)).ravel())


@pytest.mark.parametrize(
    "discr_class, method",
    [
        (pg.RT0, "assemble_mass_matrix"),
        (pg.RT0, "assemble_lumped_matrix"),
        (pg.BDM1, "assemble_mass_matrix"),
        (pg.BDM1, "assemble_lumped_matrix"),
        (pg.VRT0, "assemble_mass_matrix"),
        (pg.VecRT0, "assemble_mass_matrix_elasticity"),
        (pg.VecRT0, "assemble_lumped_matrix_elasticity"),
        (pg.VecRT0, "assemble_mass_matrix_cosserat"),
        (pg.VecRT0, "assemble_lumped_matrix_cosserat"),
        (pg.VecBDM1, "assemble_mass_matrix_elasticity"),
        (pg.VecBDM1, "assemble_lumped_matrix_elasticity"),
    ],
)
def test_inner_products_are_invariant_under_rotation(
    discr_class, method, unit_sd, tilted_sd, tilt
):
    # The tilted grid is the unit grid rotated by Q: rotating the data with it, a field
    # f to Q f(Q^T x), a tensor field F to Q F(Q^T x) Q^T and a tensor K to Q K Q^T,
    # the inner products of the interpolants do not change, since all the data are
    # given in the ambient coordinates
    if "cosserat" in method and unit_sd.dim == 1:
        pytest.skip("No Cosserat in 1D")

    Q = sp.Matrix(tilt)
    discr = discr_class("test")

    x = sp.Matrix(pg.exact.coordinates())
    rotate_x = dict(zip(x, Q.T * x))
    A = sp.Matrix([[1.0, 2.0, -1.0], [0.5, -1.0, 2.0], [3.0, 1.0, 1.0]])
    a = sp.Matrix([0.3, -0.7, 1.1])

    if discr.tensor_order == pg.VECTOR:
        u = a + A * x
        u_rot = Q * u.subs(rotate_x, simultaneous=True)
        K = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, 0.3], [0.5, 0.3, 1.0]])
        K_rot = tilt @ K @ tilt.T
        params = [
            {pg.SECOND_ORDER_TENSOR: pg.constant_tensor(sd.num_cells, K_sd)}
            for sd, K_sd in ((unit_sd, K), (tilted_sd, K_rot))
        ]
    else:
        u = A + (x[0] + 2 * x[1] + 3 * x[2]) * A.T
        u_rot = Q * u.subs(rotate_x, simultaneous=True) * Q.T
        lame = {pg.LAME_MU: 0.7, pg.LAME_LAMBDA: 1.3, pg.LAME_MU_COSSERAT: 0.4}
        params = [lame, lame]

    energies = []
    for sd, u_ex, param in zip((unit_sd, tilted_sd), (u, u_rot), params):
        data = pp.initialize_data({}, "test", param)
        u_h = discr.interpolate(sd, pg.exact.to_callable(u_ex))
        energies.append(u_h @ getattr(discr, method)(sd, data) @ u_h)

    assert np.isclose(*energies)
