"""Module contains general tests for all discretizations."""

import numpy as np
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


def test_constant_interpolation_at_three_points():
    # On a triangle, RT0 evaluates the function at 3 face centers and Lagrange1 at 3
    # nodes, as many as the components of a vector: a constant vector is repeated at
    # each point, and the values of a scalar function at the 3 points are kept
    sd = pg.reference_element(2)
    sd.compute_geometry()
    rt0 = pg.RT0()
    vec = np.array([1.0, 2.0, 0.0])

    constant = rt0.interpolate(sd, lambda _: vec)
    vectorized = rt0.interpolate(
        sd, lambda x: np.tile(vec.reshape((-1, 1)), x.shape[1])
    )
    assert np.allclose(constant, vectorized)

    p1 = pg.Lagrange1()
    assert np.allclose(p1.interpolate(sd, lambda x: x[0]), sd.nodes[0])


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
