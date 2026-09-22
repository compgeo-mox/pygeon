"""Module contains general tests for all discretizations."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture(
    params=[
        pg.Lagrange1,
        pg.Lagrange2,
        pg.RT0,
        pg.BDM1,
        pg.RT1,
        pg.Nedelec0,
        pg.Nedelec1,
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
    # Skip the Nedelec spaces in 1D
    if isinstance(discr, (pg.Nedelec0, pg.Nedelec1)) and unit_sd.dim == 1:
        return

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


def test_proj_to_pw_polynomials_methods_are_cached(discr):
    assert hasattr(discr.proj_to_PwPolynomials, "cache_info"), (
        f"{discr}.proj_to_PwPolynomials should be cached"
    )


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
        (pg.Nedelec0, 3, [0, 1, 0, 0]),
        (pg.Nedelec1, 3, [0, 2, 0, 0]),
        (pg.RT0, 1, [1, 0, 0, 0]),
        (pg.RT0, 2, [0, 1, 0, 0]),
        (pg.RT0, 3, [0, 0, 1, 0]),
        (pg.BDM1, 2, [0, 2, 0, 0]),
        (pg.BDM1, 3, [0, 0, 3, 0]),
        (pg.RT1, 2, [0, 2, 2, 0]),
        (pg.RT1, 3, [0, 0, 3, 3]),
        (pg.PwConstants, 2, [0, 0, 1, 0]),
        (pg.PwConstants, 3, [0, 0, 0, 1]),
        (pg.PwLinears, 2, [1, 0, 0, 0]),
        (pg.PwQuadratics, 3, [1, 1, 0, 0]),
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
        pg.Nedelec0,
        pg.Nedelec1,
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
    # the single cell of a point grid is not a node
    assert np.array_equal(ref_sd_0d.num_entities(), [0, 0, 0, 0])
