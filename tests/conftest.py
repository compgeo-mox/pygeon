"""
Module containing the grids to be used in the different tests. All fixtures in this file
are available to the tests in this directory and its subdirectories.
"""

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

import pygeon as pg


# ------------------------- Unit simplicial grids -------------------------

param_list = [(dim, is_str) for dim in range(1, 4) for is_str in [True, False]]
# Remove the (1, True) entry because it's the same as (1, False)
param_list = param_list[1:]
ids = ["1D", "2D_struct", "2D_unstruct", "3D_struct", "3D_unstruct"]


@pytest.fixture(scope="session")
def _unit_grids_dict() -> dict[tuple[int, bool], pg.Grid]:
    grids = {}
    for dim, is_str in param_list:
        sd = pg.unit_grid(dim, 1 / (5 - dim), structured=is_str, as_mdg=False)
        sd.compute_geometry()
        grids[dim, is_str] = sd

    return grids


@pytest.fixture(params=param_list, ids=ids)
def unit_sd(_unit_grids_dict: dict, request: pytest.FixtureRequest) -> pg.Grid:
    return _unit_grids_dict[request.param]


@pytest.fixture
def unit_sd_1d(_unit_grids_dict: dict) -> pg.Grid:
    """
    1D grid of the unit line segment with 4 elements
    """
    return _unit_grids_dict[1, False]


@pytest.fixture
def unit_sd_2d(_unit_grids_dict: dict) -> pg.Grid:
    """
    Unstructured triangle grid of the unit square
    """
    return _unit_grids_dict[2, False]


@pytest.fixture
def unit_sd_3d(_unit_grids_dict: dict) -> pg.Grid:
    """
    Unstructured tetrahedral grid of the unit cube
    """
    return _unit_grids_dict[3, False]


# ------------------------- Unit Cartesian grids -------------------------


@pytest.fixture(scope="session")
def _unit_cart_dict() -> dict[int, pg.Grid]:
    grids = {}
    for dim in [1, 2, 3]:
        sd = pp.CartGrid([5 - dim] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()
        grids[dim] = sd

    return grids


@pytest.fixture(params=[1, 2, 3], ids=["1D", "2D", "3D"])
def unit_cart_sd(_unit_cart_dict: dict, request: pytest.FixtureRequest) -> pg.Grid:
    return _unit_cart_dict[request.param]


# ------------------------- Unit Polygon grids -------------------------


@pytest.fixture(scope="session")
def _unit_poly_dict() -> dict[int, pg.Grid]:
    grids = {}

    sd = pp.CartGrid([3] * 2, [1] * 2)
    pg.convert_from_pp(sd)
    sd.compute_geometry()
    grids["Cartgrid"] = sd

    sd = pg.OctagonGrid([3] * 2, [1] * 2)
    pg.convert_from_pp(sd)
    sd.compute_geometry()
    grids["Octgrid"] = sd

    return grids


@pytest.fixture(params=["Cartgrid", "Octgrid"])
def unit_poly_sd(_unit_poly_dict: dict, request: pytest.FixtureRequest) -> pg.Grid:
    return _unit_poly_dict[request.param]


# ------------------------- Reference elements -------------------------
ids = ["{:}D".format(dim) for dim in range(1, 4)]


@pytest.fixture(scope="session")
def _ref_elements_dict() -> dict[int, pg.Grid]:
    grids = {}
    for dim in range(1, 4):
        sd = pg.reference_element(dim)
        sd.compute_geometry()
        grids[dim] = sd

    return grids


@pytest.fixture(params=range(1, 4), ids=ids)
def ref_sd(_ref_elements_dict: dict, request: pytest.FixtureRequest) -> pg.Grid:
    return _ref_elements_dict[request.param]


@pytest.fixture
def ref_sd_3d(_ref_elements_dict: dict) -> pg.Grid:
    return _ref_elements_dict[3]


@pytest.fixture
def ref_sd_0d() -> pg.Grid:
    sd = pp.PointGrid([0, 0, 0])
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    return sd


# ------------------------- Polygonal elements -------------------------
@pytest.fixture(scope="session")
def pentagon_sd() -> pg.Grid:
    nodes = np.array([[0, 3, 3, 3.0 / 2.0, 0], [0, 0, 2, 4, 4], np.zeros(5)])
    indptr = np.arange(0, 11, 2)
    indices = np.roll(np.repeat(np.arange(5), 2), -1)
    face_nodes = sps.csc_array((np.ones(10), indices, indptr))
    cell_faces = sps.csc_array(np.ones((5, 1)))

    sd = pg.Grid(2, nodes, face_nodes, cell_faces, "pentagon")
    sd.compute_geometry()

    return sd


@pytest.fixture(scope="session")
def ref_square() -> pg.Grid:
    sd = pp.CartGrid([1] * 2)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    return sd


@pytest.fixture(scope="session")
def ref_octagon() -> pg.Grid:
    sd = pg.OctagonGrid([1] * 2)
    sd.compute_geometry()

    return sd


# ------------------------- Mixed-dimensional grids -------------------------
mdg_names = ["mdg_2D", "embedded_frac_2d", "mdg_cube"]


@pytest.fixture(scope="session")
def _mdg_dict() -> dict[int, pg.MixedDimensionalGrid]:
    mdg_dict = {}

    mesh_args = {"cell_size": 0.5, "cell_size_fracture": 0.5}

    mdg_2, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "simplex", mesh_args, [0, 1]
    )
    pg.convert_from_pp(mdg_2)
    mdg_2.compute_geometry()
    mdg_dict["mdg_2D"] = mdg_2

    end_points = np.array([0.25, 0.75])
    mdg_frac, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "simplex", mesh_args, [0], fracture_endpoints=[end_points]
    )
    pg.convert_from_pp(mdg_frac)
    mdg_frac.compute_geometry()
    mdg_dict["embedded_frac_2d"] = mdg_frac

    mdg_3, _ = pp.mdg_library.cube_with_orthogonal_fractures(
        "simplex", mesh_args, [0, 1, 2]
    )
    pg.convert_from_pp(mdg_3)
    mdg_3.compute_geometry()
    mdg_dict["mdg_cube"] = mdg_3

    return mdg_dict


@pytest.fixture(params=mdg_names)
def mdg(_mdg_dict: dict, request: pytest.FixtureRequest) -> pg.MixedDimensionalGrid:
    return _mdg_dict[request.param]
