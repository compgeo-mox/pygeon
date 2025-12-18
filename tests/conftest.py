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

# Create a list with inputs for the grid generation. Grids are specified by dimension
# and whether the grid is structured.
param_list = [
    (dim, is_structured) for dim in range(1, 4) for is_structured in [True, False]
]
# Remove the first (1, True) entry because it's the same as (1, False)
param_list = param_list[1:]
ids = ["1D", "2D_struct", "2D_unstruct", "3D_struct", "3D_unstruct"]


@pytest.fixture(scope="session")
def _unit_grids_dict() -> dict[tuple[int, bool], pg.Grid]:
    """
    Helper fixture that generates a dictionary of grids, once per testing session.
    """
    grids = {}
    for dim, is_structured in param_list:
        sd = pg.unit_grid(dim, 1 / (5 - dim), structured=is_structured, as_mdg=False)
        sd.compute_geometry()
        grids[dim, is_structured] = sd

    return grids


@pytest.fixture(scope="session", params=param_list, ids=ids)
def unit_sd(_unit_grids_dict: dict, request: pytest.FixtureRequest) -> pg.Grid:
    """
    The five most common structured/unstructed grids of the cube in 1D-3D.
    """
    return _unit_grids_dict[request.param]


@pytest.fixture
def unit_sd_1d(_unit_grids_dict: dict) -> pg.Grid:
    """
    1D grid of the unit line segment with 4 elements
    """
    return _unit_grids_dict[1, False]


@pytest.fixture(scope="session")
def unit_sd_2d(_unit_grids_dict: dict) -> pg.Grid:
    """
    Unstructured triangle grid of the unit square
    """
    return _unit_grids_dict[2, False]


@pytest.fixture(scope="session")
def unit_sd_3d(_unit_grids_dict: dict) -> pg.Grid:
    """
    Unstructured tetrahedral grid of the unit cube
    """
    return _unit_grids_dict[3, False]


# ------------------------- Unit Cartesian grids -------------------------


@pytest.fixture(scope="session")
def _unit_cart_dict() -> dict[int, pg.Grid]:
    """
    Helper fixture that generates a dictionary of cartesian grids.
    """
    grids = {}
    for dim in [1, 2, 3]:
        sd = pp.CartGrid(
            np.array([5 - dim] * dim),
            np.array([1] * dim),
        )
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        grids[dim] = sd

    return grids


@pytest.fixture(params=[1, 2, 3], ids=["1D", "2D", "3D"])
def unit_cart_sd(_unit_cart_dict: dict, request: pytest.FixtureRequest) -> pg.Grid:
    """
    Representative Cartesian grids in 1D, 2D, and 3D.
    """
    return _unit_cart_dict[request.param]


# ------------------------- Unit Polygon grids -------------------------


@pytest.fixture(scope="session")
def octagon_sd_2d():
    sd = pg.OctagonGrid(
        np.array([3] * 2),
        np.array([1] * 2),
    )
    sd.compute_geometry()

    return sd


@pytest.fixture(scope="session")
def cart_sd_2d():
    sd = pp.CartGrid(
        np.array([3] * 2),
        np.array([1] * 2),
    )
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    return sd


@pytest.fixture(scope="session")
def _unit_poly_dict(cart_sd_2d, octagon_sd_2d) -> dict[str, pg.Grid]:
    """
    Helper fixture that generates a dictionary of polygonal grids in 2D.
    """
    grids = {}

    grids["Cartgrid"] = cart_sd_2d
    grids["Octgrid"] = octagon_sd_2d

    return grids


@pytest.fixture(params=["Cartgrid", "Octgrid"])
def unit_poly_sd(_unit_poly_dict: dict, request: pytest.FixtureRequest) -> pg.Grid:
    """
    Polygonal grids on the unit square.
    """
    return _unit_poly_dict[request.param]


# ------------------------- Reference elements -------------------------
ids = ["{:}D".format(dim) for dim in range(1, 4)]


@pytest.fixture(scope="session")
def _ref_elements_dict() -> dict[int, pg.Grid]:
    """
    Helper fixture that generates a dictionary of simplicial reference elements.
    """
    grids = {}
    for dim in range(1, 4):
        sd = pg.reference_element(dim)
        sd.compute_geometry()
        grids[dim] = sd

    return grids


@pytest.fixture(params=range(1, 4), ids=ids)
def ref_sd(_ref_elements_dict: dict, request: pytest.FixtureRequest) -> pg.Grid:
    """
    The simplicial reference elements in 1D, 2D, and 3D.
    """
    return _ref_elements_dict[request.param]


@pytest.fixture
def ref_sd_3d(_ref_elements_dict: dict) -> pg.Grid:
    """
    Unit polygonal grids in 2D.
    """
    return _ref_elements_dict[3]


@pytest.fixture
def ref_sd_0d() -> pg.Grid:
    """
    The point grid.
    """
    sd = pp.PointGrid(
        np.array([0, 0, 0]),
    )
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    return sd


# ------------------------- Polygonal elements -------------------------
@pytest.fixture(scope="session")
def pentagon_sd() -> pg.Grid:
    """
    The pentagon grid from the Hitchhiker's guide to VEM.
    """
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
    """
    The reference square.
    """
    sd = pp.CartGrid(
        np.array([1] * 2),
    )
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    return sd


@pytest.fixture(scope="session")
def ref_octagon() -> pg.Grid:
    """
    A grid of the unit square consisting of one regular octagon and four triangles in
    the corners
    """
    sd = pg.OctagonGrid(
        np.array([1] * 2),
    )
    sd.compute_geometry()

    return sd


# ------------------------- Mixed-dimensional grids -------------------------
mdg_names = ["fracs_2D", "embedded_frac_2D", "fracs_3D", "embedded_frac_3D"]


@pytest.fixture(scope="session")
def _mdg_dict() -> dict[str, pg.MixedDimensionalGrid]:
    """
    Helper fixture that generates a dictionary of mixed-dimensional grids, once per
    testing session.
    """
    mdg_dict = {}

    mesh_args = {"cell_size": 0.5, "cell_size_fracture": 0.5}

    # Square with two fractures
    mdg_2D, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "simplex", mesh_args, [0, 1]
    )
    pg.convert_from_pp(mdg_2D)
    mdg_2D.compute_geometry()

    mdg_dict["fracs_2D"] = mdg_2D

    # Square with one embedded fracture
    end_points = np.array([0.25, 0.75])
    mdg_frac_2D, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "simplex", mesh_args, [0], fracture_endpoints=[end_points]
    )
    pg.convert_from_pp(mdg_frac_2D)
    mdg_frac_2D.compute_geometry()

    mdg_dict["embedded_frac_2D"] = mdg_frac_2D

    # Cube with three fractures
    mdg_3D, _ = pp.mdg_library.cube_with_orthogonal_fractures(
        "simplex", mesh_args, [0, 1, 2]
    )
    pg.convert_from_pp(mdg_3D)
    mdg_3D.compute_geometry()

    mdg_dict["fracs_3D"] = mdg_3D

    # Cube with one embedded fracture
    fracture = pp.fracture_sets.orthogonal_fractures_3d(0.5)[2]
    fracture.pts += 1 / 4
    domain = pp.domains.nd_cube_domain(3, 1.0)
    fracture_network = pp.create_fracture_network([fracture], domain)
    mdg_frac_3D = pp.create_mdg("simplex", mesh_args, fracture_network)
    pg.convert_from_pp(mdg_frac_3D)
    mdg_frac_3D.compute_geometry()

    mdg_dict["embedded_frac_3D"] = mdg_frac_3D

    # Collect and return
    return mdg_dict


@pytest.fixture(scope="session", params=mdg_names)
def mdg(_mdg_dict: dict, request: pytest.FixtureRequest) -> pg.MixedDimensionalGrid:
    """
    Typical mixed-dimensional grids in 2D and 3D, with embedded and crossing fractures.
    """
    return _mdg_dict[request.param]


@pytest.fixture(scope="session")
def mdg_embedded_frac_2d(_mdg_dict):
    """
    The mixed-dimensional grid of the unit square with a single embedded fracture.
    """
    return _mdg_dict["embedded_frac_2D"]
