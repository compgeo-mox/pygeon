import numpy as np
import pytest
import scipy.sparse as sps

import pygeon as pg

""" 
Module containing the grids to be used in the different tests. All fixtures in this file
are available to the tests in this directory and its subdirectories.
"""


# ------------------------- Unit grids -------------------------

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
    return _unit_grids_dict[1, False]


@pytest.fixture
def unit_sd_2d(_unit_grids_dict: dict) -> pg.Grid:
    return _unit_grids_dict[2, False]


@pytest.fixture
def unit_sd_3d(_unit_grids_dict: dict) -> pg.Grid:
    return _unit_grids_dict[3, False]


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
def octagon_sd() -> pg.Grid:
    sd = pg.OctagonGrid([1] * 2)
    sd.compute_geometry()

    return sd
