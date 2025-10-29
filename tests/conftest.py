import pytest
import pygeon as pg


# ------------------------- Unit grids -------------------------
param_list = [(dim, is_str) for dim in range(1, 4) for is_str in [True, False]]
param_list = param_list[1:]
ids = ["1D", "2D_struct", "2D_unstruct", "3D_struct", "3D_unstruct"]


@pytest.fixture(scope="session")
def _unit_grids_dict():
    grids = {}
    for dim, is_str in param_list:
        sd = pg.unit_grid(dim, 1 / (5 - dim), structured=is_str, as_mdg=False)
        sd.compute_geometry()
        grids[dim, is_str] = sd

    return grids


@pytest.fixture(params=param_list, ids=ids)
def unit_sd(_unit_grids_dict, request):
    return _unit_grids_dict[request.param]


@pytest.fixture
def unit_sd_2d(_unit_grids_dict):
    return _unit_grids_dict[2, False]


@pytest.fixture
def unit_sd_3d(_unit_grids_dict):
    return _unit_grids_dict[3, False]


# ------------------------- Reference elements -------------------------
ids = ["{:}D".format(dim) for dim in range(1, 4)]


@pytest.fixture(scope="session")
def _ref_elements_dict():
    grids = {}
    for dim in range(1, 4):
        sd = pg.reference_element(dim)
        sd.compute_geometry()
        grids[dim] = sd

    return grids


@pytest.fixture(params=range(1, 4), ids=ids)
def ref_sd(_ref_elements_dict, request):
    return _ref_elements_dict[request.param]


@pytest.fixture
def ref_sd_3d(_ref_elements_dict):
    return _ref_elements_dict[3]
