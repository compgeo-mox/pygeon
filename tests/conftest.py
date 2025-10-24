import pytest
import pygeon as pg

param_list = [(dim, is_str) for dim in range(1, 4) for is_str in [True, False]]


@pytest.fixture(scope="session")
def unit_grids_dict():
    grids = {}
    for dim, is_str in param_list:
        sd = pg.unit_grid(dim, 1 / 3, structured=is_str, as_mdg=False)
        sd.compute_geometry()
        grids[dim, is_str] = sd

    return grids


@pytest.fixture(params=param_list)
def unit_sd(unit_grids_dict, request):
    return unit_grids_dict[request.param]


@pytest.fixture
def unit_sd_2d(unit_grids_dict):
    return unit_grids_dict[2, False]
