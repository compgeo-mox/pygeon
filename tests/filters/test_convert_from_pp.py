"""Module contains specific tests for grid conversion from porepy to pygeon."""

import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture()
def pp_mdg():
    mesh_args = {"cell_size": 0.25, "cell_size_fracture": 0.125}
    x_endpoints = [np.array([0, 0.5])]
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "simplex", mesh_args, [1], x_endpoints
    )

    return mdg


def test_convert_grid():
    sd = pp.StructuredTriangleGrid([2] * 2, [1] * 2)

    mdg = pg.as_mdg(sd)
    assert isinstance(mdg, pp.MixedDimensionalGrid)

    pg.convert_from_pp(sd)
    assert isinstance(sd, pg.Grid)


def test_convert_mortar_grid(pp_mdg):
    mg = pp_mdg.interfaces()[0]
    pg.convert_from_pp(mg)

    # The object should be converted to pg.MortarGrid
    assert isinstance(mg, pg.MortarGrid)


def test_convert_mixed_dimensional_grid(pp_mdg):
    pg.convert_from_pp(pp_mdg)

    # The subdomains and interfaces should be recursively converted
    assert isinstance(pp_mdg.subdomains(dim=2)[0], pg.Grid)
    assert isinstance(pp_mdg.subdomains(dim=1)[0], pg.Grid)
    assert isinstance(pp_mdg.interfaces()[0], pg.MortarGrid)

    # The object should be converted to pg.MixedDimensionalGrid
    assert isinstance(pp_mdg, pg.MixedDimensionalGrid)


def test_wrong_type():
    with pytest.raises(ValueError):
        pg.as_mdg(None)

    with pytest.raises(TypeError):
        pg.convert_from_pp(None)
