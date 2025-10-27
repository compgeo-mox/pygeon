import pytest

import numpy as np
import porepy as pp

import pygeon as pg


def test_convert_grid():
    dim = 2
    sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)

    mdg = pg.as_mdg(sd)
    assert isinstance(mdg, pp.MixedDimensionalGrid)

    pg.convert_from_pp(sd)
    assert isinstance(sd, pg.Grid)


def test_convert_mortar_grid():
    mesh_args = {"cell_size": 0.25, "cell_size_fracture": 0.125}
    x_endpoints = [np.array([0, 0.5])]
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "simplex", mesh_args, [1], x_endpoints
    )

    mg = mdg.interfaces()[0]
    pg.convert_from_pp(mg)

    # The object should be converted to pg.MortarGrid
    assert isinstance(mg, pg.MortarGrid)


def test_convert_mixed_dimensional_grid():
    mesh_args = {"cell_size": 0.25, "cell_size_fracture": 0.125}
    x_endpoints = [np.array([0, 0.5])]
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "simplex", mesh_args, [1], x_endpoints
    )

    mdg = pg.as_mdg(mdg)
    assert isinstance(mdg, pp.MixedDimensionalGrid)

    pg.convert_from_pp(mdg)

    # The subdomains and interfaces should be recursively converted
    assert isinstance(mdg.subdomains(dim=2)[0], pg.Grid)
    assert isinstance(mdg.subdomains(dim=1)[0], pg.Grid)
    assert isinstance(mdg.interfaces()[0], pg.MortarGrid)

    # The object should be converted to pg.MixedDimensionalGrid
    assert isinstance(mdg, pg.MixedDimensionalGrid)
