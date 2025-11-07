import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

import pygeon as pg
import pygeon.grids.levelset_remesh as remesh


def line_at_y80(x):
    return x[1] - 0.80


def line_at_x55(x):
    return x[0] - 0.55


def circle_at_0505(x):
    return 0.41 - np.linalg.norm(x - np.array([0.5, 0.5, 0]))


def circle_at_0808(x):
    return 0.51 - np.linalg.norm(x - np.array([0.8, 0.8, 0]))


def test_merge_connectivities():
    indices = np.arange(1, 5)[::-1]
    old_con = sps.csc_array((np.ones(4), indices, [0, 1, 4]))
    new_con = sps.hstack((sps.csc_array(old_con.shape), old_con))
    con = remesh.merge_connectivities(old_con, new_con)

    assert np.all(con.indices == np.tile(indices, 2))


def test_create_node_loop():
    nodes = np.array([3, 6, 4, 7, 3, 4, 6, 7])
    faces = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    orients = np.array([1, -1, -1, 1, -1, 1, 1, -1])

    loop = remesh.create_oriented_node_loop(nodes, faces, orients)

    assert np.all(loop == np.array([3, 4, 7, 6]))


def test_levelset_through_node():
    sd = pp.StructuredTriangleGrid([5] * 2, [1] * 2)
    with pytest.raises(NotImplementedError):
        pg.levelset_remesh(sd, line_at_y80)


def test_structtrianglegrid():
    sd = pp.StructuredTriangleGrid([2] * 2, [1] * 2)
    pg.convert_from_pp(sd)

    sd = pg.levelset_remesh(sd, line_at_y80)
    assert sd.num_cells == 12

    sd = pg.levelset_remesh(sd, line_at_x55)
    assert sd.num_cells == 17

    sd = pg.levelset_remesh(sd, circle_at_0505)
    assert sd.num_cells == 29

    sd.compute_geometry()
    assert np.isclose(sd.cell_volumes.sum(), 1)


def test_cartgrid():
    sd = pp.CartGrid([2] * 2, [1] * 2)
    pg.convert_from_pp(sd)

    sd = pg.levelset_remesh(sd, line_at_y80)
    assert sd.num_cells == 6
    assert sd.cell_faces.nnz == 24

    sd = pg.levelset_remesh(sd, line_at_x55)
    assert sd.num_cells == 9

    sd = pg.levelset_remesh(sd, circle_at_0505)
    assert sd.num_cells == 17

    sd.compute_geometry()
    assert np.isclose(sd.cell_volumes.sum(), 1)


def test_trianglegrid():
    sd = pg.unit_grid(2, 0.25, as_mdg=False)

    sd = pg.levelset_remesh(sd, line_at_y80)
    assert sd.num_cells == 53

    sd = pg.levelset_remesh(sd, line_at_x55)
    assert sd.num_cells == 62

    sd = pg.levelset_remesh(sd, circle_at_0505)
    assert sd.num_cells == 90

    sd.compute_geometry()
    assert np.isclose(sd.cell_volumes.sum(), 1)


def test_oct_grid():
    sd = pg.OctagonGrid([6, 6])

    sd = pg.levelset_remesh(sd, circle_at_0505)
    assert sd.num_cells == 113

    sd = pg.levelset_remesh(sd, circle_at_0808)
    assert sd.num_cells == 130

    sd.compute_geometry()
    assert np.isclose(sd.cell_volumes.sum(), 1)
