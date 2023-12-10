import unittest

import numpy as np
import scipy.sparse as sps

import pygeon as pg
import pygeon.grids.levelset_remesh as remesh
import porepy as pp


class LevelsetGridTest(unittest.TestCase):
    def line_at_y80(self, x):
        return x[1] - 0.80

    def line_at_x55(self, x):
        return x[0] - 0.55

    def circle_at_0505(self, x):
        return 0.41 - np.linalg.norm(x - np.array([0.5, 0.5, 0]))

    def circle_at_0808(self, x):
        return 0.51 - np.linalg.norm(x - np.array([0.8, 0.8, 0]))

    def test_merge_connectivities(self):
        indices = np.arange(1, 5)[::-1]
        old_con = sps.csc_matrix((np.ones(4), indices, [0, 1, 4]))
        new_con = sps.hstack((sps.csc_matrix(old_con.shape), old_con))
        con = remesh.merge_connectivities(old_con, new_con)

        self.assertTrue(np.all(con.indices == np.tile(indices, 2)))

    def test_create_node_loop(self):
        nodes = np.array([3, 6, 4, 7, 3, 4, 6, 7])
        faces = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        orients = np.array([1, -1, -1, 1, -1, 1, 1, -1])

        loop = remesh.create_oriented_node_loop(nodes, faces, orients)

        self.assertTrue(np.all(loop == np.array([3, 4, 7, 6])))

    def test_levelset_through_node(self):
        sd = pp.StructuredTriangleGrid([5] * 2, [1] * 2)
        self.assertRaises(NotImplementedError, pg.levelset_remesh, sd, self.line_at_y80)

    def test_structtrianglegrid(self):
        sd = pp.StructuredTriangleGrid([2] * 2, [1] * 2)

        sd = pg.levelset_remesh(sd, self.line_at_y80)
        self.assertEqual(sd.num_cells, 12)

        sd = pg.levelset_remesh(sd, self.line_at_x55)
        self.assertEqual(sd.num_cells, 17)

        sd = pg.levelset_remesh(sd, self.circle_at_0505)
        self.assertEqual(sd.num_cells, 29)

        sd.compute_geometry()
        self.assertAlmostEqual(sd.cell_volumes.sum(), 1)

    def test_cartgrid(self):
        sd = pp.CartGrid([2] * 2, [1] * 2)

        sd = pg.levelset_remesh(sd, self.line_at_y80)
        self.assertEqual(sd.num_cells, 6)

        sd = pg.levelset_remesh(sd, self.line_at_x55)
        self.assertEqual(sd.num_cells, 9)

        sd = pg.levelset_remesh(sd, self.circle_at_0505)
        self.assertEqual(sd.num_cells, 17)

        sd.compute_geometry()
        self.assertAlmostEqual(sd.cell_volumes.sum(), 1)

    def test_trianglegrid(self):
        mdg = pg.unit_grid(2, 0.25)
        sd = mdg.subdomains()[0]

        sd = pg.levelset_remesh(sd, self.line_at_y80)
        self.assertEqual(sd.num_cells, 53)

        sd = pg.levelset_remesh(sd, self.line_at_x55)
        self.assertEqual(sd.num_cells, 62)

        sd = pg.levelset_remesh(sd, self.circle_at_0505)
        self.assertEqual(sd.num_cells, 90)

        sd.compute_geometry()
        self.assertAlmostEqual(sd.cell_volumes.sum(), 1)

    def test_oct_grid(self):
        sd = pg.OctagonGrid([6, 6])

        sd = pg.levelset_remesh(sd, self.circle_at_0505)
        self.assertEqual(sd.num_cells, 113)

        sd = pg.levelset_remesh(sd, self.circle_at_0808)
        self.assertEqual(sd.num_cells, 130)

        sd.compute_geometry()
        self.assertAlmostEqual(sd.cell_volumes.sum(), 1)


if __name__ == "__main__":
    LevelsetGridTest().test_cartgrid()
