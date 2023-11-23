import unittest

import numpy as np
import porepy as pp

import pygeon as pg
import porepy as pp


class LevelsetGridTest(unittest.TestCase):
    def line_at_y75(self, x):
        return x[1] - 0.80

    def line_at_x60(self, x):
        return x[0] - 0.55

    def circle_at_0505(self, x):
        return 0.41 - np.linalg.norm(x - np.array([0.5, 0.5, 0]))

    def circle_at_0808(self, x):
        return 0.51 - np.linalg.norm(x - np.array([0.8, 0.8, 0]))

    def test_structtrianglegrid(self):
        sd = pp.StructuredTriangleGrid([2] * 2, [1] * 2)

        sd = pg.levelset_remesh(sd, self.line_at_y75)
        self.assertEqual(sd.num_cells, 12)

        sd = pg.levelset_remesh(sd, self.line_at_x60)
        self.assertEqual(sd.num_cells, 17)

        sd = pg.levelset_remesh(sd, self.circle_at_0505)
        self.assertEqual(sd.num_cells, 29)

        sd.compute_geometry()
        self.assertAlmostEqual(sd.cell_volumes.sum(), 1)

    def test_cartgrid(self):
        sd = pp.CartGrid([2] * 2, [1] * 2)

        sd = pg.levelset_remesh(sd, self.line_at_y75)
        self.assertEqual(sd.num_cells, 6)

        sd = pg.levelset_remesh(sd, self.line_at_x60)
        self.assertEqual(sd.num_cells, 9)

        sd = pg.levelset_remesh(sd, self.circle_at_0505)
        self.assertEqual(sd.num_cells, 17)

        sd.compute_geometry()
        self.assertAlmostEqual(sd.cell_volumes.sum(), 1)

    def test_trianglegrid(self):
        mdg = pg.unit_grid(2, 0.25)
        sd = mdg.subdomains()[0]

        sd = pg.levelset_remesh(sd, self.line_at_y75)
        self.assertEqual(sd.num_cells, 53)

        sd = pg.levelset_remesh(sd, self.line_at_x60)
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
    unittest.main()
