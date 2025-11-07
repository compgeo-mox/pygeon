import unittest

import numpy as np

import pygeon as pg


class RefinementTest(unittest.TestCase):
    def unit_cube_test_unstr(self, dim):
        sd_old = pg.unit_grid(dim, 1 / 2, as_mdg=False)
        sd_old.compute_geometry()

        sd = pg.barycentric_split(sd_old)
        sd.compute_geometry()

        assert sd.num_cells == (sd.dim + 1) * sd_old.num_cells
        assert sd.num_nodes == sd_old.num_nodes + sd_old.num_cells

        assert np.isclose(sd.cell_volumes.sum(), 1)

        assert (sd.face_ridges @ sd.cell_faces).nnz == 0
        assert (sd.ridge_peaks @ sd.face_ridges).nnz == 0

    def test_unit_line(self):
        self.unit_cube_test_unstr(1)

    def test_unit_square(self):
        self.unit_cube_test_unstr(2)

    def test_unit_cube(self):
        self.unit_cube_test_unstr(3)


if __name__ == "__main__":
    unittest.main()
