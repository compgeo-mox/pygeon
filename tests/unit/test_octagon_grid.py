import unittest
import numpy as np

import porepy as pp
import pygeon as pg
import scipy.sparse as sps


class OctagonGridTest(unittest.TestCase):
    def test_single_octagon(self):
        nx = [1, 1]
        sd = pg.OctagonGrid(nx)
        sd.compute_geometry()

        face_areas_known = np.full_like(sd.face_areas, np.sqrt(2) - 1)
        face_areas_known[8:] = 1 / (2 + np.sqrt(2))

        self.assertTrue(np.allclose(sd.face_areas, face_areas_known))

        cell_volumes_known = np.full_like(sd.cell_volumes, 0.75 - 1 / np.sqrt(2))
        cell_volumes_known[0] = 2 * np.sqrt(2) - 2

        self.assertTrue(np.allclose(sd.cell_volumes, cell_volumes_known))


if __name__ == "__main__":
    unittest.main()
