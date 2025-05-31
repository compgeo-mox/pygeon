import unittest

import numpy as np
import porepy as pp

import pygeon as pg


class RefinementTest(unittest.TestCase):
    def test_reference_element(self):
        dim = 2
        # sd = pg.reference_element(dim)
        sd = pg.unit_grid(dim, 1 / 5, as_mdg=False)
        sd.compute_geometry()

        sd = pg.barycentric_split(sd)
        sd.compute_geometry()

        self.assertTrue((sd.face_ridges @ sd.cell_faces).nnz == 0)

        pass


if __name__ == "__main__":
    unittest.main()
