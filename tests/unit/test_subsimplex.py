""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np

import porepy as pp
import pygeon as pg
import scipy.sparse as sps


class SubSimplexTest(unittest.TestCase):
    def test_subsimplices_quads(self):
        sd = pp.CartGrid([4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sd.compute_subsimplices()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sd.subsimplices, 0)))

    def test_subsimplices_tris(self):
        sd = pp.StructuredTriangleGrid([4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sd.compute_subsimplices()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sd.subsimplices, 0)))


if __name__ == "__main__":
    unittest.main()
