import unittest

import numpy as np
import porepy as pp

import pygeon as pg

"""
Module contains a unit tests to validate the opposite_node computations on simplicial grids.
"""


class OppositeNode_Test(unittest.TestCase):
    def test_grid_2d_tris(self):
        N = 1
        sd = pp.StructuredTriangleGrid([N] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        opposite_node = sd.compute_opposite_nodes()
        known_data = np.array([3, 1, 0, 3, 2, 0])

        self.assertTrue(opposite_node.nnz == sd.num_cells * (sd.dim + 1))
        self.assertTrue(np.all(known_data == opposite_node.data))

    def test_grid_3d_tets(self):
        N = 1
        sd = pp.StructuredTetrahedralGrid([N] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        opposite_node = sd.compute_opposite_nodes()
        known_data = np.array(
            [4, 2, 1, 0, 6, 4, 2, 1, 6, 5, 4, 1, 6, 3, 2, 1, 6, 5, 3, 1, 7, 6, 5, 3]
        )

        self.assertTrue(opposite_node.nnz == sd.num_cells * (sd.dim + 1))
        self.assertTrue(np.all(known_data == opposite_node.data))

    def test_non_simplicial_grid(self):
        N = 1
        sd = pp.CartGrid([N] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        self.assertRaises(NotImplementedError, sd.compute_opposite_nodes)


if __name__ == "__main__":
    unittest.main()
