""" 
Module contains a unit test for the standard grid computation.
"""
import unittest
import numpy as np

import pygeon as pg


class StandardGridTest(unittest.TestCase):
    def test_unit_square(self):
        mesh_size = 0.5
        sd = pg.grid_unitary(2, mesh_size, as_mdg=False)

        self.assertTrue(np.isclose(sd.num_cells, 14))
        self.assertTrue(np.isclose(sd.num_faces, 25))
        self.assertTrue(np.isclose(sd.num_nodes, 12))

    def test_unit_cube(self):
        mesh_size = 0.5
        sd = pg.grid_unitary(3, mesh_size, as_mdg=False)

        self.assertTrue(np.isclose(sd.num_cells, 100))
        self.assertTrue(np.isclose(sd.num_faces, 242))
        self.assertTrue(np.isclose(sd.num_nodes, 45))


if __name__ == "__main__":
    unittest.main()
