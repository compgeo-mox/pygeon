""" 
Module contains a unit test for the standard grid computation.
"""
import unittest
import numpy as np

import pygeon as pg


class StandardGridTest(unittest.TestCase):
    def test_unit_square(self):
        coords = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        mesh_size = 0.5
        sd = pg.grid_unit_square(mesh_size, as_mdg=False)

        self.assertTrue(np.isclose(sd.num_cells, 162))
        self.assertTrue(np.isclose(sd.num_faces, 259))
        self.assertTrue(np.isclose(sd.num_nodes, 98))

    def test_pentagon(self):
        coords = np.array([[0, 1, 1, 0.5, 0], [0, 0, 1, 1.5, 1], [0, 0, 0, 0, 0]])
        mesh_size = 0.5
        sd = pg.grid_2d_from_coords(coords, mesh_size, as_mdg=False)

        self.assertTrue(np.isclose(sd.num_cells, 110))
        self.assertTrue(np.isclose(sd.num_faces, 178))
        self.assertTrue(np.isclose(sd.num_nodes, 69))

    def test_plane(self):
        coords = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]])
        mesh_size = 0.5
        sd = pg.grid_2d_from_coords(coords, mesh_size, as_mdg=False)

        self.assertTrue(np.isclose(sd.num_cells, 65))
        self.assertTrue(np.isclose(sd.num_faces, 110))
        self.assertTrue(np.isclose(sd.num_nodes, 46))


if __name__ == "__main__":
    unittest.main()
