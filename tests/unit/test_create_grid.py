""" 
Module contains a unit test for the standard grid computation.
"""
import unittest
import numpy as np

import porepy as pp
import pygeon as pg


class StandardGridTest(unittest.TestCase):
    def test_unit_square(self):
        mesh_size = 0.5
        sd = pg.unit_grid(2, mesh_size, as_mdg=False)

        self.assertTrue(np.isclose(sd.num_cells, 14))
        self.assertTrue(np.isclose(sd.num_faces, 25))
        self.assertTrue(np.isclose(sd.num_nodes, 12))

    def test_unit_cube(self):
        mesh_size = 0.5
        sd = pg.unit_grid(3, mesh_size, as_mdg=False)

        self.assertTrue(np.isclose(sd.num_cells, 100))
        self.assertTrue(np.isclose(sd.num_faces, 242))
        self.assertTrue(np.isclose(sd.num_nodes, 45))

    def test_triangle(self):
        line_1 = np.array([[0, 1], [0, 0]])
        line_2 = np.array([[1, 0], [0, 1]])
        line_3 = np.array([[0, 0], [1, 0]])
        lines = [line_1, line_2, line_3]
        domain = pp.Domain(polytope=lines)

        mesh_size = 0.5
        sd = pg.grid_from_domain(domain, mesh_size, as_mdg=False)

        self.assertTrue(np.isclose(sd.num_cells, 7))
        self.assertTrue(np.isclose(sd.num_faces, 14))
        self.assertTrue(np.isclose(sd.num_nodes, 8))

    def test_concave_pentagon(self):
        line_1 = np.array([[0, 1], [0, 0]])
        line_2 = np.array([[1, 1], [0, 1]])
        line_3 = np.array([[1, 0.5], [1, 0.5]])
        line_4 = np.array([[0.5, 0], [0.5, 1]])
        line_5 = np.array([[0, 0], [1, 0]])
        lines = [line_1, line_2, line_3, line_4, line_5]
        domain = pp.Domain(polytope=lines)

        mesh_size = 0.5
        sd = pg.grid_from_domain(domain, mesh_size, as_mdg=False)

        self.assertTrue(np.isclose(sd.num_cells, 14))
        self.assertTrue(np.isclose(sd.num_faces, 26))
        self.assertTrue(np.isclose(sd.num_nodes, 13))


if __name__ == "__main__":
    unittest.main()
