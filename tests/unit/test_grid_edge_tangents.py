import unittest

import numpy as np
import porepy as pp

import pygeon as pg  # type: ignore[import-untyped]

"""
Module contains a unit tests to validate the edge length computations for grids
"""


class EdgeTangentTest(unittest.TestCase):
    def test_grid_0d(self):
        sd = pp.PointGrid([0, 0, 0])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        self.assertTrue(len(sd.edge_tangents) == 0)
        self.assertTrue(len(sd.edge_lengths) == 0)
        self.assertTrue(sd.mesh_size == 0)

    def test_cartgrids(self):
        for dim in np.arange(1, 4):
            sd = pp.CartGrid([2] * dim, [1] * dim)
            pg.convert_from_pp(sd)
            sd.compute_geometry()

            self.assertTrue(np.allclose(sd.edge_lengths, 0.5))
            self.assertTrue(np.allclose(sd.mesh_size, 0.5))

    def test_grid_2d_tris(self):
        sd = pp.StructuredTriangleGrid([1] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        unique_lengths = np.unique(sd.edge_lengths)
        known_lengths = np.sqrt(np.arange(1, 3))

        self.assertTrue(np.allclose(unique_lengths, known_lengths))
        self.assertTrue(np.allclose(sd.edge_lengths, sd.face_areas))
        self.assertTrue(
            np.allclose(np.sum(sd.edge_tangents * sd.face_normals, axis=0), 0)
        )

        known_meshsize = (4 + np.sqrt(2)) / 5
        self.assertTrue(np.allclose(sd.mesh_size, known_meshsize))

    def test_grid_3d_tets(self):
        sd = pp.StructuredTetrahedralGrid([1] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        unique_lengths = np.unique(sd.edge_lengths)
        known_lengths = np.sqrt(np.arange(1, 4))

        self.assertTrue(np.allclose(unique_lengths, known_lengths))

        known_meshsize = (12 + 6 * np.sqrt(2) + np.sqrt(3)) / 19
        self.assertTrue(np.allclose(sd.mesh_size, known_meshsize))


if __name__ == "__main__":
    unittest.main()
