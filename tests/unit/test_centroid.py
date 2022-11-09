""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np

import porepy as pp
import pygeon as pg
import scipy.sparse as sps


class CentroidTest(unittest.TestCase):
    def test_centroid_1_quad(self):
        sd = pp.CartGrid([4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        self.assertTrue(np.allclose(sd.cell_centroids, sd.cell_centers))

    def test_centroid_1_tri(self):
        sd = pp.StructuredTriangleGrid([4] * 2, [4] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        self.assertTrue(np.allclose(sd.cell_centroids, sd.cell_centers))

    def test_hitchhiker_pentagon(self):

        nodes = np.array([[0, 3, 3, 3.0 / 2, 0], [0, 0, 2, 4, 4], np.zeros(5)])
        cols = np.repeat(np.arange(5), 2)
        rows = np.roll(cols, -1)
        face_nodes = sps.csc_matrix((np.ones(10), (rows, cols)))
        cell_faces = sps.csc_matrix(np.ones((5, 1)))

        sd = pp.Grid(2, nodes, face_nodes, cell_faces, "pentagon")
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        self.assertTrue(np.allclose(sd.cell_centroids, sd.cell_centers))

    # def test_stretched_pentagon(self):

    #     nodes = np.array([[0, 3, 30, 3.0 / 2, 0], [0, 0, 2, 2, 4], np.zeros(5)])
    #     cols = np.repeat(np.arange(5), 2)
    #     rows = np.roll(cols, 1)
    #     face_nodes = sps.csc_matrix((np.ones(10), (rows, cols)))
    #     cell_faces = sps.csc_matrix(np.ones((5, 1)))

    #     sd = pp.Grid(2, nodes, face_nodes, cell_faces, "pentagon")
    #     pg.convert_from_pp(sd)
    #     sd.compute_geometry()

    #     self.assertFalse(np.allclose(sd.cell_centroids, sd.cell_centers))
    #     self.assertTrue(
    #         np.allclose(
    #             sd.cell_centroids, np.array([[7.46644295], [1.02908277], [0.0]])
    #         )
    #     )


if __name__ == "__main__":
    unittest.main()
