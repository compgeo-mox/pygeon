""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np

import porepy as pp
import pygeon as pg
import scipy.sparse as sps


class SubVolumeTest(unittest.TestCase):
    def test_quads(self):
        sd = pp.CartGrid([4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sd.sub_volumes, 0)))

    def test_tris(self):
        sd = pp.StructuredTriangleGrid([4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sd.sub_volumes, 0)))

    def test_tets(self):
        sd = pp.StructuredTetrahedralGrid([4, 4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sd.sub_volumes, 0)))

    def test_hexes(self):
        sd = pp.CartGrid([4, 4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sd.sub_volumes, 0)))

    def test_hitchhiker_pentagon(self):

        nodes = np.array([[0, 3, 3, 3.0 / 2, 0], [0, 0, 2, 4, 4], np.zeros(5)])
        cols = np.repeat(np.arange(5), 2)
        rows = np.roll(cols, -1)
        face_nodes = sps.csc_matrix((np.ones(10), (rows, cols)))
        cell_faces = sps.csc_matrix(np.ones((5, 1)))

        sd = pp.Grid(2, nodes, face_nodes, cell_faces, "pentagon")
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sd.compute_subvolumes()
        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sd.sub_volumes, 0)))

    def test_concave_quad(self):

        nodes = np.array([[0, 0.5, 1, 0.5], [0, 0.5, 0, 1], np.zeros(4)])
        indices = np.array([0, 1, 1, 2, 2, 3, 3, 0])
        face_nodes = sps.csc_matrix((np.ones(8), indices, np.arange(0, 9, 2)))
        cell_faces = sps.csc_matrix(np.ones((4, 1)))

        sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sd.compute_subvolumes()
        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sd.sub_volumes, 0)))

    def test_multiple_concave_quads(self):
        nodes = np.array(
            [[0, 0.5, 1, 0.5, 0.5, 0.5], [0, 0.5, 0, 1, -0.5, -1], np.zeros(6)]
        )
        indices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 4, 2, 2, 5, 5, 0])
        face_nodes = sps.csc_matrix((np.ones(16), indices, np.arange(0, 17, 2)))
        cell_faces_j = np.repeat(np.arange(3), 4)
        cell_faces_i = np.array([0, 1, 2, 3, 4, 5, 1, 0, 4, 5, 6, 7])
        cell_faces_v = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1])
        cell_faces = sps.csc_matrix((cell_faces_v, (cell_faces_i, cell_faces_j)))

        sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sd.compute_subvolumes()
        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sd.sub_volumes, 0)))


if __name__ == "__main__":
    unittest.main()
