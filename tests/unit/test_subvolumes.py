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

        sub_volumes = sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

    def test_tris(self):
        sd = pp.StructuredTriangleGrid([4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

    def test_oct(self):
        sd = pg.OctagonGrid([5, 5])
        sd.compute_geometry()
        sub_volumes = sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

    def test_tets(self):
        sd = pp.StructuredTetrahedralGrid([4, 4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

    def test_hexes(self):
        sd = pp.CartGrid([4, 4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

    def test_hitchhiker_pentagon(self):
        nodes = np.array([[0, 3, 3, 3.0 / 2, 0], [0, 0, 2, 4, 4], np.zeros(5)])
        indptr = np.arange(0, 11, 2)
        indices = np.roll(np.repeat(np.arange(5), 2), -1)
        face_nodes = sps.csc_matrix((np.ones(10), indices, indptr))
        cell_faces = sps.csc_matrix(np.ones((5, 1)))

        sd = pp.Grid(2, nodes, face_nodes, cell_faces, "pentagon")
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()
        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

    def test_concave_quad(self):
        nodes = np.array([[0, 0.5, 1, 0.5], [0, 0.5, 0, 1], np.zeros(4)])
        indices = np.array([0, 1, 1, 2, 2, 3, 3, 0])
        face_nodes = sps.csc_matrix((np.ones(8), indices, np.arange(0, 9, 2)))
        cell_faces = sps.csc_matrix(np.ones((4, 1)))

        sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()
        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

    def test_multiple_concave_quads(self):
        nodes = np.array(
            [[0, 0.5, 1, 0.5, 0.5, 0.5], [0, 0.5, 0, 1, -0.5, -1], np.zeros(6)]
        )
        indices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 5, 5, 2, 2, 4, 4, 0])
        face_nodes = sps.csc_matrix((np.ones(16), indices, np.arange(0, 17, 2)))
        cell_faces_j = np.repeat(np.arange(3), 4)
        cell_faces_i = np.array([0, 1, 2, 3, 7, 6, 1, 0, 4, 5, 6, 7])
        cell_faces_v = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1])
        cell_faces = sps.csc_matrix((cell_faces_v, (cell_faces_i, cell_faces_j)))

        sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()

        known_sub_volumes = np.array(
            [
                [6.25e-02, 1.25e-01, 6.25e-02],
                [0, 1.25e-01, 0],
                [6.25e-02, 1.25e-01, 6.25e-02],
                [1.25e-01, 0, 0],
                [0, 1.25e-01, 0],
                [0, 0, 1.25e-01],
            ]
        )
        self.assertTrue(np.allclose(sub_volumes.todense(), known_sub_volumes))


if __name__ == "__main__":
    SubVolumeTest().test_hexes()
    unittest.main()
