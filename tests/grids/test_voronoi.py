import unittest

import numpy as np

import pygeon as pg


class VoronoiTest(unittest.TestCase):
    def test_simple_voronoi_grid(self):
        seed = 0
        sd = pg.VoronoiGrid(8, seed=seed)
        sd.compute_geometry()

        # fmt: off
        sd_nodes = np.array([
        [ 6.62130112e-01,  4.10152948e-01,  4.95728831e-01,
          7.67887542e-01,  5.23934774e-01,  3.53342180e-01,
          5.70451037e-01,  0.00000000e+00,  0.00000000e+00,
          6.67482256e-01, -6.34121216e-17,  1.00000000e+00,
          1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
          6.19064968e-01, -1.92844452e-17,  1.00000000e+00],
        [ 1.51048971e-01,  3.18966538e-01,  2.98912438e-01,
          6.17573336e-01,  6.71296377e-01,  8.07895678e-01,
          8.76019152e-01,  3.07468908e-01,  9.19679841e-01,
          0.00000000e+00,  0.00000000e+00,  3.52398389e-01,
          7.37627116e-01,  6.81488414e-01,  0.00000000e+00,
          1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00]])

        sd_face_nodes_indptr = np.array(
        [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
        34, 36, 38, 40, 42, 44, 46, 48, 50])

        sd_face_nodes_indices = np.array(
        [ 0,  2,  1,  2,  3,  4,  4,  5,  5,  6,  2,  3,  1,  4,  7,  8,  1,
        7,  5,  8,  9, 10,  0,  9,  7, 10, 12, 13, 11, 13, 11, 14,  0, 11,
        6, 12,  3, 13,  9, 14, 15, 16, 15, 17,  6, 15,  8, 16, 12, 17])

        sd_cell_faces_data = np.array(
        [ 1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1, -1, -1,
        -1, -1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1,  1,
        -1,  1, -1, -1,  1])

        sd_cell_faces_indptr = np.array([ 0,  4,  9, 15, 21, 26, 30, 35, 39])

        sd_cell_faces_indices = np.array(
        [ 1,  2,  5,  6,  3,  6,  7,  8,  9,  0,  1,  8, 10, 11, 12,  2,  3,
        4, 13, 17, 18,  0,  5, 14, 16, 18, 11, 15, 16, 19,  4,  9, 20, 22,
        23, 17, 21, 22, 24])

        # fmt: on

        assert np.allclose(sd.nodes, sd_nodes)

        assert np.allclose(sd.face_nodes.data, np.ones_like(sd_face_nodes_indices))
        assert np.allclose(sd.face_nodes.indptr, sd_face_nodes_indptr)
        assert np.allclose(sd.face_nodes.indices, sd_face_nodes_indices)

        assert np.allclose(sd.cell_faces.data, sd_cell_faces_data)
        assert np.allclose(sd.cell_faces.indptr, sd_cell_faces_indptr)
        assert np.allclose(sd.cell_faces.indices, sd_cell_faces_indices)


if __name__ == "__main__":
    unittest.main()
