import numpy as np
import porepy as pp
import pygeon as pg
import unittest


class VoronoiTest(unittest.TestCase):
    def test_simple_voronoi_grid(self):
        seed = 0
        sd = pg.VoronoiGrid(15, seed=seed)
        sd.compute_geometry()
        pp.plot_grid(sd, alpha=0, info="cfn", plot_2d=True)

        # fmt: off
        sd_nodes = np.array(
        [[0.24991698, 0.20243673, 0.33333333, 0.66666667, 0.33333333,
        0.26119357, 0.83875204, 0.76131217, 0.66666667, 0.58634662,
        0.53752143, 0.25262052, 0.24471338, 0.25941151, 0.44611037,
        0.78611729, 0.68697346, 0.72862514, 0.8060911 , 0.80086028,
        1.        , 1.        , 1.        , 0.66666667, 0.33333333,
        0.        , 0.        , 0.        , 0.        , 0.33333333,
        0.66666667, 1.        ],
       [0.24991698, 0.33333333, 0.19412692, 0.92833401, 0.87608422,
        0.73880643, 0.33333333, 0.23868783, 0.19948401, 0.22462121,
        0.48750883, 0.62337271, 0.66666667, 0.69982361, 0.7021328 ,
        0.78611729, 0.89605752, 0.64014441, 0.66666667, 0.658025  ,
        0.33333333, 0.66666667, 1.        , 1.        , 1.        ,
        1.        , 0.66666667, 0.33333333, 0.        , 0.        ,
        0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ]])

        sd_face_nodes_indptr = np.array(
        [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
        34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66,
        68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96])

        sd_face_nodes_indices = np.array(
        [ 0, 28,  0,  1,  1, 27,  0,  2,  2, 29,  4, 24,  3, 23,  3,  4,  5,
        25,  4,  5,  7, 31,  6, 20,  6,  7,  8, 30,  7,  8,  2,  9,  8,  9,
         9, 10,  1, 11, 10, 11, 12, 26, 11, 12,  5, 13, 12, 13, 14, 17, 14,
        16, 15, 18, 15, 16, 17, 19, 18, 19, 10, 17, 13, 14,  3, 16, 15, 22,
        18, 21,  6, 19, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26,
        27, 27, 28, 28, 29, 29, 30, 30, 31, 20, 31])

        sd_cell_faces_data = np.array(
        [ 1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,
        -1, -1,  1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1,  1,  1,
         1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,
        -1, -1, -1, -1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,
         1, -1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1])

        sd_cell_faces_indptr = np.array(
        [ 0,  4,  8, 12, 16, 20, 24, 29, 35, 40, 45, 51, 57, 63, 67, 72, 79,
        84])

        sd_cell_faces_indices = np.array(
        [43,  0,  1,  2,  3,  0, 44,  4,  6, 39,  5,  7,  5, 40,  8,  9, 10,
        47, 11, 12, 13, 46, 10, 14, 15,  4, 45, 13, 16, 18,  1,  3, 15, 17,
        19, 20, 42,  2, 18, 21, 22,  8, 41, 20, 23, 26, 27, 25, 24, 28, 29,
        31, 23, 21, 19, 30, 24, 32,  7,  9, 22, 31, 25, 34, 37, 33, 26, 33,
        38,  6, 32, 27, 30, 17, 16, 14, 12, 35, 28, 35, 11, 36, 34, 29])

        # fmt: on

        self.assertTrue(np.allclose(sd.nodes, sd_nodes))

        self.assertTrue(
            np.allclose(sd.face_nodes.data, np.ones_like(sd_face_nodes_indices))
        )
        self.assertTrue(np.allclose(sd.face_nodes.indptr, sd_face_nodes_indptr))
        self.assertTrue(np.allclose(sd.face_nodes.indices, sd_face_nodes_indices))

        self.assertTrue(np.allclose(sd.cell_faces.data, sd_cell_faces_data))
        self.assertTrue(np.allclose(sd.cell_faces.indptr, sd_cell_faces_indptr))
        self.assertTrue(np.allclose(sd.cell_faces.indices, sd_cell_faces_indices))


if __name__ == "__main__":
    unittest.main()
