import unittest

import numpy as np
import scipy.sparse as sps
import porepy as pp

import pygeon as pg


class ProjectionsUnitTest(unittest.TestCase):
    def test0(self):
        N, dim = 2, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        discr = pg.Lagrange1("p1")
        P = pg.eval_at_cell_centers(mdg, discr)

        # fmt: off
        P_known_data = np.array(
        [0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333]
        )

        P_known_indices = np.array(
        [0, 1, 0, 2, 3, 2, 1, 4, 5, 0, 1, 3, 4, 6, 7, 2, 3, 6, 5, 4, 5, 7,
        6, 7]
        )

        P_known_indptr = np.array(
        [ 0,  2,  5,  6,  9, 15, 18, 19, 22, 24]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indices, P_known_indices))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))

    def test1(self):
        N, dim = 2, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        mdg.initialize_data()
        discr = pg.RT0("rt0")
        P = pg.eval_at_cell_centers(mdg, discr)

        data = pg.RT0.create_unitary_data(discr.keyword, sd, None)
        discr_pp = pp.RT0("rt0")
        discr_pp.discretize(sd, data)
        P_pp = data[pp.DISCRETIZATION_MATRICES][discr_pp.keyword][
            discr_pp.vector_proj_key
        ]
        indices = np.reshape(np.arange(3 * sd.num_cells), (3, -1), order="F").ravel()

        self.assertEqual((P_pp.tolil()[indices] - P).nnz, 0)

    def test2(self):
        mesh_args = {"cell_size": 0.5, "cell_size_fracture": 0.125}
        x_endpoints = [np.array([0, 0.5])]
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "simplex", mesh_args, [1], x_endpoints
        )
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        discr = pg.Lagrange1("p1")
        P = pg.eval_at_cell_centers(mdg, discr)

        # fmt: off
        P_known_data = np.array(
        [0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.5       , 0.5       ,
        0.5       , 0.5       , 0.5       , 0.5       , 0.5       ,
        0.5       ]
        )

        P_known_indices = np.array(
        [10, 11, 18, 19,  5, 13, 21, 29, 41, 42, 24, 39,  2,  7, 14, 15, 16,
        40, 43, 12, 14, 37, 39,  7, 15, 35, 16, 28, 40,  9, 10, 25,  9, 24,
        33,  8, 19, 22,  8, 43, 44,  1,  3, 11, 18, 20, 34,  1,  4, 17, 20,
        23, 31,  4, 13, 23, 41,  9, 25, 26, 30, 32, 33,  6,  8, 22, 36, 44,
        45, 12, 24, 30, 33, 39, 12, 26, 27, 30, 37, 38,  2,  5, 14, 21, 27,
        37,  4,  5, 13, 17, 27, 38,  0,  6, 23, 31, 41, 42, 45,  1,  3, 17,
        26, 32, 38, 20, 31, 34, 36, 45,  2,  7, 21, 29, 35,  0, 15, 16, 28,
        29, 35, 42,  0,  6, 28, 40, 43, 44,  3, 10, 11, 25, 32, 18, 19, 22,
        34, 36, 46, 46, 47, 47, 48, 48, 49, 49]
        )

        P_known_indptr = np.array(
        [  0,   2,   4,  10,  12,  15,  17,  19,  23,  26,  29,  32,  35,
         38,  41,  44,  47,  50,  53,  55,  57,  63,  69,  74,  80,  86,
         92,  99, 105, 110, 115, 122, 128, 133, 138, 139, 141, 143, 145,
        146]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indices, P_known_indices))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))

    def test3(self):
        mesh_args = {"cell_size": 0.5, "cell_size_fracture": 0.125}
        x_endpoints = [np.array([0, 0.5])]
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "simplex", mesh_args, [1], x_endpoints
        )
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        mdg.initialize_data()
        discr = pg.RT0("rt0")
        P = pg.eval_at_cell_centers(mdg, discr)

        # Generate a matrix R that reorders from the pp to the pg convention
        R_list = []
        for sd in mdg.subdomains():
            arange = np.arange(3 * sd.num_cells)
            indices = np.reshape(arange, (3, -1), order="F").ravel()
            R_sd = sps.csc_array((np.ones_like(indices), (arange, indices)))
            R_list.append(R_sd)

        R = sps.block_diag(R_list, "csc")

        # fmt: off
        P_known_data = np.array(
        [-5.47798957e+00,  2.17053036e+00,  5.47798957e+00,  1.10399040e+00,
         2.06932322e+00, -5.33333333e+00, -2.73899478e+00, -3.82861931e+00,
        -3.90767540e+00, -2.66666667e+00,  5.49202156e+00, -2.66666667e+00,
         2.73899478e+00, -5.12276827e+00,  2.71350646e+00,  5.33333333e+00,
        -2.74904183e+00,  2.52945168e-01, -2.41096652e+00, -7.85165261e-01,
        -2.42796357e+00,  3.36122469e+00, -3.17476807e+00,  2.66666667e+00,
         1.53960072e+00,  2.66666667e+00,  3.14826338e+00,  1.73790485e+00,
        -5.02071302e-01, -2.55251350e+00, -2.73646126e-01, -2.60444927e+00,
         2.12587893e+00, -1.81668262e+00,  3.40792278e+00, -1.89787355e-01,
         5.43656250e-01, -1.33333333e+00,  3.08467361e+00,  6.80222350e-01,
         1.54233681e+00, -2.30609582e+00,  3.61122320e+00, -6.66666667e-01,
        -8.99114524e-01,  1.33333333e+00,  1.33333333e+00, -1.03018074e+00,
        -2.20641060e-01,  1.72444952e+00,  1.55044274e+00,  6.66666667e-01,
         1.59125129e+00, -1.54041946e-01,  6.66666667e-01,  1.48490963e+00,
        -1.33333333e+00, -2.36908667e-01,  4.36441382e-01,  1.33333333e+00,
        -6.66666667e-01,  1.30069327e+00, -1.38125986e+00,  6.66666667e-01,
         2.06574682e-01, -1.33333333e+00, -3.08467361e+00,  8.98850382e-01,
        -2.53575513e+00, -6.66666667e-01, -1.54233681e+00, -1.34042082e+00,
        -9.63539564e-02,  2.18965714e+00,  3.06756695e+00,  6.66666667e-01,
         3.03131031e+00, -3.87518790e-01,  1.80001118e+00,  1.43300362e+00,
         2.44955726e+00, -6.66666667e-01,  2.24998259e+00, -1.27949033e+00,
        -6.66666667e-01,  2.51509037e+00, -2.26612179e+00,  2.15143114e+00,
         6.66666667e-01,  1.53760194e+00,  7.36066518e-01,  1.50068869e+00,
        -1.81770125e+00, -6.66666667e-01, -2.02491896e+00, -2.28026399e-01,
        -1.77404070e+00,  1.49810857e+00, -2.74232981e+00,  6.66666667e-01,
        -4.11069492e+00,  7.88921534e-02, -2.05534746e+00, -3.92979254e+00,
        -2.33188680e+00, -3.74646980e+00,  2.73899478e+00, -5.99914967e+00,
         4.51678317e+00, -5.64041328e+00,  2.05534746e+00, -4.00868470e+00,
         3.61523122e+00, -3.03448494e+00, -1.54233681e+00, -2.98631817e+00,
        -9.03881455e-01, -3.21066338e+00,  4.11069492e+00, -3.03563057e-01,
         2.05534746e+00, -3.90784722e+00,  1.19574987e+00, -4.36799520e+00,
        -2.73899478e+00, -6.22675866e+00, -4.94178666e+00, -5.14941315e+00,
        -2.05534746e+00, -3.60428417e+00, -4.90068286e+00, -1.66091855e+00,
         1.54233681e+00, -2.23927120e+00, -1.51050043e-01, -2.40087654e+00,
         1.91909546e+00, -5.33333333e+00, -3.03263292e+00, -2.66666667e+00,
        -3.72296949e+00, -1.86287767e+00,  6.55212371e+00, -2.66666667e+00,
         5.90590820e+00, -3.54857575e+00, -5.76058231e+00,  3.60845058e+00,
        -5.97699862e+00,  2.66666667e+00,  7.22958226e+00,  2.66666667e+00,
         6.83036239e+00,  3.51732381e+00,  3.32852194e+00, -5.33333333e+00,
        -2.86726033e+00, -2.66666667e+00, -3.01532221e+00, -2.54230114e+00,
         4.61880215e+00, -2.66666667e+00,  5.39421267e+00, -1.32361626e+00,
        -4.95172838e+00,  2.66666667e+00, -5.08195222e+00,  1.73109803e+00,
         6.20704661e+00,  2.66666667e+00,  5.25581023e+00,  3.76999702e+00,
        -6.19578226e+00,  2.66666667e+00, -5.88827452e+00, -2.66666667e+00,
         4.61880215e+00,  2.66666667e+00,  7.69800359e+00, -2.66666667e+00,
        -4.39919214e+00, -4.52395677e-01, -4.51911268e+00, -1.76178441e-01,
        -2.92792165e-02, -3.14897133e+00, -1.04333491e+00, -2.88945528e+00,
         4.64772700e+00, -2.76767353e+00,  3.96697964e+00, -3.27831433e+00,
         6.84866997e+00, -1.89394348e+00,  6.07499721e+00,  1.42995592e+00,
        -1.12598977e+00, -3.18476703e+00, -1.64374393e+00, -2.94028148e+00,
        -4.30184094e+00, -3.52352823e+00, -6.29843011e+00,  1.11591153e+00,
         2.11606940e+00, -2.85744819e+00,  4.74963282e+00, -7.39957987e-01,
        -6.13753653e+00, -7.81417952e-01, -4.43467881e+00,  3.54496568e+00,
         3.12766427e+00, -2.57717593e+00,  3.35585723e+00, -2.43705960e+00,
         1.44470032e+00, -2.67478917e+00,  4.49971414e-01, -2.71249394e+00,
         4.11805704e+00, -1.20868176e+00,  3.92877081e+00, -1.84662633e+00,
         4.67700622e+00,  3.81297797e-01,  2.99198325e+00,  2.53996299e+00,
         3.21078259e-01,  3.10827952e+00,  2.67335671e+00,  1.46610741e+00,
         1.81189235e+00, -1.87849147e+00,  1.90889522e+00, -1.76734824e+00,
        -2.06663001e+00,  4.27339917e+00, -9.36787567e-01,  4.38658932e+00,
        -1.38402436e-01,  5.09361328e+00, -4.65468618e+00,  4.05619301e+00,
         1.90024107e+00, -1.54030034e+00,  2.59659398e-01, -1.92769220e+00,
         3.32981459e+00, -2.08231105e-01,  3.24205917e+00,  3.27318835e-01,
        -2.03761281e+00,  5.47132826e+00,  2.10801757e+00,  4.70827026e+00,
         9.24454194e-01,  7.06589956e+00, -1.32837875e-01,  7.06849391e+00,
         2.39952505e+00,  7.87766644e-01,  3.00218831e+00, -6.50742446e-01,
         1.42957352e+00,  1.33206923e+00,  2.50878261e-01,  1.72613497e+00,
        -1.73756070e+00, -5.33333333e+00, -6.15840287e+00,  5.33333333e+00,
         3.45077103e-01, -5.33333333e+00,  4.10232040e-15, -5.33333333e+00,
         5.00000000e-01,  5.55111512e-17,  5.00000000e-01,  5.55111512e-17,
         5.00000000e-01,  5.55111512e-17,  5.00000000e-01,  5.55111512e-17,
         5.00000000e-01,  5.55111512e-17,  5.00000000e-01,  5.55111512e-17,
         5.00000000e-01,  5.55111512e-17,  5.00000000e-01,  5.55111512e-17]
        )

        P_known_indices = np.array(
        [ 30,  31,  57,  58,  33,  34,  30,  31,  33,  34,  54,  55,  57,
         58,  39,  40,  15,  16,  63,  64,  15,  16,  39,  40, 123, 124,
        126, 127,  63,  64,  87,  88,  87,  88, 126, 127, 117, 118,  72,
         73,  72,  73, 117, 118,  42,  43,  21,  22,   6,   7,  42,  43,
          6,   7,  21,  22,  45,  46,  48,  49,  45,  46,  48,  49, 120,
        121, 129, 130, 120, 121, 129, 130,  36,  37, 117, 118,  36,  37,
        111, 112,  42,  43, 111, 112,  21,  22, 105, 106,  45,  46, 105,
        106,  48,  49,  84,  85,  84,  85, 120, 121,  27,  28,  27,  28,
         75,  76,  30,  31,  75,  76,  27,  28,  99, 100,  72,  73,  99,
        100,  24,  25,  24,  25,  66,  67,  57,  58,  66,  67,  24,  25,
        132, 133, 129, 130, 132, 133,   3,   4,   3,   4,   9,  10,  60,
         61, 102, 103,   9,  10,  33,  34,  54,  55, 102, 103,  12,  13,
         12,  13,  51,  52,  69,  70,  93,  94,   3,   4,  51,  52,  60,
         61,  93,  94,  12,  13,  39,  40,  69,  70, 123, 124,  90,  91,
         99, 100,  78,  79,  90,  91,  78,  79,  96,  97,  75,  76,  96,
         97,  18,  19, 135, 136, 108, 109, 135, 136,  18,  19, 132, 133,
         66,  67, 108, 109,  36,  37,  90,  91,  81,  82, 111, 112,  81,
         82, 114, 115,  78,  79, 114, 115,  15,  16,  81,  82,   6,   7,
         63,  64,  51,  52, 114, 115,  93,  94, 135, 136,   0,   1, 126,
        127,   0,   1,  18,  19,   9,  10,  96,  97, 102, 103, 108, 109,
         87,  88, 105, 106,   0,   1,  84,  85,  54,  55, 123, 124,  60,
         61,  69,  70, 138, 140, 138, 140, 141, 143, 141, 143, 144, 146,
        144, 146, 147, 149, 147, 149]
        )

        P_known_indptr = np.array(
        [  0,   2,   4,   6,  10,  14,  16,  20,  24,  28,  32,  36,  38,
         40,  44,  46,  48,  52,  56,  58,  60,  64,  66,  68,  72,  76,
         80,  84,  88,  92,  96, 100, 102, 106, 110, 114, 118, 120, 124,
        128, 132, 136, 138, 142, 146, 150, 154, 156, 160, 164, 168, 172,
        176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224,
        228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 270, 272,
        274, 276, 278, 282, 286, 290, 292]
        )
        # fmt: on

        # Assemble the sparse matrix and reorder the rows to the pg convention
        P_known = sps.csc_array((P_known_data, P_known_indices, P_known_indptr))
        P_known = R @ P_known

        self.assertTrue(np.allclose((P_known - P).data, 0))


if __name__ == "__main__":
    unittest.main()
