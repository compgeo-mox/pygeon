"""Module contains a unit test for the Einstein grid class."""

import unittest

import numpy as np
import scipy.sparse as sps

import pygeon as pg
import os


class EinSteinGridTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(EinSteinGridTest, self).__init__(*args, **kwargs)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.folder = os.path.join(dir_path, "einstein_svg_grids")

    def test_T1(self):
        file_name = os.path.join(self.folder, "T1.svg")

        sd = pg.EinSteinGrid(file_name)
        sd.compute_geometry()

        # cell_faces known data
        cf_indices = np.array([13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 0, 1], dtype=int)
        cf_indptr = np.array([0, 14], dtype=int)
        cf_data = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1])

        # check the cell_faces
        cf_known = sps.csc_array((cf_data, cf_indices, cf_indptr))
        self.assertTrue(np.allclose(sps.find(sd.cell_faces), sps.find(cf_known)))

        # fmt: off
        fr_indices = np.array(
            [ 1,  0, 13,  0,  2,  1,  3,  2,  4,  3,  5,  4,  6,  5,  7,  6,  8,
              7,  9,  8, 10,  9, 11, 10, 12, 11, 13, 12])

        fr_indptr = np.array(
            [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

        fr_data = np.array(
            [ 1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1.,
             -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,
              1., -1.])
        # fmt: on

        # check the face_ridges
        fr_known = sps.csc_array((fr_data, fr_indices, fr_indptr))
        self.assertTrue(np.allclose(sps.find(sd.face_ridges), sps.find(fr_known)))

        # check the cell_volumes
        cv_kown = 0.38490017945975075
        self.assertTrue(np.allclose(sd.cell_volumes, cv_kown))

    def test_H1(self):
        file_name = os.path.join(self.folder, "H1.svg")

        sd = pg.EinSteinGrid(file_name)
        sd.compute_geometry()

        # fmt: off
        cf_indices = np.array(
            [16, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  2,  0,  1, 33, 32, 31,
             30, 29, 27, 26, 25, 24, 22, 20, 19, 17, 18, 42,  4,  3, 41, 40, 39,
             38, 37, 36, 35, 21, 20, 23, 34, 26, 28, 15, 14, 16,  1,  0,  4, 42,
             34, 23, 22, 24, 25])

        cf_indptr = np.array([ 0, 14, 28, 42, 56])

        cf_data = np.array(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1,
             -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,
             -1, -1,  1,  1,  1])
        # fmt: on

        # check the cell_faces
        cf_known = sps.csc_array((cf_data, cf_indices, cf_indptr))
        self.assertTrue(np.allclose(sps.find(sd.cell_faces), sps.find(cf_known)))

        # fmt: off
        fr_indices = np.array(
            [ 1,  0, 13,  0,  2,  1, 36,  1, 37,  1,  3,  2,  4,  3,  5,  4,  6,
              5,  7,  6,  8,  7,  9,  8, 10,  9, 11, 10, 12, 11, 39, 11, 13, 12,
             15, 14, 27, 14, 16, 15, 17, 16, 29, 16, 18, 17, 28, 17, 19, 18, 20,
             19, 21, 20, 22, 21, 39, 21, 23, 22, 24, 23, 25, 24, 26, 25, 27, 26,
             38, 28, 30, 29, 31, 30, 32, 31, 33, 32, 34, 33, 35, 34, 36, 35, 38,
             37])

        fr_indptr = np.array(
            [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
             34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66,
             68, 70, 72, 74, 76, 78, 80, 82, 84, 86])

        fr_data = np.array(
            [ 1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1.,
             -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,
              1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1.,
             -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,
              1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1.,
             -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,
              1., -1.,  1., -1.,  1., -1.,  1., -1.])
        # fmt: on

        # check the face_ridges
        fr_known = sps.csc_array((fr_data, fr_indices, fr_indptr))
        self.assertTrue(np.allclose(sps.find(sd.face_ridges), sps.find(fr_known)))

        # check the cell_volumes
        cv_kown = 0.12568169125216347
        self.assertTrue(np.allclose(sd.cell_volumes, cv_kown))

    def test_H2(self):
        file_name = os.path.join(self.folder, "H2.svg")

        sd = pg.EinSteinGrid(file_name)
        sd.compute_geometry()

        # cell_faces known data
        # fmt: off
        cf_indices = np.array([
            19,  17,  16,  15,  13,  12,  10,   9,   8,   6,   5,   2,   0,
             1,  38,  37,  35,  34,  33,  30,  29,  28,  27,  25,  23,  22,
            20,  21,  51,   4,   3,  50,  48,  47,  45,  44,  43,  41,  24,
            23,  26,  40,  29,  31,  18,  17,  19,   1,   0,   4,  51,  40,
            26,  25,  27,  28,  64,  62,  61,  60,  58,  57,  36,  35,  37,
            38,  39,  54,  52,  53,  83,  82,  80,  79,  78,  75,  74,  73,
            72,  70,  68,  67,  65,  66,  96,  56,  55,  95,  93,  92,  90,
            89,  87,  86,  69,  68,  71,  85,  74,  76,  63,  62,  64,  53,
            52,  56,  96,  85,  71,  70,  72,  73, 116, 114, 113, 112, 110,
           109, 107, 106, 105, 103, 102,  99,  97,  98,  41,  43,  44,  46,
           129, 126, 125, 124, 123, 121, 119, 118, 117,  42, 135, 101, 100,
           134, 132, 131,  91,  90,  92,  94, 120, 119, 122, 130, 125, 127,
           115, 114, 116,  98,  97, 101, 135, 130, 122, 121, 123, 124,  54,
            39,  21,  20,  22,  24,  42, 117, 118, 120,  94,  93,  95,  55,
           126, 129,  46,  45,  47,  49, 143, 142, 141, 140, 138, 137, 136,
           128, 115, 127, 128, 136, 137, 139, 148, 147, 146, 145, 111, 110,
           112, 113,   6,   8,   9,  11, 158, 157, 156, 155, 154, 153, 151,
           150, 149,   7,   2,   5,   7, 149, 150, 152, 161, 160, 159, 144,
            49,  48,  50,   3,  30,  33,  34,  36,  57,  59, 169, 168, 167,
           166, 164, 163, 162,  32,  18,  31,  32, 162, 163, 165, 174, 173,
           172, 171,  14,  13,  15,  16,  75,  78,  79,  81, 184, 183, 182,
           181, 180, 179, 177, 176, 175,  77,  63,  76,  77, 175, 176, 178,
           187, 186, 185, 170,  59,  58,  60,  61, 194, 193, 192, 191, 190,
            84,  66,  65,  67,  69,  86,  88, 188, 189, 201, 195, 189, 188,
            88,  87,  89,  91, 131, 133, 199, 198, 196, 197, 103, 105, 106,
           108, 211, 210, 209, 208, 207, 206, 204, 203, 202, 104,  99, 102,
           104, 202, 203, 205, 213, 212, 200, 199, 133, 132, 134, 100],
           dtype=int)
        cf_indptr = np.array(
            [  0,  14,  28,  42,  56,  70,  84,  98, 112, 126, 140, 154, 168,
             182, 196, 210, 224, 238, 252, 266, 280, 294, 308, 322, 336, 350],
            dtype=int)
        cf_data = np.array(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1,
             -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,
             -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1,
             -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1,
             -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,
              1, -1,  1,  1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,  1,  1, -1,  1,  1,
              1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1,
             -1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,
              1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1,
             -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
              1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
              1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,
             -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,
             -1,  1, -1, -1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,
              1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
              1,  1, -1, -1, -1,  1, -1,  1,  1, -1])
        # fmt: off

        # check the cell_faces
        cf_known = sps.csc_array((cf_data, cf_indices, cf_indptr))
        self.assertTrue(np.allclose(sps.find(sd.cell_faces), sps.find(cf_known)))

        # face_ridges known data
        # fmt: off
        fr_indices = np.array(
            [  0,   1,   0,  13,   1,   2,   1,  36,   1,  37,   2,   3,   3,
               4,   3, 122,   4,   5,   5,   6,   6,   7,   6, 131,   7,   8,
               8,   9,   8, 144,   9,  10,  10,  11,  11,  12,  11,  39,  12,
              13,  14,  15,  14,  27,  15,  16,  16,  17,  16,  29,  17,  18,
              17,  28,  18,  19,  19,  20,  20,  21,  21,  22,  21,  39,  21,
             136,  22,  23,  23,  24,  24,  25,  24,  43,  25,  26,  26,  27,
              27,  42,  28,  38,  29,  30,  29,  90,  30,  31,  31,  32,  32,
              33,  32,  99,  33,  34,  34,  35,  34, 116,  35,  36,  37,  38,
              40,  41,  40,  49,  41,  42,  41,  72,  41,  73,  43,  44,  44,
              45,  44, 143,  45,  46,  46,  47,  47,  48,  47,  75,  48,  49,
              50,  51,  50,  63,  51,  52,  52,  53,  52,  65,  53,  54,  53,
              64,  54,  55,  55,  56,  56,  57,  57,  58,  57,  75,  57, 149,
              58,  59,  59,  60,  60,  61,  60, 158,  61,  62,  62,  63,  63,
             165,  64,  74,  65,  66,  66,  67,  66, 164,  67,  68,  68,  69,
              68, 102,  69,  70,  70,  71,  70, 101,  71,  72,  73,  74,  76,
              77,  76,  89,  77,  78,  77, 105,  77, 106,  78,  79,  79,  80,
              79, 177,  80,  81,  81,  82,  82,  83,  82, 186,  83,  84,  84,
              85,  84, 117,  85,  86,  86,  87,  87,  88,  87, 108,  88,  89,
              90,  91,  91,  92,  92,  93,  92, 101,  93,  94,  93, 100,  94,
              95,  95,  96,  96,  97,  97,  98,  97, 108,  97, 109,  98,  99,
             100, 107, 102, 103, 103, 104, 103, 174, 104, 105, 106, 107, 109,
             110, 110, 111, 111, 112, 111, 121, 112, 113, 113, 114, 114, 115,
             115, 116, 116, 132, 117, 118, 118, 119, 119, 120, 120, 121, 122,
             123, 123, 124, 124, 125, 124, 135, 125, 126, 126, 127, 127, 128,
             128, 129, 129, 130, 130, 131, 132, 133, 133, 134, 134, 135, 136,
             137, 137, 138, 138, 139, 138, 148, 139, 140, 140, 141, 141, 142,
             142, 143, 143, 159, 144, 145, 145, 146, 146, 147, 147, 148, 149,
             150, 150, 151, 151, 152, 151, 162, 152, 153, 153, 154, 154, 155,
             155, 156, 156, 157, 157, 158, 159, 160, 160, 161, 161, 162, 163,
             164, 163, 170, 165, 166, 166, 167, 167, 168, 168, 169, 169, 170,
             170, 175, 171, 172, 171, 176, 172, 173, 173, 174, 173, 187, 175,
             176, 177, 178, 178, 179, 179, 180, 179, 189, 180, 181, 181, 182,
             182, 183, 183, 184, 184, 185, 185, 186, 187, 188, 188, 189],
            dtype=int)
        fr_indptr = np.array(
            [  0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,
              26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
              52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,
              78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100, 102,
             104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128,
             130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154,
             156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180,
             182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206,
             208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232,
             234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258,
             260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280, 282, 284,
             286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310,
             312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336,
             338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362,
             364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388,
             390, 392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414,
             416, 418, 420, 422, 424, 426, 428], dtype=int)
        fr_data = np.array(
            [-1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1])
        # fmt: on

        # check the face_ridges
        fr_known = sps.csc_array((fr_data, fr_indices, fr_indptr))
        self.assertTrue(np.allclose(sps.find(sd.face_ridges), sps.find(fr_known)))

        # check the cell_volumes
        cv_kown = 0.023565317109780656
        self.assertTrue(np.allclose(sd.cell_volumes, cv_kown))


if __name__ == "__main__":
    unittest.main()
