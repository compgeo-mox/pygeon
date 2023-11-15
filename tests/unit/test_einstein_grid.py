""" Module contains a unit test for the Einstein grid class.
"""
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
        cf_known = sps.csc_matrix((cf_data, cf_indices, cf_indptr))
        self.assertTrue(np.allclose(sps.find(sd.cell_faces), sps.find(cf_known)))

        # face_ridges known data
        # fmt: off
        fr_indices = np.array([ 0,  1,  0, 13,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,
                                8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13], dtype=int)

        fr_indptr = np.array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28], dtype=int)

        fr_data = np.array([-1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,
                            1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1.,
                           -1.,  1.])
        # fmt: on

        # check the face_ridges
        fr_known = sps.csc_matrix((fr_data, fr_indices, fr_indptr))
        self.assertTrue(np.allclose(sps.find(sd.face_ridges), sps.find(fr_known)))

        # check the cell_volumes
        cv_kown = 0.38490018
        self.assertTrue(np.allclose(sd.cell_volumes, cv_kown))


if __name__ == "__main__":
    unittest.main()
