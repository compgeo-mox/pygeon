import unittest

import numpy as np
import pygeon as pg
import scipy.sparse as sps


class DummyTest(unittest.TestCase):
    def create_block_mat(self):
        """
        Creates a single block matrix for testing
        """

        M = np.empty(shape=(3, 3), dtype=sps.spmatrix)

        # Fill matrix with matrices of different sizes
        for i in np.arange(M.shape[0]):
            for j in np.arange(M.shape[1]):
                M[i, j] = (i + 2) * sps.eye(i + 1, j + 1, format="csc")

        # Put nones on the diagonal
        M[[0, 1, 2], [0, 1, 2]] = None

        return M

    def test_replace_nones(self):
        """
        Test the replacement of Nones by zeros
        """
        M = self.create_block_mat()
        pg.bmat.replace_nones_with_zeros(M)

        M_known = np.array(
            [
                [0.0, 2.0, 0.0, 2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                [4.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.all(M_known == sps.bmat(M).A))

    def test_mat_multiplication(self):
        """
        Test if the produce of the bmat is the same as the bmat of the product
        """

        M = self.create_block_mat()
        pg.bmat.replace_nones_with_zeros(M)

        M_full = sps.bmat(M)

        block_prod = sps.bmat(M @ M)
        full_prod = M_full @ M_full

        assert np.all(block_prod.A == full_prod.A)

    def test_transpose(self):
        """
        Test if the transpose of the bmat is the same as the bmat of the transpose
        """

        M = self.create_block_mat()
        M_T = pg.bmat.transpose(M)

        self.assertTrue(np.all(sps.bmat(M_T).A == sps.bmat(M).A.T))


if __name__ == "__main__":
    unittest.main()
