import unittest

import numpy as np
import pygeon as pg
import scipy.sparse as sps


class BlockMatrixTest(unittest.TestCase):
    def create_block_mat(self):
        """
        Creates a single block matrix for testing
        """

        M = np.empty(shape=(3, 3), dtype=sps.sparray)

        # Fill matrix with matrices of different sizes
        for i in np.arange(M.shape[0]):
            for j in np.arange(M.shape[1]):
                M[i, j] = (i + 2 * j) * sps.eye_array(i + 1, j + 1).tocsc()

        # Put nones on the diagonal
        M[[0, 1, 2], [0, 1, 2]] = None

        return M

    def test_row_col_lengths(self):
        """
        Test the row and column lengths of a known matrix
        with a row of Nones.
        """

        M = np.empty(shape=(4, 3), dtype=sps.sparray)
        M[np.ix_([0, 1, 2], [0, 1, 2])] = self.create_block_mat()

        row, col = pg.bmat.find_row_col_lengths(M)

        known_row = np.array([1, 2, 3, 0])
        known_col = np.array([1, 2, 3])

        self.assertTrue(np.all(known_row == row))
        self.assertTrue(np.all(known_col == col))

    def test_replace_nones(self):
        """
        Test the replacement of Nones by zeros
        """
        M = self.create_block_mat()
        pg.bmat.replace_nones_with_zeros(M)

        M_known = np.array(
            [
                [0.0, 2.0, 0.0, 4.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
                [2.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.all(M_known == sps.block_array(M).toarray()))

    def test_mat_addition(self):
        """
        Test if the sum of the bmat is the same as the bmat of the sum
        """

        M = self.create_block_mat()
        pg.bmat.replace_nones_with_zeros(M)

        M_full = sps.block_array(M)

        block_sum = sps.block_array(M + 4 * M)
        full_sum = M_full + 4 * M_full

        assert np.all(block_sum.toarray() == full_sum.toarray())

    def test_transpose(self):
        """
        Test if the transpose of the bmat is the same as the bmat of the transpose
        """

        M = self.create_block_mat()
        M_T = pg.bmat.transpose(M)

        self.assertTrue(
            np.all(sps.block_array(M_T).toarray() == sps.block_array(M).toarray().T)
        )


if __name__ == "__main__":
    unittest.main()
