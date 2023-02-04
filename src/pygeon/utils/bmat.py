import numpy as np
import scipy.sparse as sps


def replace_nones_with_zeros(M: np.ndarray):
    """
    Replace each None in the block matrix by a zero sparse matrix of the right shape.
    This is done in-place.
    """

    # Do nothing if there are no Nones
    if None not in M:
        return

    # Otherwise, we first find the right shapes
    row_lengths, col_lengths = find_row_col_lengths(M)

    # and then we replace each None that we find with a coo_matrix
    for (i, j), block in np.ndenumerate(M):
        if block is None:
            M[i, j] = sps.coo_matrix((row_lengths[i], col_lengths[j]))


def find_row_col_lengths(M: np.ndarray):
    """
    Find shapes of the blocks in M
    """

    rows = np.zeros(M.shape[0], int)
    cols = np.zeros(M.shape[1], int)

    for (i, j), block in np.ndenumerate(M):
        if block is not None:
            rows[i], cols[j] = block.shape

    return rows, cols


def transpose(M: np.ndarray):
    """
    Compute the transpose of a block matrix.

    This function should be used instead of M.T,
    because M.T does not transpose the blocks themselves.
    """

    # Initialize and loop through all blocks
    M_T = np.empty_like(M.T)

    for (i, j), block in np.ndenumerate(M):
        if block is not None:
            M_T[j, i] = block.T

    return M_T
