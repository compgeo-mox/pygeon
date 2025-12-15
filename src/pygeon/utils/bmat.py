"""Block matrix utilities."""

from typing import Tuple

import numpy as np
import scipy.sparse as sps


def replace_nones_with_zeros(mat: np.ndarray) -> None:
    """
    Replace each None in the block matrix by a zero sparse matrix of the right shape.
    This is done in-place.

    Args:
        mat (np.ndarray): The block matrix to modify.

    Returns:
        None: This function modifies the input matrix in-place.
    """
    # Do nothing if there are no Nones
    if None not in mat:
        return

    # Otherwise, we retrieve the row and column lengths for the shapes
    row_lengths, col_lengths = find_row_col_lengths(mat)

    # We then replace each None with a csc_array
    for (i, j), block in np.ndenumerate(mat):
        if block is None:
            mat[i, j] = sps.csc_array((row_lengths[i], col_lengths[j]))


def find_row_col_lengths(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find shapes of the blocks in mat.

    Args:
        mat (np.ndarray): The input matrix.

    Returns:
        tuple: A tuple containing two numpy arrays - rows and cols,
        representing the lengths of rows and columns respectively.
    """
    rows = np.zeros(mat.shape[0], int)
    cols = np.zeros(mat.shape[1], int)

    for (i, j), block in np.ndenumerate(mat):
        if block is not None:
            rows[i], cols[j] = block.shape

    return rows, cols


def transpose(mat: np.ndarray) -> np.ndarray:
    """
    Compute the transpose of a block matrix.

    This function should be used instead of mat.T,
    because mat.T does not transpose the blocks themselves.

    Args:
        mat (np.ndarray): The block matrix to be transposed.

    Returns:
        np.ndarray: The transposed block matrix.
    """
    # Initialize and loop through all blocks
    mat_T = np.empty_like(mat.T)

    for (i, j), block in np.ndenumerate(mat):
        if block is not None:
            mat_T[j, i] = block.T

    return mat_T


def multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the multiplication of two block matrices.
    The @-symbol does not always work because numpy misunderstands the dimensions

    Args:
        A (np.ndarray): The first block matrix in the product.
        B (np.ndarray): The second block matrix in the product.

    Returns:
        np.ndarray: The product block matrix.
    """
    # Make sure the dimensions match
    assert A.shape[1] == B.shape[0]

    # Preallocation
    C = np.empty((A.shape[0], B.shape[1]), dtype=A.dtype)

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i, j] = sum(A[i, k] @ B[k, j] for k in range(B.shape[0]))

    return C
