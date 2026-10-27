"""Set membership utilities."""

import numpy as np


def match_coordinates(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compare and match columns of a and b.
    We assume that each column has a match.
    NOTE: This code is not optimized so only use this for "small" matrices.

    Args:
        a (np.ndarray): The first matrix to compare, of shape (m, n).
        b (np.ndarray): The second matrix to compare, of shape (m, n).

    Returns:
        np.ndarray: The indices ind, of shape (n,), such that b[:, ind] = a.
    """
    n = a.shape[1]
    ind = np.empty((n,), dtype=int)
    for i in np.arange(n):
        for j in np.arange(n):
            if np.allclose(a[:, i], b[:, j]):
                ind[i] = j
                break

    return ind
