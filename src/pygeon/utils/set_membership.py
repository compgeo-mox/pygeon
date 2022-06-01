import numpy as np


def match_coordinates(a, b):
    """
    Compare and match columns of a and b.
    We assume that each column has a match.
    NOTE: This code is not optimized so only use this for "small" matrices.

    Parameters:
        a (np.array, m x n)
        b (np.array, m x n)

    return:
        np.array, (n, ): The indices ind such that b[:, ind] = a

    """
    n = a.shape[1]
    ind = np.empty((n,), dtype=int)
    for i in np.arange(n):
        for j in np.arange(n):
            if np.allclose(a[:, i], b[:, j]):
                ind[i] = j
                break

    return ind
