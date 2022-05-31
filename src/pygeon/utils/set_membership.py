import numpy as np


def match_coordinates(a, b):
    """
    Compare and match columns of a and b
    We assume that each column has a match
    and a and b match in shape

    return: ind s.t. b[ind] = a

    """
    n = a.shape[1]
    ind = np.empty((n,), dtype=int)
    for i in np.arange(n):
        for j in np.arange(n):
            if np.allclose(a[:, i], b[:, j]):
                ind[i] = j
                break

    return ind
