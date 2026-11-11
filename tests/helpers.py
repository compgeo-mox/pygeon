import numpy as np
import porepy as pp

import pygeon as pg


def matrix_equals(A: np.ndarray, B: np.ndarray, **kwargs) -> bool:
    """Checks if two matrices are equal in shape and values within a tolerance.

    Args:
        A (np.ndarray): The first matrix to compare.
        B (np.ndarray): The second matrix to compare.
        **kwargs: Additional keyword arguments to pass to np.allclose.

    Returns:
        bool: True if the matrices are equal in shape and values within the specified
        tolerance, False otherwise.

    Note:
        The reason of having this function is to handle empty matrices, which are
        considered equal if they have the same shape, even if np.allclose would return
        True for any two empty matrices regardless of their shape.

    """
    return (A.shape == B.shape) and np.allclose(A, B, **kwargs)


def constant_tensor(sd: pg.Grid, K: np.ndarray) -> pp.SecondOrderTensor:
    """The second order tensor equal to the symmetric matrix K on every cell.

    Args:
        sd (pg.Grid): The grid.
        K (np.ndarray): The symmetric matrix, of shape (3, 3).

    Returns:
        pp.SecondOrderTensor: The tensor.
    """
    ones = np.ones(sd.num_cells)
    return pp.SecondOrderTensor(
        kxx=K[0, 0] * ones,
        kyy=K[1, 1] * ones,
        kzz=K[2, 2] * ones,
        kxy=K[0, 1] * ones,
        kxz=K[0, 2] * ones,
        kyz=K[1, 2] * ones,
    )
