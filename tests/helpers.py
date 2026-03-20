import numpy as np


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
