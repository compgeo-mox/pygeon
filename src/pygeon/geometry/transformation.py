"""This module contains functions for computing transformation matrices."""

import numpy as np


def rotation(vect: np.ndarray) -> np.ndarray:
    """
    Compute the rotation matrix for a vector.

    Args:
        vect (np.ndarray): The input vector.

    Returns:
        np.ndarray: The rotation matrix.
    """
    d = np.linalg.norm(vect)
    dx, dy, dz = vect

    dxy = dx * dx + dy * dy
    r0 = (dx * dx * dz / d + dy * dy) / dxy
    r1 = dx * dy * (dz / d - 1) / dxy
    r2 = (dy * dy * dz / d + dx * dx) / dxy

    return np.array([[r0, r1, -dx / d], [r1, r2, -dy / d], [dx / d, dy / d, dz / d]])


def scaling(vect: np.ndarray) -> np.ndarray:
    """
    Returns a scaling matrix based on the given vector.

    Args:
        vect (np.ndarray): The vector containing scaling factors for each dimension.

    Returns:
        np.ndarray: The scaling matrix.
    """
    return np.diag(vect)
