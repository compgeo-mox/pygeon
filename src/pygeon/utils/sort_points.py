import numpy as np


def argsort_ccw_convex(pts: np.ndarray) -> np.ndarray:
    """
    Sorts the given points in counterclockwise order based on their angles with respect to the
    centroid.

    Args:
        pts (np.ndarray): Array of points with shape (N, 2), where N is the number of points.

    Returns:
        np.ndarray: Array of indices that represents the sorted order of the points in
            counterclockwise direction.

    """
    # Compute the centroid and the radii of the points
    centre = np.mean(pts, axis=0)
    r = np.linalg.norm(pts - centre, axis=1)

    # Compute the angles of the points with respect to the centroid
    angles = np.where(
        (pts[:, 1] - centre[1]) > 0,
        np.arccos((pts[:, 0] - centre[0]) / r),
        2 * np.pi - np.arccos((pts[:, 0] - centre[0]) / r),
    )

    # Sort the points based on the angles
    return np.argsort(angles)
