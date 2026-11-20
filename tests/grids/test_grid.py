"""Module contains the tests of the Grid class."""

import numpy as np

import pygeon as pg


def test_sorted_topological_maps(unit_sd: pg.Grid):
    # The topological maps have sorted indices, so that abs(), which sorts unsorted
    # indices in place, leaves them unchanged
    for name in ("cell_faces", "face_ridges", "ridge_peaks"):
        conn = getattr(unit_sd, name)
        indices = conn.indices.copy()
        abs(conn)
        assert np.array_equal(conn.indices, indices), f"{name} is changed by abs()"
        assert conn.has_sorted_indices, f"{name} has unsorted indices"
