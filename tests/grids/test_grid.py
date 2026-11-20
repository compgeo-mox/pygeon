"""Module contains the tests of the Grid class."""

import pygeon as pg


def test_sorted_topological_maps(unit_sd: pg.Grid):
    for name in ("cell_faces", "face_ridges", "ridge_peaks"):
        conn = getattr(unit_sd, name)
        assert conn.has_sorted_indices, f"{name} has unsorted indices"
