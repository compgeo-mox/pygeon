import numpy as np
import porepy as pp
import pytest

import pygeon as pg

"""
Module contains tests to validate the edge length computations for grids
"""


def test_grid_0d(ref_sd_0d):
    assert len(ref_sd_0d.edge_tangents) == 0
    assert len(ref_sd_0d.edge_lengths) == 0
    assert ref_sd_0d.mesh_size == 0


def test_cartgrids(unit_cart_sd):
    h = unit_cart_sd.edge_lengths[0]
    assert np.allclose(unit_cart_sd.edge_lengths, h)
    assert np.allclose(unit_cart_sd.mesh_size, h)


def test_grid_2d_tris():
    sd = pp.StructuredTriangleGrid([1] * 2)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    unique_lengths = np.unique(sd.edge_lengths)
    known_lengths = np.sqrt(np.arange(1, 3))

    assert np.allclose(unique_lengths, known_lengths)
    assert np.allclose(sd.edge_lengths, sd.face_areas)
    assert np.allclose(np.sum(sd.edge_tangents * sd.face_normals, axis=0), 0)

    known_meshsize = (4 + np.sqrt(2)) / 5
    assert np.allclose(sd.mesh_size, known_meshsize)


def test_grid_3d_tets():
    sd = pp.StructuredTetrahedralGrid([1] * 3)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    unique_lengths = np.unique(sd.edge_lengths)
    known_lengths = np.sqrt(np.arange(1, 4))

    assert np.allclose(unique_lengths, known_lengths)

    known_meshsize = (12 + 6 * np.sqrt(2) + np.sqrt(3)) / 19
    assert np.allclose(sd.mesh_size, known_meshsize)
