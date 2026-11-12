"""Module contains tests of the barycentric coordinates of the grid cells."""

import numpy as np
import pytest


def test_shape(unit_sd):
    grads = unit_sd.compute_barycentric_gradients()
    assert grads.shape == (3, unit_sd.dim + 1, unit_sd.num_cells)


def test_gradient_of_coordinates(unit_sd):
    # the coordinates are linear, x = sum_j x_j lambda_j, so their gradients are the
    # identity on the plane of the grid, i.e. the first dim axes
    grads = unit_sd.compute_barycentric_gradients()
    nodes = unit_sd.cell_nodes().indices.reshape((unit_sd.num_cells, -1))
    vertices = unit_sd.nodes[:, nodes]
    known = np.diag(np.arange(3) < unit_sd.dim).astype(float)
    assert np.allclose(np.einsum("acj,bjc->cab", vertices, grads), known)


def test_partition_of_unity(unit_sd):
    # the barycentric coordinates sum to one, so their gradients to zero
    grads = unit_sd.compute_barycentric_gradients()
    assert np.allclose(grads.sum(axis=1), 0)


def test_stored(unit_sd):
    grads = unit_sd.compute_barycentric_gradients()
    assert unit_sd.compute_barycentric_gradients() is grads
    assert unit_sd.compute_barycentric_gradients(recompute=True) is not grads


def test_recomputed_with_geometry(unit_sd):
    # a copy with moved nodes gets the gradients of the new geometry
    grads = unit_sd.compute_barycentric_gradients()
    sd = unit_sd.copy(keep_history=False)
    sd.nodes = 2 * sd.nodes
    sd.compute_geometry()
    assert np.allclose(sd.compute_barycentric_gradients(), grads / 2)


def test_non_simplicial_grid(ref_square):
    with pytest.raises(NotImplementedError):
        ref_square.compute_barycentric_gradients()


def test_0d_grid(ref_sd_0d):
    with pytest.raises(NotImplementedError):
        ref_sd_0d.compute_barycentric_gradients()


def test_is_simplicial(unit_sd):
    assert unit_sd.is_simplicial()


def test_is_simplicial_cartesian(unit_cart_sd):
    # the 1d cartesian grids are simplicial
    assert unit_cart_sd.is_simplicial() == (unit_cart_sd.dim == 1)


def test_is_not_simplicial(ref_square, ref_sd_0d):
    assert not ref_square.is_simplicial()
    assert not ref_sd_0d.is_simplicial()


def test_barycentric_to_coords(unit_sd):
    # The vertices map to the nodes of each cell, and the barycenter to its center
    nodes = unit_sd.cell_nodes().indices.reshape((unit_sd.num_cells, -1))
    vertices = unit_sd.barycentric_to_coords(np.eye(unit_sd.dim + 1))
    assert np.allclose(vertices, np.moveaxis(unit_sd.nodes[:, nodes], 1, 2))

    barycenter = np.full((1, unit_sd.dim + 1), 1 / (unit_sd.dim + 1))
    centers = unit_sd.barycentric_to_coords(barycenter)
    assert np.allclose(centers[:, 0], unit_sd.cell_centers)


def test_barycentric_to_coords_non_simplicial(ref_square):
    with pytest.raises(NotImplementedError):
        ref_square.barycentric_to_coords(np.eye(3))
