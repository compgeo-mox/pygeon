"""Module contains tests of the gradients of the barycentric coordinates."""

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
