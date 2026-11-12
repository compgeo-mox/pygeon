"""Module contains tests of the Bernstein basis on the reference simplex."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture(params=[(dim, order) for dim in [1, 2, 3] for order in range(5)])
def basis(request):
    return pg.BernsteinBasis(*request.param)


def random_points(dim, num=7):
    # Points in barycentric coordinates
    pts = np.random.default_rng(0).random((num, dim + 1))
    return pts / pts.sum(axis=1, keepdims=True)


def test_partition_of_unity(basis):
    values = basis.eval(random_points(basis.dim))
    assert np.allclose(values.sum(axis=1), 1)


def test_change_of_basis(basis):
    assert np.allclose(
        basis.to_lagrange() @ basis.to_bernstein(),
        np.eye(len(basis.multi_indices)),
    )
    if basis.order <= 1:
        # The Bernstein and the Lagrange bases coincide
        assert np.allclose(basis.to_lagrange(), np.eye(len(basis.multi_indices)))


def test_mass(basis):
    # The integral of the partition of unity is the measure of the simplex
    assert np.isclose(basis.mass(basis).sum(), 1)


def test_derivatives(basis):
    if basis.order == 0:
        return
    # (sum_i lambda_i)^k, with all the coefficients equal to one, has the
    # derivatives k (sum_i lambda_i)^(k - 1)
    ones = np.ones(len(basis.multi_indices))
    for der in basis.derivatives():
        assert np.allclose(der @ ones, basis.order)


def test_multi_indices_order():
    # The order 3 on a triangle, as in the docstring of pg.PwPolynomials: the
    # vertices, the edges x0 x1, x0 x2, x1 x2, and the barycenter
    known = [
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [0, 2, 1],
        [0, 1, 2],
        [1, 1, 1],
    ]
    assert np.array_equal(pg.BernsteinBasis.compute_multi_indices(2, 3), known)
