"""Module contains tests of the Bernstein basis on the reference simplex."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture(params=[(dim, order) for dim in [1, 2, 3] for order in range(5)])
def basis(request):
    return pg.BernsteinBasis(*request.param)


def random_points(dim, num=7):
    # points in barycentric coordinates
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
        # the Bernstein and the Lagrange bases coincide
        assert np.allclose(basis.to_lagrange(), np.eye(len(basis.multi_indices)))


def test_mass(basis):
    # the integral of the partition of unity is the measure of the simplex
    assert np.isclose(basis.mass(basis).sum(), 1)


def test_derivatives(basis):
    if basis.order == 0:
        return
    # (sum_i lambda_i)^k, with all the coefficients equal to one, has the
    # derivatives k (sum_i lambda_i)^(k - 1)
    ones = np.ones(len(basis.multi_indices))
    for der in basis.derivatives():
        assert np.allclose(der @ ones, basis.order)
