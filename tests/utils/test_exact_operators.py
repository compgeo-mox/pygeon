"""Module contains tests for the symbolic differential operators."""

import numpy as np
import pytest
import sympy as sp

import pygeon as pg


@pytest.fixture
def coords():
    return pg.exact.coordinates()


def test_gradient(coords):
    x, y, z = coords
    grad = pg.exact.gradient(x**2 * y + z)

    assert list(grad) == [2 * x * y, x**2, 1]


def test_divergence(coords):
    x, y, z = coords

    assert pg.exact.divergence(sp.Matrix([x**2, y * z, z])) == 2 * x + z + 1
    assert pg.exact.divergence(sp.Matrix([y, -x, 0])) == 0


def test_curl(coords):
    x, y, z = coords

    assert list(pg.exact.curl(sp.Matrix([y, -x, 0]))) == [0, 0, -2]
    assert list(pg.exact.curl(sp.Matrix([0, 0, x * y]))) == [x, -y, 0]


def test_rotated_gradient(coords):
    x, y, _ = coords

    # the rotated gradient turns the gradient by ninety degrees
    assert list(pg.exact.rotated_gradient(x * y)) == [x, -y, 0]
    assert list(pg.exact.rotated_gradient(x)) == [0, -1, 0]


def test_laplacian(coords):
    x, y, z = coords

    assert pg.exact.laplacian(x**2 * y + z**3) == 2 * y + 6 * z
    assert list(pg.exact.laplacian(sp.Matrix([x**2, y**2, 0]))) == [2, 2, 0]


def test_curl_of_gradient_vanishes(coords):
    x, y, z = coords
    scalar = sp.sin(x * y) + z**3

    assert list(pg.exact.curl(pg.exact.gradient(scalar))) == [0, 0, 0]


def test_divergence_of_curl_vanishes(coords):
    x, y, z = coords
    vector = sp.Matrix([y * z, x**2, sp.cos(z)])

    assert pg.exact.divergence(pg.exact.curl(vector)) == 0


def test_laplacian_is_divergence_of_gradient(coords):
    x, y, z = coords
    scalar = x**3 * y + sp.exp(z)

    assert pg.exact.laplacian(scalar) == pg.exact.divergence(pg.exact.gradient(scalar))


def test_vector_gradient(coords):
    x, y, z = coords
    grad = pg.exact.vector_gradient(sp.Matrix([x * y, z, 0]))

    assert grad.tolist() == [[y, x, 0], [0, 0, 1], [0, 0, 0]]


def test_matrix_divergence(coords):
    x, y, z = coords
    matrix = sp.Matrix([[x**2, y, 0], [0, y * z, 0], [0, 0, z]])

    assert list(pg.exact.matrix_divergence(matrix)) == [2 * x + 1, z, 1]


def test_sym_and_skew(coords):
    x, y, z = coords
    matrix = sp.Matrix([[0, x, 0], [y, 0, 0], [0, 0, z]])

    assert pg.exact.sym(matrix) == pg.exact.sym(matrix).T
    assert pg.exact.skew(matrix) == -pg.exact.skew(matrix).T
    assert pg.exact.skew(matrix) == (matrix - matrix.T) / 2
    assert pg.exact.sym(matrix) + pg.exact.skew(matrix) == matrix


def test_vsk_of_msk(coords):
    x, y, z = coords
    vector = sp.Matrix([x, y, z])

    # vsk collects the entries of the difference with the transpose, hence the two
    assert list(pg.exact.vsk(pg.exact.msk(vector))) == list(2 * vector)


def test_vsk_and_msk_in_two_dimensions(coords):
    x, y, _ = coords
    matrix = sp.Matrix([[x, y, 0], [x * y, 1, 0], [0, 0, 0]])

    # the two-dimensional vsk is the scalar sigma_21 - sigma_12, in the third component
    assert list(pg.exact.vsk(matrix)) == [0, 0, y * (x - 1)]
    assert pg.exact.msk(sp.Matrix([0, 0, x])) == sp.Matrix(
        [[0, -x, 0], [x, 0, 0], [0, 0, 0]]
    )


def test_msk_is_the_cross_product(coords):
    x, y, z = coords
    vector, other = sp.Matrix([x, y, z]), sp.Matrix([z, 1, x * y])

    assert sp.simplify(pg.exact.msk(vector) @ other - vector.cross(other)) == sp.zeros(
        3, 1
    )


def test_vsk_and_msk_are_adjoint(coords):
    x, y, z = coords
    matrix = sp.Matrix([[x, y, z], [y * z, 1, x], [0, x**2, z]])
    vector = sp.Matrix([z, x, y])

    # (vsk sigma) . w = sigma : msk w
    lhs = pg.exact.vsk(matrix).dot(vector)
    rhs = pg.exact.double_dot(matrix, pg.exact.msk(vector))

    assert sp.simplify(lhs - rhs) == 0


def test_sym_skew_decomposition_with_vsk_and_msk(coords):
    x, y, z = coords
    matrix = sp.Matrix([[x, y, z], [y * z, 1, x], [0, x**2, z]])

    # sigma = sym sigma + msk vsk sigma / 2, and vsk vanishes on symmetric matrices
    rebuilt = pg.exact.sym(matrix) + pg.exact.msk(pg.exact.vsk(matrix)) / 2

    assert sp.simplify(rebuilt - matrix) == sp.zeros(3)
    assert pg.exact.vsk(pg.exact.sym(matrix)) == sp.zeros(3, 1)


def test_identity():
    assert pg.exact.identity() == sp.eye(3)
    assert pg.exact.identity(2) == sp.diag(1, 1, 0)


def test_double_dot(coords):
    x, y, z = coords
    matrix = sp.Matrix([[x, y, 0], [0, z, 1], [2, 0, x]])

    assert pg.exact.double_dot(matrix, sp.eye(3)) == matrix.trace()
    assert pg.exact.double_dot(matrix, matrix) == 2 * x**2 + y**2 + z**2 + 5
    # symmetric and skew-symmetric matrices are orthogonal
    assert pg.exact.double_dot(pg.exact.sym(matrix), pg.exact.skew(matrix)) == 0


def test_dev(coords):
    x, y, z = coords
    matrix = sp.Matrix([[x, y, 0], [y, z, 0], [0, 0, x * y]])

    deviator = pg.exact.dev(matrix)

    assert sp.simplify(deviator.trace()) == 0
    assert deviator == deviator.T
    assert pg.exact.dev(sp.eye(3)) == sp.zeros(3)
    # the deviator is a projection
    assert sp.simplify(pg.exact.dev(deviator) - deviator) == sp.zeros(3)


def test_dev_in_two_dimensions(coords):
    x, y, _ = coords
    matrix = sp.Matrix([[x, y, 0], [y, 1, 0], [0, 0, 0]])

    deviator = pg.exact.dev(matrix, 2)

    assert deviator == sp.Matrix([[(x - 1) / 2, y, 0], [y, (1 - x) / 2, 0], [0, 0, 0]])
    assert pg.exact.dev(sp.diag(1, 1, 0), 2) == sp.zeros(3)


def test_to_callable_shapes(coords):
    x, y, z = coords
    pts = np.array([[0.0, 1.0, 2.0], [0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])

    scalar = pg.exact.to_callable(x + y)
    vector = pg.exact.to_callable(sp.Matrix([x, y, z]))
    matrix = pg.exact.to_callable(sp.Matrix([[x, y, 0], [0, z, 0], [0, 0, 1]]))

    assert scalar(pts).shape == (3,)
    assert vector(pts).shape == (3, 3)
    assert matrix(pts).shape == (3, 3, 3)

    assert np.allclose(scalar(pts), pts[0] + pts[1])
    assert np.allclose(vector(pts), pts)


def test_to_callable_at_a_single_point(coords):
    x, y, _ = coords
    point = [0.25, 0.75, 3.0]

    scalar = pg.exact.to_callable(sp.sin(2 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    vector = pg.exact.to_callable(sp.Matrix([x, y, 0]))
    matrix = pg.exact.to_callable(sp.Matrix([[x, 0, 0], [0, y, 0], [0, 0, 1]]))

    # a point carries no axis of its own, so the value of a scalar is a scalar
    assert scalar(point).shape == ()
    assert vector(point).shape == (3,)
    assert matrix(point).shape == (3, 3)

    assert np.isclose(scalar(point), -1.0)
    assert np.allclose(vector(point), [0.25, 0.75, 0.0])


def test_to_callable_rejects_wrong_coordinates(coords):
    x, _, _ = coords

    with pytest.raises(ValueError):
        pg.exact.to_callable(x)([0.25, 0.75])


def test_to_callable_in_two_dimensions(coords):
    x, y, _ = coords
    pts = np.array([[0.0, 1.0], [0.0, 0.5], [0.0, 0.0]])

    vector = pg.exact.to_callable(sp.Matrix([x, y, 0]), 2)
    matrix = pg.exact.to_callable(pg.exact.vector_gradient(sp.Matrix([x * y, y, 0])), 2)

    assert vector(pts).shape == (2, 2)
    assert matrix(pts).shape == (2, 2, 2)
    assert np.allclose(vector(pts), pts[:2])


def test_to_callable_of_constants():
    pts = np.array([[0.0, 1.0, 2.0], [0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])

    # constant entries are not vectorized by sympy, so they are broadcast
    known = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert np.allclose(pg.exact.to_callable(sp.Integer(2))(pts), 2 * np.ones(3))
    assert np.allclose(pg.exact.to_callable(sp.Matrix([1, 0, 0]))(pts), known)


def test_interpolation_of_exact_solution(coords, unit_sd_2d):
    x, y, _ = coords
    scalar = sp.sin(sp.pi * x) * sp.cos(sp.pi * y)

    interp = pg.Lagrange1().interpolate(unit_sd_2d, pg.exact.to_callable(scalar))
    known = np.sin(np.pi * unit_sd_2d.nodes[0]) * np.cos(np.pi * unit_sd_2d.nodes[1])

    assert np.allclose(interp, known)


def test_linear_manufactured_solution_is_reproduced(coords, unit_sd_2d):
    x, y, _ = coords
    pressure = x + 2 * y
    source = -pg.exact.laplacian(pressure)

    assert source == 0

    # the finite elements reproduce a linear solution exactly
    discr = pg.Lagrange1()
    interpolated = discr.interpolate(unit_sd_2d, pg.exact.to_callable(pressure))

    rhs = discr.assemble_mass_matrix(unit_sd_2d) @ discr.interpolate(
        unit_sd_2d, pg.exact.to_callable(source)
    )
    ls = pg.LinearSystem(discr.assemble_stiff_matrix(unit_sd_2d), rhs)
    ls.flag_ess_bc(unit_sd_2d.tags["domain_boundary_nodes"], interpolated)

    assert np.allclose(ls.solve(), interpolated)


def test_manufactured_solution_converges(coords):
    x, y, _ = coords
    pressure = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    source = -pg.exact.laplacian(pressure)

    discr = pg.Lagrange1()
    errors = []
    for mesh_size in (0.2, 0.1):
        sd = pg.unit_grid(2, mesh_size, as_mdg=False, structured=True)
        sd.compute_geometry()

        mass = discr.assemble_mass_matrix(sd)
        rhs = mass @ discr.interpolate(sd, pg.exact.to_callable(source))
        ls = pg.LinearSystem(discr.assemble_stiff_matrix(sd), rhs)
        ls.flag_ess_bc(sd.tags["domain_boundary_nodes"], np.zeros(discr.ndof(sd)))

        diff = ls.solve() - discr.interpolate(sd, pg.exact.to_callable(pressure))
        errors.append(np.sqrt(diff @ (mass @ diff)))

    # halving the mesh size reduces the error by almost the second order
    assert errors[1] < errors[0] / 3


def test_rotated_gradient_matches_the_discrete_differential(coords, unit_sd_2d):
    x, y, _ = coords
    scalar = x + 2 * y  # linear, so that both sides are exact

    discr = pg.Lagrange1()
    discrete = discr.assemble_diff_matrix(unit_sd_2d) @ discr.interpolate(
        unit_sd_2d, pg.exact.to_callable(scalar)
    )
    interpolated = pg.RT0().interpolate(
        unit_sd_2d, pg.exact.to_callable(pg.exact.rotated_gradient(scalar))
    )

    assert np.allclose(discrete, interpolated)


def test_gradient_matches_the_discrete_differential(coords, unit_sd_3d):
    x, y, z = coords
    scalar = 2 * x - y + 3 * z

    discr = pg.Lagrange1()
    discrete = discr.assemble_diff_matrix(unit_sd_3d) @ discr.interpolate(
        unit_sd_3d, pg.exact.to_callable(scalar)
    )
    interpolated = pg.NedelecR0().interpolate(
        unit_sd_3d, pg.exact.to_callable(pg.exact.gradient(scalar))
    )

    assert np.allclose(discrete, interpolated)


def test_curl_matches_the_discrete_differential(coords, unit_sd_3d):
    x, y, _ = coords
    vector = sp.Matrix([-y, x, 0])

    discr = pg.NedelecR0()
    discrete = discr.assemble_diff_matrix(unit_sd_3d) @ discr.interpolate(
        unit_sd_3d, pg.exact.to_callable(vector)
    )
    interpolated = pg.RT0().interpolate(
        unit_sd_3d, pg.exact.to_callable(pg.exact.curl(vector))
    )

    assert np.allclose(discrete, interpolated)


def test_divergence_matches_the_discrete_differential(coords, unit_sd_3d):
    x, y, z = coords
    vector = sp.Matrix([x, 2 * y, -z])

    discr = pg.RT0()
    discrete = discr.assemble_diff_matrix(unit_sd_3d) @ discr.interpolate(
        unit_sd_3d, pg.exact.to_callable(vector)
    )
    interpolated = pg.PwConstants().interpolate(
        unit_sd_3d, pg.exact.to_callable(pg.exact.divergence(vector))
    )

    assert np.allclose(discrete, interpolated)
