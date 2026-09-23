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
    assert pg.exact.sym(matrix) + pg.exact.skew(matrix) == matrix


def test_asym_of_asym_T(coords):
    x, y, z = coords
    vector = sp.Matrix([x, y, z])

    # asym collects the entries of the difference with the transpose, hence the two
    assert list(pg.exact.asym(pg.exact.asym_T(vector))) == list(2 * vector)


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
    interpolated = pg.Nedelec0().interpolate(
        unit_sd_3d, pg.exact.to_callable(pg.exact.gradient(scalar))
    )

    assert np.allclose(discrete, interpolated)


def test_curl_matches_the_discrete_differential(coords, unit_sd_3d):
    x, y, _ = coords
    vector = sp.Matrix([-y, x, 0])

    discr = pg.Nedelec0()
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
