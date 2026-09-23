"""Symbolic differential operators, to build manufactured solutions.

The operators act on sympy expressions written in the three coordinates returned by
coordinates(), so that scalars, vectors and matrices are always expressed in the
ambient dimension. A two-dimensional problem is recovered by using expressions that
do not depend on the third coordinate, and by asking to_callable for the components
of interest.
"""

from typing import Callable

import numpy as np
import sympy as sp

import pygeon as pg


def coordinates() -> tuple:
    """
    Returns the symbols of the coordinates the operators differentiate with respect to.

    Args:
        None

    Returns:
        tuple: The symbols (x, y, z).
    """
    return sp.symbols("x y z")


def gradient(scalar: sp.Expr) -> sp.Matrix:
    r"""
    Computes the gradient :math:`\nabla f` of a scalar function.

    Args:
        scalar (sp.Expr): The scalar function.

    Returns:
        sp.Matrix: The gradient, a vector of size three.
    """
    return sp.simplify(sp.Matrix([sp.diff(scalar, var) for var in coordinates()]))


def divergence(vector: sp.Matrix) -> sp.Expr:
    r"""
    Computes the divergence :math:`\nabla \cdot u` of a vector function.

    Args:
        vector (sp.Matrix): The vector function, of size three.

    Returns:
        sp.Expr: The divergence.
    """
    return sp.simplify(
        sum(sp.diff(vector[i], var) for i, var in enumerate(coordinates()))
    )


def curl(vector: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the curl :math:`\nabla \times u` of a vector function, the rotor of a
    three-dimensional problem. It matches the differential of Nedelec0, while for a
    two-dimensional problem the third component is the scalar rotor of the vector.

    Args:
        vector (sp.Matrix): The vector function, of size three.

    Returns:
        sp.Matrix: The curl, a vector of size three.
    """
    x, y, z = coordinates()
    return sp.simplify(
        sp.Matrix(
            [
                sp.diff(vector[2], y) - sp.diff(vector[1], z),
                sp.diff(vector[0], z) - sp.diff(vector[2], x),
                sp.diff(vector[1], x) - sp.diff(vector[0], y),
            ]
        )
    )


def rotated_gradient(scalar: sp.Expr) -> sp.Matrix:
    r"""
    Computes the rotated gradient :math:`\nabla^\perp f` of a scalar function, the
    differential of a two-dimensional problem. It is the curl of the vector whose
    third component is the given function, and it matches the differential of
    Lagrange1 in two dimensions.

    Args:
        scalar (sp.Expr): The scalar function.

    Returns:
        sp.Matrix: The rotated gradient, a vector of size three.
    """
    return curl(sp.Matrix([0, 0, scalar]))


def laplacian(function: sp.Expr | sp.Matrix) -> sp.Expr | sp.Matrix:
    r"""
    Computes the Laplacian :math:`\nabla \cdot \nabla f` of a scalar function, or the
    componentwise Laplacian of a vector function.

    Args:
        function (sp.Expr | sp.Matrix): The scalar or vector function.

    Returns:
        sp.Expr | sp.Matrix: The Laplacian, of the same type as the input.
    """
    if isinstance(function, sp.MatrixBase):
        return sp.simplify(sp.Matrix([laplacian(entry) for entry in function]))
    return divergence(gradient(function))


def vector_gradient(vector: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the gradient :math:`\nabla u` of a vector function, the matrix whose
    row i is the gradient of the component i.

    Args:
        vector (sp.Matrix): The vector function, of size three.

    Returns:
        sp.Matrix: The gradient, a matrix of size three by three.
    """
    return sp.simplify(sp.Matrix([gradient(entry).T for entry in vector]))


def matrix_divergence(matrix: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the divergence :math:`\nabla \cdot \sigma` of a matrix function, the
    vector whose entry i is the divergence of the row i.

    Args:
        matrix (sp.Matrix): The matrix function, of size three by three.

    Returns:
        sp.Matrix: The divergence, a vector of size three.
    """
    return sp.simplify(sp.Matrix([divergence(matrix.row(i).T) for i in range(3)]))


def sym(matrix: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the symmetric part :math:`(\sigma + \sigma^\top) / 2` of a matrix.

    Args:
        matrix (sp.Matrix): The matrix, of size three by three.

    Returns:
        sp.Matrix: The symmetric part.
    """
    return sp.simplify((matrix + matrix.T) / 2)


def skew(matrix: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the skew-symmetric part :math:`(\sigma - \sigma^\top) / 2` of a matrix.

    Args:
        matrix (sp.Matrix): The matrix, of size three by three.

    Returns:
        sp.Matrix: The skew-symmetric part.
    """
    return sp.simplify((matrix - matrix.T) / 2)


def asym(matrix: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the axial vector of the skew-symmetric part of a matrix, the inverse of
    asym_T up to the factor two.

    Args:
        matrix (sp.Matrix): The matrix, of size three by three.

    Returns:
        sp.Matrix: The axial vector, of size three.
    """
    diff = matrix - matrix.T
    return sp.simplify(sp.Matrix([diff[2, 1], diff[0, 2], diff[1, 0]]))


def asym_T(vector: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the skew-symmetric matrix having the given vector as axial vector, so
    that the matrix times a vector is the cross product of the two vectors.

    Args:
        vector (sp.Matrix): The axial vector, of size three.

    Returns:
        sp.Matrix: The skew-symmetric matrix, of size three by three.
    """
    return sp.simplify(
        sp.Matrix(
            [
                [0, -vector[2], vector[1]],
                [vector[2], 0, -vector[0]],
                [-vector[1], vector[0], 0],
            ]
        )
    )


def to_callable(
    expression: sp.Expr | sp.Matrix, dim: int = pg.AMBIENT_DIM
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Turns a symbolic expression into a function of the coordinates, vectorized as the
    discretizations expect it: the function takes an array of shape (3, n) with the
    coordinates in its columns and returns the values with the points along the last
    axis.

    The first dim components of a vector, and the leading dim by dim block of a
    matrix, are retained, so that a two-dimensional problem is obtained by passing
    dim = 2.

    Args:
        expression (sp.Expr | sp.Matrix): The scalar, vector or matrix expression.
        dim (int): The number of components to retain. Default pg.AMBIENT_DIM.

    Returns:
        Callable[[np.ndarray], np.ndarray]: The function evaluating the expression.
    """
    symbols = coordinates()

    if isinstance(expression, sp.MatrixBase):
        rows, cols = expression.shape
        if cols == 1:  # a vector
            expression = expression[:dim, :]
            shape: tuple = (dim,)
        else:  # a matrix
            expression = expression[:dim, :dim]
            shape = (dim, dim)
        entries = [sp.lambdify(symbols, entry, "numpy") for entry in expression]
    else:
        shape = ()
        entries = [sp.lambdify(symbols, expression, "numpy")]

    def evaluate(coords: np.ndarray) -> np.ndarray:
        num_pts = np.shape(coords)[-1]
        # constant entries are not vectorized by lambdify, so they are broadcast
        values = [
            np.broadcast_to(np.asarray(entry(*coords), dtype=float), (num_pts,))
            for entry in entries
        ]
        return np.reshape(values, shape + (num_pts,))

    return evaluate
