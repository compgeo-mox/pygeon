"""Symbolic differential operators, to build manufactured solutions.

The operators act on sympy expressions written in the three coordinates returned by
coordinates(), so that scalars, vectors and matrices are always expressed in the
ambient dimension. A two-dimensional problem is recovered by using expressions that
do not depend on the third coordinate, and by asking to_callable for the components
of interest. The time derivatives, as the material and the objective ones, act on the
symbol returned by time().
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


def time() -> sp.Symbol:
    """
    Returns the symbol of the time, the variable of the time derivatives.

    Args:
        None

    Returns:
        sp.Symbol: The symbol t.
    """
    return sp.Symbol("t")


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
    three-dimensional problem. It matches the differential of NedelecR0, while for a
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


def matrix_curl(matrix: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the curl of a matrix function, the matrix whose row i is the curl of the
    row i. For a two-dimensional problem the third column holds the scalar rotor of
    the rows. Its divergence vanishes, as matrix_divergence acts on the rows.

    Args:
        matrix (sp.Matrix): The matrix function, of size three by three.

    Returns:
        sp.Matrix: The curl, a matrix of size three by three.
    """
    return sp.Matrix.vstack(*[curl(matrix.row(i).T).T for i in range(3)])


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
    Computes the skew-symmetric part :math:`(\sigma - \sigma^\top) / 2 =
    \frac{1}{2} \operatorname{msk} \operatorname{vsk} \sigma` of a matrix.

    Args:
        matrix (sp.Matrix): The matrix, of size three by three.

    Returns:
        sp.Matrix: The skew-symmetric part.
    """
    return msk(vsk(matrix)) / 2


def vsk(matrix: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the vector :math:`\operatorname{vsk} \sigma` collecting the entries of
    :math:`\sigma - \sigma^\top`, so that it vanishes for symmetric matrices. In two
    dimensions the scalar :math:`\sigma_{21} - \sigma_{12}` is the third component.
    It is the adjoint of msk, and :math:`\sigma = \operatorname{sym} \sigma +
    \frac{1}{2} \operatorname{msk} \operatorname{vsk} \sigma`.

    Args:
        matrix (sp.Matrix): The matrix, of size three by three.

    Returns:
        sp.Matrix: The vector, of size three.
    """
    diff = matrix - matrix.T
    return sp.simplify(sp.Matrix([diff[2, 1], diff[0, 2], diff[1, 0]]))


def msk(vector: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the skew-symmetric matrix :math:`\operatorname{msk} w` such that the
    matrix times a vector is the cross product of w with that vector. In two
    dimensions only the third component of w enters the leading two by two block.
    It is the adjoint of vsk.

    Args:
        vector (sp.Matrix): The vector, of size three.

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


def identity(dim: int = pg.AMBIENT_DIM) -> sp.Matrix:
    """
    Returns the identity of the leading dim by dim block, zero elsewhere, so that a
    two-dimensional problem has no entries outside that block.

    Args:
        dim (int): The dimension of the problem. Default pg.AMBIENT_DIM.

    Returns:
        sp.Matrix: The identity, of size three by three.
    """
    if not 1 <= dim <= pg.AMBIENT_DIM:
        raise ValueError(f"The dimension must be between 1 and {pg.AMBIENT_DIM}.")
    return sp.diag(*([1] * dim + [0] * (pg.AMBIENT_DIM - dim)))


def double_dot(matrix: sp.Matrix, other: sp.Matrix) -> sp.Expr:
    r"""
    Computes the double dot product :math:`\sigma : \tau = \sum_{ij} \sigma_{ij}
    \tau_{ij}` of two matrices.

    Args:
        matrix (sp.Matrix): The first matrix, of size three by three.
        other (sp.Matrix): The second matrix, of size three by three.

    Returns:
        sp.Expr: The double dot product.
    """
if matrix.shape != other.shape:
        raise ValueError(
            f"The matrices must have the same shape, got {matrix.shape} and {other.shape}."
        )
    return sp.simplify(sum(a * b for a, b in zip(matrix, other)))


def dev(matrix: sp.Matrix, dim: int = pg.AMBIENT_DIM) -> sp.Matrix:
    r"""
    Computes the deviatoric part :math:`\sigma - \frac{1}{d} \operatorname{Tr}(\sigma)
    I` of a matrix, with I the identity of the leading dim by dim block, so that a
    two-dimensional matrix stays zero outside that block.

    Args:
        matrix (sp.Matrix): The matrix, of size three by three.
        dim (int): The dimension d of the problem. Default pg.AMBIENT_DIM.

    Returns:
        sp.Matrix: The deviatoric part, of size three by three.
    """
    return sp.simplify(matrix - matrix.trace() / dim * identity(dim))


def symmetric_gradient(vector: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the symmetric gradient :math:`\epsilon(u) = \operatorname{sym} \nabla u`
    of a vector function.

    Args:
        vector (sp.Matrix): The vector function, of size three.

    Returns:
        sp.Matrix: The symmetric gradient, of size three by three.
    """
    return sym(vector_gradient(vector))


def spin_tensor(vector: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the spin tensor :math:`\Omega(u) = \operatorname{skew} \nabla u` of a
    vector function, so that :math:`\nabla u = \epsilon(u) + \Omega(u)`.

    Args:
        vector (sp.Matrix): The vector function, of size three.

    Returns:
        sp.Matrix: The spin tensor, of size three by three.
    """
    return skew(vector_gradient(vector))


def time_derivative(function: sp.Expr | sp.Matrix) -> sp.Expr | sp.Matrix:
    r"""
    Computes the partial derivative :math:`\partial_t f` in time, the symbol returned
    by time(), componentwise for vectors and matrices.

    Args:
        function (sp.Expr | sp.Matrix): The scalar, vector or matrix function.

    Returns:
        sp.Expr | sp.Matrix: The time derivative, of the same type as the input.
    """
    return sp.simplify(sp.diff(function, time()))


def advection(
    function: sp.Expr | sp.Matrix, velocity: sp.Matrix
) -> sp.Expr | sp.Matrix:
    r"""
    Computes the advective term :math:`(u \cdot \nabla) f = \nabla f \cdot u` of a
    function transported by the velocity u, componentwise for vectors and matrices.

    Args:
        function (sp.Expr | sp.Matrix): The scalar, vector or matrix function.
        velocity (sp.Matrix): The velocity, of size three.

    Returns:
        sp.Expr | sp.Matrix: The advective term, of the same type as the input.
    """
    if isinstance(function, sp.MatrixBase):
        return function.applyfunc(lambda entry: advection(entry, velocity))
    return sp.simplify(gradient(function).dot(velocity))


def material_derivative(
    function: sp.Expr | sp.Matrix, velocity: sp.Matrix
) -> sp.Expr | sp.Matrix:
    r"""
    Computes the material derivative :math:`D_t f = \partial_t f + (u \cdot \nabla) f`
    of a function transported by the velocity u, the sum of time_derivative and
    advection, componentwise for vectors and matrices.

    Args:
        function (sp.Expr | sp.Matrix): The scalar, vector or matrix function.
        velocity (sp.Matrix): The velocity, of size three.

    Returns:
        sp.Expr | sp.Matrix: The material derivative, of the same type as the input.
    """
    return sp.simplify(time_derivative(function) + advection(function, velocity))


def upper_convected_derivative(matrix: sp.Matrix, velocity: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the upper-convected derivative :math:`D_t \tau - (\nabla u) \tau - \tau
    (\nabla u)^\top` of a matrix transported by the velocity u.

    Args:
        matrix (sp.Matrix): The matrix function, of size three by three.
        velocity (sp.Matrix): The velocity, of size three.

    Returns:
        sp.Matrix: The upper-convected derivative, of size three by three.
    """
    grad = vector_gradient(velocity)
    return sp.simplify(
        material_derivative(matrix, velocity) - grad @ matrix - matrix @ grad.T
    )


def lower_convected_derivative(matrix: sp.Matrix, velocity: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the lower-convected derivative :math:`D_t \tau + (\nabla u)^\top \tau +
    \tau \nabla u` of a matrix transported by the velocity u.

    Args:
        matrix (sp.Matrix): The matrix function, of size three by three.
        velocity (sp.Matrix): The velocity, of size three.

    Returns:
        sp.Matrix: The lower-convected derivative, of size three by three.
    """
    grad = vector_gradient(velocity)
    return sp.simplify(
        material_derivative(matrix, velocity) + grad.T @ matrix + matrix @ grad
    )


def jaumann_derivative(matrix: sp.Matrix, velocity: sp.Matrix) -> sp.Matrix:
    r"""
    Computes the Jaumann, or co-rotational, derivative :math:`D_t \tau + \tau \Omega -
    \Omega \tau` of a matrix transported by the velocity u, with :math:`\Omega` the
    spin tensor of u.

    Args:
        matrix (sp.Matrix): The matrix function, of size three by three.
        velocity (sp.Matrix): The velocity, of size three.

    Returns:
        sp.Matrix: The Jaumann derivative, of size three by three.
    """
    spin = spin_tensor(velocity)
    return sp.simplify(
        material_derivative(matrix, velocity) + matrix @ spin - spin @ matrix
    )


def to_callable(
    expression: sp.Expr | sp.Matrix, dim: int = pg.AMBIENT_DIM, t: float | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Turns a symbolic expression into a function of the coordinates, vectorized as the
    discretizations expect it: the function takes an array of shape (3, n) with the
    coordinates in its columns and returns the values with the points along the last
    axis. A single point is passed as a vector of three coordinates, and the value is
    then returned without that axis.

    The first dim components of a vector, and the leading dim by dim block of a
    matrix, are retained, so that a two-dimensional problem is obtained by passing
    dim = 2.

    An expression depending on the time, the symbol returned by time(), is evaluated
    at the given time t, so that a function of the coordinates is obtained at each
    time step. An expression independent of time needs no t.

    Args:
        expression (sp.Expr | sp.Matrix): The scalar, vector or matrix expression.
        dim (int): The number of components to retain. Default pg.AMBIENT_DIM.
        t (float | None): The time at which the expression is evaluated. Default None.

    Returns:
        Callable[[np.ndarray], np.ndarray]: The function evaluating the expression.

    Raises:
        ValueError: If the expression depends on the time and no t is given.
    """
    symbols = coordinates()

    if t is not None:
        expression = expression.subs(time(), t)
    if time() in expression.free_symbols:
        raise ValueError(
            f"The expression depends on the time {time()}, pass the time t at which "
            "it is evaluated."
        )

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
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] != pg.AMBIENT_DIM:
            raise ValueError(
                f"The coordinates must be given as {pg.AMBIENT_DIM} rows, while an "
                f"array of shape {coords.shape} was passed."
            )

        # a single point is a vector of coordinates, several ones are the columns of
        # an array, and the values follow the same convention
        pts_shape = coords.shape[1:]

        # constant entries are not vectorized by lambdify, so they are broadcast
        values = [
            np.broadcast_to(np.asarray(entry(*coords), dtype=float), pts_shape)
            for entry in entries
        ]
        return np.reshape(values, shape + pts_shape)

    return evaluate
