"""Module contains tests to validate the block diagonal solver."""

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

import pygeon as pg


@pytest.fixture()
def M_sparse() -> sps.csc_array:
    # Create a sparse matrix M
    M = np.array(
        [
            [2, 1, 0, 0, 0],
            [1, 2, 0, 0, 0],
            [0, 0, 6, 1, 0],
            [0, 0, 1, 8, 2],
            [0, 0, 0, 2, 2],
        ]
    )
    return sps.csc_array(M)


@pytest.fixture()
def B_sparse() -> sps.csc_array:
    B = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    return sps.csc_array(B)


@pytest.fixture()
def b_vector() -> np.ndarray:
    return np.array([1, 2, 3, 0, 1])


@pytest.fixture()
def b_matrix(b_vector: np.ndarray) -> np.ndarray:
    return np.tile(b_vector, (3, 1)).T


def test_block_diag_solver_dense_vec(M_sparse, b_vector):
    # Solve the system of equations using the block diagonal solver
    sol = pg.block_diag_solver_dense(M_sparse, b_vector)

    # Check if the solution is correct
    assert np.allclose(M_sparse @ sol, b_vector)

    # Check if the shape of the solution is correct
    assert sol.shape == (5,)

    # Check if the solution is a 1D array
    assert sol.ndim == 1


def test_block_diag_solver_dense_mat(M_sparse, b_matrix):
    # Solve the system of equations using the block diagonal solver
    sol = pg.block_diag_solver_dense(M_sparse, b_matrix)

    # Check if the solution is correct
    assert np.allclose(M_sparse @ sol, b_matrix)

    # Check if the shape of the solution is correct
    assert sol.shape == (5, 3)

    # Check if the solution is a 2D array
    assert sol.ndim == 2


def test_block_diag_solver(M_sparse, B_sparse):
    # Solve the system of equations using the block diagonal solver
    sol = pg.block_diag_solver(M_sparse, B_sparse)

    # Check if the solution is correct
    assert np.allclose((M_sparse @ sol - B_sparse).data, 0)

    # Check if the shape of the solution is correct
    assert np.allclose(sol.shape, (5, 6))

    # Check if the solution is a sparse matrix
    assert sps.issparse(sol)


def test_assemble_inverse(M_sparse):
    # Solve the system of equations using the block diagonal solver
    invM = pg.assemble_inverse(M_sparse)
    expected_invM = np.linalg.inv(M_sparse.toarray())

    # Check if the solution is correct
    assert np.allclose(invM.toarray(), expected_invM)


@pytest.mark.parametrize(
    "data",
    [
        None,
        {
            pp.PARAMETERS: {
                "test": {pg.LAME_MU: 0.5, pg.LAME_LAMBDA: 1.0, pg.LAME_MU_COSSERAT: 1.0}
            }
        },
    ],
)
def test_lumped_inv_cosserat(ref_sd, data):
    if ref_sd.dim == 1:
        return  # There is no Cosserat implementation in 1D

    max_nnz = [0, 0, 52, 235]

    discr = pg.VecRT1("test")

    # check for data and without data, so we use default parameters
    L = discr.assemble_lumped_matrix_cosserat(ref_sd, data)
    L_inv = pg.assemble_inverse(L)

    L_inv.data[np.abs(L_inv.data) < 1e-10] = 0
    L_inv.eliminate_zeros()

    assert L_inv.nnz <= max_nnz[ref_sd.dim]
