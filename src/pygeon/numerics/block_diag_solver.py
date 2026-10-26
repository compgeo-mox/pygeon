"""Solvers for linear systems with block-diagonal structure."""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.csgraph as csgraph


def drop_small_entries(M: sps.sparray, rtol: float = 1e-10) -> sps.sparray:
    """
    Returns a copy of M without the entries whose absolute value is at most rtol
    times the largest absolute value of M.

    Products such as :math:`P^T M P` leave round-off entries (about 1e-17 relative)
    where the exact value is zero. They change neither the matrix nor the solution,
    but they couple degrees of freedom that are otherwise independent, which spoils
    the block structure seen by :func:`pg.assemble_inverse` and the fill-in of the
    sparse direct solvers.

    Args:
        M (sps.sparray): A sparse matrix in a format with explicit entries (CSC, CSR
            or COO).
        rtol (float): Relative tolerance for removing small matrix entries.
            Defaults to 1e-10.

    Returns:
        sps.sparray: The matrix without the small entries, in the format of M.
    """
    M = M.copy()
    if M.nnz > 0:
        M.data[np.abs(M.data) <= rtol * np.abs(M.data).max()] = 0
        M.eliminate_zeros()
    return M


def assemble_inverse(M: sps.csc_array, rtol: float = 1e-10) -> sps.csc_array:
    """
    Assembles the block-wise inverse of a sparse matrix based on connected components.

    Args:
        M (sps.csc_array): A sparse matrix in Compressed Sparse Column (CSC) format.
        rtol (float): Relative tolerance for removing small matrix entries, see
            :func:`pg.drop_small_entries`. M is not modified. Defaults to 1e-10.

    Returns:
        sps.csc_array: The block-wise inverse of the input matrix M in CSC format.

    Notes:
        - The function uses the connected components of the graph represented by M
          to determine the blocks for inversion.
        - The components are many (e.g. one per node patch) and small, so the
          cost of a Python loop over them is dominated by the fixed overhead of each
          sparse slicing call, not by the arithmetic. The blocks are therefore
          grouped by size: each group is a regular (k, s, s) stack that
          np.linalg.inv inverts in a single LAPACK call. The groups are few (the
          distinct patch sizes), so the cost is linear in the size of M.
    """
    # Remove small entries in the matrix
    M = drop_small_entries(M, rtol)

    # Get connected components
    n_patches, labels = csgraph.connected_components(M, directed=False)

    # Sort the indices by component: the component c owns the global indices
    # order[starts[c] : starts[c] + sizes[c]], and local is the position of each
    # index within its component, i.e. its row/column in the dense block
    order = np.argsort(labels, kind="stable")
    sizes = np.bincount(labels, minlength=n_patches)
    starts = np.cumsum(sizes) - sizes
    local = np.empty_like(order)
    local[order] = np.arange(order.size) - np.repeat(starts, sizes)

    # The entries of M as triplets, without duplicates since they are assigned (not
    # summed) into the blocks; row and column of an entry share the component
    M_coo = M.tocoo()
    M_coo.sum_duplicates()
    size_of_entry = sizes[labels[M_coo.row]]

    rows_list, cols_list, data_list = [], [], []
    for size in np.unique(sizes):
        # The components of this size and their position in the stack
        patches = np.flatnonzero(sizes == size)
        stack_id = np.full(n_patches, -1)
        stack_id[patches] = np.arange(patches.size)

        # Populate the stack of dense blocks with entries
        mask = size_of_entry == size
        rows, cols = M_coo.row[mask], M_coo.col[mask]
        blocks = np.zeros((patches.size, size, size))
        blocks[stack_id[labels[rows]], local[rows], local[cols]] = M_coo.data[mask]

        # Inversion of all patches with equal size
        block_invs = np.linalg.inv(blocks)

        # Global indices of each block: the inverse of the block i is a dense s x s
        # matrix, stored row-wise to match the ravel of the inverted stack
        indices = order[starts[patches, None] + np.arange(size)]
        rows_list.append(np.repeat(indices, size, axis=1).ravel())
        cols_list.append(np.tile(indices, size).ravel())
        data_list.append(block_invs.ravel())

    rows_I = np.concatenate(rows_list)
    cols_J = np.concatenate(cols_list)
    data_IJ = np.concatenate(data_list)

    return sps.csc_array((data_IJ, (rows_I, cols_J)), shape=M.shape)


def block_diag_solver(
    M: sps.csc_array, B: sps.csc_array, rtol: float = 1e-10
) -> sps.csc_array:
    """
    Solves a block diagonal system of linear equations for each connected component.

    Args:
        M (sps.csc_array): A symmetric and positive definite sparse matrix in Compressed
            Sparse Column (CSC) format, representing the block diagonal system. It is
            assumed to be symmetric and positive definite.
        B (sps.csc_array): The right-hand side matrix in Compressed Sparse Column (CSC)
            format.
        rtol (float): Relative tolerance for removing small matrix entries.
            Defaults to 1e-10.

    Returns:
        sps.csc_array: The solution matrix X.

    Notes:
        - The function identifies connected components of the matrix M using
          :func:`scipy.sparse.csgraph.connected_components`.
        - For each connected component, the corresponding submatrix is
          inverted with :func:`numpy.linalg.inv`, stacking the blocks of equal size.
        - The solution is the product of the block-wise inverse with B.
    """
    # The solution is the product with the block-wise inverse, which is zero on the
    # components where B vanishes; the sparse product avoids a loop over the
    # components, see assemble_inverse
    return (assemble_inverse(M, rtol) @ B).tocsc()


def block_diag_solver_dense(
    M: sps.csc_array, b: np.ndarray, rtol: float = 1e-10
) -> np.ndarray:
    """
    Solves a system of linear equations where the coefficient matrix M is symmetric
    and positive definite block diagonal, and the right-hand side is given by b a
    numpy array.

    Args:
        M (sps.csc_array): A symmetric and positive definite sparse matrix in Compressed
            Sparse Column (CSC) format representing the block diagonal coefficient
            matrix.
        b (np.ndarray): The right-hand side of the equation. Can be a 1D array (vector)
            or a 2D array (matrix). If 1D, it will be treated as a column vector.
        rtol (float): Relative tolerance for removing small matrix entries.
            Defaults to 1e-10.

    Returns:
        np.ndarray: The solution to the system of equations. If b is a 1D array, the
        solution will also be returned as a 1D array. If b is a 2D array, the
        solution will be returned as a 2D array.
    """
    # If b is a 1D array, convert it to a 2D column vector
    # This is necessary for the solver to work correctly
    is_b_1d = b.ndim == 1
    if is_b_1d:
        b = np.atleast_2d(b).T

    # Transform the right hand side to a sparse matrix
    b_csc = sps.csc_array(b)
    # Compute the solution by using the block diagonal solver
    sol = block_diag_solver(M, b_csc, rtol=rtol).toarray()

    # If b was a 1D array, convert the solution back to a 1D array
    if is_b_1d:
        sol = np.squeeze(sol)

    return sol
