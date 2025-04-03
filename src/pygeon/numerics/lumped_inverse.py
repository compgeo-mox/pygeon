import numpy as np
import scipy.linalg
import scipy.sparse as sps
import scipy.sparse.csgraph as csgraph


def assemble_inverse(M: sps.csc_array) -> sps.csc_array:
    """
    Assembles the block-wise inverse of a sparse matrix based on connected components.

    This function computes the inverse of a sparse matrix M by dividing it into
    submatrices corresponding to connected components. Each submatrix is inverted
    independently, and the results are assembled into the final inverse matrix.

    Args:
        M (sps.csc_array): A sparse matrix in Compressed Sparse Column (CSC) format.

    Returns:
        sps.csc_array: The block-wise inverse of the input matrix M in CSC format.

    Notes:
        - The function uses the connected components of the graph represented by M
          to determine the blocks for inversion.
        - The inversion is performed using dense matrix operations for each block,
          which may be computationally expensive for large blocks.
    """
    # Get connected components
    n_components, labels = csgraph.connected_components(M, directed=False)

    # Convert M to LIL format for efficient row and column access
    M_lil = M.tolil()
    inv_M_lil = sps.lil_array(M_lil.shape)

    # Iterate over each connected component of the matrix M
    for patch in np.arange(n_components):
        # Get the indices of the connected component
        indices = np.where(labels == patch)[0]

        # Create a submatrix for the connected component
        submat = M_lil[np.ix_(indices, indices)].toarray()
        inv_submat = np.linalg.inv(submat)

        # Store the inverse in the corresponding positions
        inv_M_lil[np.ix_(indices, indices)] = inv_submat

    # Convert the inverse matrix back to CSC format
    return inv_M_lil.tocsc()


def block_diag_solver(M: sps.csc_array, B: sps.csc_array) -> sps.csc_array:
    """
    Solves a block diagonal system of linear equations for each connected component.

    This function takes a sparse block diagonal matrix M assumed to be symmetric and
    positive defined and a right-hand side matrix B, and solves the system MX = B for
    each connected component of the matrix M.

    The connected components are determined using a graph representation of the matrix.

    Args:
        M (sps.csc_array): A symmetric and positive defined sparse matrix in Compressed
            Sparse Column (CSC) format, representing the block diagonal system. It is
            assumed to be symmetric and positive definite.
        B (sps.csc_array): The right-hand side matrix in Compressed Sparse Column (CSC)
            format.

    Returns:
        sps.csc_array: The solution matrix X.

    Notes:
        - The function identifies connected components of the matrix M using the
          connected_components function from `scipy.sparse.csgraph`.
        - For each connected component, the corresponding submatrix and subvector are
          extracted, and the system is solved using `numpy.linalg.solve`.
        - If the right-hand side B is sparse and contains zero entries for a connected
          component, the solution for that component is skipped.
    """
    # Get connected components
    n_components, labels = csgraph.connected_components(M, directed=False)

    # Convert M and B to LIL format for efficient row and column access
    M_lil = M.tolil()
    B_lil = B.tolil()
    sol = sps.lil_array(B.shape)

    # Iterate over each connected component of the matrix M
    for patch in np.arange(n_components):
        # Get the indices of the connected component
        rows = np.where(labels == patch)[0]

        # Create a submatrix for the connected component
        sub_B = B_lil[rows, :]
        # Get the non-zero columns of the submatrix
        cols = np.unique(np.nonzero(sub_B)[1])

        # If there are no non-zero columns, skip this component
        if cols.size == 0:
            continue

        # Create a dense submatrix for the connected component
        sub_M = M_lil[np.ix_(rows, rows)].toarray()

        # Solve the dense system and distribute to the solution matrix
        sol[np.ix_(rows, cols)] = scipy.linalg.solve(
            sub_M, sub_B[:, cols].toarray(), assume_a="pos"
        )

    # Convert the inverse matrix back to CSC format
    return sol.tocsc()


def block_diag_solver_dense(M: sps.csc_array, b: np.ndarray) -> np.ndarray:
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
    sol = block_diag_solver(M, b_csc).toarray()

    # If b was a 1D array, convert the solution back to a 1D array
    if is_b_1d:
        sol = np.squeeze(sol)

    return sol
