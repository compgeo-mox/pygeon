import numpy as np
import scipy.sparse as sps
import scipy.sparse.csgraph as csgraph
from typing import Union


import pygeon as pg


def assemble_inverse(M: sps.csc_array) -> sps.csc_array:
    """
    Assembles the block-wise inverse of a sparse matrix based on connected components.

    This function computes the inverse of a sparse matrix `M` by dividing it into
    submatrices corresponding to connected components. Each submatrix is inverted
    independently, and the results are assembled into the final inverse matrix.

    Args:
        M (sps.csc_array): A sparse matrix in Compressed Sparse Column (CSC) format.

    Returns:
        sps.csc_array: The block-wise inverse of the input matrix `M` in CSC format.

    Raises:
        AssertionError: If the number of connected components in `M` does not match
                        the number of nodes in the mesh `sd`.

    Notes:
        - The function uses the connected components of the graph represented by `M`
          to determine the blocks for inversion.
        - The inversion is performed using dense matrix operations for each block,
          which may be computationally expensive for large blocks.
    """
    # Get connected components
    n_components, labels = csgraph.connected_components(M, directed=False)

    M_lil = M.tolil()
    inv_M_lil = sps.lil_array(M_lil.shape)

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

    This function takes a sparse block diagonal matrix `M` and a right-hand side vector or matrix `b`,
    and solves the system `Mx = b` for each connected component of the matrix `M`. The connected
    components are determined using a graph representation of the matrix.

    Parameters:
        M (sps.csc_array): A sparse matrix in Compressed Sparse Column (CSC) format, representing
            the block diagonal system. It is assumed to be symmetric and positive definite.
        b (Union[np.ndarray, sps.csc_array]): The right-hand side vector or matrix. It can be either
            a dense NumPy array or a sparse CSC array.

    Returns:
        Union[np.ndarray, sps.csc_array]: The solution vector or matrix `x` in the same format as `b`.
            If `b` is a dense NumPy array, the solution will be a NumPy array. If `b` is a sparse
            CSC array, the solution will also be returned as a sparse CSC array.

    Notes:
        - The function identifies connected components of the matrix `M` using the `connected_components`
          function from `scipy.sparse.csgraph`.
        - For each connected component, the corresponding submatrix and subvector are extracted, and
          the system is solved using `numpy.linalg.solve`.
        - If the right-hand side `b` is sparse and contains zero entries for a connected component,
          the solution for that component is skipped.
    """
    # Get connected components
    n_components, labels = csgraph.connected_components(M, directed=False)

    M_lil = M.tolil()
    B_lil = B.tolil()
    sol = sps.lil_array(B.shape)

    for patch in np.arange(n_components):
        # Get the indices of the connected component
        rows = np.where(labels == patch)[0]

        sub_B = B_lil[rows, :]
        cols = np.unique(np.nonzero(sub_B)[1])

        if not np.any(cols):
            continue

        # Create a submatrix for the connected component
        sub_M = M_lil[np.ix_(rows, rows)].toarray()
        sol[np.ix_(rows, cols)] = np.linalg.solve(sub_M, sub_B[:, cols].toarray())

    # Convert the inverse matrix back to CSC format
    return sol.tocsc()
