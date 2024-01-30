""" This module contains functions for computing the differential operators. """

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg

"""
Acknowledgements:
    The functionalities related to the curl computations are modified from
    github.com/anabudisa/md_aux_precond developed by Ana Budiša and Wietse M. Boon.
"""

# ---------------------------------- Aliases ---------------------------------- #


def div(grid, **kwargs):
    """
    Compute the divergence.

    Parameters:
        grid (pp.Grid, pp.MortarGrid, or pp.MixedDimensionalGrid).
        kwargs: Optional parameters
            as_bmat: In case of mixed-dimensional, return the matrix as sparse sub-blocks.
                Default False.

    Returns:
        sps.csr_matrix. The divergence operator.
    """
    return exterior_derivative(grid, 1, **kwargs)


def curl(grid, **kwargs):
    """
    Compute the curl.

    Parameters:
        grid (pp.Grid, pp.MortarGrid, or pp.MixedDimensionalGrid).
        kwargs: Optional parameters
            as_bmat: In case of mixed-dimensional, return the matrix as sparse sub-blocks.
                Default False.

    Returns:
        sps.csr_matrix. The curl operator.
    """
    return exterior_derivative(grid, 2, **kwargs)


def grad(grid, **kwargs):
    """
    Compute the gradient.

    Parameters:
        grid (pp.Grid, pp.MortarGrid, or pp.MixedDimensionalGrid).
        kwargs: Optional parameters
            as_bmat: In case of mixed-dimensional, return the matrix as sparse sub-blocks.
                Default False.

    Returns:
        sps.csr_matrix. The gradient operator.
    """
    return exterior_derivative(grid, 3, **kwargs)


# --------------------------- MD exterior derivative --------------------------- #


def exterior_derivative(grid, n_minus_k, **kwargs):
    """
    Compute the (mixed-dimensional) exterior derivative for the differential forms of
    order n - k.

    Parameters:
        grid (pp.Grid, pp.MortarGrid, or pp.MixedDimensionalGrid).
        n_minus_k (int): The difference between the ambient dimension and the order of the
            differential form.

    Returns:
        sps.csr_matrix. The differential operator.
    """

    if isinstance(grid, (pp.Grid, pp.MortarGrid)):
        return _g_exterior_derivative(grid, n_minus_k, **kwargs)

    elif isinstance(grid, pp.MixedDimensionalGrid):
        return _mdg_exterior_derivative(grid, n_minus_k, **kwargs)

    else:
        raise TypeError(
            "Input needs to be of type pp.Grid, pp.MortarGrid, or pp.MixedDimensionalGrid"
        )


def _g_exterior_derivative(grid, n_minus_k, **kwargs):
    """
    Compute the exterior derivative on a grid.

    Parameters:
        grid (pp.Grid or pp.MortarGrid): The grid.
        n_minus_k (int): The difference between the ambient dimension and the order of the
            differential form.

    Returns:
        sps.csr_matrix. The differential operator.
    """

    if n_minus_k == 0:
        return sps.csr_matrix((0, grid.num_cells))
    elif n_minus_k == 1:
        return grid.cell_faces.T
    elif n_minus_k == 2:
        return grid.face_ridges.T
    elif n_minus_k == 3:
        return grid.ridge_peaks.T
    elif n_minus_k == 4:
        return sps.csr_matrix((grid.num_peaks, 0))
    else:
        Warning("(n - k) is not between 0 and 4")
        return sps.csr_matrix((0, 0))


def _mdg_exterior_derivative(mdg, n_minus_k, **kwargs):
    """
    Compute the mixed-dimensional exterior derivative on a grid bucket.

    Parameters:
        grid (pp.MixedDimensionalGrid): The grid bucket.
        n_minus_k (int): The difference between the ambient dimension and the order of
            the differential form.
        kwargs: Optional parameters
            as_bmat: In case of mixed-dimensional, return the matrix as sparse sub-blocks.
                Default False.
    """
    as_bmat = kwargs.get("as_bmat", False)

    # Pre-allocation of the block-matrix
    bmat = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )

    # Compute local differential operator
    for idx, sd in enumerate(mdg.subdomains()):
        bmat[idx, idx] = exterior_derivative(sd, n_minus_k)

    # Compute mixed-dimensional jump operator
    for intf in mdg.interfaces():
        pair = mdg.interface_to_subdomain_pair(intf)

        if pair[0].dim >= n_minus_k:
            # Get indices (node_numbers) in grid_bucket
            node_nrs = [mdg.subdomains().index(sd) for sd in pair]

            # Place the jump term in the block-matrix
            bmat[node_nrs[1], node_nrs[0]] = exterior_derivative(intf, n_minus_k)

    pg.bmat.replace_nones_with_zeros(bmat)
    # remove the tips
    is_tip_dof = pg.numerics.restrictions.zero_tip_dofs(mdg, n_minus_k, **kwargs)

    bmat = bmat if as_bmat else sps.bmat(bmat, format="csc")
    return bmat @ is_tip_dof
