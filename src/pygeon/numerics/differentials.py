"""This module contains functions for computing the differential operators."""

from typing import Union

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


def div(
    grid: Union[pg.Grid, pg.MortarGrid, pg.MixedDimensionalGrid], **kwargs
) -> sps.csc_array:
    """
    Compute the divergence.

    Args:
        grid (pg.Grid, pg.MortarGrid, or pg.MixedDimensionalGrid).
        kwargs: Optional parameters:

            - as_bmat: In case of mixed-dimensional, return the matrix as sparse
              sub-blocks. Default False.

    Returns:
        sps.csc_array. The divergence operator.
    """
    return exterior_derivative(grid, 1, **kwargs)


def curl(
    grid: Union[pg.Grid, pg.MortarGrid, pg.MixedDimensionalGrid], **kwargs
) -> sps.csc_array:
    """
    Compute the curl.

    Args:
        grid (pg.Grid, pg.MortarGrid, or pg.MixedDimensionalGrid).
        kwargs: Optional parameters:

            - as_bmat: In case of mixed-dimensional, return the matrix as sparse
              sub-blocks. Default False.

    Returns:
        sps.csc_array. The curl operator.
    """
    return exterior_derivative(grid, 2, **kwargs)


def grad(
    grid: Union[pg.Grid, pg.MortarGrid, pg.MixedDimensionalGrid], **kwargs
) -> sps.csc_array:
    """
    Compute the gradient.

    Args:
        grid (pg.Grid, pg.MortarGrid, or pg.MixedDimensionalGrid).
        kwargs: Optional parameters:

            - as_bmat: In case of mixed-dimensional, return the matrix as sparse
              sub-blocks. Default False.

    Returns:
        sps.csc_array. The gradient operator.
    """
    return exterior_derivative(grid, 3, **kwargs)


# --------------------------- MD exterior derivative --------------------------- #


def exterior_derivative(
    grid: Union[pg.Grid, pg.MortarGrid, pg.MixedDimensionalGrid],
    n_minus_k: int,
    **kwargs,
) -> sps.csc_array:
    """
    Compute the (mixed-dimensional) exterior derivative for the differential forms of
    order n - k.

    Args:
        grid (pg.Grid, pg.MortarGrid, or pg.MixedDimensionalGrid).
        n_minus_k (int): The difference between the ambient dimension and the order of
            the differential form.

    Returns:
        sps.csc_array. The differential operator.
    """
    if isinstance(grid, (pp.Grid, pp.MortarGrid)):
        return _g_exterior_derivative(grid, n_minus_k, **kwargs)

    elif isinstance(grid, pp.MixedDimensionalGrid):
        return _mdg_exterior_derivative(grid, n_minus_k, **kwargs)

    else:
        raise TypeError(
            "Input needs to be of type pp.Grid, pp.MortarGrid, or "
            "pp.MixedDimensionalGrid"
        )


def _g_exterior_derivative(
    grid: Union[pg.Grid, pg.MortarGrid],
    n_minus_k: int,
    **kwargs,
) -> sps.csc_array:
    """
    Compute the exterior derivative on a grid.

    Args:
        grid (pg.Grid or pg.MortarGrid): The grid.
        n_minus_k (int): The difference between the ambient dimension and the order of
            the differential form.

    Returns:
        sps.csc_array. The differential operator.
    """
    if n_minus_k == 0:
        derivative = sps.csc_array((0, grid.num_cells))
    elif n_minus_k == 1:
        derivative = grid.cell_faces.T
    elif n_minus_k == 2:
        derivative = grid.face_ridges.T  # type: ignore[has-type]
    elif n_minus_k == 3:
        derivative = grid.ridge_peaks.T  # type: ignore[has-type]
    elif n_minus_k == 4:
        derivative = sps.csc_array((grid.num_peaks, 0))  # type: ignore[type-var, union-attr]
    else:
        Warning("(n - k) is not between 0 and 4")
        derivative = sps.csc_array((0, 0))
    return derivative


def _mdg_exterior_derivative(
    mdg: pg.MixedDimensionalGrid, n_minus_k: int, **kwargs
) -> sps.csc_array:
    """
    Compute the mixed-dimensional exterior derivative on a grid bucket.

    Args:
        grid (pg.MixedDimensionalGrid): The grid bucket.
        n_minus_k (int): The difference between the ambient dimension and the order of
            the differential form.
        kwargs: Optional parameters
            as_bmat: In case of mixed-dimensional, return the matrix as sparse
                sub-blocks. Default False.
    Return:
        sps.csc_array: the differential operator.
    """
    as_bmat = kwargs.get("as_bmat", False)

    # Pre-allocation of the block-matrix
    bmat = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.sparray
    )

    # Compute local differential operator
    for idx, sd in enumerate(mdg.subdomains()):
        bmat[idx, idx] = exterior_derivative(sd, n_minus_k)  # type: ignore[arg-type]

    # Compute mixed-dimensional jump operator
    for intf in mdg.interfaces():
        pair = mdg.interface_to_subdomain_pair(intf)

        if pair[0].dim >= n_minus_k:
            # Get indices (node_numbers) in grid_bucket
            node_nrs = [mdg.subdomains().index(sd) for sd in pair]

            # Place the jump term in the block-matrix
            bmat[node_nrs[1], node_nrs[0]] = exterior_derivative(intf, n_minus_k)  # type: ignore[arg-type]

    pg.bmat.replace_nones_with_zeros(bmat)
    # remove the tips
    is_tip_dof = pg.numerics.restrictions.zero_tip_dofs(mdg, n_minus_k, **kwargs)

    if not as_bmat:
        bmat_matrix = sps.block_array(bmat)  # type: ignore[call-overload]
        return bmat_matrix @ is_tip_dof
    else:
        return bmat @ is_tip_dof
