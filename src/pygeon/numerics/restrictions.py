"""This module contains functions that compute restriction operators."""

from typing import Union

import numpy as np
import scipy.sparse as sps

import pygeon as pg


def zero_tip_dofs(
    mdg: pg.MixedDimensionalGrid, n_minus_k: int, **kwargs
) -> Union[sps.csc_array, np.ndarray]:
    """
    Compute the operator that maps the tip degrees of freedom to zero.

    Args:
        mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
        n_minus_k (int): The difference between the dimension and the order of the
            differential form.
        kwargs: Optional parameters:

            - as_bmat (bool): In case of mixed-dimensional, return the matrix as sparse
              sub-blocks. Default False.

    Returns:
        sps.csc_array or np.ndarray: The operator that maps the tip degrees of freedom
            to zero.
    """
    as_bmat = kwargs.get("as_bmat", False)

    if n_minus_k == 0:
        return sps.diags_array(np.ones(mdg.num_subdomain_cells()), dtype=int).tocsc()

    s = "tip_" + get_codim_str(n_minus_k)

    # Pre-allocation of the block-matrix
    is_tip_dof = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.csc_array
    )
    for sd in mdg.subdomains():
        if sd.dim >= n_minus_k:
            # Get indice (node_numbers) in grid_bucket
            node_nr = mdg.subdomains().index(sd)
            # Add the sparse matrix
            is_tip_dof[node_nr, node_nr] = sps.diags_array(
                np.logical_not(sd.tags[s]), dtype=int
            )

    pg.bmat.replace_nones_with_zeros(is_tip_dof)
    return is_tip_dof if as_bmat else sps.block_array(is_tip_dof).tocsc()  # type: ignore[call-overload]


def remove_tip_dofs(
    mdg: pg.MixedDimensionalGrid, n_minus_k: int, **kwargs
) -> sps.csc_array:
    """
    Compute the operator that removes the tip degrees of freedom.

    This function computes the operator that removes the tip degrees of freedom from
    a given mixed-dimensional grid. The operator is represented as a sparse matrix
    in compressed sparse column (CSC) format.

    Args:
        mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
        n_minus_k (int): The difference between the dimension and the order of the
            differential form.

    Returns:
        sps.csc_array: The operator that removes the tip degrees of freedom.
    """
    R = zero_tip_dofs(mdg, n_minus_k, **kwargs).tocsr()  # type: ignore[union-attr]
    return R[R.indices, :].tocsc()


def get_codim_str(n_minus_k: int) -> str:
    """
    Helper function that returns the name of the mesh entity

    Args:
        n_minus_k (int): The codimension of the mesh entity

    Returns:
        str: The name of the mesh entity
    """
    return ["cells", "faces", "ridges", "peaks"][n_minus_k]
