"""Module for the projection operators."""

from typing import cast

import numpy as np
import scipy.sparse as sps

import pygeon as pg


def eval_at_cell_centers(
    mdg: pg.MixedDimensionalGrid, discr: pg.Discretization, **kwargs
) -> sps.csc_array | np.ndarray:
    """
    Create an operator that evaluates a solution in the cell centers.

    This function takes a mixed-dimensional grid ``mdg`` and a discretization ``discr``
    and returns an operator that can be used to evaluate a solution in
    the cell centers of the grid.

    Args:
        mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
        discr (pg.Discretization): The discretization used for the evaluation.
        **kwargs: Optional parameters:

            - as_bmat (bool): In case of mixed-dimensional, return the matrix as sparse
              sub-blocks. Defaults to False.

    Returns:
        sps.csc_array | np.ndarray: The operator that evaluates the solution in the
        cell centers. If ``as_bmat`` is True, the operator is returned as an array of
        sparse sub-blocks, otherwise as a single sparse matrix.

    """
    as_bmat = kwargs.get("as_bmat", False)

    bmat_sd = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.csc_array
    )

    # Local mass matrices
    for nn_sd, sd in enumerate(mdg.subdomains()):
        sd = cast(pg.Grid, sd)
        bmat_sd[nn_sd, nn_sd] = discr.eval_at_cell_centers(sd)

    pg.bmat.replace_nones_with_zeros(bmat_sd)
    return bmat_sd if as_bmat else sps.block_array(bmat_sd).tocsc()
