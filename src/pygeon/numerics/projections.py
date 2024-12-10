""" This module contains functions for creating projection operators. """

from typing import Optional, Union

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


def eval_at_cell_centers(
    mdg: pg.MixedDimensionalGrid, discr: pg.Discretization, **kwargs
) -> Union[sps.csc_matrix, sps.bmat]:
    """
    Create an operator that evaluates a solution in the cell centers.

    This function takes a mixed-dimensional grid `mdg` and a discretization `discr`
    (optional) and returns an operator that can be used to evaluate a solution in
    the cell centers of the grid.

    Parameters:
        mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
        discr (pg.Discretization): The discretization used for the evaluation.
        kwargs (dict): Optional parameters.
            as_bmat (bool): In case of mixed-dimensional, return the matrix as sparse
                sub-blocks. Default is False.

    Returns:
        sps.spmatrix or sps.bmat: The operator that evaluates the solution in the cell centers.
        If `as_bmat` is True, the operator is returned as sparse sub-blocks in `sps.spmatrix`
        format. Otherwise, the operator is returned as a block matrix in `sps.bmat` format.

    """
    as_bmat = kwargs.get("as_bmat", False)

    bmat_sd = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )

    # Local mass matrices
    for nn_sd, sd in enumerate(mdg.subdomains()):
        bmat_sd[nn_sd, nn_sd] = discr.eval_at_cell_centers(sd)

    pg.bmat.replace_nones_with_zeros(bmat_sd)
    return bmat_sd if as_bmat else sps.bmat(bmat_sd, format="csc")


def proj_faces_to_cells(
    mdg: pg.MixedDimensionalGrid, discr: Optional[pg.Discretization] = None, **kwargs
) -> Union[sps.csc_matrix, sps.bmat]:
    """
    Create an operator that evaluates a solution defined on the faces in the cell centers.

    Parameters:
        mdg (pg.MixedDimensionalGrid): The mixed dimensional grid.
        discr (pg.Discretization, optional): The discretization used for the evaluation.
            If not provided, a standard discretization is used. Default is None.
        kwargs (dict): Optional parameters.
            as_bmat (bool): In case of mixed-dimensional, return the matrix as sparse
                sub-blocks. Default is False.

    Returns:
        sps.spmatrix or sps.bmat: The operator matrix. If `as_bmat` is True, the matrix
            is returned as sparse sub-blocks, otherwise it is returned as a block matrix
            in CSC format.
    """
    as_bmat = kwargs.get("as_bmat", False)

    bmat_sd = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )

    if discr is None:
        discr = pg.RT0("unit")

    # Local mass matrices
    for nn_sd, (sd, d_sd) in enumerate(mdg.subdomains(return_data=True)):
        bmat_sd[nn_sd, nn_sd] = discr.eval_at_cell_centers(sd)

    pg.bmat.replace_nones_with_zeros(bmat_sd)
    return bmat_sd if as_bmat else sps.bmat(bmat_sd, format="csc")
