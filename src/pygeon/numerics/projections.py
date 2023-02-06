import numpy as np
import porepy as pp
import scipy.sparse as sps


def eval_at_cell_centers(mdg, discr=None, **kwargs):
    """
    Create an operator that evalates a solution in the cell centers.

    Parameters:
        mdg (pp.MixedDimensionalGrid): The mixed dimensional grid.
        discr: The discretization used for th evaluation, if not provided a standard discretization is used. Default to
        None.
        kwargs: Optional parameters
            to_sparse: Convert to a sparse matrix from sparse sub-blocks. Default True.
            sps_format: Format of the sparse matrix when converted. Delfault csc.
    """

    to_sparse = kwargs.get("to_sparse", True)
    sps_format = kwargs.get("sps_format", "csc")

    bmat_sd = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )

    # Local mass matrices
    for (nn_sd, sd) in enumerate(mdg.subdomains()):
        bmat_sd[nn_sd, nn_sd] = discr.eval_at_cell_centers(sd)

    pg.bmat.replace_nones_with_zeros(bmat_sd)
    return sps.bmat(bmat_sd, format=sps_format) if to_sparse else bmat_sd


def proj_faces_to_cells(mdg, discr=None, **kwargs):
    """
    Create an operator that evalates a solution defined on the faces in the cell centers.

    Parameters:
        mdg (pp.MixedDimensionalGrid): The mixed dimensional grid.
        discr: The discretization used for th evaluation, if not provided a standard discretization is used. Default to
        None.
        kwargs: Optional parameters
            to_sparse: Convert to a sparse matrix from sparse sub-blocks. Default True.
            sps_format: Format of the sparse matrix when converted. Delfault csc.
    """

    to_sparse = kwargs.get("to_sparse", True)
    sps_format = kwargs.get("sps_format", "csc")

    bmat_sd = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )

    if discr is None:
        discr = pp.RT0("flow")

    # Local mass matrices
    for (nn_sd, (sd, d_sd)) in enumerate(mdg.subdomains(return_data=True)):
        discr.discretize(sd, d_sd)
        bmat_sd[nn_sd, nn_sd] = d_sd[pp.DISCRETIZATION_MATRICES][discr.keyword][
            "vector_proj"
        ]

    pg.bmat.replace_nones_with_zeros(bmat_sd)
    return sps.bmat(bmat_sd, format=sps_format) if to_sparse else bmat_sd
