import numpy as np
import porepy as pp
import scipy.sparse as sps


def eval_at_cell_centers(mdg, discr=None, **kwargs):
    bmat_sd = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )

    # Local mass matrices
    for nn_sd, sd in enumerate(mdg.subdomains()):
        bmat_sd[nn_sd, nn_sd] = discr.eval_at_cell_centers(sd)

    return sps.bmat(bmat_sd, format="csc")


def proj_faces_to_cells(mdg, discr=None, **kwargs):
    """
    Computes the block matrices of the mass matrix
    """
    bmat_sd = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )

    if discr is None:
        discr = pp.RT0("flow")

    # Local mass matrices
    for nn_sd, (sd, d_sd) in enumerate(mdg.subdomains(return_data=True)):
        discr.discretize(sd, d_sd)
        bmat_sd[nn_sd, nn_sd] = d_sd[pp.DISCRETIZATION_MATRICES][discr.keyword][
            "vector_proj"
        ]

    return sps.bmat(bmat_sd, format="csc")
