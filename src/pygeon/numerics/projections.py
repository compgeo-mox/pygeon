import numpy as np
import scipy.sparse as sps
import porepy as pp

def proj_cells_to_faces(mdg, discr=None, **kwargs):
    """
    Computes the block matrices of the mass matrix
    """
    bmat_sd = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )

    # Local mass matrices
    for sd, d_sd in mdg.subdomains(return_data=True):
        nn_sd = d_sd["node_number"]
        bmat_sd[nn_sd, nn_sd] = _sd_proj_matrix(sd, discr, d_sd)

    return sps.bmat(bmat_sd, format="csc")

# ---------------------------------- General ---------------------------------- #

def default_discr(keyword="flow"):
    """
    Construct the default discretization operator depending on n_minus_k.
    These correspond to the Whitney forms.
    """
    return pp.RT0(keyword)

def _sd_proj_matrix(sd, discr=None, data=None):
    """
    Compute the projection matrix on a single grid

    Parameters:
        sd (pp.Grid).
        discr (pp discretization object).
        data (dict): the data object associated to the grid.

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """
    if discr is None:
        discr = default_discr()

    discr.discretize(sd, data)
    return data[pp.DISCRETIZATION_MATRICES][discr.keyword]["vector_proj"].T
