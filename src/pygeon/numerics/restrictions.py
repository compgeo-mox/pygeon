import numpy as np
import scipy.sparse as sps


def zero_tip_dofs(gb, n_minus_k):
    """
    Compute the operator that maps the tip degrees of freedom to zero.

    Parameters:
        gb (pp.GridBucket).
        n_minus_k (int): The difference between the dimension and the order of the differential form

    Returns:
        sps.dia_matrix
    """

    str = "tip_" + get_codim_str(n_minus_k)

    is_tip_dof = []
    for g in gb.get_grids():
        if g.dim >= n_minus_k:
            is_tip_dof.append(g.tags[str])

    if len(is_tip_dof) > 0:
        is_tip_dof = np.concatenate(is_tip_dof)

    return sps.diags(np.logical_not(is_tip_dof), dtype=np.int)


def remove_tip_dofs(gb, n_minus_k):
    """
    Compute the operator that removes the tip degrees of freedom.

    Parameters:
        gb (pp.GridBucket).
        n_minus_k (int): The difference between the dimension and the order of the differential form

    Returns:
        sps.csr_matrix
    """
    R = zero_tip_dofs(gb, n_minus_k).tocsr()
    return R[R.indices, :]


def get_codim_str(n_minus_k):
    """
    Helper function that returns the name of the mesh entity

    Parameters:
        n_minus_k (int): The codimension of the mesh entity

    Returns:
        str
    """
    return ["cells", "faces", "ridges", "peaks"][n_minus_k]
