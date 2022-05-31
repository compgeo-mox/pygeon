import numpy as np
import scipy.sparse as sps


def zero_tip_dofs(gb, n_minus_k):
    str = "tip_" + get_codim_str(n_minus_k)

    is_tip_dof = []
    for g in gb.get_grids():
        if g.dim >= n_minus_k:
            is_tip_dof.append(g.tags[str])

    if len(is_tip_dof) > 0:
        is_tip_dof = np.concatenate(is_tip_dof)

    return sps.diags(np.logical_not(is_tip_dof), dtype=np.int)


def remove_tip_dofs(gb, n_minus_k):
    R = zero_tip_dofs(gb, n_minus_k).tocsr()
    return R[R.indices, :]


def get_codim_str(n_minus_k):
    return ["cells", "faces", "ridges", "peaks"][n_minus_k]
