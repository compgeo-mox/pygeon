import numpy as np
import pytest

import pygeon as pg


@pytest.mark.parametrize("n_minus_k", range(0, 4))
def test_stiffness_aliases(mdg, n_minus_k):
    if n_minus_k > mdg.dim_max():
        return

    match n_minus_k:
        case 0:
            S = pg.cell_stiff(mdg)
            n = mdg.num_subdomain_cells()
        case 1:
            S = pg.face_stiff(mdg)
            n = mdg.num_subdomain_faces()
        case 2:
            S = pg.ridge_stiff(mdg)
            n = mdg.num_subdomain_ridges()
        case 3:
            S = pg.peak_stiff(mdg)
            n = mdg.num_subdomain_peaks()

    # Symmetry and shape checks
    assert np.allclose((S - S.T).data, 0)
    assert S.shape == (n, n)
