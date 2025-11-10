import numpy as np
import pytest

import pygeon as pg


def test_barycentric_split(unit_sd):
    sd = pg.barycentric_split(unit_sd)
    sd.compute_geometry()

    assert sd.num_cells == (sd.dim + 1) * unit_sd.num_cells
    assert sd.num_nodes == unit_sd.num_nodes + unit_sd.num_cells

    assert np.isclose(sd.cell_volumes.sum(), 1)

    assert (sd.face_ridges @ sd.cell_faces).nnz == 0
    assert (sd.ridge_peaks @ sd.face_ridges).nnz == 0
