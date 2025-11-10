import numpy as np
import pytest
import scipy.sparse as sps

"""
Module contains tests to validate the opposite_node computations on simplicial 
grids.
"""


def test_opposite_nodes(unit_sd):
    opposite_node = unit_sd.compute_opposite_nodes()
    assert opposite_node.nnz == unit_sd.num_cells * (unit_sd.dim + 1)

    faces, cells, nodes = sps.find(opposite_node)

    cell_nodes = unit_sd.cell_nodes()
    assert np.all(cell_nodes[nodes, cells])
    assert not np.any(unit_sd.face_nodes[nodes, faces])


def test_non_simplicial_grid(ref_square):
    with pytest.raises(NotImplementedError):
        ref_square.compute_opposite_nodes()
