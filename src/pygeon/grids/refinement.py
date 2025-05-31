import numpy as np
import scipy.sparse as sps

import pygeon as pg


def barycentric_split(sd: pg.Grid) -> pg.Grid:
    """
    Performs a barycentric split of the input grid, subdividing each cell into smaller
    cells using barycentric coordinates.

    This function constructs a new grid where each original cell is split into (dim+1)
    subcells by introducing a new node at the cell center and connecting it to the
    original cell's nodes and faces. The resulting grid has updated nodes, face-node
    connectivity, and cell-face connectivity.

    Args:
        sd (pg.Grid): The input grid to be barycentrically split.

    Returns:
        pg.Grid: A new grid object representing the barycentric split of the input grid.
    """
    oppisite_nodes = sd.compute_opposite_nodes()

    new_nodes = np.arange(sd.num_cells) + sd.num_nodes

    rows_I = np.concatenate((oppisite_nodes.data, np.repeat(new_nodes, sd.dim + 1)))
    cols_J = np.tile(np.arange(oppisite_nodes.data.size), sd.dim)
    data_IJ = np.ones(rows_I.size)

    new_fn = sps.csc_array((data_IJ, (rows_I, cols_J)))

    shape = (new_fn.shape[0], sd.face_nodes.shape[1])
    extended_fn = sps.csc_array(sd.face_nodes, shape=shape)

    fn = sps.hstack((extended_fn, new_fn)).tocsc()

    nodes = np.hstack((sd.nodes, sd.cell_centers))

    new_face_inds = sd.cell_faces.copy()
    new_face_inds.data = np.arange(new_face_inds.nnz)

    size = np.square(sd.dim + 1) * sd.num_cells
    rows_I = np.empty(size, dtype=np.int64)
    cols_J = np.empty(size, dtype=np.int64)
    data_IJ = np.empty(size)
    idx = 0

    for c in range(sd.num_cells):
        loc_slice = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
        loc_faces = sd.cell_faces.indices[loc_slice]

        for f in loc_faces:
            cols_J[idx : idx + sd.dim + 1] = new_face_inds[f, c]
            rows_I[idx] = f
            data_IJ[idx] = sd.cell_faces[f, c]
            idx += 1

            other_f = loc_faces[loc_faces != f]
            mask = np.argsort(fn.indices[fn.indptr[other_f]])
            other_f = other_f[mask]

            rows_I[idx : idx + sd.dim] = (
                new_face_inds[other_f, c].todense() + sd.num_faces
            )
            data_IJ[idx : idx + sd.dim] = np.array([1, -1]) * sd.cell_faces[f, c]
            idx += sd.dim

    cf = sps.csc_array((data_IJ, (rows_I, cols_J)))

    return pg.Grid(sd.dim, nodes, fn, cf, "barycentric split")
