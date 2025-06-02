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

    # New nodes are introduced at the cell-centers and appended to the node list
    new_nodes = np.arange(sd.num_cells) + sd.num_nodes
    nodes = np.hstack((sd.nodes, sd.cell_centers))

    ## Face-node connectivity
    # Extend the existing face_nodes to accommodate the new nodes
    shape = (nodes.shape[1], sd.face_nodes.shape[1])
    extended_fn = sps.csc_array(sd.face_nodes, shape=shape)

    new_fn = compute_face_nodes(sd, new_nodes)
    face_nodes = sps.hstack((extended_fn, new_fn)).tocsc()

    ## Cell-face connectivity
    cell_faces = compute_cell_faces(sd, face_nodes)

    return pg.Grid(sd.dim, nodes, face_nodes, cell_faces, "Barycentric split")


def compute_cell_faces(sd: pg.Grid, face_nodes: sps.csc_array) -> sps.csc_array:
    new_cell_inds = sd.cell_faces.copy()
    new_cell_inds.data = np.arange(new_cell_inds.nnz)

    size = np.square(sd.dim + 1) * sd.num_cells
    rows_I = np.empty(size, dtype=np.int64)
    cols_J = np.empty(size, dtype=np.int64)
    data_IJ = np.empty(size)
    idx = 0

    if sd.dim == 3:
        new_face_inds = np.abs(sd.face_ridges) @ np.abs(sd.cell_faces)
        new_face_inds.data = sd.num_faces + np.arange(new_face_inds.nnz)

    for c in range(sd.num_cells):
        loc_slice = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
        loc_faces = sd.cell_faces.indices[loc_slice]

        # Loop over each sub-simplex given by a cell-face pair in the original grid
        for f in loc_faces:
            # Connect the original face to the sub-simplex
            cols_J[idx : idx + sd.dim + 1] = new_cell_inds[f, c]
            rows_I[idx] = f
            data_IJ[idx] = sd.cell_faces[f, c]
            idx += 1

            # Connect each new face to the sub-simplex
            if sd.dim == 1:
                rows_I[idx] = sd.num_faces + c
                data_IJ[idx] = -sd.cell_faces[f, c]

            elif sd.dim == 2:
                other_f = loc_faces[loc_faces != f]
                mask = np.argsort(face_nodes.indices[face_nodes.indptr[other_f]])
                other_f = other_f[mask]

                rows_I[idx : idx + sd.dim] = (
                    new_cell_inds[other_f, c].todense() + sd.num_faces
                )
                data_IJ[idx : idx + sd.dim] = np.array([1, -1]) * sd.cell_faces[f, c]

            elif sd.dim == 3:
                fr = sd.face_ridges[:, [f]]
                loc_ridges = fr.indices
                rows_I[idx : idx + sd.dim] = new_face_inds[loc_ridges, c].todense()
                data_IJ[idx : idx + sd.dim] = -sd.cell_faces[f, c] * fr.data

            idx += sd.dim

    return sps.csc_array((data_IJ, (rows_I, cols_J)))


def compute_face_nodes(sd: pg.Grid, new_nodes: np.ndarray) -> sps.csc_array:
    if sd.dim == 1:
        # Each new face is at the location of a new node
        rows_I = new_nodes
        cols_J = np.arange(sd.num_cells)

    elif sd.dim == 2:
        # Each new face connects a node to a cell-center
        opposite_nodes = sd.compute_opposite_nodes()
        rows_I = np.concatenate((opposite_nodes.data, np.repeat(new_nodes, sd.dim + 1)))
        cols_J = np.tile(np.arange(opposite_nodes.data.size), sd.dim)

    elif sd.dim == 3:
        # Each new face connects a ridge to a cell-center
        cell_ridges = np.abs(sd.face_ridges) @ np.abs(sd.cell_faces)
        ridges = cell_ridges.tocsc().indices

        nodes = sd.ridge_peaks[:, ridges].tocsc().indices
        rows_I = nodes
        cols_J = np.repeat(np.arange(ridges.size), 2)

        cells = sd.num_nodes + np.repeat(np.arange(sd.num_cells), 6)
        rows_I = np.concatenate((rows_I, cells))
        cols_J = np.concatenate((cols_J, np.arange(ridges.size)))

    data_IJ = np.ones_like(rows_I)

    return sps.csc_array((data_IJ, (rows_I, cols_J)))
