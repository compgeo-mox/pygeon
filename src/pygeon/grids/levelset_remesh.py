import numpy as np
import scipy.sparse as sps
import pygeon as pg
import porepy as pp


def levelset_remesh(sd: pg.Grid, levelset: callable):
    """
    Remeshes a polygonal grid such that it conforms to a level-set function

    Args:
        sd: grid to remesh
        levelset: function that returns the level-set value for each x
    """

    # Mark the cut faces and cells
    cut_cells, cut_faces, new_nodes = mark_intersections(sd, levelset)

    # Include the new nodes in the node list
    nodes = np.hstack((sd.nodes, new_nodes))

    # Create a dictionary of entity maps according to the following conventions:
    # (0,1) => node on face,
    # (1,2) => face on cell,
    # (2,2) => cells on cell,
    # (1,1) => faces on face.
    entity_maps = {}
    entity_maps[0, 1] = create_new_entity_map(cut_faces, sd.num_nodes)
    entity_maps[1, 2] = create_new_entity_map(cut_cells, sd.num_faces)
    entity_maps[2, 2] = create_splitting_map(cut_cells, sd.num_cells)
    entity_maps[1, 1] = create_splitting_map(cut_faces, entity_maps[1, 2].shape[0])

    # Compute the new face-node connectivity
    new_face_nodes = create_new_face_nodes(sd, cut_cells, cut_faces, entity_maps)
    face_nodes = merge_connectivities(sd.face_nodes, new_face_nodes)

    # Compute new cell-face connectivity
    new_cell_faces = create_new_cell_faces(
        sd, cut_cells, cut_faces, entity_maps, face_nodes
    )
    cell_faces = merge_connectivities(sd.cell_faces, new_cell_faces)

    # Decide which entities to keep
    new_cells = np.ones(2 * sum(cut_cells), dtype="bool")
    new_faces = np.ones(2 * sum(cut_faces) + sum(cut_cells), dtype="bool")

    keep_cells = np.hstack((np.logical_not(cut_cells), new_cells))
    keep_faces = np.hstack((np.logical_not(cut_faces), new_faces))

    # Restrict cell_faces using restriction operators
    restrict = pg.numerics.linear_system.create_restriction
    restrict_cells = restrict(keep_cells)
    restrict_faces = restrict(keep_faces)

    cell_faces = restrict_faces @ cell_faces @ restrict_cells.T

    # We restrict face_nodes by slicing to keep the ordering of indices intact
    face_nodes = face_nodes[:, keep_faces]

    return pg.Grid(sd.dim, nodes, face_nodes, cell_faces, sd.name)


def merge_connectivities(old_con, new_con):
    """
    Concatenates two connectivity matrices without reordering their indices
    Args:
        old_con: the old connectivity matrix
        new_con: the additional connectivities using new numbering
    """
    data = np.hstack((old_con.data, new_con.data))
    indices = np.hstack((old_con.indices, new_con.indices))

    indptr = new_con.indptr + old_con.indptr[-1]
    indptr[: old_con.indptr.size] = old_con.indptr

    result = sps.csc_matrix(
        (data, indices, indptr),
        shape=new_con.shape,
    )

    return result


def create_new_entity_map(cut_entities, offset=0):
    """
    Mapping of n_new x n_old in which (i_new, i_old) = 1 if i_new is a new entity placed on i_old
    """
    n_cuts = np.sum(cut_entities)

    rows = np.arange(n_cuts) + offset
    cols = np.flatnonzero(cut_entities)
    data = np.ones(n_cuts)

    return sps.csc_matrix(
        (data, (rows, cols)), shape=(n_cuts + offset, len(cut_entities))
    )


def create_splitting_map(cut_entities, offset=0):
    """
    Mapping of n_new x n_old in which (i_new, i_old) = 1 if i_new is a split of i_old
    """
    n = 2 * np.sum(cut_entities)

    rows = np.arange(n) + offset
    cols = np.repeat(np.flatnonzero(cut_entities), 2)
    data = np.ones(n)

    return sps.csc_matrix((data, (rows, cols)), shape=(n + offset, len(cut_entities)))


## introducing new nodes and marking
def mark_intersections(sd: pg.Grid, levelset):
    new_nodes = []
    cut_faces = np.zeros(sd.num_faces, dtype=bool)

    for face in np.arange(sd.num_faces):  # loop over faces
        node_indices = sd.face_nodes[:, face].indices

        levelset_vals = [levelset(sd.nodes[:, id]) for id in node_indices]

        if np.prod(levelset_vals) < 0:
            v_coords = sd.nodes[:, node_indices]

            t0 = levelset_vals[0] / (levelset_vals[0] - levelset_vals[1])
            cut_point = v_coords[:, 0] + t0 * (v_coords[:, 1] - v_coords[:, 0])
            new_nodes.append(cut_point)
            cut_faces[face] = True

        elif np.prod(levelset_vals) == 0:
            raise NotImplementedError("Level set passes exactly through a node.")

    cell_finder = cut_faces @ np.abs(sd.cell_faces)

    if np.any(cell_finder > 2):
        raise NotImplementedError("A cell has more than two cut faces.")

    cut_cells = cell_finder > 0
    new_nodes = np.vstack(new_nodes).T

    return cut_cells, cut_faces, new_nodes


## Create new faces based on connected nodes
def create_new_face_nodes(sd, cut_cells, cut_faces, entity_maps):
    rows = []
    cols = []

    # Introduce a new face inside each cut cell that connects the two cut faces
    for cell in np.flatnonzero(cut_cells):
        faces_el = sd.cell_faces[:, cell].indices

        cut_faces_el = faces_el[cut_faces[faces_el]]
        new_nodes_el = entity_maps[0, 1][:, cut_faces_el].indices
        new_faces = entity_maps[1, 2][:, cell].indices

        rows.append(np.sort(new_nodes_el))
        cols.append(np.repeat(new_faces, 2))

    # Introduce two new faces on top of a cut face
    for face in np.flatnonzero(cut_faces):
        old_nodes = sd.face_nodes[:, face].indices

        new_faces = entity_maps[1, 1][:, face].indices
        new_node = entity_maps[0, 1][:, face].indices[0]

        rows.append([old_nodes[0], new_node, old_nodes[1], new_node])
        cols.append(np.repeat(new_faces, 2))

    rows = np.hstack(rows)
    cols = np.hstack(cols)
    data = np.ones_like(rows)

    return sps.csc_matrix((data, (rows, cols)))


def create_new_cell_faces(sd, cut_cells, cut_faces, entity_maps, face_nodes):
    """
    Add new cells
    """

    # If face_ridges is missing, we generate one based on face_nodes.
    if hasattr(sd, "face_ridges"):
        face_ridges = sd.face_ridges
    else:
        face_ridges = sps.csc_matrix(sd.face_nodes, copy=True)
        face_ridges.data = -np.power(-1, np.arange(face_ridges.nnz))

    assert (face_ridges @ sd.cell_faces).nnz == 0, "Inconsistent connectivities."

    rows = []
    cols = []
    data = []

    for el in np.flatnonzero(cut_cells):
        new_cells = entity_maps[2, 2][:, el].indices
        faces_el = sd.cell_faces[:, el].indices

        face_nodes_el = face_ridges[:, faces_el] * sps.diags(
            sd.cell_faces[faces_el, el].A.ravel()
        )  # Extract positively oriented face_node connectivity

        (I, J, V) = sps.find(face_nodes_el)
        nodes_el = create_oriented_node_loop(I, J, V)

        loop_starts = I[np.logical_and(V == 1, cut_faces[faces_el[J]])]
        loop_ends = np.flip(I[np.logical_and(V == -1, cut_faces[faces_el[J]])])

        for i in [0, 1]:
            # Faces that are uncut
            nodes_el = np.roll(nodes_el, -np.argmax(nodes_el == loop_starts[i]))
            sub_nodes = nodes_el[: np.argmax(nodes_el == loop_ends[i])]
            sub_faces = np.array(
                [faces_el[J[np.logical_and(V == -1, I == sn)]][0] for sn in sub_nodes]
            )
            rows.append(sub_faces)
            data.append(sd.cell_faces[sub_faces, el].A.ravel())

            # Faces that are cut at the start/end of the loop
            start_face = faces_el[J[np.logical_and(I == loop_starts[i], V == 1)]][0]
            splits_at_start = entity_maps[1, 1][:, start_face].indices
            face_at_start = splits_at_start[
                np.argmax(face_nodes[loop_starts[i], splits_at_start])
            ]

            end_face = faces_el[J[np.logical_and(I == loop_ends[i], V == -1)]][0]
            splits_at_end = entity_maps[1, 1][:, end_face].indices
            face_at_end = splits_at_end[
                np.argmax(face_nodes[loop_ends[i], splits_at_end])
            ]

            rows.append([face_at_start, face_at_end])
            data.append([-1, 1])

            # The new face cutting through the element
            cutting_face = entity_maps[1, 2][:, el].indices[0]
            rows.append(cutting_face)

            oriented_ccw = (
                face_nodes[:, face_at_end].indices[1]
                == face_nodes[:, cutting_face].indices[0]
            )
            data.append(2 * oriented_ccw - 1)

            cols.append(np.repeat(new_cells[i], 3 + len(sub_faces)))

    rows = np.hstack(rows).astype(int)
    cols = np.hstack(cols)
    data = np.hstack(data)

    return sps.csc_matrix((data, (rows, cols)))


def create_oriented_node_loop(I, J, V):
    node_loop = np.zeros(len(I) // 2, dtype="int")
    node_loop[0] = I[0]

    for i in np.arange(1, len(node_loop)):
        next_face = J[np.logical_and(I == node_loop[i - 1], V == -1)]
        node_loop[i] = I[np.logical_and(J == next_face, V == 1)]

    return node_loop
