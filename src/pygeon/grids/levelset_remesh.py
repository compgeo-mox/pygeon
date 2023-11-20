import numpy as np
import scipy.sparse as sps
import pygeon as pg
import porepy as pp


def create_new_entity_map(cut_entities, offset=0):
    # Mapping of n_new x n_old in which (i_new, i_old) = 1 if i_new is a new entity placed on i_old
    n = np.sum(cut_entities)

    I = np.arange(n) + offset
    J = np.flatnonzero(cut_entities)
    V = np.ones(n)

    return sps.csc_matrix((V, (I, J)), shape=(n + offset, len(cut_entities)))


def create_splitting_map(cut_entities, offset=0):
    # Mapping of n_new x n_old in which (i_new, i_old) = 1 if i_new is a split of i_old
    n = 2 * np.sum(cut_entities)

    rows = np.arange(n) + offset
    cols = np.repeat(np.flatnonzero(cut_entities), 2)
    data = np.ones(n)

    return sps.csc_matrix((data, (rows, cols)), shape=(n + offset, len(cut_entities)))


def remesh(sd, levelset, name="LevelsetGrid"):
    cut_cells, cut_faces, new_nodes = mark_intersections(sd, levelset)

    nodes = np.hstack((sd.nodes, new_nodes))

    # Create a dictionary of entity maps according to the following conventions:
    # (0,1) => node_on_face,
    # (1,2) => face_on_cell,
    # (2,2) => cell_on_cell,
    # (1,1) => face_on_face.
    entity_maps = {}
    entity_maps[0, 1] = create_new_entity_map(cut_faces, sd.num_nodes)
    entity_maps[1, 2] = create_new_entity_map(cut_cells, sd.num_faces)
    entity_maps[2, 2] = create_splitting_map(cut_cells, sd.num_cells)
    entity_maps[1, 1] = create_splitting_map(cut_faces, entity_maps[1, 2].shape[0])

    # Compute the new face-node connectivity
    new_face_nodes = create_new_face_nodes(sd, cut_cells, cut_faces, entity_maps)

    fn_rows, fn_cols, fn_data = sps.find(sd.face_nodes)
    old_face_nodes = sps.csc_matrix(
        (fn_data, (fn_rows, fn_cols)),
        shape=new_face_nodes.shape,
    )
    face_nodes = old_face_nodes + new_face_nodes

    # Compute new cell-face connectivity
    new_cell_faces = create_new_cell_faces(
        sd, cut_cells, cut_faces, entity_maps, face_nodes
    )

    cf_rows, cf_cols, cf_data = sps.find(sd.cell_faces)
    old_cell_faces = sps.csc_matrix(
        (cf_data, (cf_rows, cf_cols)),
        shape=new_cell_faces.shape,
    )
    cell_faces = old_cell_faces + new_cell_faces

    # Eliminate old mesh entities
    restrict = pg.numerics.linear_system.create_restriction

    keep_cells = np.hstack(
        (np.logical_not(cut_cells), np.ones(2 * sum(cut_cells), dtype="bool"))
    )
    keep_faces = np.hstack(
        (
            np.logical_not(cut_faces),
            np.ones(2 * sum(cut_faces) + sum(cut_cells), dtype="bool"),
        )
    )

    restrict_cells = restrict(keep_cells)
    restrict_faces = restrict(keep_faces)

    cell_faces = restrict_faces @ cell_faces @ restrict_cells.T
    face_nodes = face_nodes @ restrict_faces.T
    face_nodes.sort_indices()

    return pg.Grid(2, nodes, face_nodes, cell_faces, name)


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
            raise NotImplementedError("Level set passes exactly through a node")

    cell_finder = np.abs(sd.cell_faces.T) @ cut_faces

    if np.any(cell_finder > 2):
        raise NotImplementedError("A cell has more than two cut faces")

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


## Add new cells
def create_new_cell_faces(sd, cut_cells, cut_faces, entity_maps, face_nodes):
    rows = []
    cols = []
    data = []

    for el in np.flatnonzero(cut_cells):
        new_cells = entity_maps[2, 2][:, el].indices
        faces_el = sd.cell_faces[:, el].indices

        face_nodes_el = sd.face_ridges[:, faces_el] * sps.diags(
            sd.cell_faces[faces_el, el].data
        )  # Extract positively oriented face_node connectivity

        (I, J, V) = sps.find(face_nodes_el)
        loop_starts = I[np.logical_and(V == 1, cut_faces[faces_el[J]])]
        loop_ends = np.flip(I[np.logical_and(V == -1, cut_faces[faces_el[J]])])

        nodes_el = positively_oriented_node_loop(I, J, V)

        for i in [0, 1]:
            # Faces that are uncut
            nodes_el = np.roll(nodes_el, -np.argmax(nodes_el == loop_starts[i]))
            sub_nodes = nodes_el[: np.argmax(nodes_el == loop_ends[i])]
            sub_faces = np.array(
                [faces_el[J[np.logical_and(V == -1, I == sn)]][0] for sn in sub_nodes]
            )
            rows.append(sub_faces)
            data.append(sd.cell_faces[sub_faces, el].data)

            # Face that is cut at the start of the loop
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


def positively_oriented_node_loop(I, J, V):
    node_loop = np.zeros(len(I) // 2, dtype="int")
    node_loop[0] = I[0]

    for i in np.arange(len(node_loop) - 1):
        next_face = J[np.logical_and(I == node_loop[i], V == -1)]
        node_loop[i + 1] = I[np.logical_and(J == next_face, V == 1)]

    return node_loop


if __name__ == "__main__":

    def level_set(x):
        # return x[1] - 0.75
        return 0.4 - np.linalg.norm(x - np.array([0.5, 0.5, 0]))

    sd = pg.unit_grid(2, 1 / 10, as_mdg=False)
    sd.compute_geometry()

    map = create_new_entity_map(sd.tags["domain_boundary_faces"])
    map = create_splitting_map(sd.tags["domain_boundary_faces"])

    sd_new = remesh(sd, level_set)
    sd_new.compute_geometry()

    pp.plot_grid(sd_new, alpha=0)
