from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sps
from scipy.optimize import brentq

import pygeon as pg


def levelset_remesh(sd: pg.Grid, levelset: Callable) -> pg.Grid:
    """
    Remeshes a polygonal grid such that it conforms to a level-set function

    Args:
        sd (pg.Grid): The grid to remesh.
        levelset (Callable): A function that returns the level-set value for each x.

    Returns:
        pg.Grid: A new grid conforming to the level set.
    """

    # Mark the cut faces and cells
    cut_faces, new_nodes = intersect_faces(sd, levelset)
    cut_cells = intersect_cells(sd, cut_faces)

    # Include the new nodes in the node list
    nodes = np.hstack((sd.nodes, new_nodes))

    # Create a dictionary of entity maps according to the following conventions:
    # "n_on_f" => node on face,
    # "f_on_c" => face on cell,
    # "c_on_c" => cells on cell,
    # "f_on_f" => faces on face.
    entity_maps = {}
    entity_maps["n_on_f"] = create_new_entity_map(cut_faces, sd.num_nodes)
    entity_maps["f_on_c"] = create_new_entity_map(cut_cells, sd.num_faces)
    entity_maps["c_on_c"] = create_splitting_map(cut_cells, sd.num_cells)
    entity_maps["f_on_f"] = create_splitting_map(
        cut_faces, entity_maps["f_on_c"].shape[0]
    )

    # Compute the new face-node connectivity
    new_face_nodes = create_new_face_nodes(sd, cut_cells, cut_faces, entity_maps)
    face_nodes = merge_connectivities(sd.face_nodes, new_face_nodes)

    # Compute new cell-face connectivity
    new_cell_faces = create_new_cell_faces(
        sd, cut_cells, cut_faces, entity_maps, face_nodes
    )
    cell_faces = merge_connectivities(sd.cell_faces, new_cell_faces)

    # Mark which entities to keep in the new grid
    new_cells = np.ones(2 * sum(cut_cells), dtype=bool)
    new_faces = np.ones(2 * sum(cut_faces) + sum(cut_cells), dtype=bool)

    keep_cells = np.hstack((np.logical_not(cut_cells), new_cells))
    keep_faces = np.hstack((np.logical_not(cut_faces), new_faces))

    # Restrict cell_faces using restriction operators
    restrict = pg.numerics.linear_system.create_restriction
    restrict_cells = restrict(keep_cells)
    restrict_faces = restrict(keep_faces)
    cell_faces = restrict_faces @ cell_faces @ restrict_cells.T

    # Restrict face_nodes by slicing to keep the ordering of indices intact
    face_nodes = face_nodes[:, keep_faces]

    return pg.Grid(sd.dim, nodes, face_nodes, cell_faces, sd.name)


def merge_connectivities(
    old_con: sps.csc_matrix, new_con: sps.csc_matrix
) -> sps.csc_matrix:
    """
    Concatenates two connectivity matrices without reordering their indices

    Args:
        old_con (sps.csc_matrix): The old connectivity matrix.
        new_con (sps.csc_matrix): The additional connectivities using global numbering.

    Returns:
        sps.csc_matrix: The merged connectivity matrix.

    """
    data = np.hstack((old_con.data, new_con.data))
    indices = np.hstack((old_con.indices, new_con.indices))

    indptr = new_con.indptr + old_con.indptr[-1]
    indptr[: old_con.indptr.size] = old_con.indptr

    return sps.csc_matrix(
        (data, indices, indptr),
        shape=new_con.shape,
    )


def create_new_entity_map(
    cut_entities: np.ndarray[Any, bool], offset: Optional[int] = 0
) -> sps.csc_matrix:
    """
    Creates a mapping matrix of size n_new x n_old in which
    (i_new, i_old) = 1 if i_new is a new entity placed on i_old

    Args:
        cut_entities (np.ndarray[Any, bool]): Boolean array indicating which entities are cut
        offset (int, optional): Offset value for the mapping matrix. Defaults to 0.

    Returns:
        sps.csc_matrix: Mapping matrix of size n_new x n_old
    """
    n_cuts = np.sum(cut_entities)

    rows = np.arange(n_cuts) + offset
    cols = np.flatnonzero(cut_entities)
    data = np.ones(n_cuts)

    return sps.csc_matrix(
        (data, (rows, cols)), shape=(n_cuts + offset, len(cut_entities))
    )


def create_splitting_map(
    cut_entities: np.ndarray[Any, bool], offset: Optional[int] = 0
) -> sps.csc_matrix:
    """
    Creates a mapping matrix of size n_new x n_old in which
    (i_new, i_old) = 1 if i_new is a new entity from a splitting of i_old

    Args:
        cut_entities (np.ndarray[Any, bool]): Boolean array indicating which entities are cut
        offset (int, optional): Offset value for the rows of the mapping matrix. Defaults to 0.

    Returns:
        sps.csc_matrix: Mapping matrix of size n_new x n_old
    """
    n_cuts = 2 * np.sum(cut_entities)

    rows = np.arange(n_cuts) + offset
    cols = np.repeat(np.flatnonzero(cut_entities), 2)
    data = np.ones(n_cuts)

    return sps.csc_matrix(
        (data, (rows, cols)), shape=(n_cuts + offset, len(cut_entities))
    )


def intersect_faces(
    sd: pg.Grid, levelset: Callable, root_finder=brentq
) -> Tuple[np.ndarray[Any, bool], np.ndarray]:
    """
    Marks the cells and faces cut by the level set and
    finds the new nodes at the intersection points.

    Args:
        sd (pg.Grid): The grid object.
        levelset (Callable): The level set function.
        root_finder (brentq): The root finder function. Default is brentq.

    Returns:
        Tuple[np.ndarray[Any, bool], np.ndarray]: A tuple containing the cut_faces array
        and the new_nodes array.
    """
    new_nodes = []
    cut_faces = np.zeros(sd.num_faces, dtype=bool)

    for face in np.arange(sd.num_faces):
        node_indices = sd.face_nodes[:, face].indices

        levelset_vals = [levelset(sd.nodes[:, id]) for id in node_indices]

        if np.prod(levelset_vals) < 0:
            cut_faces[face] = True

            v_coords = sd.nodes[:, node_indices]
            level_set_loc = lambda t: levelset(
                v_coords[:, 0] + t * (v_coords[:, 1] - v_coords[:, 0])
            )
            t_0 = root_finder(level_set_loc, 0, 1)
            cut_point = v_coords[:, 0] + t_0 * (v_coords[:, 1] - v_coords[:, 0])
            new_nodes.append(cut_point)

        elif np.prod(levelset_vals) == 0:
            raise NotImplementedError("Level set passes exactly through a node.")

    new_nodes = np.vstack(new_nodes).T

    return cut_faces, new_nodes


def intersect_cells(
    sd: pg.Grid, cut_faces: np.ndarray[Any, bool]
) -> np.ndarray[Any, bool]:
    """
    Marks the cells that are cut and checks if each cut cell is only cut once.

    Args:
        sd (pg.Grid): The grid object representing the spatial domain.
        cut_faces (np.ndarray[Any, bool]): An array indicating which faces are cut.

    Returns:
        np.ndarray[Any, bool]: An array indicating which cells are cut.
    """
    cell_finder = cut_faces @ np.abs(sd.cell_faces)

    if np.any(cell_finder > 2):
        print(cell_finder)
        raise NotImplementedError("A cell has more than two cut faces.")

    return cell_finder.astype(bool)


def create_new_face_nodes(
    sd: pg.Grid,
    cut_cells: np.ndarray[Any, bool],
    cut_faces: np.ndarray[Any, bool],
    entity_maps: Dict,
) -> sps.csc_matrix:
    """
    Creates new faces through the cut cells and on top of cut faces
    and generates the corresponding face-node connectivity matrix.

    Args:
        sd (pg.Grid): The grid object.
        cut_cells (np.ndarray[Any, bool]): Boolean array indicating the cut cells.
        cut_faces (np.ndarray[Any, bool]): Boolean array indicating the cut faces.
        entity_maps (Dict): Dictionary containing entity maps.

    Returns:
        sps.csc_matrix: The face-node connectivity matrix.
    """
    rows = []
    cols = []

    # Introduce a new face that connects the two cut faces of a cut cell
    for cell in np.flatnonzero(cut_cells):
        faces_el = sd.cell_faces[:, cell].indices

        cut_faces_el = faces_el[cut_faces[faces_el]]
        new_nodes_el = entity_maps["n_on_f"][:, cut_faces_el].indices
        new_faces = entity_maps["f_on_c"][:, cell].indices

        rows.append(np.sort(new_nodes_el))
        cols.append(np.repeat(new_faces, 2))

    # Introduce two new faces on top of a cut face
    for face in np.flatnonzero(cut_faces):
        old_nodes = sd.face_nodes[:, face].indices

        new_faces = entity_maps["f_on_f"][:, face].indices
        new_node = entity_maps["n_on_f"][:, face].indices[0]

        rows.append([old_nodes[0], new_node, old_nodes[1], new_node])
        cols.append(np.repeat(new_faces, 2))

    rows = np.hstack(rows)
    cols = np.hstack(cols)
    data = np.ones_like(rows, dtype=bool)

    return sps.csc_matrix((data, (rows, cols)))


def create_new_cell_faces(
    sd: pg.Grid,
    cut_cells: np.ndarray[Any, bool],
    cut_faces: np.ndarray[Any, bool],
    entity_maps: sps.csc_matrix,
    face_nodes: sps.csc_matrix,
) -> sps.csc_matrix:
    """
    Creates two new cells on top of each cut cell
    and generates the corresponding cell-face connectivity matrix.

    Args:
        sd (pg.Grid): The grid object.
        cut_cells (np.ndarray[Any, bool]): Boolean array indicating which cells are cut.
        cut_faces (np.ndarray[Any, bool]): Boolean array indicating which faces are cut.
        entity_maps (sps.csc_matrix): Sparse matrix representing the entity maps.
        face_nodes (sps.csc_matrix): Sparse matrix representing the face nodes.

    Returns:
        sps.csc_matrix: The cell-face connectivity matrix.
    """

    # If face_ridges is missing, we generate one based on face_nodes.
    if hasattr(sd, "face_ridges"):
        face_ridges = sd.face_ridges
    else:
        face_ridges = sps.csc_matrix(sd.face_nodes, copy=True)
        face_ridges.data = -np.power(-1, np.arange(face_ridges.nnz))

    # Check connectivity consistency
    assert (face_ridges @ sd.cell_faces).nnz == 0, "Inconsistent connectivities."

    rows = []
    cols = []
    data = []

    for el in np.flatnonzero(cut_cells):
        new_cells = entity_maps["c_on_c"][:, el].indices
        faces_el = sd.cell_faces[:, el].indices

        # Extract positively oriented face_node connectivity
        face_nodes_el = face_ridges[:, faces_el] * sps.diags(
            sd.cell_faces[faces_el, el].A.ravel()
        )

        (I_node, J_face, V_orient) = sps.find(face_nodes_el)
        node_loop = create_oriented_node_loop(I_node, J_face, V_orient)

        loop_starts = I_node[np.logical_and(V_orient == 1, cut_faces[faces_el[J_face]])]
        loop_ends = np.flip(
            I_node[np.logical_and(V_orient == -1, cut_faces[faces_el[J_face]])]
        )

        for i in [0, 1]:  # Loop over the two subcells
            # Faces that are uncut
            node_loop = np.roll(node_loop, -np.argmax(node_loop == loop_starts[i]))
            sub_nodes = node_loop[: np.argmax(node_loop == loop_ends[i])]
            sub_faces = np.array(
                [
                    faces_el[J_face[np.logical_and(V_orient == -1, I_node == sn)]][0]
                    for sn in sub_nodes
                ],
                dtype=int,
            )
            rows.append(sub_faces)
            data.append(sd.cell_faces[sub_faces, el].A.ravel())

            # Faces that are cut at the start/end of the loop
            start_face = faces_el[
                J_face[np.logical_and(I_node == loop_starts[i], V_orient == 1)][0]
            ]
            splits_at_start = entity_maps["f_on_f"][:, start_face].indices
            face_at_start = splits_at_start[
                np.argmax(face_nodes[loop_starts[i], splits_at_start])
            ]

            end_face = faces_el[
                J_face[np.logical_and(I_node == loop_ends[i], V_orient == -1)][0]
            ]
            splits_at_end = entity_maps["f_on_f"][:, end_face].indices
            face_at_end = splits_at_end[
                np.argmax(face_nodes[loop_ends[i], splits_at_end])
            ]

            rows.append([face_at_start, face_at_end])
            data.append([-1, 1])

            # The new face cutting through the element
            cutting_face = entity_maps["f_on_c"][:, el].indices[0]
            rows.append(cutting_face)

            oriented_ccw = (
                face_nodes[:, face_at_end].indices[1]
                == face_nodes[:, cutting_face].indices[0]
            )
            data.append(2 * oriented_ccw - 1)

            cols.append(np.repeat(new_cells[i], 3 + len(sub_faces)))

    rows = np.hstack(rows)
    cols = np.hstack(cols)
    data = np.hstack(data)

    return sps.csc_matrix((data, (rows, cols)))


def create_oriented_node_loop(
    I_node: np.ndarray[Any, int],
    J_face: np.ndarray[Any, int],
    V_orient: np.ndarray[Any, float],
) -> np.ndarray[Any, int]:
    """
    Creates a node loop for the cell according to a positive orientation.

    Args:
        I_node (np.ndarray[Any, int]): Array of node indices.
        J_face (np.ndarray[Any, int]): Array of face indices.
        V_orient (np.ndarray[Any, float]): Array of orientation values.

    Returns:
        np.ndarray[Any, int]: Array of node indices representing the node loop.

    Notes:
        The input corresponds to (node, face, orient) triplets such that
        orient = plus/minus 1 means that the node is at the end/start of the face
        according to the ccw orientation of the cell.
    """

    node_loop = np.zeros(len(I_node) // 2, dtype=int)
    node_loop[0] = I_node[0]

    for i in np.arange(1, len(node_loop)):
        next_face = J_face[np.logical_and(I_node == node_loop[i - 1], V_orient == -1)]
        node_loop[i] = I_node[np.logical_and(J_face == next_face, V_orient == 1)]

    return node_loop
