import numpy as np
import scipy.sparse as sps
import porepy as pp

"""
Acknowledgments:
    The functionalities related to the edge computations are modified from
    github.com/anabudisa/md_aux_precond developed by Ana Budiša and Wietse M. Boon.
"""

"""
TODO: Refactor to ridges and peaks
"""


def compute_geometry(gb):
    compute_edges(gb)
    assign_smtp_to_mg(gb)
    assign_cell_faces_to_mg(gb)
    tag_edges(gb)


def compute_edges(grid):
    if isinstance(grid, pp.Grid):
        if grid.dim == 3:
            _compute_edges_3d(grid)
        elif grid.dim == 2:
            _compute_edges_2d(grid)
        elif grid.dim == 1:
            _compute_edges_1d(grid)
        elif grid.dim == 0:
            _compute_edges_0d(grid)

    if isinstance(grid, pp.GridBucket):
        for g in grid.get_grids():
            compute_edges(g)

        for e, d_e in grid.edges():
            if d_e["mortar_grid"].dim >= 1:
                _compute_edges_md(grid, e)


def _compute_edges_0d(g):
    g.num_edges = 0
    g.edge_nodes = sps.csc_matrix((0, g.num_edges), dtype=np.int)
    g.face_edges = sps.csc_matrix((g.num_edges, g.num_faces), dtype=np.int)


def _compute_edges_1d(g):
    g.num_edges = 0
    g.edge_nodes = sps.csc_matrix((0, g.num_edges), dtype=np.int)
    g.face_edges = sps.csc_matrix((g.num_edges, g.num_faces), dtype=np.int)


def _compute_edges_2d(g):
    # Edges in 2D are nodes
    g.num_edges = g.num_nodes

    R = pp.map_geometry.project_plane_matrix(g.nodes)
    rot = np.dot(
        R.T, np.dot(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), R)
    )
    face_tangential = rot.dot(g.face_normals)

    face_edges = g.face_nodes.copy().astype(np.int)

    nodes = sps.find(g.face_nodes)[0]
    for face in np.arange(g.num_faces):
        loc = slice(g.face_nodes.indptr[face], g.face_nodes.indptr[face + 1])
        nodes_loc = np.sort(nodes[loc])

        tangent = g.nodes[:, nodes_loc[1]] - g.nodes[:, nodes_loc[0]]
        sign = np.sign(np.dot(face_tangential[:, face], tangent))

        face_edges.data[loc] = [-sign, sign]

    g.edge_nodes = sps.csc_matrix((0, g.num_edges), dtype=np.int)
    g.face_edges = face_edges


def _compute_edges_3d(g):
    # Number of edges per face, assumed to be constant.
    # TODO: Relax this assumption
    n_e = g.face_nodes[:, 0].nnz

    # Pre-allocation
    edges = np.ndarray((2, n_e * g.num_faces), dtype=np.int)

    for face in np.arange(g.num_faces):
        # find indices for nodes of this face
        loc = g.face_nodes.indices[
            g.face_nodes.indptr[face] : g.face_nodes.indptr[face + 1]
        ]
        # Define edges between each pair of nodes
        # assuming ordering in face_nodes is done
        # according to right-hand rule
        edges[:, n_e * face : n_e * (face + 1)] = np.row_stack((loc, np.roll(loc, -1)))

    # Save orientation of each edge w.r.t. the face
    orientations = np.sign(edges[1, :] - edges[0, :])

    # Edges are oriented from low to high node indices
    edges.sort(axis=0)
    edges, _, indices = pp.utils.setmembership.unique_columns_tol(edges)
    g.num_edges = np.size(edges, 1)

    # Generate edge-node connectivity such that
    # edge_nodes(i, j) = +/- 1:
    # edge j points to/away from node i
    indptr = np.arange(0, edges.size + 1, 2)
    ind = np.ravel(edges, order="F")
    data = -((-1) ** np.arange(edges.size))
    g.edge_nodes = sps.csc_matrix((data, ind, indptr))

    # Generate face_edges such that
    # face_edges(i, j) = +/- 1:
    # face j has edge i with same/opposite orientation
    # with the orientation defined according to the right-hand rule
    indptr = np.arange(0, indices.size + 1, n_e)
    g.face_edges = sps.csc_matrix((orientations, indices, indptr))


def _compute_edges_md(gb, e):
    """
    Computes the mixed-dimensional face-edge and edge-node connectivities
    and saves them as properties of the mortar grid
    """

    # Find high-dim faces matching to low-dim cell
    mg = gb.edge_props(e, "mortar_grid")
    cell_faces = mg.mortar_to_primary_int() * mg.secondary_to_mortar_int()

    g_down, g_up = gb.nodes_of_edge(e)

    # High-dim edges matching to low-dim face
    face_edges = sps.lil_matrix((g_up.num_edges, g_down.num_faces), dtype=int)
    # High-dim nodes matching to low-dim edge
    edge_nodes = sps.lil_matrix((g_up.num_nodes, g_down.num_edges), dtype=int)

    # Find information about the two-dimensional grid
    if mg.dim == 1:
        R = pp.map_geometry.project_plane_matrix(g_up.nodes)
        rot = np.dot(
            R.T,
            np.dot(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), R),
        )
    else:  # mg.dim == 2
        R = pp.map_geometry.project_plane_matrix(g_down.nodes)
        normal_to_g_down = np.dot(R.T, [0, 0, 1])

    for (face_up, cell_down) in zip(*sps.find(cell_faces)[:-1]):
        # Faces of cell in lower-dim grid
        cf_down = g_down.cell_faces
        faces_down = cf_down.indices[
            cf_down.indptr[cell_down] : cf_down.indptr[cell_down + 1]
        ]

        # Edges of face in higher-dim grid
        fe_up = g_up.face_edges
        edges_up = fe_up.indices[fe_up.indptr[face_up] : fe_up.indptr[face_up + 1]]

        # Swap edges around so they match with lower-dim faces
        if mg.dim == 1:
            face_xyz = g_down.face_centers[:, faces_down]
            edge_xyz = g_up.nodes[:, edges_up]
        else:  # mg.dim == 2
            face_xyz = g_down.nodes * abs(g_down.face_edges[:, faces_down]) / 2
            edge_xyz = g_up.nodes * abs(g_up.edge_nodes[:, edges_up]) / 2

        edges_up = edges_up[match_coordinates(face_xyz, edge_xyz)]

        # Edge-node connectivity in 3D
        if mg.dim == 2:
            # Edges of cell in lower-dim grid
            ce_down = g_down.cell_nodes()
            edges_down = ce_down.indices[
                ce_down.indptr[cell_down] : ce_down.indptr[cell_down + 1]
            ]
            edge_xyz = g_down.nodes[:, edges_down]

            # Nodes of face in higher-dim grid
            fn_up = g_up.face_nodes
            nodes_up = fn_up.indices[fn_up.indptr[face_up] : fn_up.indptr[face_up + 1]]
            node_xyz = g_up.nodes[:, nodes_up]

            # Swap nodes around so they match with lower-dim edges
            nodes_up = nodes_up[match_coordinates(edge_xyz, node_xyz)]

        # Take care of orientations
        # NOTE:this computation is done here so that we have access to the normal vector

        # Find the normal vector oriented outward wrt the higher-dim grid
        is_outward = g_up.cell_faces.tocsr()[face_up, :].data[0]
        normal_up = g_up.face_normals[:, face_up] * is_outward

        # Find the normal to the lower-dim face
        normal_down = g_down.face_normals[:, faces_down]

        # Identify orientation
        if mg.dim == 1:
            # we say that orientations align if the rotated mortar
            # normal corresponds to the normal of the
            # lower-dimensional face
            orientations_fe = np.dot(np.dot(rot, normal_up), normal_down)

        else:  # mg.dim == 2
            # we say that orientations align if the cross product
            # between the edge tangent and the mortar normal corresponds
            # to the normal of the lower-dimensional face
            tangents = g_up.nodes * g_up.edge_nodes[:, edges_up]
            products = np.cross(tangents, normal_up, axisa=0, axisc=0)
            orientations_fe = [
                np.dot(products[:, i], normal_down[:, i])
                for i in np.arange(np.size(tangents, 1))
            ]

            # The (virtual) line connecting the low-dim edge to
            # the high-dim is oriented according to the normal to the fracture plane
            orientations_en = -np.dot(normal_up, normal_to_g_down) * np.ones(
                nodes_up.shape
            )
            edge_nodes[nodes_up, edges_down] += np.sign(orientations_en)

        face_edges[edges_up, faces_down] += np.sign(orientations_fe)

    # Ensure that double indices are mapped to +-1
    # This step ensures that the jump maps to zero at tips.
    face_edges = sps.csc_matrix(face_edges, dtype=int)
    edge_nodes = sps.csc_matrix(edge_nodes, dtype=int)

    face_edges.data = np.sign(face_edges.data)
    edge_nodes.data = np.sign(edge_nodes.data)

    # Set face_edges and edge_nodes as properties of the mortar grid
    mg.face_edges = face_edges
    mg.edge_nodes = edge_nodes


# ------------------------------------------------------------------------ #


def tag_edges(gb):
    for g in gb.get_grids():
        if g.dim == 2:
            fe_bool = g.face_edges.astype("bool")
            g.tags["tip_edges"] = fe_bool * g.tags["tip_faces"]
        else:
            g.tags["tip_edges"] = np.zeros(g.num_edges, dtype=np.bool)


# ------------------------------------------------------------------------ #


def assign_cell_faces_to_mg(gb):
    for mg in gb.get_mortar_grids():
        mg.cell_faces = -mg.signed_mortar_to_primary * mg.secondary_to_mortar_int()


# ------------------------------------------------------------------------ #


def match_coordinates(a, b):
    """
    Compare and match columns of a and b
    return: ind s.t. b[ind] = a
    NOTE: we assume that each column has a match
          and a and b match in shape
    TODO: Move this function to utils
    """
    n = a.shape[1]
    ind = np.empty((n,), dtype=int)
    for i in np.arange(n):
        for j in np.arange(n):
            if np.allclose(a[:, i], b[:, j]):
                ind[i] = j
                break

    return ind


# ------------------------------------------------------------------------ #


def assign_smtp_to_mg(gb):
    for e, d_e in gb.edges():
        # Get adjacent grids and mortar_grid
        g = gb.nodes_of_edge(e)[1]
        mg = d_e["mortar_grid"]

        mg.signed_mortar_to_primary = signed_mortar_to_primary(mg, g)


def signed_mortar_to_primary(mg, g):
    cells, faces, _ = sps.find(mg.primary_to_mortar_int())
    signs = [g.cell_faces.tocsr()[face, :].data[0] for face in faces]

    return sps.csc_matrix((signs, (faces, cells)), (g.num_faces, mg.num_cells))


# ------------------------------------------------------------------------ #


def tag_mesh_entities(gb):

    for g in gb.get_grids():
        # Tag the faces that correspond to a codim 1 domain
        g.tags["leaf_faces"] = g.tags["tip_faces"] + g.tags["fracture_faces"]

        # Initialize the other tags
        g.tags["leaf_edges"] = np.zeros(g.num_edges, dtype=bool)
        num_nodes = [0, 0, 0, g.num_nodes]
        g.tags["leaf_nodes"] = np.zeros(num_nodes[g.dim], dtype=bool)

    # Tag the edges that correspond to a codim 2 domain
    for e, d in gb.edges():
        mg = d["mortar_grid"]

        if mg.dim >= 1:
            g_down, g_up = gb.nodes_of_edge(e)
            g_up.tags["leaf_edges"] += (
                abs(mg.face_edges) * g_down.tags["leaf_faces"]
            ).astype("bool")

    # Tag the nodes that correspond to a codim 3 domain
    for e, d in gb.edges():
        mg = d["mortar_grid"]

        if mg.dim >= 2:
            g_down, g_up = gb.nodes_of_edge(e)
            g_up.tags["leaf_nodes"] += (
                abs(mg.edge_nodes) * g_down.tags["leaf_edges"]
            ).astype("bool")
