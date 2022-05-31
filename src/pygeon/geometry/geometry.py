import numpy as np
import scipy.sparse as sps
import porepy as pp

"""
Acknowledgments:
    The functionalities related to the ridge computations are modified from
    github.com/anabudisa/md_aux_precond developed by Ana Budiša and Wietse M. Boon.
"""


def compute_geometry(gb):
    compute_ridges(gb)
    if isinstance(gb, pp.GridBucket):
        assign_smtp_to_mg(gb)
        assign_cell_faces_to_mg(gb)
        tag_tips(gb)


def compute_ridges(grid):
    if isinstance(grid, pp.Grid):
        if grid.dim == 3:
            _compute_ridges_3d(grid)
        elif grid.dim == 2:
            _compute_ridges_2d(grid)
        elif grid.dim <= 1:
            _compute_ridges_01d(grid)

    if isinstance(grid, pp.GridBucket):
        for g in grid.get_grids():
            compute_ridges(g)

        for e, d_e in grid.edges():
            if d_e["mortar_grid"].dim >= 1:
                _compute_ridges_md(grid, e)


def _compute_ridges_01d(g):
    g.num_peaks = 0
    g.num_ridges = 0
    g.ridge_peaks = sps.csc_matrix((g.num_peaks, g.num_ridges), dtype=np.int)
    g.face_ridges = sps.csc_matrix((g.num_ridges, g.num_faces), dtype=np.int)


def _compute_ridges_2d(g):
    # Edges in 2D are nodes
    g.num_peaks = 0
    g.num_ridges = g.num_nodes

    R = pp.map_geometry.project_plane_matrix(g.nodes)
    rot = np.dot(
        R.T, np.dot(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), R)
    )
    face_tangential = rot.dot(g.face_normals)

    face_ridges = g.face_nodes.copy().astype(np.int)

    nodes = sps.find(g.face_nodes)[0]
    for face in np.arange(g.num_faces):
        loc = slice(g.face_nodes.indptr[face], g.face_nodes.indptr[face + 1])
        nodes_loc = np.sort(nodes[loc])

        tangent = g.nodes[:, nodes_loc[1]] - g.nodes[:, nodes_loc[0]]
        sign = np.sign(np.dot(face_tangential[:, face], tangent))

        face_ridges.data[loc] = [-sign, sign]

    g.ridge_peaks = sps.csc_matrix((g.num_peaks, g.num_ridges), dtype=np.int)
    g.face_ridges = face_ridges


def _compute_ridges_3d(g):
    g.num_peaks = g.num_nodes

    # Number of ridges per face, assumed to be constant.
    # TODO: Relax this assumption
    n_r = g.face_nodes[:, 0].nnz

    # Pre-allocation
    ridges = np.ndarray((2, n_r * g.num_faces), dtype=np.int)

    for face in np.arange(g.num_faces):
        # find indices for nodes of this face
        loc = g.face_nodes.indices[
            g.face_nodes.indptr[face] : g.face_nodes.indptr[face + 1]
        ]
        # Define ridges between each pair of nodes
        # assuming ordering in face_nodes is done
        # according to right-hand rule
        ridges[:, n_r * face : n_r * (face + 1)] = np.row_stack((loc, np.roll(loc, -1)))

    # Save orientation of each ridge w.r.t. the face
    orientations = np.sign(ridges[1, :] - ridges[0, :])

    # Ridges are oriented from low to high node indices
    ridges.sort(axis=0)
    ridges, _, indices = pp.utils.setmembership.unique_columns_tol(ridges)
    g.num_ridges = np.size(ridges, 1)

    # Generate ridge-peak connectivity such that
    # ridge_peaks(i, j) = +/- 1:
    # ridge j points to/away from peak i
    indptr = np.arange(0, ridges.size + 1, 2)
    ind = np.ravel(ridges, order="F")
    data = -((-1) ** np.arange(ridges.size))
    g.ridge_peaks = sps.csc_matrix((data, ind, indptr))

    # Generate face_ridges such that
    # face_ridges(i, j) = +/- 1:
    # face j has ridge i with same/opposite orientation
    # with the orientation defined according to the right-hand rule
    indptr = np.arange(0, indices.size + 1, n_r)
    g.face_ridges = sps.csc_matrix((orientations, indices, indptr))


def _compute_ridges_md(gb, e):
    """
    Computes the mixed-dimensional face-ridge and ridge-peak connectivities
    and saves them as properties of the mortar grid
    """

    # Find high-dim faces matching to low-dim cell
    mg = gb.edge_props(e, "mortar_grid")
    cell_faces = mg.mortar_to_primary_int() * mg.secondary_to_mortar_int()

    g_down, g_up = gb.nodes_of_edge(e)

    # High-dim ridges matching to low-dim face
    face_ridges = sps.lil_matrix((g_up.num_ridges, g_down.num_faces), dtype=int)
    # High-dim peaks matching to low-dim ridge
    ridge_peaks = sps.lil_matrix((g_up.num_peaks, g_down.num_ridges), dtype=int)

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

        # Ridges of face in higher-dim grid
        fr_up = g_up.face_ridges
        ridges_up = fr_up.indices[fr_up.indptr[face_up] : fr_up.indptr[face_up + 1]]

        # Swap ridges around so they match with lower-dim faces
        if mg.dim == 1:
            face_xyz = g_down.face_centers[:, faces_down]
            ridge_xyz = g_up.nodes[:, ridges_up]
        else:  # mg.dim == 2
            face_xyz = g_down.nodes * abs(g_down.face_ridges[:, faces_down]) / 2
            ridge_xyz = g_up.nodes * abs(g_up.ridge_peaks[:, ridges_up]) / 2

        ridges_up = ridges_up[match_coordinates(face_xyz, ridge_xyz)]

        # Ridge-peak connectivity in 3D
        if mg.dim == 2:
            # Ridges of cell in lower-dim grid
            cr_down = g_down.cell_nodes()
            ridges_down = cr_down.indices[
                cr_down.indptr[cell_down] : cr_down.indptr[cell_down + 1]
            ]
            ridge_xyz = g_down.nodes[:, ridges_down]

            # Nodes of face in higher-dim grid
            fn_up = g_up.face_nodes
            peaks_up = fn_up.indices[fn_up.indptr[face_up] : fn_up.indptr[face_up + 1]]
            peak_xyz = g_up.nodes[:, peaks_up]

            # Swap peaks around so they match with lower-dim ridges
            peaks_up = peaks_up[match_coordinates(ridge_xyz, peak_xyz)]

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
            orientations_fr = np.dot(np.dot(rot, normal_up), normal_down)

        else:  # mg.dim == 2
            # we say that orientations align if the cross product
            # between the ridge tangent and the mortar normal corresponds
            # to the normal of the lower-dimensional face
            tangents = g_up.nodes * g_up.ridge_peaks[:, ridges_up]
            products = np.cross(tangents, normal_up, axisa=0, axisc=0)
            orientations_fr = [
                np.dot(products[:, i], normal_down[:, i])
                for i in np.arange(np.size(tangents, 1))
            ]

            # The (virtual) line connecting the low-dim ridge to
            # the high-dim is oriented according to the normal to the fracture plane
            orientations_rp = -np.dot(normal_up, normal_to_g_down) * np.ones(
                peaks_up.shape
            )
            ridge_peaks[peaks_up, ridges_down] += np.sign(orientations_rp)

        face_ridges[ridges_up, faces_down] += np.sign(orientations_fr)

    # Ensure that double indices are mapped to +-1
    # This step ensures that the jump maps to zero at tips.
    face_ridges = sps.csc_matrix(face_ridges, dtype=int)
    ridge_peaks = sps.csc_matrix(ridge_peaks, dtype=int)

    face_ridges.data = np.sign(face_ridges.data)
    ridge_peaks.data = np.sign(ridge_peaks.data)

    # Set face_ridges and ridge_peaks as properties of the mortar grid
    mg.face_ridges = face_ridges
    mg.ridge_peaks = ridge_peaks


# ------------------------------------------------------------------------ #


def tag_tips(gb):
    for g in gb.get_grids():
        g.tags["tip_peaks"] = np.zeros(g.num_peaks, dtype=np.bool)
        if g.dim == 2:
            fr_bool = g.face_ridges.astype("bool")
            g.tags["tip_ridges"] = fr_bool * g.tags["tip_faces"]
        else:
            g.tags["tip_ridges"] = np.zeros(g.num_ridges, dtype=np.bool)


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
        g.tags["leaf_ridges"] = np.zeros(g.num_ridges, dtype=bool)
        g.tags["leaf_peaks"] = np.zeros(g.num_peaks, dtype=bool)

    # Tag the ridges that correspond to a codim 2 domain
    for e, d in gb.edges():
        mg = d["mortar_grid"]

        if mg.dim >= 1:
            g_down, g_up = gb.nodes_of_edge(e)
            g_up.tags["leaf_ridges"] += (
                abs(mg.face_ridges) * g_down.tags["leaf_faces"]
            ).astype("bool")

    # Tag the peaks that correspond to a codim 3 domain
    for e, d in gb.edges():
        mg = d["mortar_grid"]

        if mg.dim >= 2:
            g_down, g_up = gb.nodes_of_edge(e)
            g_up.tags["leaf_peaks"] += (
                abs(mg.ridge_peaks) * g_down.tags["leaf_ridges"]
            ).astype("bool")
