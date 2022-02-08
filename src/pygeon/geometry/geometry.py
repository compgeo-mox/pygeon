import numpy as np
import scipy.sparse as sps
import porepy as pp

def compute_edges(g):
    if g.dim == 3:
        return _compute_edges_3d(g)
    elif g.dim == 2:
        return _compute_edges_2d(g)
    else:
        return None, None

def _compute_edges_2d(g):
    rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    face_tangential = rot.dot(g.face_normals)

    face_edges = g.face_nodes.copy().astype(np.int)

    nodes, faces, _ = sps.find(g.face_nodes)
    for face in np.arange(g.num_faces):
        loc = slice(g.face_nodes.indptr[face], g.face_nodes.indptr[face + 1])
        nodes_loc = nodes[loc]

        tangent = g.nodes[:, nodes_loc[1]] - g.nodes[:, nodes_loc[0]]
        sign = np.sign(np.dot(face_tangential[:, face], tangent))

        face_edges.data[loc] = [-sign, sign]

    return sps.identity(g.num_nodes, dtype=np.int), face_edges

def _compute_edges_3d(g):
    # Pre-allocation
    edges = np.ndarray((2, 3*g.num_faces), dtype=np.int)
    orientations = np.ones(3*g.num_faces, dtype=np.int)

    for face in np.arange(g.num_faces):
        # find indices for nodes of this face
        loc = g.face_nodes.indices[g.face_nodes.indptr[face]:\
                                   g.face_nodes.indptr[face + 1]]
        # Define edges between each pair of nodes
        # according to right-hand rule
        edges[:, 3*face] = [loc[0], loc[1]]
        edges[:, 3*face+1] = [loc[1], loc[2]]
        edges[:, 3*face+2] = [loc[2], loc[0]]

        # Save orientation of each edge w.r.t. the face
        orientation_loc = np.sign(loc[[1, 2, 0]] - loc)
        orientations[3*face:3*face + 3] = orientation_loc

    # Edges are oriented from low to high node indices
    edges.sort(axis=0)
    edges, _, indices = pp.utils.setmembership.unique_columns_tol(edges)

    # Generate edge-node connectivity such that
    # edge_nodes(i, j) = +/- 1:
    # edge j points to/away from node i
    indptr = np.arange(0, edges.size + 1, 2)
    ind = np.ravel(edges, order="F")
    data = -(-1)**np.arange(edges.size)
    edge_nodes = sps.csc_matrix((data, ind, indptr))

    # Generate face_edges such that
    # face_edges(i, j) = +/- 1:
    # face j has edge i with same/opposite orientation
    # with the orientation defined according to the right-hand rule
    indptr = np.arange(0, indices.size + 1, 3)
    face_edges = sps.csc_matrix((orientations, indices, indptr))

    return edge_nodes, face_edges
