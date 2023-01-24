import numpy as np
import porepy as pp
import scipy.sparse as sps
import scipy.spatial

import pygeon as pg


class VoronoiGrid(pg.Grid):
    """docstring for VoronoiGrid."""

    def __init__(self, bdry_mesh_size, num_pts, seed=None, name="VoronoiGrid"):

        if seed is not None:
            np.random.seed(seed)
        pts = np.random.rand(2, num_pts)

        scaling = np.sqrt(0.75) * bdry_mesh_size
        pts = scaling + pts * (1 - 2 * scaling)

        bdr_pts = np.linspace(
            bdry_mesh_size, 1 - bdry_mesh_size, int(1 / bdry_mesh_size - 1)
        )
        east = np.vstack((np.ones_like(bdr_pts), bdr_pts))
        north = np.vstack((bdr_pts[::-1], np.ones_like(bdr_pts)))
        west = np.vstack((np.zeros_like(bdr_pts), bdr_pts[::-1]))
        south = np.vstack((bdr_pts, np.zeros_like(bdr_pts)))
        pts = np.hstack((east, north, west, south, pts))

        # Use Scipy to generate the Voronoi grid
        vor = scipy.spatial.Voronoi(pts[:2, :].T)

        ridge_dict = dict((tuple(sorted(k)), v) for k, v in vor.ridge_dict.items())

        bdry_ind_0 = np.arange(4 * bdr_pts.size)
        bdry_ind_1 = np.roll(bdry_ind_0, -1)

        new_verts = np.zeros((2, bdry_ind_0.size))
        num_verts = vor.vertices.shape[0]

        int_to_bdry = {}

        for id0, id1 in zip(bdry_ind_0, bdry_ind_1):
            new_verts[:, id0] = (pts[:, id0] + pts[:, id1]) / 2

            dashed_line = ridge_dict[tuple(sorted((id0, id1)))]
            face_indx = vor.ridge_vertices.index(dashed_line)

            int_to_bdry[vor.ridge_vertices[face_indx][1]] = num_verts + id0
            vor.ridge_vertices[face_indx][0] = num_verts + id0
            vor.ridge_vertices.append([num_verts + id0, num_verts + id1])

        [rv.sort() for rv in vor.ridge_vertices]

        bdry_ind_prev = np.roll(bdry_ind_0, 1)

        for id0, id_prev in zip(bdry_ind_0, bdry_ind_prev):

            region_nodes = vor.regions[vor.point_region[id0]]

            idx = region_nodes.index(-1)
            region_nodes[idx : idx + 1] = [
                int_to_bdry[region_nodes[idx - 1]],
                int_to_bdry[region_nodes[idx - len(region_nodes) + 1]],
            ]

        corner_idx = np.arange(bdr_pts.size - 1, new_verts.shape[1], bdr_pts.size)
        new_verts[:, corner_idx] = np.round(new_verts[:, corner_idx])

        vor.vertices = np.vstack((vor.vertices, new_verts.T))

        # Get the node coordinates
        nodes = vor.vertices.T
        nodes = np.vstack((nodes, np.zeros(nodes.shape[1])))

        # Derive face-node connectivity
        internal_faces = [f for f in vor.ridge_vertices]
        indices = np.hstack(internal_faces)
        indptr = 2 * np.arange(len(internal_faces) + 1)
        data = np.ones(2 * len(internal_faces), dtype=int)
        vor.ridge_points
        face_nodes = sps.csc_matrix((data, indices, indptr))

        # Compute cell-face connectivity

        # Extract the start and end nodes of the region faces
        internal_regions = [r for r in vor.regions if len(r) > 0]

        for (indx, r) in enumerate(internal_regions):
            check = pp.geometry_property_checks.is_ccw_polygon(nodes[:2, r])
            internal_regions[indx] = r[:: 2 * check - 1]

        start_node = np.hstack(internal_regions)
        end_node = np.hstack([np.roll(r, -1) for r in internal_regions])

        # Construct a matrix with ones on the nodes for each region face
        face_finder_indices = np.vstack((start_node, end_node)).ravel("F")
        face_finder_indptr = 2 * np.arange(start_node.size + 1)
        face_finder = sps.csc_matrix(
            (np.ones_like(face_finder_indices), face_finder_indices, face_finder_indptr)
        )

        # Multiply with face_nodes to match region faces with their global number
        FaFi = face_finder.T @ face_nodes

        # Extract the indices, indptr and orientation
        cf_data = np.sign(end_node - start_node).astype(int)
        cf_indices = FaFi.indices[FaFi.data == 2]
        cf_indptr = np.hstack((0, np.cumsum([len(r) for r in internal_regions])))

        cell_faces = sps.csc_matrix((cf_data, cf_indices, cf_indptr))

        # Generate a PyGeon grid
        super().__init__(2, nodes, face_nodes, cell_faces, name)
