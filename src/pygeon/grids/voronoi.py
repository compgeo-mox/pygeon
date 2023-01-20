import numpy as np
import scipy.spatial
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class VoronoiGrid(pg.Grid):
    """docstring for VoronoiGrid."""

    def __init__(self, pts, name="VoronoiGrid"):

        # Use Scipy to generate the Voronoi grid
        vor = scipy.spatial.Voronoi(pts[:2, :].T)
        scipy.spatial.voronoi_plot_2d(vor)

        # Get the node coordinates
        nodes = vor.vertices.T
        nodes = np.vstack((nodes, np.zeros(nodes.shape[1])))

        # Derive face-node connectivity
        internal_faces = [f for f in vor.ridge_vertices if np.all(np.asarray(f) >= 0)]
        indices = np.hstack(internal_faces)
        indptr = 2 * np.arange(len(internal_faces) + 1)
        data = np.ones(2 * len(internal_faces), dtype=int)

        face_nodes = sps.csc_matrix((data, indices, indptr))

        # Compute cell-face connectivity

        # Extract the start and end nodes of the region faces
        internal_regions = [
            r for r in vor.regions if np.all(np.asarray(r) >= 0) and len(r) > 0
        ]

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
