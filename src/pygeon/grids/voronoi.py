from typing import Union

import numpy as np
import porepy as pp
import scipy.sparse as sps
import scipy.spatial

import pygeon as pg

import matplotlib.pyplot as plt


class VoronoiGrid(pg.Grid):
    """docstring for VoronoiGrid."""

    def __init__(self, num_bdry_els: int, num_pts=None, vrt=None, **kwargs) -> None:
        """
        Initialize a VoronoiGrid object.

        Args:
            num_bdry_els (int): The number of boundary elements.
            num_pts (int, optional): The number of internal seed points. Defaults to None.
            pts (ndarray, optional): The internal seed points. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.num_bdry_els = num_bdry_els
        mesh_size = 1.0 / num_bdry_els

        # Generate the internal seed points for the Voronoi grid
        if vrt is None:
            vrt = self.generate_internal_pts(num_pts, mesh_size, **kwargs)

        # Use Scipy to generate the Voronoi grid
        vor = scipy.spatial.Voronoi(vrt[:2, :].T)

        # extend the edges that are at infinity, strategy taken from the plot of scipy
        vrt = []
        center = vor.points.mean(axis=0)
        # MAGARI QUANDO VISITO POSSO CREARE GIA LE MAPPE?
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                vrt_loc = vor.vertices[simplex]
            else:
                # finite end Voronoi vertex
                i = simplex[simplex >= 0][0]

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                if vor.furthest_site:
                    direction *= -1
                far_pt = vor.vertices[i] + direction
                vrt_loc = np.array([vor.vertices[i], far_pt])

            vrt.append(vrt_loc.T)

        # add the points of the bounding box and create the edges
        vrt.append([[0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 1, 1, 0]])
        vrt = np.hstack(vrt)
        edg = np.arange(vrt.shape[1]).reshape((2, -1), order="F")

        # mi faccio restituire le mappe inverse e poi uso la classe vor per ricostruire effettivamente le celle. magari
        # uso lo split solo sui lati di bordo (ovvero quelli che hanno un vertice che cade fuori dal b-box)

        # select only the edges that intersect the bounding box
        is_bd = np.zeros(edg.shape[1], dtype=bool)
        for pos, e in enumerate(edg.T):
            is_bd[pos] = self.is_outside(vrt[0, e], vrt[1, e])

        # split the edges that intersect the bounding box
        bd_vrt, bd_edg, _, arg_sort = pp.intersections.split_intersecting_segments_2d(
            vrt, edg[:, is_bd], return_argsort=True
        )

        # remove the edges that are out from the bounding box
        is_in = np.zeros(bd_edg.shape[1], dtype=bool)
        for pos, e in enumerate(bd_edg.T):
            is_in[pos] = not self.is_outside(bd_vrt[0, e], bd_vrt[1, e], 0)

        bd_edg = bd_edg[:, is_in]
        arg_sort = arg_sort[is_in]

        # fix the boundary elements
        self.update_boundary_elements(vor, vrt, num_bdry_els)

        # Get the node coordinates
        nodes = vor.vertices.T
        nodes = np.vstack((nodes, np.zeros(nodes.shape[1])))

        # construct the grid topology
        face_nodes, cell_faces = self.grid_topology(vor, nodes)

        # Generate a PyGeoN grid
        name = kwargs.get("name", "VoronoiGrid")
        super().__init__(2, nodes, face_nodes, cell_faces, name)

    def is_outside(self, x, y, tol=1e-10):
        return (
            np.any(x < tol)
            or np.any(x > 1 - tol)
            or np.any(y < tol)
            or np.any(y > 1 - tol)
        )

    def generate_internal_pts(
        self, num_pts: int, mesh_size: float, **kwargs
    ) -> np.ndarray:
        """
        Generate internal points within the Voronoi grid.

        Args:
            num_pts (int): The number of points to generate.
            mesh_size (float): The size of the mesh.
            **kwargs: Additional keyword arguments.
                seed (int, optional): The seed for the random number generator.
                margin_coeff (float, optional): The margin coefficient.

        Returns:
            np.ndarray: An array of generated internal points.
        """
        seed = kwargs.get("seed", None)
        margin = 0
        # margin = kwargs.get("margin_coeff", 0.5) * mesh_size

        if seed is not None:
            np.random.seed(seed)

        pts = np.random.rand(2, num_pts)
        return margin + pts * (1 - 2 * margin)

    def grid_topology(
        self, vor: scipy.spatial.Voronoi, nodes: np.ndarray
    ) -> Union[sps.csc_matrix, sps.csc_matrix]:
        """
        Computes the grid topology for a given Voronoi diagram.

        Args:
            vor (scipy.spatial.Voronoi): The Voronoi diagram.
            nodes (np.ndarray): The array of node coordinates.

        Returns:
            Tuple[sps.csc_matrix, sps.csc_matrix]: A tuple containing the face-node
            connectivity matrix and the cell-face connectivity matrix.
        """
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

        for indx, r in enumerate(internal_regions):
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

        assert (
            cf_data.size == cf_indices.size
        ), "Try coarsening the boundaries or increasing the number of interior points"

        cell_faces = sps.csc_matrix((cf_data, cf_indices, cf_indptr))

        return face_nodes, cell_faces

    def update_boundary_elements(
        self, vor: scipy.spatial.Voronoi, pts: np.ndarray, num_bdry_els: int
    ) -> None:
        """
        Update the boundary elements of the Voronoi diagram.

        Args:
            vor (scipy.spatial.Voronoi): The Voronoi diagram.
            pts (np.ndarray): The array of points used to construct the Voronoi diagram.
            num_bdry_els (int): The number of boundary elements.

        Returns:
            None
        """
        # Connect infinite faces to a boundary node
        ridge_dict = dict((tuple(sorted(k)), v) for k, v in vor.ridge_dict.items())

        bdry_ind_0 = np.arange(4 * num_bdry_els)
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

        # Sort ridge_vertices for quick lookup
        [rv.sort() for rv in vor.ridge_vertices]

        # Complete regions at the boundary
        for id0 in bdry_ind_0:
            region_nodes = vor.regions[vor.point_region[id0]]

            idx = region_nodes.index(-1)
            region_nodes[idx : idx + 1] = [
                int_to_bdry[region_nodes[idx - 1]],
                int_to_bdry[region_nodes[idx - len(region_nodes) + 1]],
            ]

        corner_idx = np.arange(num_bdry_els - 1, 4 * num_bdry_els, num_bdry_els)
        new_verts[:, corner_idx] = np.round(new_verts[:, corner_idx])

        vor.vertices = np.vstack((vor.vertices, new_verts.T))

    def regularize(self, num_iter=1) -> None:
        """
        Regularizes the Voronoi grid by performing a specified number of iterations.

        Args:
            num_iter (int): The number of iterations to perform for regularization.
                Default is 1.

        Returns:
            None
        """
        for _ in np.arange(num_iter):
            self.compute_geometry()

            face_cells = self.cell_faces.astype(bool).T
            bd_cells = face_cells @ self.tags["domain_boundary_faces"]

            pts = self.cell_centers[:2, np.logical_not(bd_cells)]

            self.__init__(self.num_bdry_els, vrt=pts)

            pp.plot_grid(self, alpha=0, plot_2d=True)
