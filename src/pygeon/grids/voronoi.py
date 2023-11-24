from typing import Union
import numpy as np
import porepy as pp
import scipy.sparse as sps
import scipy.spatial

import pygeon as pg


class VoronoiGrid(pg.Grid):
    """docstring for VoronoiGrid."""

    def __init__(self, num_bdry_els: int, num_pts=None, pts=None, **kwargs) -> None:
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
        if pts is None:
            pts = self.generate_internal_pts(num_pts, mesh_size, **kwargs)

        # Generate the boundary points for the Voronoi grid
        bd_pts = self.generate_boundary_pts(mesh_size, num_bdry_els)

        pts = np.hstack((bd_pts, pts))

        # Use Scipy to generate the Voronoi grid
        vor = scipy.spatial.Voronoi(pts[:2, :].T)

        # fix the boundary elements
        self.update_boundary_elements(vor, pts, num_bdry_els)

        # Get the node coordinates
        nodes = vor.vertices.T
        nodes = np.vstack((nodes, np.zeros(nodes.shape[1])))

        # construct the grid topology
        face_nodes, cell_faces = self.grid_topology(vor, nodes)

        # Generate a PyGeoN grid
        name = kwargs.get("name", "VoronoiGrid")
        super().__init__(2, nodes, face_nodes, cell_faces, name)

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
        margin = kwargs.get("margin_coeff", 0.5) * mesh_size

        if seed is not None:
            np.random.seed(seed)

        pts = np.random.rand(2, num_pts)
        return margin + pts * (1 - 2 * margin)

    def generate_boundary_pts(self, mesh_size: float, num_elem: int) -> np.ndarray:
        """
        Generate boundary points for the Voronoi grid.

        Args:
            mesh_size (float): The size of each mesh element.
            num_elem (int): The number of elements on each boundary.

        Returns:
            np.ndarray: An array of boundary points.
        """
        # Append the boundary seeds
        bdry_pts = np.linspace(mesh_size / 2, 1 - mesh_size / 2, num_elem)

        east = np.vstack((np.ones_like(bdry_pts), bdry_pts))
        north = np.vstack((bdry_pts[::-1], np.ones_like(bdry_pts)))
        west = np.vstack((np.zeros_like(bdry_pts), bdry_pts[::-1]))
        south = np.vstack((bdry_pts, np.zeros_like(bdry_pts)))

        return np.hstack((east, north, west, south))

    def grid_topology(
        self, vor: scipy.spatial.Voronoi, nodes: np.ndarray
    ) -> Union[sps.csc_matrix, sps.csc_matrix]:
        """
        Computes the grid topology for a given Voronoi diagram.

        Args:
            vor (scipy.spatial.Voronoi): The Voronoi diagram.
            nodes (np.ndarray): The array of node coordinates.

        Returns:
            Tuple[sps.csc_matrix, sps.csc_matrix]: A tuple containing the face-node connectivity matrix
            and the cell-face connectivity matrix.
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

            self.__init__(self.num_bdry_els, pts=pts)

            pp.plot_grid(self, alpha=0, plot_2d=True)
