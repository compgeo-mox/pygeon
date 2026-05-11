"""Module for the Voronoi grid generation."""

from typing import Tuple

import numpy as np
import porepy as pp
import scipy.sparse as sps
import scipy.spatial

import pygeon as pg


class VoronoiGrid(pg.Grid):
    """Voronoi grid implementation."""

    def __init__(self, num_pts=None, vrt=None, **kwargs) -> None:
        """
        Initialize a VoronoiGrid object.

        Args:
            num_pts (int, optional): The number of internal seed points. Defaults to
                None.
            pts (ndarray, optional): The internal seed points, to be in the unit square.
                Defaults to None.
            **kwargs: Additional keyword arguments, like the seed for the random number
                and a parameter to fit the grid to a bounding box in case the standard
                value does not work as expected. The former with key "seed" and the
                latter with key "factor".

        Returns:
            None
        """
        tol = kwargs.get("tol", 1e-8)
        # Generate the internal seed points for the Voronoi grid
        if vrt is None:
            vrt = self.generate_internal_pts(num_pts, **kwargs)
        else:
            assert np.amin(vrt) >= -tol and np.amax(vrt) <= 1 + tol, (
                "Points must be in the unit square"
            )

        # Use Scipy to generate the Voronoi grid
        vor = scipy.spatial.Voronoi(vrt[:2, :].T)

        # extend the edges that are at infinity, strategy taken from the plot of scipy
        map_vrt = {}
        center = vor.points.mean(axis=0)
        for idx, (pt_idx, simplex_idx) in enumerate(
            zip(vor.ridge_points, vor.ridge_vertices)
        ):
            simplex = np.asarray(simplex_idx)
            if not np.all(simplex >= 0):
                # finite end Voronoi vertex
                i = simplex[simplex >= 0][0]

                # find the tangent and normal to the line
                t = vor.points[pt_idx[1]] - vor.points[pt_idx[0]]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])

                # find the point that is furthest from the center
                midpoint = vor.points[pt_idx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                # This if-statement (copied from scipy) is never true because we don't
                # consider furthest-site Voronoi grids.
                # if vor.furthest_site:
                #     direction *= -1
                far_pt = vor.vertices[i] + direction * kwargs.get("factor", 1)

                # add the far point to the list of vertices
                vor.vertices = np.vstack((vor.vertices, far_pt))

                # add the far point to the list of ridge vertices
                mask = np.where(simplex < 0)[0][0]
                vor.ridge_vertices[idx][mask] = vor.vertices.shape[0] - 1

                # add the far point to the list of connected points
                map_vrt[i] = vor.vertices.shape[0] - 1

        # remove the infinite vertices and construct the regions that are open
        for idx, reg_idx in enumerate(vor.regions):
            reg = np.array(reg_idx)
            mask = reg < 0
            # consider only the regions that are open
            if np.any(mask):
                reg = np.delete(reg, mask)
                # add a new ridge for the open region composed of the new vertices
                ridge = [map_vrt[v] for v in reg if v in map_vrt]
                vor.ridge_vertices.append(ridge)

                # add the new ridges, sorted counter-clockwise to the current region
                pts = np.append(reg, ridge)
                mask = pg.sort_points.argsort_ccw_convex(vor.vertices[pts])
                vor.regions[idx] = pts[mask].tolist()

        # Get the node coordinates
        nodes = vor.vertices.T
        nodes = np.vstack((nodes, np.zeros(nodes.shape[1])))

        # construct the grid topology
        face_nodes, cell_faces = self.grid_topology(vor, nodes)

        # Generate a PyGeoN grid
        name = kwargs.get("name", "VoronoiGrid")
        sd = pg.Grid(2, nodes, face_nodes, cell_faces, name)

        # Add the bounding box with the levelset remesh function
        sd = pg.levelset_remesh(sd, lambda pt: pt[0])
        sd = pg.levelset_remesh(sd, lambda pt: pt[1])
        sd = pg.levelset_remesh(sd, lambda pt: pt[0] - 1)
        sd = pg.levelset_remesh(sd, lambda pt: pt[1] - 1)
        sd.compute_geometry()

        # Partition the grid removing the elements outside the bounding box
        ind = np.logical_and.reduce(
            (
                sd.cell_centers[0] > 0,
                sd.cell_centers[0] < 1,
                sd.cell_centers[1] > 0,
                sd.cell_centers[1] < 1,
            )
        )
        [_, sd_part], _, _ = pp.partition.partition_grid(sd, ind.astype(int))

        # Initialize the PyGeoN grid with the cut Voronoi grid
        super().__init__(2, sd_part.nodes, sd_part.face_nodes, sd_part.cell_faces, name)

    def generate_internal_pts(self, num_pts: int, **kwargs) -> np.ndarray:
        """
        Generate internal points within the Voronoi grid.

        Args:
            num_pts (int): The number of points to generate.
            **kwargs: Additional keyword arguments.
                seed (int, optional): The seed for the random number generator.

        Returns:
            np.ndarray: An array of generated internal points.
        """
        seed = kwargs.get("seed", None)
        if seed is not None:
            np.random.seed(seed)

        return np.random.rand(2, num_pts)

    def grid_topology(
        self, vor: scipy.spatial.Voronoi, nodes: np.ndarray
    ) -> Tuple[sps.csc_array, sps.csc_array]:
        """
        Computes the grid topology for a given Voronoi diagram.

        Args:
            vor (scipy.spatial.Voronoi): The Voronoi diagram.
            nodes (np.ndarray): The array of node coordinates.

        Returns:
            Tuple[sps.csc_array, sps.csc_array]: A tuple containing the face-node
            connectivity matrix and the cell-face connectivity matrix.
        """
        # Derive face-node connectivity
        internal_faces = [np.sort(f) for f in vor.ridge_vertices]
        indices = np.hstack(internal_faces)
        indptr = 2 * np.arange(len(internal_faces) + 1)
        data = np.ones(2 * len(internal_faces), dtype=int)
        face_nodes = sps.csc_array((data, indices, indptr), dtype=int)

        # Compute cell-face connectivity

        # Extract the start and end nodes of the region faces
        regions = [r for r in vor.regions if len(r) > 0]

        for indx, r in enumerate(regions):
            check = pp.geometry_property_checks.is_ccw_polygon(nodes[:2, r])
            regions[indx] = r[:: 2 * check - 1]

        start_node = np.hstack(regions)
        end_node = np.hstack([np.roll(r, -1) for r in regions])

        # Construct a matrix with ones on the nodes for each region face
        face_finder_indices = np.vstack((start_node, end_node)).ravel("F")
        face_finder_indptr = 2 * np.arange(start_node.size + 1)
        face_finder = sps.csc_array(
            (np.ones_like(face_finder_indices), face_finder_indices, face_finder_indptr)
        )

        # Multiply with face_nodes to match region faces with their global number
        FaFi = face_finder.T @ face_nodes

        # Extract the indices, indptr and orientation
        cf_data = np.sign(end_node - start_node).astype(int)
        cf_indices = FaFi.indices[FaFi.data == 2]
        cf_indptr = np.hstack((0, np.cumsum([len(r) for r in regions])))

        assert cf_data.size == cf_indices.size, (
            "Try increasing the number of interior points"
        )

        cell_faces = sps.csc_array((cf_data, cf_indices, cf_indptr), dtype=int)

        return face_nodes, cell_faces
