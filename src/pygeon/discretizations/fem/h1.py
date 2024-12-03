""" Module for the discretizations of the H1 space. """

from typing import Callable, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class Lagrange1(pg.Discretization):
    """
    Class representing the Lagrange1 finite element discretization.

    Attributes:
        None

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the mass matrix for the lowest order Lagrange element.

        local_mass(c_volume: np.ndarray, dim: int) -> np.ndarray:
            Computes the local mass matrix.

        assemble_stiffness_matrix(sd: pg.Grid, data: dict) -> sps.csc_matrix:
            Assembles the stiffness matrix for the finite element method.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the differential matrix based on the dimension of the grid.

        local_stiff(K: np.ndarray, c_volume: np.ndarray, coord: np.ndarray, dim: int)
            -> np.ndarray:
            Computes the local stiffness matrix for P1.

        local_grads(coord: np.ndarray, dim: int) -> np.ndarray:
            Calculates the local gradients of the finite element basis functions.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the lumped mass matrix for the finite element method.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_matrix:
            Constructs the matrix for evaluating a Lagrangian function at the cell centers of
            the given grid.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a given function over the nodes of a grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the 'natural' boundary condition.

        get_range_discr_class(dim: int) -> object:
            Returns the appropriate range discretization class based on the dimension.
    """

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case number of nodes.

        Args
            sd: grid, or a subclass.

        Returns
            ndof: the number of degrees of freedom.
        """
        return sd.num_nodes

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Returns the mass matrix for the lowest order Lagrange element

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly process.

        Returns:
            sps.csc_matrix: The mass matrix obtained from the discretization.
        """

        # Data allocation
        size = np.power(sd.dim + 1, 2) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            # Compute the mass-H1 local matrix
            A = self.local_mass(sd.cell_volumes[c], sd.dim)

            # Save values for mass-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrix
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def local_mass(self, c_volume: np.ndarray, dim: int) -> np.ndarray:
        """Compute the local mass matrix.

        Args:
            c_volume (np.ndarray): Cell volume.
            dim (int): Dimension of the matrix.

        Returns:
            np.ndarray: Local mass matrix of shape (num_nodes_of_cell, num_nodes_of_cell).
        """

        M = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        return c_volume * M / ((dim + 1) * (dim + 2))

    def assemble_stiffness_matrix(self, sd: pg.Grid, data: dict) -> sps.csc_matrix:
        """
        Assembles the stiffness matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (dict): A dictionary containing the necessary data for assembling the matrix.

        Returns:
            sps.csc_matrix: The assembled stiffness matrix.
        """
        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the permeability, boundary conditions
        k = parameter_dictionary["second_order_tensor"]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        _, _, _, R, dim, node_coords = pp.map_geometry.map_grid(sd)

        if not data.get("is_tangential", False):
            # Rotate the permeability tensor and delete last dimension
            if sd.dim < 3:
                k = k.copy()
                k.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                k.values = np.delete(k.values, (remove_dim), axis=0)
                k.values = np.delete(k.values, (remove_dim), axis=1)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(sd.dim + 1, 2) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            A = self.local_stiff(
                k.values[0 : sd.dim, 0 : sd.dim, c],
                sd.cell_volumes[c],
                coord_loc,
                sd.dim,
            )

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the differential matrix based on the dimension of the grid.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        if sd.dim == 3:
            return sd.ridge_peaks.T.tocsc()
        elif sd.dim == 2:
            return sd.face_ridges.T.tocsc()
        elif sd.dim == 1:
            return sd.cell_faces.T.tocsc()
        elif sd.dim == 0:
            return sps.csc_matrix((0, 1))
        else:
            raise ValueError

    def local_stiff(
        self, K: np.ndarray, c_volume: np.ndarray, coord: np.ndarray, dim: int
    ) -> np.ndarray:
        """
        Compute the local stiffness matrix for P1.

        Args:
            K (np.ndarray): permeability of the cell of (dim, dim) shape.
            c_volume (np.ndarray): scalar cell volume.
            coord (np.ndarray): coordinates of the cell vertices of (dim+1, dim) shape.
            dim (int): dimension of the problem.

        Returns:
            np.ndarray: local stiffness matrix of (dim+1, dim+1) shape.
        """

        dphi = self.local_grads(coord, dim)

        return c_volume * np.dot(dphi.T, np.dot(K, dphi))

    @staticmethod
    def local_grads(coord: np.ndarray, dim: int) -> np.ndarray:
        """
        Calculate the local gradients of the finite element basis functions.

        Args:
            coord (np.ndarray): The coordinates of the nodes in the element.
            dim (int): The dimension of the element.

        Returns:
            np.ndarray: The local gradients of the finite element basis functions.
        """
        Q = np.hstack((np.ones((dim + 1, 1)), coord.T))
        invQ = np.linalg.inv(Q)
        return invQ[1:, :]

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the lumped mass matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (Optional[dict]): Optional data dictionary.

        Returns:
            sps.csc_matrix: The assembled lumped mass matrix.
        """
        volumes = sd.cell_nodes() * sd.cell_volumes / (sd.dim + 1)
        return sps.diags(volumes).tocsc()

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Construct the matrix for evaluating a Lagrangian function at the
        cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_matrix: The matrix representing the projection at the cell centers.
        """
        eval = sd.cell_nodes()
        num_nodes = sps.diags(1.0 / sd.num_cell_nodes())

        return (eval @ num_nodes).T.tocsc()

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a given function over the nodes of a grid.

        Args:
            sd (pg.Grid): The grid on which to interpolate the function.
            func (Callable[[np.ndarray], np.ndarray]): The function to be interpolated.

        Returns:
            np.ndarray: An array containing the interpolated values at each node of the grid.
        """
        return np.array([func(x) for x in sd.nodes.T])

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the 'natural' boundary condition
        (u, func)_Gamma with u a test function in Lagrange1

        Args:
            sd (pg.Grid): The grid object representing the computational domain
            func (Callable[[np.ndarray], np.ndarray]): The function used to evaluate
                the 'natural' boundary condition
            b_faces (np.ndarray): The array of boundary faces

        Returns:
            np.ndarray: The assembled 'natural' boundary condition values
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        vals = np.zeros(self.ndof(sd))

        for face in b_faces:
            loc = slice(sd.face_nodes.indptr[face], sd.face_nodes.indptr[face + 1])
            loc_n = sd.face_nodes.indices[loc]

            vals[loc_n] += (
                func(sd.face_centers[:, face]) * sd.face_areas[face] / loc_n.size
            )

        return vals

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the appropriate range discretization class based on the dimension.

        Args:
            dim (int): The dimension of the problem.

        Returns:
            object: The range discretization class.

        Raises:
            NotImplementedError: If there's no zero discretization in PyGeoN.
        """
        if dim == 3:
            return pg.Nedelec0
        elif dim == 2:
            return pg.RT0
        elif dim == 1:
            return pg.PwConstants
        else:
            raise NotImplementedError("There's no zero discretization in PyGeoN")
