"""Module for the discretizations of the H1 space."""

from typing import Optional, Type

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class VLagrange1(pg.Lagrange1):
    """
    Discretization class for the VLagrange1 method.

    Attributes:
        keyword (str): The keyword for the discretization method.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Assembles and returns the mass matrix.

        assemble_loc_mass_matrix(sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray)
            -> np.ndarray:
            Computes the local VEM mass matrix on a given cell.

        assemble_loc_proj_to_mon(sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray)
            -> np.ndarray:
            Computes the local projection onto the monomials.

        assemble_loc_L2proj_lhs(sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray)
            -> np.ndarray:
            Returns the system matrix G for the local L2 projection.

        assemble_loc_L2proj_rhs(sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray)
            -> np.ndarray:
            Returns the righthand side B for the local L2 projection.

        assemble_loc_monomial_mass(sd: pg.Grid, cell: int, diam: float) -> np.ndarray:
            Computes the inner products of the monomials.

        assemble_loc_dofs_of_monomials(sd: pg.Grid, cell: int, diam: float,
            nodes: np.ndarray) -> np.ndarray:
            Returns the matrix D for the local dofs of monomials.

        assemble_stiff_matrix(sd: pg.Grid, data: Optional[dict] = None)
            -> sps.csc_array:
            Assembles and returns the stiffness matrix.

        assemble_loc_stiff_matrix(sd: pg.Grid, cell: int, diam: float,
            nodes: np.ndarray) -> np.ndarray:
            Computes the local VEM stiffness matrix on a given cell.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_array:
            Returns the differential mapping in the discrete cochain complex.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_array:
            Evaluate the function at the cell centers of the given grid.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray])
            -> np.ndarray:
            Interpolates a function over the given grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the 'natural' boundary condition.
    """

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles and returns the mass matrix.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly process.

        Returns:
            sps.csc_array: The sparse mass matrix obtained from the discretization.
        """
        # Precomputations
        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters(cell_nodes)

        # Data allocation
        size = np.sum(np.square(cell_nodes.sum(0)))
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_V = np.empty(size)
        idx = 0

        for cell, diam in enumerate(cell_diams):
            loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
            nodes_loc = cell_nodes.indices[loc]

            A = self.assemble_loc_mass_matrix(sd, cell, diam, nodes_loc)

            # Save values for local mass matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_V[loc_idx] = A.ravel()
            idx += cols.size

        return sps.csc_array((data_V, (rows_I, cols_J)))

    def assemble_loc_mass_matrix(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Computes the local VEM mass matrix on a given cell
        according to the Hitchhiker's (6.5)

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            cell (int): The index of the cell on which to compute the mass matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of nodes associated with the cell.

        Returns:
            np.ndarray: The computed local VEM mass matrix.
        """
        proj = self.assemble_loc_proj_to_mon(sd, cell, diam, nodes)
        H = self.assemble_loc_monomial_mass(sd, cell, diam)

        D = self.assemble_loc_dofs_of_monomials(sd, cell, diam, nodes)
        I_minus_Pi = np.eye(nodes.size) - D @ proj

        return proj.T @ H @ proj + sd.cell_volumes[cell] * I_minus_Pi.T @ I_minus_Pi

    def assemble_loc_proj_to_mon(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Computes the local projection onto the monomials
        Returns the coefficients {a_i} in a_0 + [a_1, a_2] dot (x - c) / d
        for each VL1 basis function.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            cell (int): The index of the cell in which the projection is computed.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The coordinates of the nodes in the cell.

        Returns:
            np.ndarray: The coefficients of the local projection onto the monomials.
        """
        G = self.assemble_loc_L2proj_lhs(sd, cell, diam, nodes)
        B = self.assemble_loc_L2proj_rhs(sd, cell, diam, nodes)

        return np.linalg.solve(G, B)

    def assemble_loc_L2proj_lhs(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Returns the system G from the hitchhiker's (3.9)

        Args:
            sd (pg.Grid): The grid object.
            cell (int): The index of the cell.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.

        Returns:
            np.ndarray: The system matrix G.
        """

        G = sd.cell_volumes[cell] / (diam**2) * np.eye(3)
        G[0, 0] = 1
        G[0, 1:] = (
            sd.nodes[: sd.dim, nodes].mean(1) - sd.cell_centers[: sd.dim, cell]
        ) / diam

        return G

    def assemble_loc_L2proj_rhs(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Returns the righthand side B from the hitchhiker's (3.14)

        Args:
            sd (pg.Grid): The grid object.
            cell (int): The cell index.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.

        Returns:
            np.ndarray: The righthand side B.
        """
        normals = (
            sd.face_normals[: sd.dim] * sd.cell_faces[:, cell].toarray().ravel()
        ) @ sd.face_nodes[nodes, :].T

        B = np.empty((3, nodes.size))
        B[0, :] = 1.0 / nodes.size
        B[1:, :] = normals / diam / 2

        return B

    def assemble_loc_monomial_mass(
        self, sd: pg.Grid, cell: int, diam: float
    ) -> np.ndarray:
        """
        Computes the inner products of the monomials
        {1, (x - c)/d, (y - c)/d}
        Hitchhiker's (5.3)

        Args:
            sd (pg.Grid): The grid object.
            cell (int): The index of the cell.
            diam (float): The diameter of the cell.

        Returns:
            np.ndarray: The computed inner products matrix.
        """
        H = np.zeros((3, 3))
        H[0, 0] = sd.cell_volumes[cell]

        M = np.ones((2, 2)) + np.eye(2)

        cell_col = np.array([cell])
        for face in sd.cell_faces[:, cell_col].indices:
            sub_volume = (
                np.dot(
                    sd.face_centers[:, face] - sd.cell_centers[:, cell],
                    sd.face_normals[:, face] * sd.cell_faces[face, cell],
                )
                / 2
            )

            vals = (
                sd.nodes[:2, sd.face_nodes[:, [face]].indices]
                - sd.cell_centers[:2, [cell] * 2]
            ) / diam

            H[1:, 1:] += sub_volume * vals @ M @ vals.T / 12

        return H

    def assemble_loc_dofs_of_monomials(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Returns the matrix D from the hitchhiker's (3.17)

        Args:
            sd (pg.Grid): The grid object.
            cell (int): The index of the cell.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.

        Returns:
            np.ndarray: The matrix D.
        """
        D = np.empty((nodes.size, 3))
        D[:, 0] = 1.0
        D[:, 1:] = (
            sd.nodes[: sd.dim, nodes] - sd.cell_centers[: sd.dim, [cell] * nodes.size]
        ).T / diam

        return D

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles and returns the stiffness matrix.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly process.

        Returns:
            sps.csc_array: The stiffness matrix obtained from the discretization.
        """
        # Precomputations
        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters(cell_nodes)

        # Data allocation
        size = np.sum(np.square(cell_nodes.sum(0)))
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_V = np.empty(size)
        idx = 0

        for cell, diam in enumerate(cell_diams):
            loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
            nodes_loc = cell_nodes.indices[loc]

            M_loc = self.assemble_loc_stiff_matrix(sd, cell, diam, nodes_loc)

            # Save values for local mass matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_V[loc_idx] = M_loc.ravel()
            idx += cols.size

        return sps.csc_array((data_V, (rows_I, cols_J)))

    def assemble_loc_stiff_matrix(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Computes the local VEM stiffness matrix on a given cell
        according to the Hitchhiker's (3.25)

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            cell (int): The index of the cell on which to compute the stiffness matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of nodal values on the cell.

        Returns:
            np.ndarray: The computed local VEM stiffness matrix.
        """
        proj = self.assemble_loc_proj_to_mon(sd, cell, diam, nodes)
        G = self.assemble_loc_L2proj_lhs(sd, cell, diam, nodes)
        G[0, :] = 0.0

        D = self.assemble_loc_dofs_of_monomials(sd, cell, diam, nodes)
        I_minus_Pi = np.eye(nodes.size) - D @ proj

        return proj.T @ G @ proj + I_minus_Pi.T @ I_minus_Pi

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range.

        Returns:
            pg.Discretization: The range discretization class.

        Raises:
            NotImplementedError: This method is not implemented and should be
                overridden in a subclass.
        """
        raise NotImplementedError
