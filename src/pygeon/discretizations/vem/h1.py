""" Module for the discretizations of the H1 space. """

from typing import Callable, Optional, Union

import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps

import pygeon as pg


class VLagrange1(pg.Discretization):
    """
    Discretization class for the VLagrange1 method.

    Attributes:
        keyword (str): The keyword for the discretization method.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
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

        assemble_loc_dofs_of_monomials(sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray)
            -> np.ndarray:
            Returns the matrix D for the local dofs of monomials.

        assemble_stiff_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles and returns the stiffness matrix.

        assemble_loc_stiff_matrix(sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray)
            -> np.ndarray:
            Computes the local VEM stiffness matrix on a given cell.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Returns the differential mapping in the discrete cochain complex.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_matrix:
            Evaluate the function at the cell centers of the given grid.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function over the given grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the 'natural' boundary condition.
    """

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case number of nodes.

        Args:
            sd (pg.Grid): grid, or a subclass.

        Returns:
            int: the number of degrees of freedom.
        """
        return sd.num_nodes

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles and returns the mass matrix.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly process.

        Returns:
            sps.csc_matrix: The sparse mass matrix obtained from the discretization.
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

            M_loc = self.assemble_loc_mass_matrix(sd, cell, diam, nodes_loc)

            # Save values for local mass matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_V[loc_idx] = M_loc.ravel()
            idx += cols.size

        return sps.csc_matrix((data_V, (rows_I, cols_J)))

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
        Returns the coefficients {a_i} in a_0 + [a_1, a_2] \dot (x - c) / d
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
            sd.face_normals[: sd.dim] * sd.cell_faces[:, cell].A.ravel()
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

        for face in sd.cell_faces[:, cell].indices:
            sub_volume = (
                np.dot(
                    sd.face_centers[:, face] - sd.cell_centers[:, cell],
                    sd.face_normals[:, face] * sd.cell_faces[face, cell],
                )
                / 2
            )

            vals = (
                sd.nodes[:2, sd.face_nodes[:, face].indices]
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
    ) -> sps.csc_matrix:
        """
        Assembles and returns the stiffness matrix.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly process.

        Returns:
            sps.csc_matrix: The stiffness matrix obtained from the discretization.
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

        return sps.csc_matrix((data_V, (rows_I, cols_J)))

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

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Returns the differential mapping in the discrete cochain complex.

        Args:
            sd (pg.Grid): The grid on which the differential mapping is computed.

        Returns:
            sps.csc_matrix: The differential mapping matrix.
        """
        p1 = pg.Lagrange1(self.keyword)
        return p1.assemble_diff_matrix(sd)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Evaluate the function at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to evaluate the function.

        Returns:
            sps.csc_matrix: The evaluated function values at the cell centers.
        """
        eval = sd.cell_nodes()
        num_nodes = sps.diags(1.0 / sd.num_cell_nodes())

        return (eval @ num_nodes).T.tocsc()

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function over the given grid.

        Args:
            sd (pg.Grid): The grid to interpolate the function on.
            func (Callable[[np.ndarray], np.ndarray]): The function to interpolate.

        Returns:
            np.ndarray: The interpolated values of the function on the grid.
        """
        return np.array([func(x) for x in sd.nodes.T])

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the 'natural' boundary condition
        (u, func)_Gamma with u a test function in Lagrange1

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function defining the
                'natural' boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled 'natural' boundary condition.
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


class VLagrange1_vec(VLagrange1):
    """
    Vectorized version of the VLagrange1 class.

    This class represents a vectorized version of the VLagrange1 class,
    which is a specific discretization method for solving partial differential
    equations. It inherits from the VLagrange1 class and overrides some of its
    methods to provide vectorized implementations.

    Attributes:
        Inherits all attributes from the VLagrange1 class.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles and returns the mass matrix.

        assemble_stiff_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the stiffness matrix for the H1 discretization.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Returns the differential mapping in the discrete cochain complex.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_matrix:
            Evaluate the function at the cell centers of the given grid.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function over the given grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the 'natural' boundary condition.

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the range discretization class for the given dimension.
    """

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case dim x number of nodes.

        Args:
            sd (pg.Grid): Grid object representing the discretized domain.

        Returns:
            int: The number of degrees of freedom.
        """
        return sd.dim * super().ndof(sd)

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles and returns the mass matrix.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly.

        Returns:
            sps.csc_matrix: The mass matrix obtained from the discretization.
        """
        M_VL1 = super().assemble_mass_matrix(sd, data)
        return sps.block_diag([M_VL1] * sd.dim, "csc")

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the stiffness matrix for the H1 discretization.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly.

        Returns:
            sps.csc_matrix: The stiffness matrix obtained from the discretization.
        """
        A_VL1 = super().assemble_stiff_matrix(sd)
        return sps.block_diag([A_VL1] * sd.dim, "csc")

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Returns the differential mapping in the discrete cochain complex.

        Args:
            sd (pg.Grid): The grid on which the differential mapping is assembled.

        Returns:
            sps.csc_matrix: The differential mapping matrix in the discrete cochain complex.
        """
        p1 = pg.Lagrange1(self.keyword)
        diff = p1.assemble_diff_matrix(sd)

        return sps.block_diag([diff] * sd.dim, "csc")

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Evaluate the function at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to evaluate the function.

        Returns:
            sps.csc_matrix: The evaluation of the function at the cell centers.
        """
        eval = super().eval_at_cell_centers(sd)
        return sps.block_diag([eval] * sd.dim, "csc")

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function over the given grid.

        Args:
            sd (pg.Grid): The grid to interpolate the function on.
            func (Callable[[np.ndarray], np.ndarray]): The function to be interpolated.

        Returns:
            np.ndarray: The interpolated values of the function on the grid.
        """
        return np.array([func(x) for x in sd.nodes.T])

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the 'natural' boundary condition
        (u, func)_Gamma with u a test function in Lagrange1

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function defining the
                'natural' boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled 'natural' boundary condition.

        Raises:
            NotImplementedError: This method is not implemented and should be
                overridden in a subclass.
        """
        raise NotImplementedError

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
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


class VecVLagrange1(pg.VecDiscretization):
    """
    Vector Lagrange virtual element discretization for H1 space in 2d.

    This class represents a virtual element discretization for the H1 space using
    vector virtual Lagrange elements. It provides methods for assembling various matrices
    and operators, such as the mass matrix, divergence matrix, symmetric gradient
    matrix, and more.

    Convention for the ordering is first all the x then all the y.

    The stress tensor and strain tensor are represented as vectors unrolled row-wise.
    In 2D, the stress tensor has a length of 4.

    We are considering the following structure of the stress tensor in 2d

    sigma = [[sigma_xx, sigma_xy],
             [sigma_yx, sigma_yy]]

    which is represented in the code unrolled row-wise as a vector of length 4

    sigma = [sigma_xx, sigma_xy,
             sigma_yx, sigma_yy]

    The strain tensor follows the same approach.

    Args:
        keyword (str): The keyword for the H1 class.

    Attributes:
        scalar_discr (pg.VLagrange1): A local virtual Lagrange1 class for performing some of
            the computations.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated with the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles and returns the mass matrix for the lowest order Lagrange element.

        assemble_div_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Returns the divergence matrix operator for the lowest order vector Lagrange
            element.

        local_div(c_volume: float, coord: np.ndarray, dim: int) -> np.ndarray:
            Computes the local divergence matrix for P1.

        assemble_div_div_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Returns the div-div matrix operator for the lowest order vector Lagrange element.

        assemble_symgrad_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Returns the symmetric gradient matrix operator for the lowest order vector Lagrange
            element.

        local_symgrad(c_volume: float, coord: np.ndarray, dim: int, sym: np.ndarray)
            -> np.ndarray:
            Computes the local symmetric gradient matrix for P1.

        assemble_symgrad_symgrad_matrix(sd: pg.Grid, data: Optional[dict] = None)
            -> sps.csc_matrix:
            Returns the symgrad-symgrad matrix operator for the lowest order vector Lagrange
            element.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the matrix corresponding to the differential operator.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function onto the finite element space.
    """

    def __init__(self, keyword: str) -> None:
        """
        Initialize the vector discretization class.
        The scalar discretization class is pg.Lagrange1.

        Args:
            keyword (str): The keyword for the vector discretization class.

        Returns:
            None
        """
        super().__init__(keyword, pg.VLagrange1)

    def assemble_div_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Returns the div matrix operator for the lowest order
        vector Lagrange element

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_matrix: The div matrix obtained from the discretization.
        """
        # If a 0-d grid is given then we return a zero matrix
        if sd.dim == 0:
            return sps.csc_matrix((1, 1))

        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters(cell_nodes)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = cell_nodes.sum() * sd.dim
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # shift to comply with the ordering convention of (x, y, z) components
        shift = np.atleast_2d(np.arange(sd.dim)).T * sd.num_nodes
        for cell, diam in enumerate(cell_diams):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])

            nodes_loc = cell_nodes.indices[loc]

            # Compute the div local matrix
            A = self.local_div(sd, cell, diam, nodes_loc)

            # Save values for the local matrix in the global structure
            cols = nodes_loc + shift
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cell * np.ones(cols.size)
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def local_div(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Compute the local div matrix for vector P1.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            cell (int): The index of the cell on which to compute the div matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.

        Returns:
            ndarray: Local mass Hdiv matrix.
        """
        proj = self.scalar_discr.assemble_loc_proj_to_mon(sd, cell, diam, nodes)

        return sd.cell_volumes[cell] * proj[1:] / diam

    def assemble_div_div_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Returns the div-div matrix operator for the lowest order
        vector Lagrange element. The matrix is multiplied by the Lame' parameter lambda.

        Args:
            sd (pg.Grid): The grid object.
            data (Optional[dict]): Additional data, the Lame' parameter lambda.
                Defaults to None.

        Returns:
            matrix: sparse (sd.num_nodes, sd.num_nodes)
                Div-div matrix obtained from the discretization.
        """
        coeff = data.get("lambda", 1) if data is not None else 1
        p0 = pg.PwConstants(self.keyword)

        div = self.assemble_div_matrix(sd)
        mass = p0.assemble_mass_matrix(sd)

        return div.T @ (coeff * mass) @ div

    def assemble_symgrad_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Returns the symmetric gradient matrix operator for the
        lowest order vector Lagrange element

        Args:
            sd (pg.Grid): The grid object representing the domain.

        Returns:
            sps.csc_matrix: The sparse symmetric gradient matrix operator.

        Raises:
            None

        Notes:
            - If a 0-dimensional grid is given, a zero matrix is returned.
            - The method maps the domain to a reference geometry.
            - The method allocates data to store matrix entries efficiently.
            - The symmetrization matrix is constructed differently for 2D and 3D cases.
            - The method computes the symgrad local matrix for each cell and saves
              the values in the global structure.
            - Finally, the method constructs the global matrices using the saved values.

        """
        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters(cell_nodes)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = cell_nodes.sum() * np.power(sd.dim, 3)
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        dim2 = np.square(sd.dim)
        # construct the symmetrization matrix
        sym = np.eye(dim2)
        if sd.dim == 2:
            sym[np.ix_([1, 2], [1, 2])] = 0.5
        else:
            raise ValueError("Grid dimension should be 2.")

        # shift to comply with the ordering convention of (x, y, z) components
        shift = np.atleast_2d(np.arange(sd.dim)).T * sd.num_nodes
        for cell, diam in enumerate(cell_diams):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
            nodes_loc = cell_nodes.indices[loc]

            # Compute the symgrad local matrix
            A = self.local_symgrad(sd, cell, diam, nodes_loc, sym)

            # Save values for the local matrix in the global structure
            cols = (nodes_loc + shift).ravel()
            cols = cols * np.ones((dim2, 1), dtype=int)

            rows = cell + np.arange(dim2) * sd.num_cells
            rows = np.ones(nodes_loc.size * sd.dim, dtype=int) * rows.reshape((-1, 1))

            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = rows.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def local_symgrad(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray, sym: np.ndarray
    ) -> np.ndarray:
        """
        Compute the local symgrad matrix for vector virtual Lagrangian.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            cell (int): The index of the cell on which to compute the div matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.
            sym (np.ndarray): Symmetric matrix.

        Returns:
            np.ndarray: Local symmetric gradient matrix.
        """

        proj = self.scalar_discr.assemble_loc_proj_to_mon(sd, cell, diam, nodes)
        grad = spl.block_diag(*([proj[1:]] * sd.dim))

        return sd.cell_volumes[cell] * sym @ grad / diam

    def assemble_symgrad_symgrad_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Returns the symgrad-symgrad matrix operator for the lowest order
        vector Lagrange element. The matrix is multiplied by twice the Lame' parameter mu.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Additional data, the Lame' parameter mu. Defaults to None.

        Returns:
            sps.csc_matrix: Sparse symgrad-symgrad matrix of shape
                (sd.num_nodes, sd.num_nodes).
                The matrix obtained from the discretization.
        """
        coeff = 2 * data.get("mu", 1) if data is not None else 1
        p0 = pg.PwConstants(self.keyword)

        symgrad = self.assemble_symgrad_matrix(sd)
        mass = p0.assemble_mass_matrix(sd)
        tensor_mass = sps.block_diag([coeff * mass] * np.square(sd.dim), format="csc")

        return symgrad.T @ tensor_mass @ symgrad

    def assemble_penalisation_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles and returns the penalisation matrix.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly process.

        Returns:
            sps.csc_matrix: The penalisation matrix obtained from the discretization.
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

            A = self.assemble_loc_penalisation_matrix(sd, cell, diam, nodes_loc)

            # Save values for local mass matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_V[loc_idx] = A.ravel()
            idx += cols.size

        scalar_pen = sps.csc_matrix((data_V, (rows_I, cols_J)))
        return sps.block_diag([scalar_pen] * sd.dim, format="csc")

    def assemble_loc_penalisation_matrix(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Computes the local penalisation VEM matrix on a given cell
        according to the Hitchhiker's (6.5)

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            cell (int): The index of the cell on which to compute the mass matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of nodes associated with the cell.

        Returns:
            np.ndarray: The computed local VEM mass matrix.
        """
        proj = self.scalar_discr.assemble_loc_proj_to_mon(sd, cell, diam, nodes)

        D = self.scalar_discr.assemble_loc_dofs_of_monomials(sd, cell, diam, nodes)
        I_minus_Pi = np.eye(nodes.size) - D @ proj

        return I_minus_Pi.T @ I_minus_Pi

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        div = self.assemble_div_matrix(sd)
        symgrad = self.assemble_symgrad_matrix(sd)

        return sps.bmat([[symgrad], [div]], format="csc")

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the global stiffness matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid on which the finite element method is defined.
            data (Optional[dict]): Additional data required for the assembly process.

        Returns:
            sps.csc_matrix: The assembled global stiffness matrix.
        """
        # compute the two parts of the global stiffness matrix
        sym_sym = self.assemble_symgrad_symgrad_matrix(sd, data)
        div_div = self.assemble_div_div_matrix(sd, data)

        # penalisation
        dofi_dofi = self.assemble_penalisation_matrix(sd)

        # return the global stiffness matrix
        return sym_sym + div_div + dofi_dofi

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): grid, or a subclass.
                The grid onto which the function will be interpolated.
            func (function): a function that returns the function values at coordinates
                The function to be interpolated.

        Returns:
            array: the values of the degrees of freedom
                The interpolated values of the function on the finite element space.

        NOTE: We are assuming the sd grid in the (x,y) coordinates
        """
        return self.scalar_discr.interpolate(sd, func).ravel(order="F")

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix by evaluating the Lagrange basis functions at the cell
        centers of the given grid.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The evaluation matrix.
        """
        proj = self.scalar_discr.eval_at_cell_centers(sd)
        return sps.block_diag([proj] * sd.dim, format="csc")

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (Tr q, p)_Gamma

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function that defines the
                natural boundary condition.
            b_faces (np.ndarray): List of boundary faces where the natural boundary
                condition is applied.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        bc_val = []
        for d in np.arange(sd.dim):
            f = lambda x: func(x)[d]
            bc_val.append(self.scalar_discr.assemble_nat_bc(sd, f, b_faces))
        return np.hstack(bc_val)

    def get_range_discr_class(self, dim: int) -> object:
        """
        Returns the discretization class that contains the range of the differential.

        Args:
            dim (int): The dimension of the range.

        Returns:
            Discretization: The discretization class that contains the range of
                the differential.

        Raises:
            NotImplementedError: If there is no range discretization for the vector
                Lagrangian 1 in PyGeoN.
        """
        raise NotImplementedError(
            "There's no range discr for the vector Lagrangian 1 in PyGeoN"
        )

    def compute_stress(
        self,
        sd: pg.Grid,
        u: np.ndarray,
        labda: Union[float, np.ndarray],
        mu: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Compute the stress tensor for a given displacement field.

        Args:
            sd (pg.Grid): The spatial discretization object.
            u (ndarray): The displacement field.
            labda (float or ndarray): The first Lamé parameter.
            mu (float or ndarray): The second Lamé parameter.

        Returns:
            ndarray: The stress tensor.
        """
        print("DA METTERE I DATI IN UN DATA; IDEM PER IL CASO FEM")
        # construct the differentials
        symgrad = self.assemble_symgrad_matrix(sd)
        div = self.assemble_div_matrix(sd)

        p0 = pg.PwConstants(self.keyword)
        proj = p0.eval_at_cell_centers(sd)

        # compute the two terms and split on each component
        sigma = np.array(np.split(2 * mu * symgrad @ u, np.square(sd.dim)))
        sigma[:: (sd.dim + 1)] += labda * div @ u

        # compute the actual dofs
        sigma = sigma @ proj

        # create the indices to re-arrange the components for the second
        # order tensor
        idx = np.arange(np.square(sd.dim)).reshape((sd.dim, -1), order="F")

        return sigma[idx].T
