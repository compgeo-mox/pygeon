"""Module for the discretizations of the L2 space."""

import abc
from typing import Callable, Optional, Type

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class PieceWisePolynomial(pg.Discretization):
    """
    PieceWisePolynomial is a subclass of pg.Discretization that represents
    an abstract element wise polynomial discretization.
    """

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case, it returns the number of cells in the grid.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom.

        """
        return sd.num_cells * self.ndof_per_cell(sd)

    def local_dofs_of_cell(self, sd: pg.Grid, c: int) -> np.ndarray:
        return sd.num_cells * np.arange(self.ndof_per_cell(sd)) + c

    @abc.abstractmethod
    def ndof_per_cell(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom per cell.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom per cell.
        """

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Computes the mass matrix for piecewise linears

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (Optional[dict]): Dictionary with possible scaling.

        Returns:
            sps.csc_array: Sparse csc matrix of shape (sd.num_cells, sd.num_cells).
        """
        local_mass = self.assemble_local_mass(sd.dim)

        weight = np.ones(sd.num_cells)
        if data is not None:
            weight = (
                data.get(pp.PARAMETERS, {}).get(self.keyword, {}).get("weight", weight)
            )

        diag_weight = sps.diags_array(sd.cell_volumes * weight)
        M = sps.kron(local_mass, diag_weight)
        M.eliminate_zeros()

        return M.tocsc()

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles the lumped matrix for the given grid.

        Args:
            sd (pg.Grid): The grid object.
            data (Optional[dict]): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped matrix.
        """
        local_mass = self.assemble_local_lumped_mass(sd.dim)

        weight = np.ones(sd.num_cells)
        if data is not None:
            weight = (
                data.get(pp.PARAMETERS, {}).get(self.keyword, {}).get("weight", weight)
            )

        diag_weight = sps.diags_array(sd.cell_volumes * weight)
        M = sps.kron(local_mass, diag_weight).tocsc()
        M.eliminate_zeros()

        return M

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator.

        This method takes a grid object and returns the differential matrix
        corresponding to the given grid.

        Args:
            sd (pg.Grid): The grid object or its subclass.

        Returns:
            sps.csc_array: The differential matrix.

        """
        return sps.csc_array((0, self.ndof(sd)))

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles the stiffness matrix for the given grid.

        Args:
            sd (pg.Grid): The grid or a subclass.
            data (Optional[dict]): Additional data for the assembly process.

        Returns:
            sps.csc_array: The assembled stiffness matrix.
        """
        return sps.csc_array((self.ndof(sd), self.ndof(sd)))

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition vector, equal to zero.

        Args:
            sd (pg.Grid): The grid object.
            func (Callable[[np.ndarray], np.ndarray]): The function defining the
                 natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition vector.
        """
        return np.zeros(self.ndof(sd))

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class for the range of the differential.

        Args:
            dim (int): The dimension of the range space.

        Raises:
            NotImplementedError: There is no zero discretization available in PyGeoN.
        """
        raise NotImplementedError("There's no zero discretization in PyGeoN (yet)")

    @abc.abstractmethod
    def assemble_local_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local mass matrix for piecewise polynomials.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise polynomials.
        """

    @abc.abstractmethod
    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise polynomials

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise polynomials.
        """


class PwConstants(PieceWisePolynomial):
    """
    Discretization class for the piecewise constants.
    NB! Each degree of freedom is the integral over the cell.
    """

    def ndof_per_cell(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom per cell.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom per cell.
        """
        return 1

    def assemble_local_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local mass matrix for piecewise constants

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise constants.
        """
        return np.array([[1]])

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Computes the mass matrix for piecewise constants

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (Optional[dict]): Dictionary with possible scaling.

        Returns:
            sps.csc_array: Sparse csc matrix of shape (sd.num_cells, sd.num_cells).
        """
        M = super().assemble_mass_matrix(sd, data)
        M /= np.square(sd.cell_volumes)

        return M.tocsc()

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Computes the lumped mass matrix, which coincides with the mass matrix for P0.

        Parameters:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (Optional[dict]): Additional data for the assembly process.

        Returns:
            sps.csc_array: The assembled lumped mass matrix.
        """
        return self.assemble_mass_matrix(sd, data)

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): Grid or a subclass.
            func (Callable[[np.ndarray], np.ndarray]): A function that returns the
                function values at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """
        return np.array(
            [func(x) * vol for (x, vol) in zip(sd.cell_centers.T, sd.cell_volumes)]
        )

    def proj_to_pwLinears(self, sd: pg.Grid) -> sps.csc_array:
        """
        Projects the P0 discretization to the P1 discretization.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        return sps.vstack([self.eval_at_cell_centers(sd)] * (sd.dim + 1)).tocsc()

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix that evaluates a function at the cell centers of a grid.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            sps.csc_array: The evaluation matrix.
        """
        return sps.diags_array(1 / sd.cell_volumes).tocsc()

    def error_l2(
        self,
        sd: pg.Grid,
        num_sol: np.ndarray,
        ana_sol: Callable[[np.ndarray], np.ndarray],
        relative: bool = True,
        etype: str = "specific",
        data: Optional[dict] = None,
    ) -> float:
        """
        Returns the l2 error computed against an analytical solution given as a
        function.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            num_sol (np.ndarray): Vector of the numerical solution.
            ana_sol (Callable[[np.ndarray], np.ndarray]): Function that represents the
                analytical solution.
            relative (Optional[bool], optional): Compute the relative error or not.
                Defaults to True.
            etype (Optional[str], optional): Type of error computed. Defaults to
            "specific".

        Returns:
            float: The computed error.
        """
        if etype == "standard":
            return super().error_l2(sd, num_sol, ana_sol, relative, etype)

        int_sol = np.array([ana_sol(x) for x in sd.nodes.T])
        proj = self.eval_at_cell_centers(sd)
        num_sol = proj @ num_sol

        norm = self._cell_error(sd, np.zeros_like(num_sol), int_sol) if relative else 1
        return self._cell_error(sd, num_sol, int_sol) / norm

    def _cell_error(
        self, sd: pg.Grid, num_sol: np.ndarray, int_sol: np.ndarray
    ) -> float:
        """
        Calculate the error for each cell in the finite element mesh.

        Args:
            sd (pg.Grid): The finite element mesh.
            num_sol (np.ndarray): The numerical solution.
            int_sol (np.ndarray): The interpolated solution.

        Returns:
            float: The error for each cell.
        """
        cell_nodes = sd.cell_nodes()
        err = 0
        for c in np.arange(sd.num_cells):
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]
            diff = int_sol[nodes_loc] - num_sol[c]

            err += sd.cell_volumes[c] * diff @ diff.T
        return np.sqrt(err / (sd.dim + 1))


class PwLinears(PieceWisePolynomial):
    """
    Discretization class for piecewise linear finite element method.
    """

    def ndof_per_cell(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom per cell.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom per cell.
        """
        return sd.dim + 1

    def assemble_local_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local mass matrix for piecewise linears

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise linears.
        """
        lagrange1 = pg.Lagrange1(self.keyword)
        return lagrange1.assemble_local_mass(dim)

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise linears

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise linears.
        """
        return np.eye(dim + 1) / (dim + 1)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix for evaluating the discretization at the cell centers.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The evaluation matrix.
        """
        matr = sps.hstack([sps.eye_array(sd.num_cells)] * (sd.dim + 1)) / (sd.dim + 1)
        return matr.tocsc()

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): grid, or a subclass.
            func (Callable): a function that returns the function values at coordinates

        Returns:
            np.ndarray: the values of the degrees of freedom
        """
        cell_nodes = sd.cell_nodes()
        vals = np.zeros((sd.num_cells, sd.dim + 1))

        for c in np.arange(sd.num_cells):
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            vals[c, :] = [func(x) for x in sd.nodes[:, nodes_loc].T]

        return vals.ravel(order="F")

    def proj_to_pwQuadratics(self, sd: pg.Grid) -> sps.csc_array:
        """
        Projects the P1 discretization to the P2 discretization.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        l2 = pg.Lagrange2()
        p1 = pg.PwLinears()
        p2 = pg.PwQuadratics()

        # Local dof mapping
        num_cell_edges = l2.num_edges_per_cell(sd.dim)
        edge_nodes = l2.get_local_edge_nodes(sd.dim).ravel()
        vals = np.concatenate((np.ones(sd.dim + 1), 0.5 * np.ones(num_cell_edges * 2)))

        # Define the vectors for storing the matrix entries
        rows_I = np.empty((sd.num_cells, vals.size), dtype=int)
        cols_J = np.empty((sd.num_cells, vals.size), dtype=int)
        data_IJ = np.tile(vals, (sd.num_cells, 1))

        for c in np.arange(sd.num_cells):
            dofs_p1 = p1.local_dofs_of_cell(sd, c)
            dofs_p2 = p2.local_dofs_of_cell(sd, c)

            rows_I[c] = np.concatenate(
                (dofs_p2[: sd.dim + 1], np.repeat(dofs_p2[sd.dim + 1 :], 2))
            )
            cols_J[c] = np.concatenate((dofs_p1, dofs_p1[edge_nodes]))

        return sps.csc_array((data_IJ.ravel(), (rows_I.ravel(), cols_J.ravel())))


class PwQuadratics(PieceWisePolynomial):
    """
    PwQuadratics is a class that represents piecewise quadratic finite element
    discretizations.
    """

    def ndof_per_cell(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom per cell.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom per cell.
        """
        return (sd.dim + 1) * (sd.dim + 2) // 2

    def assemble_local_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local mass matrix for piecewise quadratics.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise quadratics.
        """
        lagrange2 = pg.Lagrange2(self.keyword)
        return lagrange2.assemble_local_mass(dim)

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise quadratics

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise quadratics.
        """
        if dim == 1:
            raise NotImplementedError(
                "Lumped mass matrix for piecewise quadratics in 1D is not implemented."
            )
        elif dim == 2:
            vals_at_center = np.array([-1, -1, -1, 4, 4, 4]) / 9
            L = 0.75 * np.outer(vals_at_center, vals_at_center)
            L += np.diag(np.array([1, 1, 1, 0, 0, 0])) / 12

        else:
            vals_at_center = np.array([-1, -1, -1, -1, 2, 2, 2, 2, 2, 2]) / 8
            L = 0.8 * np.outer(vals_at_center, vals_at_center)
            L += np.diag(np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])) / 20

        return L

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix for evaluating the discretization at the cell centers.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The evaluation matrix.
        """
        val_at_cc = 1 / (sd.dim + 1)
        eval_nodes = val_at_cc * (2 * val_at_cc - 1) * sps.eye_array(sd.num_cells)
        eval_nodes_stacked = sps.hstack([eval_nodes] * (sd.dim + 1))

        num_edges_per_cells = sd.dim * (sd.dim + 1) // 2
        eval_edges = 4 * val_at_cc * val_at_cc * sps.eye_array(sd.num_cells)
        eval_edges_stacked = sps.hstack([eval_edges] * num_edges_per_cells)

        return sps.hstack((eval_nodes_stacked, eval_edges_stacked)).tocsc()

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): grid, or a subclass.
            func (Callable): a function that returns the function values at degrees of
                freedom

        Returns:
            np.ndarray: the values of the degrees of freedom
        """
        lagrange2 = pg.Lagrange2(self.keyword)
        edge_nodes = lagrange2.get_local_edge_nodes(sd.dim)

        cell_nodes = sd.cell_nodes()
        vals = np.empty((sd.num_cells, self.ndof_per_cell(sd)))

        for c in np.arange(sd.num_cells):
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            vals[c, : sd.dim + 1] = [func(x) for x in sd.nodes[:, nodes_loc].T]

            edge_nodes_loc = nodes_loc[edge_nodes]
            edge_mid_pt = (
                sd.nodes[:, edge_nodes_loc[:, 0]] + sd.nodes[:, edge_nodes_loc[:, 1]]
            )
            edge_mid_pt /= 2

            vals[c, sd.dim + 1 :] = [func(x) for x in edge_mid_pt.T]

        return vals.ravel(order="F")
