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
    an abstract elementwise polynomial discretization.

    Attributes:
        keyword (str): The keyword for the discretization.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated with the method.

        ndof_per_cell(sd: pg.Grid) -> int:
            Abstract method that returns the number of degrees of freedom per cell.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_array:
            Assembles and returns the matrix corresponding to the differential operator for
            the given grid.

        assemble_stiff_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Assembles and returns the stiffness matrix for the given grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces:
            np.ndarray) -> np.ndarray:
            Assembles and returns the natural boundary condition vector, which is equal to
            zero.

        get_range_discr_class(dim: int) -> Type[pg.Discretization]:
            Returns the discretization class for the range of the differential.
            Raises NotImplementedError if not available.
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

    @abc.abstractmethod
    def ndof_per_cell(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom per cell.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom per cell.
        """

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


class PwConstants(PieceWisePolynomial):
    """
    Discretization class for the piecewise constants.
    NB! Each degree of freedom is the integral over the cell.

    Attributes:
        keyword (str): The keyword for the discretization.

    Methods:
        ndof_per_cell(sd: pg.Grid) -> int:
            Method that returns the number of degrees of freedom per cell.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Computes the mass matrix for piecewise constants.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Computes the lumped mass matrix, which coincides with the mass matrix for P0.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> sps.csc_array:
            Interpolates a function onto the finite element space.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_array:
            Assembles the matrix that evaluates a function at the cell centers of a grid.

        error_l2(sd: pg.Grid, num_sol: np.ndarray, ana_sol: Callable[[np.ndarray], np.ndarray],
            relative: Optional[bool] = True, etype: Optional[str] = "specific") -> float:
            Returns the l2 error computed against an analytical solution given as a function.

        _cell_error(sd: pg.Grid, num_sol: np.ndarray, int_sol: np.ndarray) -> float:
            Calculate the error for each cell in the finite element mesh.
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
        return sps.diags_array(1 / sd.cell_volumes).tocsc()

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
        Returns the l2 error computed against an analytical solution given as a function.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            num_sol (np.ndarray): Vector of the numerical solution.
            ana_sol (Callable[[np.ndarray], np.ndarray]): Function that represents the
                analytical solution.
            relative (Optional[bool], optional): Compute the relative error or not.
                Defaults to True.
            etype (Optional[str], optional): Type of error computed. Defaults to "specific".

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

    Attributes:
        keyword (str): The keyword for the discretization.

    Methods:
        ndof_per_cell(sd: pg.Grid) -> int:
            Abstract method that returns the number of degrees of freedom per cell.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Computes the mass matrix for piecewise linears.

        eval_at_cell_centers(sd: pg.Grid) -> np.ndarray:
            Assembles the matrix for evaluating the discretization at the cell centers.
            Not implemented.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function onto the finite element space. Not implemented.
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
        # Data allocation
        size = np.square(sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        lagrange1 = pg.Lagrange1(self.keyword)
        local_mass = lagrange1.local_mass(sd.dim)

        weight = np.ones(sd.num_cells)
        if data is not None:
            weight = (
                data.get(pp.PARAMETERS, {}).get(self.keyword, {}).get("weight", weight)
            )

        for c in np.arange(sd.num_cells):
            # Compute the mass local matrix
            A = local_mass * sd.cell_volumes[c] * weight[c]

            # Save values for mass local matrix in the global structure
            dofs_loc = sd.num_cells * np.arange(sd.dim + 1) + c
            cols = np.tile(dofs_loc, (dofs_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)

            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrix
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

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
        weight = np.ones(sd.num_cells)
        if data is not None:
            weight = (
                data.get(pp.PARAMETERS, {}).get(self.keyword, {}).get("weight", weight)
            )

        diag = np.tile(weight * sd.cell_volumes / (sd.dim + 1), sd.dim + 1)
        return sps.diags_array(diag).tocsc()

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


class PwQuadratics(PieceWisePolynomial):
    """
    PwQuadratics is a class that represents piecewise quadratic finite element discretizations.

    Attributes:
        keyword (str): The keyword for the discretization.

    Methods:
        ndof_per_cell(sd: pg.Grid) -> int:
            Method that returns the number of degrees of freedom per cell.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Computes the mass matrix for piecewise quadratics.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_array:
            Assembles the matrix for evaluating the discretization at the cell centers.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function onto the finite element space.
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

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Computes the mass matrix for piecewise quadratics.

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (Optional[dict]): Dictionary with possible scaling.

        Returns:
            sps.csc_array: Sparse csc matrix of shape (ndof, ndof).
        """
        # Data allocation
        ndof_per_cell = self.ndof_per_cell(sd)
        size = np.square(ndof_per_cell) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        lagrange2 = pg.Lagrange2(self.keyword)
        local_mass = lagrange2.assemble_local_mass(sd.dim)

        weight = np.ones(sd.num_cells)
        if data is not None:
            weight = (
                data.get(pp.PARAMETERS, {}).get(self.keyword, {}).get("weight", weight)
            )

        for c in np.arange(sd.num_cells):
            # Compute the mass local matrix
            A = local_mass * sd.cell_volumes[c] * weight[c]

            # Save values for mass local matrix in the global structure
            dof_loc = np.arange(ndof_per_cell) * sd.num_cells + c
            cols = np.tile(dof_loc, (dof_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)

            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrix
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

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
