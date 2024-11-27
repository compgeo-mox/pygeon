""" Module for the discretizations of the L2 space. """

from typing import Callable, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class PwConstants(pg.Discretization):
    """
    Discretization class for the piecewise constants.
    NB! Each degree of freedom is the integral over the cell.

    Attributes:
        keyword (str): The keyword for the discretization.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Computes the mass matrix for piecewise constants.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Computes the lumped mass matrix, which coincides with the mass matrix for P0.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the matrix corresponding to the differential operator.

        assemble_stiff_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the stiffness matrix for the given grid.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> sps.csc_matrix:
            Interpolates a function onto the finite element space.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the matrix that evaluates a function at the cell centers of a grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition vector, equal to zero.

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the discretization class for the range of the differential.

        error_l2(sd: pg.Grid, num_sol: np.ndarray, ana_sol: Callable[[np.ndarray], np.ndarray],
            relative: Optional[bool] = True, etype: Optional[str] = "specific") -> float:
            Returns the l2 error computed against an analytical solution given as a function.

        _cell_error(sd: pg.Grid, num_sol: np.ndarray, int_sol: np.ndarray) -> float:
            Calculate the error for each cell in the finite element mesh.
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
        return sd.num_cells

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Computes the mass matrix for piecewise constants

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (Optional[dict]): Dictionary with possible scaling.

        Returns:
            sps.csc_matrix: Sparse csc matrix of shape (sd.num_cells, sd.num_cells).
        """
        return sps.diags(1 / sd.cell_volumes).tocsc()

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Computes the lumped mass matrix, which coincides with the mass matrix for P0.

        Parameters:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (Optional[dict]): Additional data for the assembly process.

        Returns:
            sps.csc_matrix: The assembled lumped mass matrix.
        """
        return self.assemble_mass_matrix(sd, data)

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator.

        This method takes a grid object and returns the differential matrix
        corresponding to the given grid.

        Args:
            sd (pg.Grid): The grid object or its subclass.

        Returns:
            sps.csc_matrix: The differential matrix.

        """
        return sps.csc_matrix((0, self.ndof(sd)))

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the stiffness matrix for the given grid.

        Args:
            sd (pg.Grid): The grid or a subclass.
            data (Optional[dict]): Additional data for the assembly process.

        Returns:
            sps.csc_matrix: The assembled stiffness matrix.
        """
        return sps.csc_matrix((self.ndof(sd), self.ndof(sd)))

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> sps.csc_matrix:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): Grid or a subclass.
            func (Callable[[np.ndarray], np.ndarray]): A function that returns the
                function values at coordinates.

        Returns:
            sps.csc_matrix: The values of the degrees of freedom.
        """
        return np.array(
            [func(x) * vol for (x, vol) in zip(sd.cell_centers.T, sd.cell_volumes)]
        )

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix that evaluates a function at the cell centers of a grid.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            sps.csc_matrix: The evaluation matrix.
        """
        return sps.diags(1 / sd.cell_volumes).tocsc()

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

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the discretization class for the range of the differential.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The discretization class for the range of the differential.

        Raises:
            NotImplementedError: If there is no zero discretization available in PyGeoN.
        """
        raise NotImplementedError("There's no zero discretization in PyGeoN (yet)")

    def error_l2(
        self,
        sd: pg.Grid,
        num_sol: np.ndarray,
        ana_sol: Callable[[np.ndarray], np.ndarray],
        relative: Optional[bool] = True,
        etype: Optional[str] = "specific",
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


class PwLinears(pg.Discretization):
    """
    Discretization class for piecewise linear finite element method.

    Attributes:
        keyword (str): The keyword for the discretization.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Computes the mass matrix for piecewise linears.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the matrix corresponding to the differential operator.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition vector. Not implemented.

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the discretization class for the range of the differential.
            Not implemented.

        eval_at_cell_centers(sd: pg.Grid) -> np.ndarray:
            Assembles the matrix for evaluating the discretization at the cell centers.
            Not implemented.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function onto the finite element space. Not implemented.
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
        return sd.num_cells * (sd.dim + 1)

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Computes the mass matrix for piecewise linears

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (Optional[dict]): Dictionary with possible scaling.

        Returns:
            sps.csc_matrix: Sparse csc matrix of shape (sd.num_cells, sd.num_cells).
        """
        # Data allocation
        size = np.square(sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        lagrange1 = pg.Lagrange1(self.keyword)

        try:
            weight = data[pp.PARAMETERS][self.keyword]["weight"]
        except Exception:
            weight = np.ones(sd.num_cells)

        for c in np.arange(sd.num_cells):
            # Compute the mass local matrix
            A = lagrange1.local_mass(sd.cell_volumes[c], sd.dim)
            A *= weight[c]

            # Save values for mass local matrix in the global structure
            nodes_loc = np.arange((sd.dim + 1) * c, (sd.dim + 1) * (c + 1))
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)

            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrix
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the lumped matrix for the given grid.

        Args:
            sd (pg.Grid): The grid object.
            data (Optional[dict]): Optional data dictionary.

        Returns:
            sps.csc_matrix: The assembled lumped matrix.
        """
        try:
            weight = data[pp.PARAMETERS][self.keyword]["weight"]
        except Exception:
            weight = np.ones(sd.num_cells)

        diag = np.repeat(weight * sd.cell_volumes, sd.dim + 1) / (sd.dim + 1)
        return sps.diags(diag, format="csc")

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        raise NotImplementedError

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (Tr q, p)_Gamma

        Args:
            sd (pg.Grid): The grid object.
            func (Callable): The function representing the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        raise NotImplementedError

    def get_range_discr_class(self, dim: int) -> object:
        """
        Returns the discretization class that contains the range of the differential

        Args:
            dim (int): The dimension of the range

        Returns:
            pg.Discretization: The discretization class containing the range of the
                differential
        """
        raise NotImplementedError

    def eval_at_cell_centers(self, sd: pg.Grid) -> np.ndarray:
        """
        Assembles the matrix for evaluating the discretization at the cell centers.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            np.ndarray: The evaluation matrix.
        """

        rows = np.repeat(np.arange(sd.num_cells), sd.dim + 1)
        cols = np.arange(self.ndof(sd))
        data = np.ones(self.ndof(sd)) / (sd.dim + 1)

        return sps.csc_matrix((data, (rows, cols)))

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
        raise NotImplementedError
