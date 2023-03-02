import numpy as np
import scipy.sparse as sps

import pygeon as pg


class PwConstants(pg.Discretization):
    """
    Discretization class for the piecewise constants.
    NB! Each degree of freedom is the integral over the cell.
    """

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case number of cells.

        Args
            sd: grid, or a subclass.

        Returns
            dof: the number of degrees of freedom.

        """
        return sd.num_cells

    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Computes the mass matrix for piecewise constants

        Args
            sd: grid, or a subclass, with geometry fields computed.
            data: dictionary with possible scaling

        Returns
            matrix: sparse csr (sd.num_cells, sd.num_cells)
        """

        return sps.diags(1 / sd.cell_volumes).tocsc()

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict):
        """
        Computes the lumped mass matrix, which coincides with the mass matrix for P0.
        """

        return self.assemble_mass_matrix(sd, data)

    def assemble_diff_matrix(self, sd: pg.Grid):
        """
        Assembles the matrix corresponding to the differential

        Args
            sd: grid, or a subclass.

        Returns
            diff_matrix: the differential matrix.
        """

        return sps.csr_matrix((0, self.ndof(sd)))

    def assemble_stiff_matrix(self, sd: pg.Grid, data):
        """
        Returns a zero matrix.

        Args
            sd: grid, or a subclass.

        Returns
            diff_matrix: the differential matrix.
        """

        return sps.csr_matrix((self.ndof(sd), self.ndof(sd)))

    def interpolate(self, sd: pg.Grid, func):
        """
        Interpolates a function onto the finite element space

        Args
            sd: grid, or a subclass.
            func: a function that returns the function values at coordinates

        Returns
            array: the values of the degrees of freedom
        """

        return np.array(
            [func(x) * vol for (x, vol) in zip(sd.cell_centers.T, sd.cell_volumes)]
        )

    def eval_at_cell_centers(self, sd: pg.Grid):
        """
        Assembles the matrix

        Args
            sd: grid, or a subclass.

        Returns
            matrix: the evaluation matrix.
        """

        return sps.diags(1 / sd.cell_volumes).tocsc()

    def assemble_nat_bc(self, sd: pg.Grid, func, b_faces):
        """
        Returns a zero vector
        """

        return np.zeros(self.ndof(sd))

    def get_range_discr_class(self, dim: int):
        """
        Raises an error since the range of the differential is zero.
        """

        raise NotImplementedError("There's no zero discretization in PyGeoN (yet)")

    def error_l2(self, sd, num_sol, ana_sol, relative=True, etype="specific"):
        """
        Returns the l2 error computed against an analytical solution given as a function.

        Args
            sd: grid, or a subclass.
            num_sol: np.array, vector of the numerical solution
            ana_sol: callable, function that represent the analytical solution
            relative=True: boolean, compute the relative error or not
            etype="specific": string, type of error computed.

        Returns
            error: the error computed.

        """
        if etype == "standard":
            return super().error_l2(sd, num_sol, ana_sol, relative, etype)

        int_sol = np.array([ana_sol(x) for x in sd.nodes.T])
        proj = self.eval_at_cell_centers(sd)
        num_sol = proj * num_sol

        norm = self._cell_error(sd, np.zeros_like(num_sol), int_sol) if relative else 1
        return self._cell_error(sd, num_sol, int_sol) / norm

    def _cell_error(self, sd, num_sol, int_sol):
        cell_nodes = sd.cell_nodes()
        err = 0
        for c in np.arange(sd.num_cells):
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]
            diff = int_sol[nodes_loc] - num_sol[c]

            err += sd.cell_volumes[c] * diff @ diff.T
        return np.sqrt(err / (sd.dim + 1))
