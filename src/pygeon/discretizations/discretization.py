import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class Discretization:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

        # Keywords used to identify individual terms in the discretization matrix dictionary
        self.mass_matrix_key = "mass"
        self.diff_matrix_key = "diff"
        self.stiff_matrix_key = "stiff"
        self.lumped_matrix_key = "lumped"

    def discretize(self, sd: pg.Grid, data: dict):
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        matrix_dictionary[self.mass_matrix_key] = self.assemble_mass_matrix(sd, data)
        matrix_dictionary[self.diff_matrix_key] = self.assemble_diff_matrix(sd)
        matrix_dictionary[self.stiff_matrix_key] = self.assemble_stiff_matrix(sd, data)
        matrix_dictionary[self.lumped_matrix_key] = self.assemble_lumped_matrix(
            sd, data
        )

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.

        Args
            sd: grid, or a subclass.

        Returns
            dof: the number of degrees of freedom.
        """

        raise NotImplementedError

    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Assembles the mass matrix

        Args
            sd: grid, or a subclass.
            data: optional dictionary with physical parameters for scaling.

        Returns
            mass_matrix: the mass matrix.
        """

        raise NotImplementedError

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Assembles the lumped mass matrix given by the row sums on the diagonal.

        Args
            sd: grid, or a subclass.
            data: optional dictionary with physical parameters for scaling.

        Returns
            lumped_matrix: the lumped mass matrix.
        """

        diagonal = np.sum(self.assemble_mass_matrix(sd, data), axis=0)

        return sps.diags(diagonal)

    def assemble_diff_matrix(self, sd: pg.Grid):
        """
        Assembles the matrix corresponding to the differential

        Args
            sd: grid, or a subclass.

        Returns
            csr_matrix: the differential matrix.
        """

        raise NotImplementedError

    def assemble_stiff_matrix(self, sd: pg.Grid, data):
        """
        Assembles the stiffness matrix

        Args
            sd: grid, or a subclass.

        Returns
            csr_matrix: the differential matrix.
        """

        B = self.assemble_diff_matrix(sd)

        discr = self.get_range_discr_class()(self.keyword)
        A = discr.assemble_mass_matrix(sd, data)

        return B.T * A * B

    def interpolate(self, sd: pg.Grid, func):
        """
        Interpolates a function onto the finite element space

        Args
            sd: grid, or a subclass.
            func: a function that returns the function values at coordinates

        Returns
            array: the values of the degrees of freedom
        """
        raise NotImplementedError

    def eval_at_cell_centers(self, sd: pg.Grid):
        """
        Assembles the matrix

        Args
            sd: grid, or a subclass.

        Returns
            matrix: the evaluation matrix.
        """
        raise NotImplementedError

    def source_term(self, sd: pg.Grid, func):
        """
        Assembles the source term by interpolating the given function
        and multiplying by the mass matrix

        Args
            sd: grid, or a subclass.
            func: a function that returns the function values at coordinates

        Returns
            matrix: the evaluation matrix.
        """

        return self.assemble_mass_matrix(sd) * self.interpolate(sd, func)

    def assemble_nat_bc(self, sd: pg.Grid, b_dofs):
        """
        Assembles the natural boundary condition term

        """
        raise NotImplementedError

    def get_range_discr_class(self):
        """
        Returns the discretization class that contains the range of the differential

        """
        raise NotImplementedError
