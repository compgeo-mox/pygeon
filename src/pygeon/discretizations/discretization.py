import numpy as np
import scipy.sparse as sps
import abc

import porepy as pp
import pygeon as pg


class Discretization(pp.numerics.discretization.Discretization):
    """
    Abstract class for pygeon discretization methods.
    For full compatibility, a child class requires the following methods:
        ndof
        assemble_mass_matrix
        assemble_diff_matrix
        interpolate
        eval_at_cell_centers
        assemble_nat_bc
        get_range_discr_class
    """

    def __init__(self, keyword: str) -> None:
        super().__init__(keyword)

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

    @abc.abstractmethod
    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Assembles the mass matrix

        Args
            sd: grid, or a subclass.
            data: optional dictionary with physical parameters for scaling.

        Returns
            mass_matrix: the mass matrix.
        """

        pass

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Assembles the lumped mass matrix given by the row sums on the diagonal.

        Args
            sd: grid, or a subclass.
            data: optional dictionary with physical parameters for scaling.

        Returns
            lumped_matrix: the lumped mass matrix.
        """

        return sps.diags(np.sum(self.assemble_mass_matrix(sd, data), axis=0))

    @abc.abstractmethod
    def assemble_diff_matrix(self, sd: pg.Grid):
        """
        Assembles the matrix corresponding to the differential

        Args
            sd: grid, or a subclass.

        Returns
            csr_matrix: the differential matrix.
        """

        pass

    def assemble_stiff_matrix(self, sd: pg.Grid, data):
        """
        Assembles the stiffness matrix

        Args
            sd: grid, or a subclass.

        Returns
            csr_matrix: the differential matrix.
        """

        B = self.assemble_diff_matrix(sd)

        discr = self.get_range_discr_class(sd.dim)(self.keyword)
        A = discr.assemble_mass_matrix(sd, data)

        return B.T * A * B

    @abc.abstractmethod
    def interpolate(self, sd: pg.Grid, func):
        """
        Interpolates a function onto the finite element space

        Args
            sd: grid, or a subclass.
            func: a function that returns the function values at coordinates

        Returns
            array: the values of the degrees of freedom
        """
        pass

    @abc.abstractmethod
    def eval_at_cell_centers(self, sd: pg.Grid):
        """
        Assembles the matrix

        Args
            sd: grid, or a subclass.

        Returns
            matrix: the evaluation matrix.
        """
        pass

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

    @abc.abstractmethod
    def assemble_nat_bc(self, sd: pg.Grid, func, b_faces):
        """
        Assembles the natural boundary condition term
        (Tr q, p)_Gamma
        """

        pass

    @abc.abstractmethod
    def get_range_discr_class(self, dim: int):
        """
        Returns the discretization class that contains the range of the differential
        """

        pass

    def assemble_matrix_rhs(self, sd: pp.Grid, data: dict):
        """
        Returns a mass matrix and a zero rhs vector.
        This is only implemented for compatibility with Porepy.
        """

        return self.assemble_mass_matrix(sd, data), np.zeros(self.ndof(sd))
