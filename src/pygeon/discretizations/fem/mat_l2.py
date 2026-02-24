"""Module for the discretizations of the matrix L2 space."""

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class MatPwPolynomials(pg.VecPwPolynomials):
    """
    Base class for matrix-valued piecewise polynomial discretizations.
    """

    poly_order: int
    """Polynomial degree of the basis functions"""

    tensor_order = pg.MATRIX
    """Matrix-valued discretization"""

    def assemble_mass_matrix_elasticity(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles and returns the elasticity inner product matrix, which is given by
        :math:`(A \\sigma, \\tau)` where

        .. math::

            A \\sigma = \\frac{1}{2\\mu} \\left[ \\sigma - c
            \\text{Tr}(\\sigma) I\\right]

        with :math:`\\mu` and :math:`\\lambda` the Lamé constants and

        .. math::

            c = \\frac{\\lambda}{2\\mu + d \\lambda}

        where :math:`d` is the dimension.

        Args:
            sd (pg.Grid): The grid.
            data (dict): Data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """
        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        lambda_ = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        # Save 1/(2mu) so that it can be read by self
        data_self = pp.initialize_data({}, self.keyword, {pg.WEIGHT: 1 / (2 * mu)})

        # Save the coefficient for the trace contribution
        comp = ~np.isinf(lambda_)
        coeff = 1 / sd.dim / (2 * mu)
        coeff[comp] = (
            lambda_[comp] / (2 * mu[comp] + sd.dim * lambda_[comp]) / (2 * mu[comp])
        )

        data_tr_space = pp.initialize_data({}, self.keyword, {pg.WEIGHT: coeff})

        # Assemble the block diagonal mass matrix
        D = self.assemble_mass_matrix(sd, data_self)
        # Assemble the trace part
        B = self.assemble_trace_matrix(sd)

        # Assemble the piecewise linear mass matrix, to assemble the term
        # (Tr(sigma), Tr(tau))
        scalar_discr = pg.get_PwPolynomials(self.poly_order, pg.SCALAR)(self.keyword)
        M = scalar_discr.assemble_mass_matrix(sd, data_tr_space)

        # Compose all the parts and return them
        return D - B.T @ M @ B

    def assemble_mass_matrix_cosserat(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles and returns the Cosserat inner product, which is given by
        :math:`(A \\sigma, \\tau)` where

        .. math::

            A \\sigma = \\frac{1}{2\\mu} \\left( \\text{sym}(\\sigma)
            - c \\text{Tr}(\\sigma) I \\right)
            + \\frac{1}{2\\mu_c} \\text{skw}(\\sigma)

        with :math:`\\mu` and :math:`\\lambda` the Lamé constants,
        :math:`\\mu_c` the coupling Lamé modulus, and

        .. math::

            c = \\frac{\\lambda}{2\\mu + d \\lambda}

        where :math:`d` is the dimension.

        Args:
            sd (pg.Grid): The grid.
            data (dict): Data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """
        M = self.assemble_mass_matrix_elasticity(sd, data)

        # Extract the data
        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        mu_c = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU_COSSERAT)

        weight = 0.25 * (1 / mu_c - 1 / mu)
        data_ = pp.initialize_data({}, self.keyword, {pg.WEIGHT: weight})

        if sd.dim == 2:
            R_tensor_order = pg.SCALAR
        elif sd.dim == 3:
            R_tensor_order = pg.VECTOR
        else:
            raise ValueError

        R_space = pg.get_PwPolynomials(self.poly_order, R_tensor_order)(self.keyword)
        R_mass = R_space.assemble_mass_matrix(sd, data_)

        asym = self.assemble_asym_matrix(sd)

        return M + asym.T @ R_mass @ asym

    def assemble_lumped_matrix_elasticity(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the lumped matrix for the given grid.

        Args:
            sd (pg.Grid): The grid object.
            data (dict | None): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped matrix.
        """
        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        lambda_ = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        weight_M = lambda_ / (2 * mu + sd.dim * lambda_) / (2 * mu)
        weight_D = 1 / (2 * mu)

        # Assemble the block diagonal mass matrix for the base discretization class
        data_D = pp.initialize_data({}, self.keyword, {pg.WEIGHT: weight_D})
        D = self.assemble_lumped_matrix(sd, data_D)

        # Assemble the trace part
        B = self.assemble_trace_matrix(sd)

        # Assemble the piecewise linear mass matrix, to assemble the term
        # (Trace(sigma), Trace(tau))
        data_M = pp.initialize_data({}, self.keyword, {pg.WEIGHT: weight_M})

        scalar_discr = pg.get_PwPolynomials(self.poly_order, pg.SCALAR)(self.keyword)
        M = scalar_discr.assemble_lumped_matrix(sd, data_M)

        # Compose all the parts and return them
        return D - B.T @ M @ B

    def assemble_lumped_matrix_cosserat(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the lumped matrix with Cosserat terms for the given grid.

        Args:
            sd (pg.Grid): The grid object.
            data (dict | None): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped matrix.
        """
        M = self.assemble_lumped_matrix_elasticity(sd, data)

        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        mu_c = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU_COSSERAT)

        if sd.dim == 2:
            R_tensor_order = pg.SCALAR
        elif sd.dim == 3:
            R_tensor_order = pg.VECTOR
        else:
            raise ValueError

        weight = 0.25 * (1 / mu_c - 1 / mu)
        data_R = pp.initialize_data({}, self.keyword, {pg.WEIGHT: weight})

        R_space = pg.get_PwPolynomials(self.poly_order, R_tensor_order)(self.keyword)
        R_mass = R_space.assemble_lumped_matrix(sd, data_R)

        asym = self.assemble_asym_matrix(sd)

        return M + asym.T @ R_mass @ asym

    def assemble_trace_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the trace matrix for the matrix-valued piecewise
        polynomials.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The trace matrix obtained from the discretization.

        Raises:
            NotImplementedError: This method is not implemented for the base class.
        """
        # Extract the number of degrees of freedom for the underlying scalar space.
        scalar_ndof = self.ndof(sd) // (sd.dim**2)

        # If the matrix-valued function is raveled into a vector, then the trace
        # operator becomes a linear operation on that vector. This is a matrix-vector
        # product with the matrix:
        # 1D: [1]
        # 2D: [1 0 0 1]
        # 3D: [1 0 0 0 1 0 0 0 1]
        trace = np.eye(sd.dim).reshape((1, -1))

        return sps.kron(trace, sps.eye_array(scalar_ndof), format="csc")

    def assemble_asym_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the asymmetry matrix for the matrix-valued piecewise
        polynomials.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The asymmetry matrix obtained from the discretization.

        Raises:
            NotImplementedError: This method is not implemented for the base class.
        """
        # Extract the number of degrees of freedom for the underlying scalar space.
        scalar_ndof = self.ndof(sd) // (sd.dim**2)

        # If the matrix-valued function is raveled into a vector, then the trace
        # operator becomes a linear operation on that vector. This is a matrix-vector
        # product with the matrix:
        match sd.dim:
            case 2:
                asym = np.array([[0, -1, 1, 0]])
            case 3:
                # [0  0 0 0 0 -1  0 1 0]
                # [0  0 1 0 0  0 -1 0 0]
                # [0 -1 0 1 0  0  0 0 0]
                asym = np.zeros((3, 9))
                asym[[0, 1, 2], [7, 2, 3]] = 1
                asym[[0, 1, 2], [5, 6, 1]] = -1
            case _:
                raise ValueError("The grid should be either two or three-dimensional")

        return sps.kron(asym, sps.eye_array(scalar_ndof), format="csc")


class MatPwConstants(MatPwPolynomials):
    """
    A class representing the discretization using matrix piecewise constant functions.
    """

    poly_order = 0
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwConstants.

        Args:
            keyword (str): The keyword for the matrix discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwConstants(keyword)


class MatPwLinears(MatPwPolynomials):
    """
    A class representing the discretization using matrix piecewise linear functions.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwLinears.

        Args:
            keyword (str): The keyword for the matrix discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwLinears(keyword)

    def assemble_corotational_correction(
        self, sd: pg.Grid, rotation: np.ndarray
    ) -> sps.csc_array:
        """
        Assembles and returns the corotational correction matrix for the matrix-valued
        piecewise linears. We assume rotation to be a piecewise constant function in P0.

        Args:
            sd (pg.Grid): The grid.
            rotation (np.ndarray): The rotation in P0, either a scalar field in 2D or
                a vector field in 3D.

        Returns:
            sps.csc_array: The corotational correction matrix obtained from the
                discretization.
        """
        # Retrieve the discretization for the rotation, it depends on the dimension
        disc_rot: pg.Discretization
        if sd.dim == 2:
            disc_rot = pg.PwConstants(self.keyword)
        else:
            disc_rot = pg.VecPwConstants(self.keyword)

        # The idea is to project the rotation from P0 to P1 to be able to
        # perform the assembly with the P1 asymmetry matrix
        proj_p1 = disc_rot.proj_to_higher_PwPolynomials(sd)

        # Assemble the asymmetry matrix in P1 space
        asym = self.assemble_asym_matrix(sd)

        # Convert the rotation to the Omega tensor, which is in matrix P0 space
        omega = -asym.T @ proj_p1 @ rotation

        # Assemble the multiplication matrices A*Omega and Omega*A used in the
        # corotational correction
        A_omega = self.assemble_mult_matrix(sd, omega, right_mult=True)
        omega_A = self.assemble_mult_matrix(sd, omega, right_mult=False)

        # Return the corotational correction matrix
        return A_omega - omega_A

    def assemble_mult_matrix(
        self, sd: pg.Grid, mult_mat: np.ndarray, right_mult: bool
    ) -> sps.csc_array:
        """
        Assembles and returns the multiplication matrix for the matrix-valued
        piecewise constants.

        Args:
            sd (pg.Grid): The grid.
            mult_mat (np.ndarray): The matrix to multiply with. It is assumed to be
                a piecewise constant matrix.
            right_mult (bool): If True, performs right multiplication. If False, left
                multiplication.

        Returns:
            sps.csc_array: The multiplication matrix obtained from the discretization.
        """
        size = sd.dim**4 * (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Precompute the basis functions
        basis = np.eye(sd.dim**2).reshape((sd.dim, sd.dim, -1))

        if right_mult:
            # right multiplication: A @ mult_mat
            oper_str = "ijk,jp->ipk"
        else:
            # left multiplication: mult_mat @ A
            oper_str = "ijk,pi->pjk"

        for c in range(sd.num_cells):
            # Get the local degrees of freedom for the cell ordered column wise
            loc_dofs = self.local_dofs_of_cell(sd, c).reshape((-1, sd.dim + 1))

            # Iterate over the node dofs
            for node_dofs in loc_dofs.T:
                mult_mat_loc = mult_mat[node_dofs].reshape((sd.dim, sd.dim))

                # Compute the product A @ mult_mat (if right_mult is True)
                # or mult_mat @ A (if right_mult is False)
                prod = np.einsum(oper_str, basis, mult_mat_loc)

                # Reshape the product to match the degrees of freedom
                prod_at_dofs = np.array(
                    [prod[:, :, i].ravel() for i in np.arange(sd.dim**2)]
                )

                # Save only the non-zeros values of the local matrix in the global
                # structure
                rows_loc, cols_loc = prod_at_dofs.nonzero()

                # Save values of the local matrix in the global structure
                loc_idx = slice(idx, idx + rows_loc.size)
                rows_I[loc_idx] = node_dofs[cols_loc]
                cols_J[loc_idx] = node_dofs[rows_loc]
                data_IJ[loc_idx] = prod_at_dofs[rows_loc, cols_loc]
                idx += rows_loc.size

        # Construct the global matrices
        shape = (self.ndof(sd), self.ndof(sd))
        return sps.csc_array((data_IJ[:idx], (rows_I[:idx], cols_J[:idx])), shape=shape)


class MatPwQuadratics(MatPwPolynomials):
    """
    A class representing the discretization using matrix piecewise quadratic functions.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwQuadratics.

        Args:
            keyword (str): The keyword for the matrix discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwQuadratics(keyword)
