"""Module for the discretizations of the H(div) space."""

from typing import Optional, Type

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class VecBDM1(pg.VecDiscretization):
    """
    VecBDM1 is a class that represents the vector BDM1 (Brezzi-Douglas-Marini) finite
    element method. It provides methods for assembling matrices like the mass matrix,
    the trace matrix, the asymmetric matrix and the differential matrix. It also
    provides methods for evaluating the solution at cell centers, interpolating a given
    function onto the grid, assembling the natural boundary condition term, and more.

    Attributes:
        keyword (str): The keyword associated with the vector BDM1 method.

    Methods:
        ndof(sd: pp.Grid) -> int:
            Return the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Assembles the mass matrix for the given grid.

        assemble_trace_matrix(sd: pg.Grid) -> sps.csc_array:
            Assembles the trace matrix for the vector BDM1.

        assemble_asym_matrix(sd: pg.Grid) -> sps.csc_array:
            Assembles the asymmetric matrix for the vector BDM1.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_array:
            Assembles the matrix corresponding to the differential operator.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_array:
            Evaluate the finite element solution at the cell centers of the given grid.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray])
            -> np.ndarray:
            Interpolates a given function onto the grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition term.

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the range discretization class for the given dimension.
    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector BDM1 discretization class.
        The scalar discretization class is pg.BDM1.

        We are considering the following structure of the stress tensor in 2d

        sigma = [[sigma_xx, sigma_xy],
                 [sigma_yx, sigma_yy]]

        which is represented in the code unrolled row-wise as a vector of length 4

        sigma = [sigma_xx, sigma_xy,
                 sigma_yx, sigma_yy]

        While in 3d the stress tensor can be written as

        sigma = [[sigma_xx, sigma_xy, sigma_xz],
                 [sigma_yx, sigma_yy, sigma_yz],
                 [sigma_zx, sigma_zy, sigma_zz]]

        where its vectorized structure of lenght 9 is given by

        sigma = [sigma_xx, sigma_xy, sigma_xz,
                 sigma_yx, sigma_yy, sigma_yz,
                 sigma_zx, sigma_zy, sigma_zz]

        Args:
            keyword (str): The keyword for the vector discretization class.

        Returns:
            None
        """
        self.scalar_discr: pg.BDM1
        super().__init__(keyword, pg.BDM1)

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles and returns the mass matrix for vector BDM1, which is given by
        (A sigma, tau) where A sigma = (sigma - coeff * Trace(sigma) * I) / (2 mu)
        with mu and lambda the Lamé constants and coeff = lambda / (2*mu + dim*lambda)

        Args:
            sd (pg.Grid): The grid.
            data (dict): Data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """
        if data is None:
            # If the data is not provided then use default values to build a block
            # diagonal mass matrix without the trace term
            mu = 0.5 * np.ones(sd.num_cells)
            lambda_ = np.zeros(sd.num_cells)
        else:
            # Extract the data
            mu = data[pp.PARAMETERS][self.keyword]["mu"]
            lambda_ = data[pp.PARAMETERS][self.keyword]["lambda"]

        # If mu is a scalar, replace it by a vector so that it can be accessed per cell
        if isinstance(mu, np.ScalarType):
            mu = np.full(sd.num_cells, mu)

        # Save 1/(2mu) as a tensor so that it can be read by BDM1
        mu_tensor = pp.SecondOrderTensor(1 / (2 * mu))
        data_for_BDM = pp.initialize_data(
            sd, {}, self.keyword, {"second_order_tensor": mu_tensor}
        )

        # Save the coefficient for the trace contribution
        coeff = lambda_ / (2 * mu + sd.dim * lambda_) / (2 * mu)
        data_for_PwL = pp.initialize_data(sd, {}, self.keyword, {"weight": coeff})

        # Assemble the block diagonal mass matrix for the base discretization class
        D = super().assemble_mass_matrix(sd, data_for_BDM)
        # Assemble the trace part
        B = self.assemble_trace_matrix(sd)

        # Assemble the piecewise linear mass matrix, to assemble the term
        # (Trace(sigma), Trace(tau))
        discr = pg.PwLinears(self.keyword)
        M = discr.assemble_mass_matrix(sd, data_for_PwL)

        # Compose all the parts and return them
        return D - B.T @ M @ B

    def assemble_mass_matrix_cosserat(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assembles and returns the mass matrix for vector BDM1 discretizing the Cosserat
        inner product, which is given by (A sigma, tau) where
        A sigma = (sym(sigma) - coeff * Trace(sigma) * I) / (2 mu)
                  + skw(sigma) / (2 mu_c)
        with mu and lambda the Lamé constants, coeff = lambda / (2*mu + dim*lambda), and
        mu_c the coupling Lamé modulus.

        Args:
            sd (pg.Grid): The grid.
            data (dict): Data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """
        M = self.assemble_mass_matrix(sd, data)

        # Extract the data
        mu = data[pp.PARAMETERS][self.keyword]["mu"]
        mu_c = data[pp.PARAMETERS][self.keyword]["mu_c"]

        coeff = 0.25 * (1 / mu_c - 1 / mu)

        # If coeff is a scalar, replace it by a vector so that it can be accessed per
        # cell
        if isinstance(coeff, np.ScalarType):
            coeff = np.full(sd.num_cells, coeff)

        data_for_R = pp.initialize_data(sd, {}, self.keyword, {"weight": coeff})

        R_space: pg.Discretization
        if sd.dim == 2:
            R_space = pg.PwLinears(self.keyword)
        elif sd.dim == 3:
            R_space = pg.VecPwLinears(self.keyword)

        R_mass = R_space.assemble_mass_matrix(sd, data_for_R)

        asym = self.assemble_asym_matrix(sd, False)

        return M + asym.T @ R_mass @ asym

    def assemble_trace_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the trace matrix for the vector BDM1.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The trace matrix obtained from the discretization.
        """
        # overestimate the size
        size = np.square((sd.dim + 1) * sd.dim) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        opposite_nodes = sd.compute_opposite_nodes()
        scalar_ndof = self.scalar_discr.ndof(sd)

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its faces and
            # determine the location of the dof
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            opposites_loc = opposite_nodes.data[loc]

            Psi = self.scalar_discr.eval_basis_at_node(sd, opposites_loc, faces_loc)

            # Get all the components of the basis at node
            Psi_i, Psi_j = np.nonzero(Psi)
            Psi_v = Psi[Psi_i, Psi_j]  # type: ignore[call-overload]

            loc_ind = np.hstack([faces_loc] * sd.dim)
            loc_ind += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces

            cols = np.tile(loc_ind, (3, 1))
            cols[1, :] += scalar_ndof
            cols[2, :] += 2 * scalar_ndof
            cols = np.tile(cols, (sd.dim + 1, 1)).T
            cols = cols[Psi_i, Psi_j]

            nodes_loc = sd.num_cells * np.arange(sd.dim + 1) + c

            rows = np.repeat(nodes_loc, 3)[Psi_j]

            # Save values of the local matrix in the global structure
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = rows
            cols_J[loc_idx] = cols
            data_IJ[loc_idx] = Psi_v
            idx += cols.size

        # Construct the global matrices
        return sps.csc_array((data_IJ[:idx], (rows_I[:idx], cols_J[:idx])))

    def assemble_asym_matrix(self, sd: pg.Grid, as_pwconstant=True) -> sps.csc_array:
        """
        Assembles and returns the asymmetric matrix for the vector BDM1.

        The asymmetric operator `as' for a tensor is a scalar and it is defined in 2d as
        as(tau) = tau_yx - tau_xy
        while for a tensor in 3d it is a vector and given by
        as(tau) = [tau_zy - tau_yz, tau_xz - tau_zx, tau_yx - tau_xy]^T

        Note: We assume that the as(tau) is a piecewise linear.

        Args:
            sd (pg.Grid): The grid.
            as_pwconstant (bool): Compute the operator with the range on the piece-wise
                constant (default), otherwise the mapping is on the piece-wise linears.

        Returns:
            sps.csc_array: The asymmetric matrix obtained from the discretization.
        """

        # overestimate the size
        size = np.square((sd.dim + 1) * sd.dim) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Helper functions for inside the loop
        negate_col = [2, 0, 1]
        zeroed_col = [0, 1, 2]

        rot_space: pg.Discretization
        if sd.dim == 3:
            ind_list = np.arange(3)
            shift = ind_list
            rot_space = pg.VecPwLinears(self.keyword)
            scaling = sps.diags_array(np.tile(sd.cell_volumes, 3))
        elif sd.dim == 2:
            ind_list = np.array([2])
            shift = np.array([0, 0, 0])
            rot_space = pg.PwLinears(self.keyword)
            scaling = sps.diags_array(sd.cell_volumes)
        else:
            raise ValueError("The grid should be either two or three-dimensional")

        opposite_nodes = sd.compute_opposite_nodes()
        ndof_scalar = self.scalar_discr.ndof(sd)

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its faces and
            # determine the location of the dof
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            opposites_loc = opposite_nodes.data[loc]

            Psi = self.scalar_discr.eval_basis_at_node(sd, opposites_loc, faces_loc)

            # Get all the components of the basis at node
            Psi_i, Psi_j = np.nonzero(Psi)
            Psi_v = Psi[Psi_i, Psi_j]  # type: ignore[call-overload]

            for ind in ind_list:
                Psi_v_copy = Psi_v.copy()
                Psi_v_copy[np.mod(Psi_j, 3) == negate_col[ind]] *= -1
                Psi_v_copy[np.mod(Psi_j, 3) == zeroed_col[ind]] *= 0

                loc_ind = np.tile(faces_loc, sd.dim)
                loc_ind += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces

                cols = np.tile(loc_ind, (3, 1))
                cols[0, :] += np.mod(-ind, 3) * ndof_scalar
                cols[1, :] += np.mod(-ind - 1, 3) * ndof_scalar
                cols[2, :] += np.mod(-ind - 2, 3) * ndof_scalar

                cols = np.tile(cols, (sd.dim + 1, 1)).T

                cols = cols[Psi_i, Psi_j]

                nodes_loc = sd.num_cells * np.arange(sd.dim + 1) + c

                rows = np.repeat(nodes_loc, 3)[Psi_j]

                # Save values of the local matrix in the global structure
                loc_idx = slice(idx, idx + cols.size)
                rows_I[loc_idx] = rows + shift[ind] * (sd.dim + 1) * sd.num_cells
                cols_J[loc_idx] = cols
                data_IJ[loc_idx] = Psi_v_copy
                idx += cols.size

        # Construct the global matrices
        asym = sps.csc_array((data_IJ[:idx], (rows_I[:idx], cols_J[:idx])))

        # Return the operator that maps to the piece-wise constant
        if as_pwconstant:
            return scaling @ rot_space.eval_at_cell_centers(sd) @ asym
        else:
            return asym

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
        if data is None:
            raise ValueError("Data must be provided for the assembly")

        # Assemble the block diagonal mass matrix for the base discretization class
        D = super().assemble_lumped_matrix(sd)

        # Assemble the trace part
        B = self.assemble_trace_matrix(sd)

        # Assemble the piecewise linear mass matrix, to assemble the term
        # (Trace(sigma), Trace(tau))
        discr = pg.PwLinears(self.keyword)
        M = discr.assemble_lumped_matrix(sd)

        # Extract the data and compute the coefficient for the trace part
        mu = data[pp.PARAMETERS][self.keyword]["mu"]
        lambda_ = data[pp.PARAMETERS][self.keyword]["lambda"]
        coeff = lambda_ / (2 * mu + sd.dim * lambda_)

        # Compose all the parts and return them
        return (D - coeff * B.T @ M @ B) / (2 * mu)

    def assemble_lumped_matrix_cosserat(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assembles the lumped matrix with cosserat terms for the given grid.

        Args:
            sd (pg.Grid): The grid object.
            data (Optional[dict]): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped matrix.
        """
        M = self.assemble_lumped_matrix(sd, data)

        # Extract the data
        mu = data[pp.PARAMETERS][self.keyword]["mu"]
        mu_c = data[pp.PARAMETERS][self.keyword]["mu_c"]

        coeff = 0.25 * (1 / mu_c - 1 / mu)

        # If coeff is a scalar, replace it by a vector so that it can be accessed per
        # cell
        if isinstance(coeff, np.ScalarType):
            coeff = np.full(sd.num_cells, coeff)

        data_for_R = pp.initialize_data(sd, {}, self.keyword, {"weight": coeff})

        R_space: pg.Discretization
        if sd.dim == 2:
            R_space = pg.PwLinears(self.keyword)
        elif sd.dim == 3:
            R_space = pg.VecPwLinears(self.keyword)

        R_mass = R_space.assemble_lumped_matrix(sd, data_for_R)

        asym = self.assemble_asym_matrix(sd, False)

        return M + asym.T @ R_mass @ asym

    def proj_to_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the function space to the lowest order Raviart-Thomas (RT0) space.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.

        Returns:
            sps.csc_array: The projection matrix to the RT0 space.
        """
        proj = self.scalar_discr.proj_to_RT0(sd)
        return sps.block_diag([proj] * sd.dim).tocsc()

    def proj_from_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the RT0 finite element space onto the faces of the given grid.

        Args:
            sd (pg.Grid): The grid on which the projection is performed.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.scalar_discr.proj_from_RT0(sd)
        return sps.block_diag([proj] * sd.dim).tocsc()

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class that contains the range of the differential

        Args:
            dim (int): The dimension of the range

        Returns:
            pg.Discretization: The discretization class containing the range of the
                differential
        """
        return pg.VecPwConstants


class VecRT0(pg.VecDiscretization):
    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector RT0 discretization class.
        The scalar discretization class is pg.RT0.

        We are considering the following structure of the stress tensor in 2d

        sigma = [[sigma_xx, sigma_xy],
                 [sigma_yx, sigma_yy]]

        which is represented in the code unrolled row-wise as a vector of length 4

        sigma = [sigma_xx, sigma_xy,
                 sigma_yx, sigma_yy]

        While in 3d the stress tensor can be written as

        sigma = [[sigma_xx, sigma_xy, sigma_xz],
                 [sigma_yx, sigma_yy, sigma_yz],
                 [sigma_zx, sigma_zy, sigma_zz]]

        where its vectorized structure of length 9 is given by

        sigma = [sigma_xx, sigma_xy, sigma_xz,
                 sigma_yx, sigma_yy, sigma_yz,
                 sigma_zx, sigma_zy, sigma_zz]

        Args:
            keyword (str): The keyword for the vector discretization class.

        Returns:
            None
        """
        self.scalar_discr: pg.RT0
        super().__init__(keyword, pg.RT0)

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles and returns the mass matrix for vector RT0, which is given by
        (A sigma, tau) where A sigma = (sigma - coeff * Trace(sigma) * I) / (2 mu)
        with mu and lambda the Lamé constants and coeff = lambda / (2*mu + dim*lambda)

        Args:
            sd (pg.Grid): The grid.
            data (dict): Data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """
        if data is None:
            # If the data is not provided then use default values to build a block
            # diagonal mass matrix without the trace term
            mu = 0.5 * np.ones(sd.num_cells)
            lambda_ = np.zeros(sd.num_cells)
        else:
            # Extract the data
            mu = data[pp.PARAMETERS][self.keyword]["mu"]
            lambda_ = data[pp.PARAMETERS][self.keyword]["lambda"]

        # If mu is a scalar, replace it by a vector so that it can be accessed per cell
        if isinstance(mu, np.ScalarType):
            mu = np.full(sd.num_cells, mu)

        # Save 1/(2mu) as a tensor so that it can be read by BDM1
        mu_tensor = pp.SecondOrderTensor(1 / (2 * mu))
        data_for_RT0 = pp.initialize_data(
            sd, {}, self.keyword, {"second_order_tensor": mu_tensor}
        )

        # Save the coefficient for the trace contribution
        coeff = lambda_ / (2 * mu + sd.dim * lambda_) / (2 * mu)
        data_for_PwL = pp.initialize_data(sd, {}, self.keyword, {"weight": coeff})

        # Assemble the block diagonal mass matrix for the base discretization class
        D = super().assemble_mass_matrix(sd, data_for_RT0)
        # Assemble the trace part
        B = self.assemble_trace_matrix(sd)

        # Assemble the piecewise linear mass matrix, to assemble the term
        # (Trace(sigma), Trace(tau))
        discr = pg.PwLinears(self.keyword)
        M = discr.assemble_mass_matrix(sd, data_for_PwL)

        # Compose all the parts and return them
        return D - B.T @ M @ B

    def assemble_mass_matrix_cosserat(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assembles and returns the mass matrix for vector BDM1 discretizing the Cosserat
        inner product, which is given by (A sigma, tau) where
        A sigma = (sym(sigma) - coeff * Trace(sigma) * I) / (2 mu)
                  + skw(sigma) / (2 mu_c)
        with mu and lambda the Lamé constants, coeff = lambda / (2*mu + dim*lambda), and
        mu_c the coupling Lamé modulus.

        Args:
            sd (pg.Grid): The grid.
            data (dict): Data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.

        TODO: Consider using inheritance from VecBDM1.assemble_mass_matrix_cosserat
        """
        M = self.assemble_mass_matrix(sd, data)

        # Extract the data
        mu = data[pp.PARAMETERS][self.keyword]["mu"]
        mu_c = data[pp.PARAMETERS][self.keyword]["mu_c"]

        coeff = 0.25 * (1 / mu_c - 1 / mu)

        # If coeff is a scalar, replace it by a vector so that it can be accessed per
        # cell
        if isinstance(coeff, np.ScalarType):
            coeff = np.full(sd.num_cells, coeff)

        data_for_R = pp.initialize_data(sd, {}, self.keyword, {"weight": coeff})

        R_space: pg.Discretization
        if sd.dim == 2:
            R_space = pg.PwLinears(self.keyword)
        elif sd.dim == 3:
            R_space = pg.VecPwLinears(self.keyword)

        R_mass = R_space.assemble_mass_matrix(sd, data_for_R)

        asym = self.assemble_asym_matrix(sd, False)

        return M + asym.T @ R_mass @ asym

    def assemble_trace_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the trace matrix for the vector BDM1.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The trace matrix obtained from the discretization.
        """
        vec_bdm1 = VecBDM1(self.keyword)
        proj = vec_bdm1.proj_from_RT0(sd)
        return vec_bdm1.assemble_trace_matrix(sd) @ proj

    def assemble_asym_matrix(self, sd: pg.Grid, as_pwconstant=True) -> sps.csc_array:
        """
        Assembles and returns the asymmetric matrix for the vector RT0.

        The asymmetric operator `as' for a tensor is a scalar and it is defined in 2d as
        as(tau) = tau_xy - tau_yx
        while for a tensor in 3d it is a vector and given by
        as(tau) = [tau_zy - tau_yz, tau_xz - tau_zx, tau_yx - tau_xy]^T

        Note: We assume that the as(tau) is a cell variable.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The asymmetric matrix obtained from the discretization.
        """
        vec_bdm1 = VecBDM1(self.keyword)
        proj = vec_bdm1.proj_from_RT0(sd)
        return vec_bdm1.assemble_asym_matrix(sd, as_pwconstant) @ proj

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.VecPwConstants
