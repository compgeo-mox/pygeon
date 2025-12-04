"""Module for the discretizations of the H(div) space."""

import abc
from typing import Type, cast

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class VecHDiv(pg.VecDiscretization):
    """Base class for vector-valued discretizations in the H(div) space.
    This class provides methods for assembling mass matrices, trace matrices,
    asymmetric matrices, and lumped matrices for vector-valued finite element
    discretizations in the H(div) space.
    """

    def assemble_mass_matrix_elasticity(
        self, sd: pg.Grid, data: dict | None = None
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
        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        lambda_ = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        # Save 1/(2mu) as a tensor so that it can be read by self
        mu_tensor = pp.SecondOrderTensor(1 / (2 * mu))
        data_self = pp.initialize_data(
            {}, self.keyword, {pg.SECOND_ORDER_TENSOR: mu_tensor}
        )

        # Save the coefficient for the trace contribution
        coeff = lambda_ / (2 * mu + sd.dim * lambda_) / (2 * mu)
        data_tr_space = pp.initialize_data({}, self.keyword, {pg.WEIGHT: coeff})

        # Assemble the block diagonal mass matrix for the base discretization class
        D = super().assemble_mass_matrix(sd, data_self)
        # Assemble the trace part
        B = self.assemble_trace_matrix(sd)

        # Assemble the piecewise linear mass matrix, to assemble the term
        # (Trace(sigma), Trace(tau))
        scalar_discr = pg.get_PwPolynomials(self.poly_order, pg.SCALAR)(self.keyword)
        M = scalar_discr.assemble_mass_matrix(sd, data_tr_space)

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
        D = super().assemble_lumped_matrix(sd, data_D)

        # Assemble the trace part
        B = self.assemble_trace_matrix(sd)

        # Assemble the piecewise linear mass matrix, to assemble the term
        # (Trace(sigma), Trace(tau))
        data_M = pp.initialize_data({}, self.keyword, {pg.WEIGHT: weight_M})

        scalar_discr = pg.get_PwPolynomials(self.poly_order, pg.SCALAR)(self.keyword)
        M = scalar_discr.assemble_lumped_matrix(sd, data_M)

        # Compose all the parts and return them
        return D - B.T @ M @ B

    def assemble_lumped_matrix_cosserat(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assembles the lumped matrix with cosserat terms for the given grid.

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

    def assemble_asym_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assemble the asymmetric matrix for the given grid.

        This method constructs an asymmetric matrix by projecting to
        matrix piecewise polynomials and combining it with the
        discretization's asymmetric matrix.

        Args:
            sd (pg.Grid): The grid object representing the spatial discretization.

        Returns:
            sps.csc_array: The assembled asymmetric matrix in compressed sparse column
            format.
        """
        P = self.proj_to_PwPolynomials(sd)
        mat_discr = pg.get_PwPolynomials(self.poly_order, pg.MATRIX)(self.keyword)
        mat_discr = cast(pg.MatPwLinears | pg.MatPwQuadratics, mat_discr)
        asym = mat_discr.assemble_asym_matrix(sd)

        return asym @ P

    @abc.abstractmethod
    def assemble_trace_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the trace matrix for the vector HDiv.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The trace matrix obtained from the discretization.

        Note:
            This method should be implemented in subclasses.
        """


class VecBDM1(VecHDiv):
    """
    VecBDM1 is a class that represents the vector BDM1 (Brezzi-Douglas-Marini) finite
    element method. It provides methods for assembling matrices like the mass matrix,
    the trace matrix, the asymmetric matrix and the differential matrix. It also
    provides methods for evaluating the solution at cell centers, interpolating a given
    function onto the grid, assembling the natural boundary condition term, and more.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    tensor_order = pg.MATRIX
    """Matrix-valued discretization"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector BDM1 discretization class.
        The base discretization class is pg.BDM1.

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
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr: pg.BDM1 = pg.BDM1(keyword)

    def assemble_trace_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the trace matrix for the vector BDM1.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The trace matrix obtained from the discretization.
        """
        # overestimate the size
        size = (sd.dim + 1) * sd.dim**2 * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        opposite_nodes = sd.compute_opposite_nodes()
        scalar_ndof = self.base_discr.ndof(sd)

        for c in range(sd.num_cells):
            # For the current cell retrieve its faces and
            # determine the location of the dof
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            opposites_loc = opposite_nodes.data[loc]

            Psi = self.base_discr.eval_basis_at_node(sd, opposites_loc, faces_loc)

            # Get all the components of the basis at node
            Psi_i, Psi_j = np.nonzero(Psi)
            Psi_v = Psi[Psi_i, Psi_j]

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

        ndof_pwlinear = pg.PwLinears().ndof(sd)
        shape = (ndof_pwlinear, self.ndof(sd))
        # Construct the global matrices
        return sps.csc_array((data_IJ[:idx], (rows_I[:idx], cols_J[:idx])), shape=shape)

    def assemble_asym_matrix(self, sd: pg.Grid, as_pwconstant=False) -> sps.csc_array:
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
                linears (default), otherwise the mapping is on the piece-wise constant.

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
        ndof_scalar = self.base_discr.ndof(sd)

        for c in range(sd.num_cells):
            # For the current cell retrieve its faces and
            # determine the location of the dof
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            opposites_loc = opposite_nodes.data[loc]

            Psi = self.base_discr.eval_basis_at_node(sd, opposites_loc, faces_loc)

            # Get all the components of the basis at node
            Psi_i, Psi_j = np.nonzero(Psi)
            Psi_v = Psi[Psi_i, Psi_j]

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

    def proj_to_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the function space to the lowest order Raviart-Thomas (RT0) space.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.

        Returns:
            sps.csc_array: The projection matrix to the RT0 space.
        """
        proj = self.base_discr.proj_to_RT0(sd)
        return sps.block_diag([proj] * sd.dim).tocsc()

    def proj_from_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the RT0 finite element space onto the faces of the given grid.

        Args:
            sd (pg.Grid): The grid on which the projection is performed.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.base_discr.proj_from_RT0(sd)
        return sps.block_diag([proj] * sd.dim).tocsc()

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class that contains the range of the differential

        Args:
            dim (int): The dimension of the range.

        Returns:
            pg.Discretization: The discretization class containing the range of the
            differential
        """
        return pg.VecPwConstants


class VecRT0(VecHDiv):
    """
    VecRT0 is a tensor-valued discretization class for the Raviart-Thomas RT0 finite
    element, specialized for handling stress tensors in 2D and 3D.
    This class provides methods for assembling trace and asymmetric matrices
    for vector RT0 discretizations, as well as retrieving the appropriate range
    discretization class.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    tensor_order = pg.MATRIX
    """Matrix-valued discretization"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector RT0 discretization class.
        The base discretization class is pg.RT0.

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
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr: pg.RT0 = pg.RT0(keyword)

    def assemble_trace_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the trace matrix for the vector RT0.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The trace matrix obtained from the discretization.
        """
        vec_bdm1 = VecBDM1(self.keyword)
        proj = vec_bdm1.proj_from_RT0(sd)
        return vec_bdm1.assemble_trace_matrix(sd) @ proj

    def assemble_asym_matrix(self, sd: pg.Grid, as_pwconstant=False) -> sps.csc_array:
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


class VecRT1(VecHDiv):
    """
    VecRT1 is a vector Raviart-Thomas finite element discretization class of order 1.

    This class is designed for matrix-valued finite element discretizations in the
    H(div) space, specifically using the Raviart-Thomas elements of order 1 (RT1).
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

    tensor_order = pg.MATRIX
    """Matrix-valued discretization"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector RT1 discretization class.
        The base discretization class is pg.RT1.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr: pg.RT1 = pg.RT1(keyword)

    def assemble_trace_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assemble the trace matrix for the given grid.

        This method constructs a sparse matrix that represents the trace operator
        for a finite element discretization on a given grid. The trace operator
        maps the degrees of freedom associated with the elements of the grid to
        the degrees of freedom associated with the faces and edges of the grid.

        Args:
            sd (pg.Grid): The grid object containing information about the
                discretization.

        Returns:
            sps.csc_array: A sparse matrix in compressed sparse column (CSC) format
            representing the trace operator.
        """
        # overestimate the size of a local computation
        loc_size = (
            sd.dim * (sd.dim * (sd.dim + 1) ** 2 + sd.dim**2 * (sd.dim + 1) // 2)
            + sd.dim**2
        )
        size = loc_size * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Compute the opposite nodes for each face
        opposite_nodes = sd.compute_opposite_nodes()
        scalar_ndof = self.base_discr.ndof(sd)
        edges_nodes_per_cell = sd.dim + 1 + sd.dim * (sd.dim + 1) // 2

        for c in range(sd.num_cells):
            nodes_loc, faces_loc, signs_loc = self.base_discr.reorder_faces(
                sd.cell_faces, opposite_nodes, c
            )

            Psi = self.base_discr.eval_basis_functions(
                sd, nodes_loc, signs_loc, sd.cell_volumes[c]
            )

            # Get all the components of the basis at nodes and edges
            Psi_i, Psi_j = np.nonzero(Psi)
            Psi_v = Psi[Psi_i, Psi_j]

            # Get the indices for the local face and cell degrees of freedom
            loc_face = np.hstack([faces_loc] * sd.dim)
            loc_face += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces
            loc_cell = sd.dim * sd.num_faces + sd.num_cells * np.arange(sd.dim) + c
            loc_ind = np.hstack((loc_face, loc_cell))

            cols = np.tile(loc_ind, (3, 1))
            cols[1, :] += scalar_ndof
            cols[2, :] += 2 * scalar_ndof
            cols = np.tile(cols, (edges_nodes_per_cell, 1)).T
            cols = cols[Psi_i, Psi_j]

            nodes_edges_loc = np.arange(edges_nodes_per_cell) * sd.num_cells + c
            rows = np.repeat(nodes_edges_loc, 3)[Psi_j]

            # Save values of the local matrix in the global structure
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = rows
            cols_J[loc_idx] = cols
            data_IJ[loc_idx] = Psi_v
            idx += cols.size

        # Construct the global matrices
        return sps.csc_array((data_IJ[:idx], (rows_I[:idx], cols_J[:idx])))

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.VecPwLinears
