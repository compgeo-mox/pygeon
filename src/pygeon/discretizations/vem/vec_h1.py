"""Module for the virtual element discretizations of the vector H1 space."""

from typing import Type

import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps

import pygeon as pg


class VecVLagrange1(pg.VecDiscretization):
    r"""
    Vector Lagrange virtual element discretization for H1 space in 2D.

    Convention for the ordering is first all the x then all the y.

    The stress tensor and strain tensor are represented as vectors unrolled row-wise.
    In 2D, the stress tensor has a length of 4.

    We are considering the following structure of the stress tensor in 2D:

    .. math::

        \sigma = \begin{bmatrix}
            \sigma_{xx} & \sigma_{xy} \\
            \sigma_{yx} & \sigma_{yy}
        \end{bmatrix}

    which is represented in the code unrolled row-wise as a vector of length 4:

    .. math::

        \sigma = [\sigma_{xx}, \sigma_{xy}, \sigma_{yx}, \sigma_{yy}]

    The strain tensor follows the same approach.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    tensor_order = pg.VECTOR
    """Vector-valued discretization"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The base discretization class is pg.VLagrange1.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Defaults to pg.UNITARY_DATA.
        """
        super().__init__(keyword)
        self.base_discr: pg.VLagrange1 = pg.VLagrange1(keyword)

    def assemble_div_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Returns the divergence matrix :math:`\nabla \cdot u` operator for the lowest
        order vector virtual Lagrange element.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The div matrix obtained from the discretization.
        """
        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters()

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
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def local_div(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Compute the local divergence matrix for the vector virtual Lagrange1.

        Args:
            sd (pg.Grid): The grid.
            cell (int): The index of the cell on which to compute the div matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.

        Returns:
            np.ndarray: The local divergence matrix.
        """
        proj = self.base_discr.assemble_loc_proj_to_mon(sd, cell, diam, nodes)

        return sd.cell_volumes[cell] * proj[1:] / diam

    def assemble_div_div_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Returns the divergence-divergence matrix operator
        :math:`\lambda (\nabla \cdot u, \nabla \cdot v)_\Omega` for the lowest
        order vector virtual Lagrange element, multiplied by the Lamé parameter
        :math:`\lambda`.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Additional data, the Lamé parameter lambda.
                Defaults to None.

        Returns:
            sps.csc_array: The div-div matrix obtained from the discretization, of
            shape (ndof, ndof).
        """
        lambda_ = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        p0 = pg.PwConstants(self.keyword)

        div = self.assemble_div_matrix(sd)
        mass = p0.assemble_mass_matrix(sd)

        return div.T @ (lambda_ * mass) @ div

    def assemble_symgrad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Returns the symmetric gradient matrix operator
        :math:`\varepsilon(u) = \frac{1}{2}(\nabla u + \nabla u^T)` for the lowest
        order vector virtual Lagrange element.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The sparse symmetric gradient matrix operator.

        Raises:
            ValueError: If the grid is not two-dimensional.
        """
        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters()

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = cell_nodes.sum() * np.power(sd.dim, pg.AMBIENT_DIM)
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
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def local_symgrad(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray, sym: np.ndarray
    ) -> np.ndarray:
        """
        Compute the local symgrad matrix for vector virtual Lagrangian.

        Args:
            sd (pg.Grid): The grid.
            cell (int): The index of the cell on which to compute the symgrad matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.
            sym (np.ndarray): The symmetrization matrix of the unrolled tensors.

        Returns:
            np.ndarray: Local symmetric gradient matrix.
        """

        proj = self.base_discr.assemble_loc_proj_to_mon(sd, cell, diam, nodes)
        grad = spl.block_diag(*([proj[1:]] * sd.dim))

        return sd.cell_volumes[cell] * sym @ grad / diam

    def assemble_symgrad_symgrad_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Returns the symmetric-gradient inner product matrix
        :math:`2\mu (\varepsilon(u), \varepsilon(v))_\Omega` for the lowest order
        vector virtual Lagrange element, multiplied by twice the Lamé parameter
        :math:`\mu`.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Additional data, the Lamé parameter mu. Defaults to
                None.

        Returns:
            sps.csc_array: The symgrad-symgrad matrix obtained from the discretization,
            of shape (ndof, ndof).
        """
        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        p0 = pg.PwConstants(self.keyword)

        symgrad = self.assemble_symgrad_matrix(sd)
        mass = p0.assemble_mass_matrix(sd)
        tensor_mass = sps.block_diag([2 * mu * mass] * np.square(sd.dim)).tocsc()

        return symgrad.T @ tensor_mass @ symgrad

    def assemble_penalisation_matrix(
        self, sd: pg.Grid, _data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles and returns the VEM penalisation matrix, which stabilizes the virtual
        element method by penalizing the non-polynomial part of the solution.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Optional data for the assembly process.

        Returns:
            sps.csc_array: The penalisation matrix obtained from the discretization.
        """
        # Precomputations
        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters()

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

        scalar_pen = sps.csc_array((data_V, (rows_I, cols_J)))
        return sps.kron(sps.eye_array(sd.dim), scalar_pen).tocsc()

    def assemble_loc_penalisation_matrix(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        r"""
        Computes the local VEM penalisation matrix
        :math:`((I - \Pi) u, (I - \Pi) v)` on a given cell, in the degrees of freedom,
        stabilizing the non-polynomial part. Based on the Hitchhiker's (6.5).

        Args:
            sd (pg.Grid): The grid.
            cell (int): The index of the cell on which to compute the matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of nodes associated with the cell.

        Returns:
            np.ndarray: The computed local VEM penalisation matrix.
        """
        proj = self.base_discr.assemble_loc_proj_to_mon(sd, cell, diam, nodes)

        D = self.base_discr.assemble_loc_dofs_of_monomials(sd, cell, diam, nodes)
        I_minus_Pi = np.eye(nodes.size) - D @ proj

        return I_minus_Pi.T @ I_minus_Pi

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles the VEM linear elasticity stiffness matrix

        .. math::

            2\mu (\varepsilon(u), \varepsilon(v))_\Omega
            + \lambda (\nabla \cdot u, \nabla \cdot v)_\Omega

        where :math:`\mu` and :math:`\lambda` are the Lamé parameters, plus a VEM
        penalisation term.

        Args:
            sd (pg.Grid): The grid on which the finite element method is defined.
            data (dict | None): Additional data required for the assembly process.

        Returns:
            sps.csc_array: The assembled global stiffness matrix.
        """
        # compute the two parts of the global stiffness matrix
        sym_sym = self.assemble_symgrad_symgrad_matrix(sd, data)
        div_div = self.assemble_div_div_matrix(sd, data)

        # penalisation
        dofi_dofi = self.assemble_penalisation_matrix(sd)

        # return the global stiffness matrix
        return sym_sym + div_div + dofi_dofi

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Raises:
            NotImplementedError: There is no range discretization for the vector
                Lagrangian 1 in PyGeoN.
        """
        raise NotImplementedError(
            "There's no range discr for the vector VLagrangian 1 in PyGeoN"
        )

    def compute_stress(
        self,
        sd: pg.Grid,
        u: np.ndarray,
        data: dict,
    ) -> np.ndarray:
        """
        Compute the stress tensor for a given displacement field at the cell centers,
        as a (num_cells, dim, dim) array.

        Args:
            sd (pg.Grid): The grid.
            u (np.ndarray): The displacement field.
            data (dict): Data for the computation including the Lamé parameters accessed
                with the keys pg.LAME_LAMBDA and pg.LAME_MU.
                Both float and np.ndarray are accepted.

        Returns:
            np.ndarray: The stress tensor.
        """
        # construct the differentials
        symgrad = self.assemble_symgrad_matrix(sd)
        div = self.assemble_div_matrix(sd)

        p0 = pg.PwConstants(self.keyword)
        proj = p0.eval_at_cell_centers(sd)

        # retrieve Lamé parameters
        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        mu = np.tile(mu, np.square(sd.dim))
        lambda_ = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        # compute the two terms and split on each component
        sigma = np.array(np.split(2 * mu * (symgrad @ u), np.square(sd.dim)))
        sigma[:: (sd.dim + 1)] += lambda_ * (div @ u)

        # compute the actual dofs
        sigma = sigma @ proj

        # create the indices to re-arrange the components for the second
        # order tensor
        idx = np.arange(np.square(sd.dim)).reshape((sd.dim, -1), order="F")

        return sigma[idx].T
