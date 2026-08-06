"""Module for the discretizations of the L2 space."""

import abc
from typing import Callable, Type

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class PwPolynomials(pg.Discretization):
    r"""
    Abstract class implementing the finite element discretization for piecewise
    polynomials of order k :math:`\mathbb{P}_k(\Omega) \subset L^2(\Omega)`,
    for a generic domain :math:`\Omega \in \mathbb{R}^d`.
    """

    poly_order: int
    """Polynomial degree of the basis functions"""

    tensor_order = pg.SCALAR
    """Scalar-valued discretization"""

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
        """
        Compute the local degrees of freedom (DOFs) indices for a cell.

        Args:
            sd (pp.Grid): Grid object or a subclass.
            c (int): Index of the cell.

        Returns:
            np.ndarray: Array of local DOF indices associated with the cell.
        """
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
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Computes the mass matrix :math:`(\sigma u, v)_\Omega` for
        :math:`u, v \in \mathbb{P}_k(\Omega)` and :math:`\sigma` a scalar. Both
        domain and range lie in :class:`PwPolynomials`.

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (dict | None): Dictionary with possible scaling.

        Returns:
            sps.csc_array: Sparse csc matrix of shape (sd.num_cells, sd.num_cells).
        """
        local_mass = self.assemble_local_mass(sd.dim)

        weight = pg.get_cell_data(sd, data, self.keyword, pg.WEIGHT)
        diag_weight = sps.diags_array(sd.cell_volumes * weight)

        M = sps.kron(local_mass, diag_weight)
        M.eliminate_zeros()

        return M.tocsc()

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles the lumped matrix for the given grid, which is a diagonal
        approximation (in this case equivalent) of the mass matrix
        :math:`(\sigma u, v)_\Omega` for :math:`u, v \in \mathbb{P}_k(\Omega)` and
        :math:`\sigma` a scalar. Both domain and range lie in
        :class:`PwPolynomials` (L2 space).

        Args:
            sd (pg.Grid): The grid object.
            data (dict | None): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped matrix.
        """
        local_mass = self.assemble_local_lumped_mass(sd.dim)

        weight = pg.get_cell_data(sd, data, self.keyword, pg.WEIGHT)
        diag_weight = sps.diags_array(sd.cell_volumes * weight)

        M = sps.kron(local_mass, diag_weight).tocsc()
        M.eliminate_zeros()

        return M

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator.

        For piecewise polynomials (L2), the differential is the zero matrix since
        these are discontinuous functions with no global differential.

        This method takes a grid object and returns the differential matrix
        corresponding to the given grid.

        Args:
            sd (pg.Grid): The grid object or its subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        return sps.csc_array((0, self.ndof(sd)))

    def assemble_broken_grad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Assembles the broken (element-wise) gradient matrix for the given grid.
        This method should be implemented in the child class.

        The broken gradient :math:`\nabla_h` maps from
        :math:`\mathbb{P}_k(\Omega)` (L2) to :math:`[\mathbb{P}_{k-1}(\Omega)]^d`.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            sps.csc_array: The assembled broken gradient matrix.
        """
        raise NotImplementedError

    def assemble_stiff_matrix(
        self, sd: pg.Grid, _data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles the stiffness matrix for the given grid, representing the bilinear
        form :math:`(K \nabla u, \nabla v)_\Omega` for
        :math:`u,v \in \mathbb{P}_k(\Omega)` and :math:`K` a scalar.

        For piecewise polynomials, the stiffness matrix is the zero matrix since the
        differential operator is trivial.

        Args:
            sd (pg.Grid): The grid or a subclass.
            data (dict | None): Additional data for the assembly process.

        Returns:
            sps.csc_array: The assembled stiffness matrix.
        """
        return sps.csc_array((self.ndof(sd), self.ndof(sd)))

    def assemble_nat_bc(
        self,
        sd: pg.Grid,
        _func: Callable[[np.ndarray], np.ndarray],
        _b_faces: np.ndarray,
    ) -> np.ndarray:
        r"""
        Assembles the natural boundary condition vector, equal to zero.

        For piecewise polynomials, the natural boundary condition is zero since these
        are discontinuous functions with no boundary trace.

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
        r"""
        Returns the discretization class for the range of the differential.

        Args:
            dim (int): The dimension of the range space.

        Raises:
            NotImplementedError: There is no zero discretization available in PyGeoN.
        """
        raise NotImplementedError("There's no zero discretization in PyGeoN (yet)")

    @abc.abstractmethod
    def assemble_local_mass(self, dim: int) -> np.ndarray:
        r"""
        Computes the local mass matrix :math:`(\varphi_i, \varphi_j)_S` for
        :math:`\varphi_i, \varphi_j` being the local basis functions of the
        piecewise polynomials on a d-simplex :math:`S`.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise polynomials.
        """

    @abc.abstractmethod
    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        r"""
        Computes the local lumped mass matrix for piecewise polynomials, which
        is a diagonal approximation of the local mass matrix. It approximates the form
        :math:`(\varphi_i, \varphi_j)_S`, with :math:`\varphi_i, \varphi_j` being
        the local basis functions of the piecewise polynomials on a d-simplex :math:`S`.

        It approximates this form by a diagonal matrix.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise polynomials.
        """

    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Construct the matrix for projecting a piecewise function to a piecewise
        polynomial function. The (identity) projection operator :math:`\Pi` takes a
        function from :math:`\mathbb{P}_k(\Omega)` and maps it to a piecewise
        polynomial function in :math:`\mathbb{P}_k(\Omega)`.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        return sps.eye_array(self.ndof(sd)).tocsc()

    def proj_to_lower_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Projects the discretization to -1 order discretization. The projection operator
        :math:`\Pi` takes a function from :math:`\mathbb{P}_k(\Omega)` and maps it
        to a piecewise polynomial function of lower order in
        :math:`\mathbb{P}_{k-1}(\Omega)`.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def proj_to_higher_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Projects the discretization to +1 order discretization. The projection operator
        :math:`\Pi` takes a function from :math:`\mathbb{P}_k(\Omega)` and maps it
        to a piecewise polynomial function of higher order in
        :math:`\mathbb{P}_{k+1}(\Omega)`.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """


class PwConstants(PwPolynomials):
    r"""
    Class implementing the finite element discretization for piecewise polynomials of
    lowest order :math:`\mathbb{P}_0(\Omega) \subset L^2(\Omega)`, for a generic
    domain :math:`\Omega \in \mathbb{R}^d`.

    Each degree of freedom is the integral over the cell.
    """

    poly_order = 0
    """Polynomial degree of the basis functions"""

    def ndof_per_cell(self, _sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom per cell.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom per cell.
        """
        return 1

    def assemble_local_mass(self, _dim: int) -> np.ndarray:
        r"""
        Computes the local mass matrix :math:`(\varphi_i, \varphi_j)` for
        :math:`\varphi_i, \varphi_j \in \mathbb{P}_0(\Omega)`, which is the scalar
        :math:`[[1]]`.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise constants.
        """
        return np.array([[1]])

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise constants,
        which coincides with the local mass matrix since P0 has one dof per cell.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise constants.
        """
        return self.assemble_local_mass(dim)

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Computes the mass matrix for piecewise constants, representing the bilinear form
        :math:`(\sigma u, v)_\Omega` where :math:`u, v \in \mathbb{P}_0(\Omega)` and
        :math:`\sigma` is the datum. Both domain and range lie in :class:`P0`.

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (dict | None): Dictionary with possible scaling.

        Returns:
            sps.csc_array: Sparse csc matrix of shape (sd.num_cells, sd.num_cells).
        """
        M = super().assemble_mass_matrix(sd, data)
        M /= np.square(sd.cell_volumes)

        return M.tocsc()

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Computes the lumped mass matrix, which coincides with the mass matrix for P0.

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (dict | None): Additional data for the assembly process.

        Returns:
            sps.csc_array: The assembled lumped mass matrix.
        """
        M = super().assemble_lumped_matrix(sd, data)
        M /= np.square(sd.cell_volumes)

        return M.tocsc()

    def assemble_broken_grad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Assembles the broken (element-wise) gradient matrix for the given grid,
        which is zero for the piecewise constants.

        The broken gradient :math:`\nabla_h` maps from :math:`\mathbb{P}_0(\Omega)`
        to :math:`(0, 0, 0)`.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            sps.csc_array: The assembled broken gradient matrix.
        """
        ndof = self.ndof(sd)
        return sps.csc_array((pg.AMBIENT_DIM * ndof, ndof))

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

    def proj_to_higher_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Projects the P0 discretization to the P1 discretization. The projection
        operator :math:`\Pi` takes a function from :math:`\mathbb{P}_0(\Omega)` and
        maps it to a piecewise linear function in :math:`\mathbb{P}_1(\Omega)`.

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


class PwLinears(PwPolynomials):
    r"""
    Class implementing the finite element discretization for piecewise polynomials of
    order 1 :math:`\mathbb{P}_1(\Omega) \subset L^2(\Omega)`, for a generic
    domain :math:`\Omega \in \mathbb{R}^d`.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

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
        r"""
        Computes the local mass matrix :math:`(\varphi_i, \varphi_j)` for
        :math:`\varphi_i, \varphi_j \in \mathbb{P}_1(\Omega)`, where
        :math:`\varphi_i` are the local basis functions.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise linears.
        """
        M = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        return M / ((dim + 1) * (dim + 2))

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise linears, which is a
        diagonal approximation of the local mass matrix.

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
        Interpolates a function onto the finite element space by evaluating the function
        at the (sd.dim + 1) Gauss points.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable): A function that returns the function values at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """
        lookup = self.get_dof_lookup_array(sd).tocoo()
        dofs = lookup.data

        # Retrieve the (cell, node) pair for each degree of freedom
        cells = np.empty_like(lookup.col)
        nodes = np.empty_like(lookup.row)
        cells[dofs] = lookup.col
        nodes[dofs] = lookup.row

        # Compute the Gauss points as a weighted average of the node and cell center
        # coordinates.
        alpha = 1 / np.sqrt(sd.dim + 2)
        gauss_pts = alpha * sd.nodes[:, nodes] + (1 - alpha) * sd.cell_centers[:, cells]

        # Evaluate the function at the Gauss points.
        func_at_gauss = np.array([func(x) for x in gauss_pts.T])

        # To retrieve the values at the nodes, we first compute the value of the
        # interpolated function at the cell center. Since the Gauss points are
        # equidistant from the cell center, we can use eval_at_cc as the averaging
        # operator.
        interp_at_cc = self.eval_at_cell_centers(sd) @ func_at_gauss

        # Expand from cell-indices to dof-indices
        interp_at_cc = interp_at_cc[cells]

        # Extrapolate the linear function from the cell center, through the
        # Gauss point, to the node.
        return interp_at_cc + 1 / alpha * (func_at_gauss - interp_at_cc)

    def proj_to_lower_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Construct the matrix for projecting a piece-wise function to a piecewise
        constant function. The projection operator :math:`\Pi` takes a function from
        :math:`\mathbb{P}_1(\Omega)` and maps it to a piecewise polynomial function
        of lower order in :math:`\mathbb{P}_0(\Omega)`.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        matr = sps.hstack([sps.diags_array(sd.cell_volumes)] * (sd.dim + 1)) / (
            sd.dim + 1
        )
        return matr.tocsc()

    def proj_to_higher_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Projects the discretization to +1 order discretization. The projection operator
        :math:`\Pi` takes a function from :math:`\mathbb{P}_0(\Omega)` and maps it
        to a piecewise polynomial function of higher order in
        :math:`\mathbb{P}_1(\Omega)`.

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

        for c in range(sd.num_cells):
            dofs_p1 = p1.local_dofs_of_cell(sd, c)
            dofs_p2 = p2.local_dofs_of_cell(sd, c)

            rows_I[c] = np.concatenate(
                (dofs_p2[: sd.dim + 1], np.repeat(dofs_p2[sd.dim + 1 :], 2))
            )
            cols_J[c] = np.concatenate((dofs_p1, dofs_p1[edge_nodes]))

        return sps.csc_array((data_IJ.ravel(), (rows_I.ravel(), cols_J.ravel())))

    def get_dof_lookup_array(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles a lookup matrix L with the property L[cell, node] = dof_index.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            sps.csc_array: The lookup matrix.
        """
        dof_array = sd.cell_nodes().astype("int")
        ndof = self.ndof(sd)
        dof_array.data = np.reshape(np.arange(ndof), (sd.num_cells, -1), "F").ravel()

        return dof_array

    def assemble_broken_grad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Assembles the broken (element-wise) gradient matrix for the given grid.
        This operator maps to the vector-valued piecewise constants
        :math:`[\mathbb{P}_0(\Omega)]^d` from :math:`u \in \mathbb{P}_1(\Omega)`
        and is used for computing :math:`\nabla_h u` element-wise.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            sps.csc_array: The assembled broken gradient matrix.
        """
        opposite_nodes = sd.compute_opposite_nodes().tocoo()
        faces = opposite_nodes.row
        cells = opposite_nodes.col
        orien = sd.cell_faces[faces, cells]
        nodes = opposite_nodes.data

        vecp0_dofs = np.arange(sd.dim * sd.num_cells).reshape((sd.dim, -1))
        rows_I = vecp0_dofs[:, cells].ravel()

        dof_lookup = self.get_dof_lookup_array(sd)
        cols_J = np.tile(dof_lookup[nodes, cells], sd.dim)

        normals = sd.rotation_matrix @ sd.face_normals
        grads = -normals[:, faces] * orien / sd.dim
        data_IJ = grads.ravel()

        return sps.csc_array((data_IJ, (rows_I, cols_J)))


class PwQuadratics(PwPolynomials):
    r"""
    Class implementing the finite element discretization for piecewise polynomials of
    order 2 :math:`\mathbb{P}_2(\Omega) \subset L^2(\Omega)`, for a generic
    domain :math:`\Omega \in \mathbb{R}^d`.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

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
        r"""
        Computes the local mass matrix :math:`(\varphi_i, \varphi_j)` for
        :math:`\varphi_i, \varphi_j \in \mathbb{P}_2(\Omega)` the local basis
        functions.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise quadratics.
        """
        lagrange2 = pg.Lagrange2(self.keyword)
        return lagrange2.assemble_local_mass(dim)

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise quadratics, which is a
        diagonal approximation of the local mass matrix.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise quadratics.
        """
        num_edges = (dim * (dim + 1)) // 2

        # Evaluate the basis function at the cell center
        node_bf_at_cc = np.full(dim + 1, 1 - dim)
        edge_bf_at_cc = np.full(num_edges, 4)
        vals_at_center = np.concatenate((node_bf_at_cc, edge_bf_at_cc)) / (dim + 1) ** 2
        center_weight = (dim + 1) / (dim + 2)

        L = center_weight * np.outer(vals_at_center, vals_at_center)

        # Evaluate the basis functions at the nodes
        vals_at_nodes = np.zeros(dim + 1 + num_edges)
        vals_at_nodes[: dim + 1] = 1
        node_weight = 1 / ((dim + 1) * (dim + 2))

        L += node_weight * np.diag(vals_at_nodes)

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
            sd (pg.Grid): Grid, or a subclass.
            func (Callable): A function that returns the function values at degrees of
                freedom.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """
        lagrange2 = pg.Lagrange2(self.keyword)
        edge_nodes = lagrange2.get_local_edge_nodes(sd.dim)

        cell_nodes = sd.cell_nodes()
        vals = np.empty((sd.num_cells, self.ndof_per_cell(sd)))

        for c in range(sd.num_cells):
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

    def proj_to_higher_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Projects the discretization to +1 order discretization. The projection operator
        :math:`\Pi` takes a function from :math:`\mathbb{P}_2(\Omega)` and maps it
        to a piecewise polynomial function of higher order (not yet implemented) in
        :math:`\mathbb{P}_3(\Omega)`.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        raise NotImplementedError
