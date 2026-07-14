"""Module for the discretizations of the L2 space."""

import abc
from math import factorial
from typing import Callable, Type

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class PwPolynomials(pg.Discretization):
    """
    PwPolynomials is a subclass of pg.Discretization that represents
    an abstract element wise polynomial discretization.
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
        """
        Computes the mass matrix for piecewise polynomials.

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
        """
        Assembles the lumped matrix for the given grid.

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

        This method takes a grid object and returns the differential matrix
        corresponding to the given grid.

        Args:
            sd (pg.Grid): The grid object or its subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        return sps.csc_array((0, self.ndof(sd)))

    def assemble_broken_grad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the broken (element-wise) gradient matrix for the given grid.
        This method should be implemented in the child class.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            sps.csc_array: The assembled broken gradient matrix.
        """
        raise NotImplementedError

    def assemble_stiff_matrix(
        self, sd: pg.Grid, _data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the stiffness matrix for the given grid.

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

    @abc.abstractmethod
    def assemble_local_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local mass matrix for piecewise polynomials.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise polynomials.
        """

    @abc.abstractmethod
    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise polynomials

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise polynomials.
        """

    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for projecting a piecewise function to a piecewise
        polynomial function.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        return sps.eye_array(self.ndof(sd)).tocsc()

    def proj_to_lower_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Projects the discretization to -1 order discretization.

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
        """
        Projects the discretization to +1 order discretization.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """


class PwConstants(PwPolynomials):
    """
    Discretization class for the piecewise constants.
    NOTE: Each degree of freedom is the integral over the cell.
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
        """
        Computes the local mass matrix for piecewise constants

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise constants.
        """
        return np.array([[1]])

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise constants

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise constants.
        """
        return self.assemble_local_mass(dim)

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Computes the mass matrix for piecewise constants

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
        """
        Assembles the broken (element-wise) gradient matrix for the given grid,
        which is zero for the piecewise constants

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


class PwLinears(PwPolynomials):
    """
    Discretization class for piecewise linear finite element method.
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
        """
        Computes the local mass matrix for piecewise linears

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise linears.
        """
        M = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        return M / ((dim + 1) * (dim + 2))

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise linears

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
        """
        Construct the matrix for projecting a piece-wise function to a piecewise
        constant function.

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
        """
        Projects the P1 discretization to the P2 discretization.

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
        """
        Assembles the broken (element-wise) gradient matrix for the given grid.
        This operator maps to the vector-valued piecewise constants.

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
    """
    PwQuadratics is a class that represents piecewise quadratic finite element
    discretizations.
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

    def num_edges_per_cell(self, dim: int) -> int:
        """
        Compute the number of edges of a simplex of a given dimension.

        Args:
            dim (int): Dimension.

        Returns:
            int: The number of adjacent edges.
        """
        return dim * (dim + 1) // 2

    def get_local_edge_nodes(self, dim: int) -> np.ndarray:
        """
        Lists the local edge-node connectivity in the cell

        Args:
            dim (int): Dimension.

        Returns:
            np.ndarray: Row i contains the local indices of the nodes connected to the
            edge with local index i.
        """
        n_nodes = dim + 1
        n_edges = self.num_edges_per_cell(dim)
        e_nodes = np.empty((n_edges, 2), int)

        ind = 0
        for first_node in np.arange(n_nodes):
            for second_node in np.arange(first_node + 1, n_nodes):
                e_nodes[ind] = [first_node, second_node]
                ind += 1

        return e_nodes

    def assemble_local_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local mass matrix for piecewise quadratics.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise quadratics.
        """
        # Helper constants
        n_edges = self.num_edges_per_cell(dim)
        eye = np.eye(dim + 1)
        zero = np.zeros((n_edges, dim + 1))

        # List the barycentric functions up to degree 2,
        # by exponents, consisting of
        # - the linears lambda_i
        # - the cross-quadratics lambda_i lambda_j
        # - the quadratics lambda_i^2
        quads = np.zeros((dim + 1, n_edges))
        e_nodes = self.get_local_edge_nodes(dim)
        for ind, nodes in enumerate(e_nodes):
            quads[nodes, ind] = 1
        exponents = np.hstack((eye, quads, 2 * eye))

        # Compute the local mass matrix of the barycentric functions
        barycentric_mass = self.assemble_barycentric_mass(exponents)

        # Our basis functions are given by
        # - nodes: lambda_i (2 lambda_i - 1)
        # - edges: 4 lambda_i lambda_j
        # We list the coefficients in the array "basis"
        basis_nodes = np.vstack((-eye, zero, 2 * eye))
        basis_edges = np.zeros((2 * (dim + 1) + n_edges, n_edges))
        basis_edges[dim + 1 : dim + n_edges + 1, :] = 4 * np.eye(n_edges)
        basis = np.hstack((basis_nodes, basis_edges))

        return basis.T @ barycentric_mass @ basis

    def assemble_barycentric_mass(self, expnts: np.ndarray) -> np.ndarray:
        """
        Compute the inner products of all monomials up to degree 2

        Args:
            expnts (np.ndarray): Each column is an array of exponents
                alpha_i of the monomial expressed as
                prod_i lambda_i ^ alpha_i.

        Returns:
            np.ndarray: The inner products of the monomials on a simplex with measure 1.
        """
        n_monomials = expnts.shape[1]
        mass = np.empty((n_monomials, n_monomials))

        for i in np.arange(n_monomials):
            for j in np.arange(n_monomials):
                mass[i, j] = self.integrate_monomial(expnts[:, i] + expnts[:, j])

        return mass

    def integrate_monomial(self, alphas: np.ndarray) -> float:
        """
        Exact integration of products of monomials based on
        Vermolen and Segal (2018).

        Args:
            alphas (np.ndarray): Array of exponents alpha_i of the monomial
                expressed as prod_i lambda_i ^ alpha_i.

        Returns:
            float: The integral of the monomial on a simplex with measure 1.
        """
        alphas = alphas.astype(int)
        dim = len(alphas) - 1
        fac_alph = [factorial(a_i) for a_i in alphas]

        return float(
            factorial(dim) * np.prod(fac_alph) / factorial(dim + np.sum(alphas))
        )

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise quadratics

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

    def assemble_broken_grad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the broken (element-wise) gradient matrix for the given grid.
        This operator maps to the vector-valued piecewise constants.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            sps.csc_array: The assembled broken gradient matrix.
        """
        print("implement me!")

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
        edge_nodes = self.get_local_edge_nodes(sd.dim)

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
        """
        Projects the discretization to +1 order discretization.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        raise NotImplementedError
