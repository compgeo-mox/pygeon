"""Module for the discretizations of the H1 space."""

from math import factorial
from typing import Callable, Type, cast

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class Lagrange1(pg.Discretization):
    """
    Class representing the Lagrange1 finite element discretization.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    tensor_order = pg.SCALAR
    """Scalar-valued discretization"""

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case number of nodes.

        Args
            sd: grid, or a subclass.

        Returns
            ndof: the number of degrees of freedom.
        """
        return sd.num_nodes

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Returns the mass matrix for the lowest order Lagrange element

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Optional data for the assembly process.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """

        # Data allocation
        size = np.power(sd.dim + 1, 2) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()
        local_mass = self.assemble_local_mass(sd.dim)

        for c in range(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            # Compute the mass-H1 local matrix
            A = local_mass * sd.cell_volumes[c]

            # Save values for mass-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrix
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def assemble_local_mass(self, dim: int) -> np.ndarray:
        """Compute the local mass matrix on an element with measure 1.

        Args:
            dim (int): Dimension of the matrix.

        Returns:
            np.ndarray: Local mass matrix of shape (num_nodes_of_cell,
            num_nodes_of_cell).
        """

        M = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        return M / ((dim + 1) * (dim + 2))

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the stiffness matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (dict): A dictionary containing the necessary data for assembling the
                matrix.

        Returns:
            sps.csc_array: The assembled stiffness matrix.
        """
        # Get dictionary for parameter storage
        K = pp.SecondOrderTensor(np.ones(sd.num_cells))
        if data is not None:
            K = (
                data.get(pp.PARAMETERS, {})
                .get(self.keyword, {})
                .get("second_order_tensor", K)
            )
        else:
            data = {"is_tangential": True}

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        _, _, _, R, dim, node_coords = pp.map_geometry.map_grid(sd)

        if not data.get("is_tangential", False):
            # Rotate the permeability tensor and delete last dimension
            if sd.dim < 3:
                K = K.copy()
                K.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                K.values = np.delete(K.values, (remove_dim), axis=0)
                K.values = np.delete(K.values, (remove_dim), axis=1)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(sd.dim + 1, 2) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()

        for c in range(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            A = self.local_stiff(
                K.values[0 : sd.dim, 0 : sd.dim, c],
                sd.cell_volumes[c],
                coord_loc,
                sd.dim,
            )

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the differential matrix based on the dimension of the grid.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The differential matrix.
        """
        if sd.dim == 3:
            return sd.ridge_peaks.T.tocsc()
        elif sd.dim == 2:
            return sd.face_ridges.T.tocsc()
        elif sd.dim == 1:
            return sps.csc_array(sd.cell_faces.T)
        elif sd.dim == 0:
            return sps.csc_array((0, 1))
        else:
            raise ValueError

    def local_stiff(
        self, K: np.ndarray, c_volume: np.ndarray, coord: np.ndarray, dim: int
    ) -> np.ndarray:
        """
        Compute the local stiffness matrix for P1.

        Args:
            K (np.ndarray): Permeability of the cell of (dim, dim) shape.
            c_volume (np.ndarray): Scalar cell volume.
            coord (np.ndarray): Coordinates of the cell vertices of (dim+1, dim) shape.
            dim (int): Dimension of the problem.

        Returns:
            np.ndarray: Local stiffness matrix of (dim+1, dim+1) shape.
        """
        dphi = self.local_grads(coord, dim)

        return c_volume * dphi.T @ K @ dphi

    @staticmethod
    def local_grads(coord: np.ndarray, dim: int) -> np.ndarray:
        """
        Calculate the local gradients of the finite element basis functions.

        Args:
            coord (np.ndarray): The coordinates of the nodes in the element.
            dim (int): The dimension of the element.

        Returns:
            np.ndarray: The local gradients of the finite element basis functions.
        """
        Q = np.hstack((np.ones((dim + 1, 1)), coord.T))
        invQ = np.linalg.inv(Q)
        return invQ[1:, :]

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the lumped mass matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (dict | None): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped mass matrix.
        """
        volumes = sd.cell_nodes() @ sd.cell_volumes / (sd.dim + 1)
        return sps.diags_array(volumes).tocsc()

    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for projecting a Lagrangian function to a piecewise linear
        function.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        rows_I = np.arange(sd.num_cells * (sd.dim + 1))
        rows_I = rows_I.reshape((-1, sd.num_cells)).ravel(order="F")
        cols_J = sd.cell_nodes().indices
        data_IJ = np.ones_like(rows_I, dtype=float)

        # Construct the global matrix
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for evaluating a Lagrangian function at the
        cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection at the cell centers.
        """
        if sd.dim == 0:
            return sps.csc_array((1, 0))
        eval = sps.csc_array(sd.cell_nodes())
        num_nodes = sps.diags_array(1.0 / sd.num_cell_nodes())

        return (eval @ num_nodes).T.tocsc()

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a given function over the nodes of a grid.

        Args:
            sd (pg.Grid): The grid on which to interpolate the function.
            func (Callable[[np.ndarray], np.ndarray]): The function to be interpolated.

        Returns:
            np.ndarray: An array containing the interpolated values at each node of the
            grid.
        """
        return np.array([func(x) for x in sd.nodes.T])

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the 'natural' boundary condition
        (u, func)_Gamma with u a test function in Lagrange1

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function used to evaluate
                the 'natural' boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled 'natural' boundary condition values.
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        vals = np.zeros(self.ndof(sd))

        for face in b_faces:
            loc = slice(sd.face_nodes.indptr[face], sd.face_nodes.indptr[face + 1])
            loc_n = sd.face_nodes.indices[loc]

            vals[loc_n] += (
                func(sd.face_centers[:, face]) * sd.face_areas[face] / loc_n.size
            )

        return vals

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the appropriate range discretization class based on the dimension.

        Args:
            dim (int): The dimension of the problem.

        Returns:
            object: The range discretization class.

        Raises:
            NotImplementedError: If there's no zero discretization in PyGeoN.
        """
        if dim == 3:
            return pg.Nedelec0
        elif dim == 2:
            return pg.RT0
        elif dim == 1:
            return pg.PwConstants
        else:
            raise NotImplementedError("There's no zero discretization in PyGeoN")


class Lagrange2(pg.Discretization):
    """
    Class representing the Lagrange2 finite element discretization.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

    tensor_order = pg.SCALAR
    """Scalar-valued discretization"""

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case number of nodes plus the number of edges,
        where edges are one-dimensional mesh entities.

        Args
            sd: grid, or a subclass.

        Returns
            ndof: the number of degrees of freedom.
        """
        if sd.dim == 0:
            num_edges = 0
        elif sd.dim == 1:
            num_edges = sd.num_cells
        elif sd.dim == 2:
            num_edges = sd.num_faces
        elif sd.dim == 3:
            num_edges = sd.num_ridges

        return sd.num_nodes + num_edges

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Returns the mass matrix for the second order Lagrange element

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Optional data for the assembly process.

        Returns:
            sps.csc_array: The mass matrix.
        """
        weight = pg.get_cell_data(sd, data, self.keyword, pg.WEIGHT)

        # Data allocation
        size = np.square((sd.dim + 1) + self.num_edges_per_cell(sd.dim)) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        opposite_nodes = sd.compute_opposite_nodes()
        local_mass = self.assemble_local_mass(sd.dim)

        for c in range(sd.num_cells):
            loc = slice(opposite_nodes.indptr[c], opposite_nodes.indptr[c + 1])
            faces = opposite_nodes.indices[loc]
            nodes = opposite_nodes.data[loc]
            edges = self.get_edge_dof_indices(sd, c, faces)

            A = local_mass.ravel() * weight[c] * sd.cell_volumes[c]

            loc_ind = np.hstack((nodes, edges))

            cols = np.tile(loc_ind, (loc_ind.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Assemble
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def assemble_local_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local mass matrix of the basis functions
        on a d-simplex with measure 1.

        Args:
            dim (int): The dimension of the simplex.

        Returns:
            np.ndarray: The local mass matrix.
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

    def eval_grads_at_nodes(self, dphi, e_nodes) -> np.ndarray:
        """
        Evaluates the gradients of the basis functions at the nodes

        Args:
            dphi (np.ndarray): Gradients of the P1 basis functions.
            e_nodes (np.ndarray): The local edge-node connectivity.

        Returns:
            np.ndarray: The gradient of basis function i at node j is in elements
            [i, 3 * (j:J + 1)].
        """
        # the gradient of our basis functions are given by
        # - nodes: (grad lambda_i) ( 4 lambda_i - 1 )
        # - edges: 4 lambda_i (grad lambda_j) + 4 lambda_j (grad lambda_i)

        # nodal dofs
        n_nodes = dphi.shape[1]
        Psi_nodes = np.zeros((n_nodes, 3 * n_nodes))
        for ind in np.arange(n_nodes):
            Psi_nodes[ind, 3 * ind : 3 * (ind + 1)] = 4 * dphi[:, ind]
        Psi_nodes[:n_nodes] -= np.tile(dphi.T, n_nodes)

        # edge dofs
        n_edges = self.num_edges_per_cell(n_nodes - 1)
        Psi_edges = np.zeros((n_edges, 3 * n_nodes))

        for ind, (e0, e1) in enumerate(e_nodes):
            Psi_edges[ind, 3 * e0 : 3 * (e0 + 1)] = 4 * dphi[:, e1]
            Psi_edges[ind, 3 * e1 : 3 * (e1 + 1)] = 4 * dphi[:, e0]

        return np.vstack((Psi_nodes, Psi_edges))

    def get_edge_dof_indices(self, sd, cell, faces) -> np.ndarray:
        """
        Finds the indices for the edge degrees of freedom that correspond
        to the local numbering of the edges.

        Args:
            sd (pg.Grid): The grid.
            cell (int): The cell index.
            faces (np.ndarray): Face indices of the cell.

        Returns:
            np.ndarray: Indices of the edge degrees of freedom.
        """

        if sd.dim == 1:
            # The only edge in 1d is the cell
            edges = np.array([cell])
        elif sd.dim == 2:
            # The edges (0, 1), (0, 2), and (1, 2)
            # are the faces opposite nodes 2, 1, and 0, respectively.
            edges = faces[::-1]
        elif sd.dim == 3:
            # We first find the edges adjacent to the local faces
            cell_edges = abs(sd.face_ridges[:, faces]) @ np.ones((4, 1))
            edge_inds = np.where(cell_edges)[0]

            # Experimentally, we always find the following numbering
            edges = edge_inds[[5, 4, 2, 3, 1, 0]]

        # The edge dofs come after the nodal dofs
        return edges + sd.num_nodes

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the stiffness matrix for the P2 finite element method.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (dict): A dictionary containing the necessary data for assembling the
                matrix.

        Returns:
            sps.csc_array: The stiffness matrix.
        """
        sot = pg.get_cell_data(sd, data, self.keyword, "second_order_tensor", pg.VECTOR)

        size = np.square((sd.dim + 1) + self.num_edges_per_cell(sd.dim)) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        opposite_nodes = sd.compute_opposite_nodes()
        local_mass = pg.BDM1.local_inner_product(sd.dim)
        e_nodes = self.get_local_edge_nodes(sd.dim)

        for c in range(sd.num_cells):
            loc = slice(opposite_nodes.indptr[c], opposite_nodes.indptr[c + 1])
            faces = opposite_nodes.indices[loc]
            nodes = opposite_nodes.data[loc]
            edges = self.get_edge_dof_indices(sd, c, faces)

            signs = sd.cell_faces.data[loc]
            dphi = -sd.face_normals[:, faces] * signs / (sd.dim * sd.cell_volumes[c])
            Psi = self.eval_grads_at_nodes(dphi, e_nodes)

            weight = np.kron(np.eye(sd.dim + 1), sot.values[:, :, c])

            A = Psi @ local_mass @ weight @ Psi.T * sd.cell_volumes[c]

            loc_ind = np.hstack((nodes, edges))

            cols = np.tile(loc_ind, (loc_ind.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Assemble
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the differential matrix based on the dimension of the grid.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The differential matrix.
        """

        if sd.dim == 0:
            # In a point, the differential is the trivial map
            return sps.csc_array((0, 1))
        elif sd.dim == 1:
            # In 1D, the gradient of the nodal functions scales as 1/h
            diff_nodes_0 = (sd.cell_faces.T / sd.cell_volumes[:, None]).tocsr()
            diff_nodes_1 = diff_nodes_0.copy()

            # The derivative of the nodal basis functions is equal to 3
            # on one side of the element and -1 on the other
            diff_nodes_0.data[0::2] = 3 * diff_nodes_0.data[0::2]
            diff_nodes_0.data[1::2] = -diff_nodes_0.data[1::2]
            diff_nodes_1.data[0::2] = -diff_nodes_1.data[0::2]
            diff_nodes_1.data[1::2] = 3 * diff_nodes_1.data[1::2]

            diff_nodes = sps.vstack((diff_nodes_0, diff_nodes_1))

            # The derivative of the edge (cell) basis functions are 4 and -4
            diff_edges_0 = sps.diags_array(4 / sd.cell_volumes)
            diff_edges = sps.vstack((diff_edges_0, -diff_edges_0))

            return sps.hstack((diff_nodes, diff_edges)).tocsc()

        # The 2D and 3D cases can be handled in a general way
        elif sd.dim == 2:
            edge_nodes = sd.face_ridges
            num_edges = sd.num_faces
            # The second degree of freedom on an edge
            # is oriented in the same way as the first
            second_dof_scaling = 1

        elif sd.dim == 3:
            edge_nodes = sd.ridge_peaks
            num_edges = sd.num_ridges
            # By design of Nedelec1, we orient the second dof
            # on an edge opposite to the first in 3D
            second_dof_scaling = -1

        # Start of the edge
        # The nodal function associated with the start has derivative -3 here.
        # The other nodal function has derivative -1.
        diff_nodes_0_csc = edge_nodes.copy().T
        diff_nodes_0_csc.data[edge_nodes.data == -1] = -3
        diff_nodes_0_csc.data[edge_nodes.data == 1] = -1

        diff_0 = sps.hstack((diff_nodes_0_csc, 4 * sps.eye_array(num_edges)))

        # End of the edge
        # The nodal function associated with the start has derivative 1 here.
        # The other nodal function has derivative 3.
        diff_nodes_1_csr = edge_nodes.copy().T
        diff_nodes_1_csr.data[edge_nodes.data == 1] = 3
        diff_nodes_1_csr.data[edge_nodes.data == -1] = 1

        # Rescale due to design choices in Nedelec1
        diff_1 = second_dof_scaling * sps.hstack(
            (diff_nodes_1_csr, -4 * sps.eye_array(num_edges))
        )

        # Combine
        return sps.vstack((diff_0, diff_1)).tocsc()

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for evaluating a P2 function at the
        cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection at the cell centers.
        """
        val_at_cc = 1 / (sd.dim + 1)
        eval_nodes = sd.cell_nodes().T * val_at_cc * (2 * val_at_cc - 1)

        if sd.dim == 1:
            eval_edges = sps.eye_array(sd.num_cells).tocsc()
        elif sd.dim == 2:
            eval_edges = abs(sd.cell_faces).T
        elif sd.dim == 3:
            eval_edges = abs(sd.cell_faces).T @ abs(sd.face_ridges).T
            eval_edges.data[:] = 1

        eval_edges = eval_edges * 4 * val_at_cc * val_at_cc

        return sps.hstack((eval_nodes, eval_edges)).tocsc()

    def assemble_lumped_matrix(self, sd: pg.Grid, data=None) -> sps.csc_array:
        """
        Assembles the lumped mass matrix for the quadratic Lagrange space.
        This is based on the integration rule by Eggers and Radu,
        and is not block-diagonal for this space.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (dict): A dictionary containing the necessary data for assembling the
                matrix.

        Returns:
            sps.csc_array: The lumped mass matrix.
        """
        Pi = self.proj_to_PwPolynomials(sd)
        L = pg.PwQuadratics(self.keyword).assemble_lumped_matrix(sd, data)

        return Pi.T @ L @ Pi

    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for projecting a quadratic Lagrangian function to a
        piecewise quadratic function.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        opposite_nodes = sd.compute_opposite_nodes()

        # Data allocation for the nodes mapping
        rows_I = np.arange(sd.num_cells * (sd.dim + 1))
        rows_I = rows_I.reshape((-1, sd.num_cells)).ravel(order="F")
        cols_J = opposite_nodes.data
        data_IJ = np.ones_like(rows_I, dtype=float)
        proj_nodes = sps.csc_array((data_IJ, (rows_I, cols_J)))

        # Data allocation for the edges mapping
        n_edges = self.num_edges_per_cell(sd.dim)
        size = n_edges * sd.num_cells
        rows_I = np.arange(size)
        rows_I = rows_I.reshape((-1, sd.num_cells)).ravel(order="F")
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.ones(size)
        idx = 0

        for c in range(sd.num_cells):
            loc = slice(opposite_nodes.indptr[c], opposite_nodes.indptr[c + 1])
            faces = opposite_nodes.indices[loc]
            edges = self.get_edge_dof_indices(sd, c, faces)

            loc_ind = slice(idx, idx + n_edges)
            cols_J[loc_ind] = edges - sd.num_nodes
            idx += n_edges

        proj_edges = sps.csc_array((data_IJ, (rows_I, cols_J)))

        return sps.block_diag((proj_nodes, proj_edges)).tocsc()

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a given function over the nodes of a grid.

        Args:
            sd (pg.Grid): The grid on which to interpolate the function.
            func (Callable[[np.ndarray], np.ndarray]): The function to be interpolated.

        Returns:
            np.ndarray: An array containing the interpolated values at each node of the
            grid.
        """
        if sd.dim == 0:
            edge_coords = np.empty(0)
        elif sd.dim == 1:
            edge_coords = sd.cell_centers
        elif sd.dim == 2:
            edge_coords = sd.face_centers
        elif sd.dim == 3:
            edge_coords = sd.nodes @ abs(sd.ridge_peaks) / 2

        coords = np.hstack((sd.nodes, edge_coords))

        return np.array([func(x) for x in coords.T])

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the 'natural' boundary condition
        (func, u)_Gamma with u a test function in Lagrange2

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function used to evaluate
                the 'natural' boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled 'natural' boundary condition values.
        """
        # In 1D, we reuse the code from P1
        if sd.dim == 1:
            # NOTE we pass self so that ndof() is taken from P2, not P1
            return Lagrange1.assemble_nat_bc(
                cast(pg.Lagrange1, self), sd, func, b_faces
            )

        # 2D and 3D
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        vals = np.zeros(self.ndof(sd))

        M = self.assemble_local_mass(sd.dim - 1)
        edge_nodes = sd.face_ridges if sd.dim == 2 else sd.ridge_peaks

        for face in b_faces:
            loc = slice(sd.face_nodes.indptr[face], sd.face_nodes.indptr[face + 1])
            loc_n = sd.face_nodes.indices[loc]

            if sd.dim == 2:
                edges = np.array([face])
            elif sd.dim == 3:
                # List local edges
                edges = sd.face_ridges.indices[loc]

                # Swap ordering so that edge 0 is opposite node 2
                check = sd.face_nodes[:, [face] * 3].astype(bool) - edge_nodes[
                    :, edges
                ].astype(bool)
                edges = edges[np.argsort(check.indices)]

                assert not np.any(edge_nodes[loc_n, edges])

            # Evaluate f at the nodes and edges
            f_vals = np.empty(sd.dim + len(edges))
            f_vals[: sd.dim] = [func(x) for x in sd.nodes[:, loc_n].T]

            for ind, edge in enumerate(edges):
                x0, x1 = sd.nodes[:, edge_nodes[:, [edge]].indices].T
                f_vals[sd.dim + ind] = func((x0 + x1) / 2)

            # Use the local mass matrix on the boundary
            vals_loc = M @ f_vals * sd.face_areas[face]

            vals[loc_n] += vals_loc[: sd.dim]
            vals[sd.num_nodes + edges] += vals_loc[sd.dim :]

        return vals

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the appropriate range discretization class based on the dimension.

        Args:
            dim (int): The dimension of the problem.

        Returns:
            object: The range discretization class.

        Raises:
            NotImplementedError: There is no zero-dimensional discretization in PyGeoN.
        """
        if dim == 3:
            return pg.Nedelec1
        elif dim == 2:
            return pg.BDM1
        elif dim == 1:
            return pg.PwLinears
        else:
            raise NotImplementedError("There's no zero discretization in PyGeoN")
