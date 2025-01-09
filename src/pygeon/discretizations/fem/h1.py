""" Module for the discretizations of the H1 space. """

import math
from typing import Callable, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class Lagrange1(pg.Discretization):
    """
    Class representing the Lagrange1 finite element discretization.

    Attributes:
        None

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the mass matrix for the lowest order Lagrange element.

        local_mass(c_volume: np.ndarray, dim: int) -> np.ndarray:
            Computes the local mass matrix.

        assemble_stiff_matrix(sd: pg.Grid, data: dict) -> sps.csc_matrix:
            Assembles the stiffness matrix for the finite element method.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the differential matrix based on the dimension of the grid.

        local_stiff(K: np.ndarray, c_volume: np.ndarray, coord: np.ndarray, dim: int)
            -> np.ndarray:
            Computes the local stiffness matrix for P1.

        local_grads(coord: np.ndarray, dim: int) -> np.ndarray:
            Calculates the local gradients of the finite element basis functions.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the lumped mass matrix for the finite element method.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_matrix:
            Constructs the matrix for evaluating a Lagrangian function at the cell centers of
            the given grid.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a given function over the nodes of a grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the 'natural' boundary condition.

        get_range_discr_class(dim: int) -> object:
            Returns the appropriate range discretization class based on the dimension.
    """

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
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Returns the mass matrix for the lowest order Lagrange element

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly process.

        Returns:
            sps.csc_matrix: The mass matrix obtained from the discretization.
        """

        # Data allocation
        size = np.power(sd.dim + 1, 2) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()
        local_mass = self.local_mass(sd.dim)

        for c in np.arange(sd.num_cells):
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
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def local_mass(self, dim: int) -> np.ndarray:
        """Compute the local mass matrix on an element with measure 1.

        Args:
            dim (int): Dimension of the matrix.

        Returns:
            np.ndarray: Local mass matrix of shape (num_nodes_of_cell, num_nodes_of_cell).
        """

        M = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        return M / ((dim + 1) * (dim + 2))

    def assemble_stiff_matrix(self, sd: pg.Grid, data: dict) -> sps.csc_matrix:
        """
        Assembles the stiffness matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (dict): A dictionary containing the necessary data for assembling the matrix.

        Returns:
            sps.csc_matrix: The assembled stiffness matrix.
        """
        # Get dictionary for parameter storage
        try:
            K = data[pp.PARAMETERS][self.keyword]["second_order_tensor"]
        except Exception:
            K = pp.SecondOrderTensor(np.ones(sd.num_cells))
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

        for c in np.arange(sd.num_cells):
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
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the differential matrix based on the dimension of the grid.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        if sd.dim == 3:
            return sd.ridge_peaks.T.tocsc()
        elif sd.dim == 2:
            return sd.face_ridges.T.tocsc()
        elif sd.dim == 1:
            return sd.cell_faces.T.tocsc()
        elif sd.dim == 0:
            return sps.csc_matrix((0, 1))
        else:
            raise ValueError

    def local_stiff(
        self, K: np.ndarray, c_volume: np.ndarray, coord: np.ndarray, dim: int
    ) -> np.ndarray:
        """
        Compute the local stiffness matrix for P1.

        Args:
            K (np.ndarray): permeability of the cell of (dim, dim) shape.
            c_volume (np.ndarray): scalar cell volume.
            coord (np.ndarray): coordinates of the cell vertices of (dim+1, dim) shape.
            dim (int): dimension of the problem.

        Returns:
            np.ndarray: local stiffness matrix of (dim+1, dim+1) shape.
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
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the lumped mass matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (Optional[dict]): Optional data dictionary.

        Returns:
            sps.csc_matrix: The assembled lumped mass matrix.
        """
        volumes = sd.cell_nodes() * sd.cell_volumes / (sd.dim + 1)
        return sps.diags(volumes).tocsc()

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Construct the matrix for evaluating a Lagrangian function at the
        cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_matrix: The matrix representing the projection at the cell centers.
        """
        eval = sd.cell_nodes()
        num_nodes = sps.diags(1.0 / sd.num_cell_nodes())

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
            np.ndarray: An array containing the interpolated values at each node of the grid.
        """
        return np.array([func(x) for x in sd.nodes.T])

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the 'natural' boundary condition
        (u, func)_Gamma with u a test function in Lagrange1

        Args:
            sd (pg.Grid): The grid object representing the computational domain
            func (Callable[[np.ndarray], np.ndarray]): The function used to evaluate
                the 'natural' boundary condition
            b_faces (np.ndarray): The array of boundary faces

        Returns:
            np.ndarray: The assembled 'natural' boundary condition values
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

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
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
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Returns the mass matrix for the second order Lagrange element

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly process.

        Returns:
            sps.csc_matrix: The mass matrix.
        """

        try:
            weight = data[pp.PARAMETERS][self.keyword]["weight"]
        except Exception:
            weight = np.ones(sd.num_cells)

        # Data allocation
        size = np.square((sd.dim + 1) + self.num_edges_per_cell(sd.dim)) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        opposite_nodes = sd.compute_opposite_nodes()
        local_mass = self.assemble_local_mass(sd.dim)

        for c in np.arange(sd.num_cells):
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
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_local_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local mass matrix of the basis functions
        on a d-simplex with measure 1.

        Args:
            dim (int): The dimension of the simplex.

        Returns:
            np.ndarray: the local mass matrix.
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
            expnts (np.ndarray): each column is an array of exponents
                alpha_i of the monomial expressed as
                prod_i lambda_i ^ alpha_i.

        Returns:
            np.ndarray: the inner products of the monomials
                on a simplex with measure 1.
        """
        n_monomials = expnts.shape[1]
        mass = np.empty((n_monomials, n_monomials))

        for i in np.arange(n_monomials):
            for j in np.arange(n_monomials):
                mass[i, j] = self.integrate_monomial(expnts[:, i] + expnts[:, j])

        return mass

    @staticmethod
    def factorial(n: float) -> int:
        """
        Compute the factorial of a float by first rounding to an int.
        Args:
            n (float): the input float

        Returns:
            int: the factorial n!
        """
        return math.factorial(int(n))

    def integrate_monomial(self, alphas: np.ndarray) -> float:
        """
        Exact integration of products of monomials based on
        Vermolen and Segal (2018).

        Args:
            alphas (np.ndarray): array of exponents alpha_i of the monomial
                expressed as prod_i lambda_i ^ alpha_i

        Returns:
            float: the integral of the monomial on a simplex with measure 1
        """
        dim = len(alphas) - 1
        fac_alph = [self.factorial(a_i) for a_i in alphas]

        return (
            self.factorial(dim)
            * np.prod(fac_alph)
            / self.factorial(dim + np.sum(alphas))
        )

    def num_edges_per_cell(self, dim: int) -> int:
        """
        Compute the number of edges of a simplex of a given dimension.

        Args:
            dim (int): dimension

        Returns:
            int: the number of adjacent edges
        """

        return dim * (dim + 1) // 2

    def get_local_edge_nodes(self, dim: int) -> np.ndarray:
        """
        Lists the local edge-node connectivity in the cell

        Args:
            dim (int): dimension

        Returns:
            np.ndarray: row i contains the local indices of the
                nodes connected to the edge with local index i
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
            dphi (np.ndarray): Gradients of the P1 basis functions
            e_nodes (np.ndarray): The local edge-node connectivity

        Returns:
            np.ndarray: the gradient of basis function i at node j is
                in elements [i, 3 * (j:j + 1)]
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

    def get_edge_dof_indices(self, sd, cell, faces):
        """
        Finds the indices for the edge degrees of freedom that correspond
        to the local numbering of the edges.

        Args:
            sd (pg.Grid): The grid
            cell (int): The cell index
            faces (np.ndarray): Face indices of the cell

        Returns:
            np.ndarray: Indices of the edge degrees of freedom
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
            cell_edges = np.abs(sd.face_ridges[:, faces]) @ np.ones((4, 1))
            edge_inds = np.where(cell_edges)[0]

            # Experimentally, we always find the following numbering
            edges = edge_inds[[5, 4, 2, 3, 1, 0]]

        # The edge dofs come after the nodal dofs
        return edges + sd.num_nodes

    def assemble_stiff_matrix(self, sd: pg.Grid, data: dict) -> sps.csc_matrix:
        """
        Assembles the stiffness matrix for the P2 finite element method.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (dict): A dictionary containing the necessary data for assembling the matrix.

        Returns:
            sps.csc_matrix: The stiffness matrix.
        """

        try:
            K = data[pp.PARAMETERS][self.keyword]["second_order_tensor"]
        except Exception:
            K = pp.SecondOrderTensor(np.ones(sd.num_cells))

        size = np.square((sd.dim + 1) + self.num_edges_per_cell(sd.dim)) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        opposite_nodes = sd.compute_opposite_nodes()
        local_mass = pg.BDM1.local_inner_product(sd.dim)
        e_nodes = self.get_local_edge_nodes(sd.dim)

        for c in np.arange(sd.num_cells):
            loc = slice(opposite_nodes.indptr[c], opposite_nodes.indptr[c + 1])
            faces = opposite_nodes.indices[loc]
            nodes = opposite_nodes.data[loc]
            edges = self.get_edge_dof_indices(sd, c, faces)

            signs = sd.cell_faces.data[loc]
            dphi = -sd.face_normals[:, faces] * signs / (sd.dim * sd.cell_volumes[c])
            Psi = self.eval_grads_at_nodes(dphi, e_nodes)

            weight = np.kron(np.eye(sd.dim + 1), K.values[:, :, c])

            A = Psi @ local_mass @ weight @ Psi.T * sd.cell_volumes[c]

            loc_ind = np.hstack((nodes, edges))

            cols = np.tile(loc_ind, (loc_ind.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Assemble
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the differential matrix based on the dimension of the grid.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_matrix: The differential matrix.
        """

        if sd.dim == 0:
            # In a point, the differential is the trivial map
            return sps.csc_matrix((0, 1))
        elif sd.dim == 1:
            # In 1D, the gradient of the nodal functions scales as 1/h
            diff_nodes = sd.cell_faces.T / sd.cell_volumes[:, None]

            # Because of the numbering of the pw linears, the
            # first dof of Lagrange2 maps to the first two dofs of PwLinears
            diff_nodes = diff_nodes.tocsr()[
                np.repeat(np.arange(diff_nodes.shape[0]), 2), :
            ]
            # The derivative of the nodal basis functions is equal to 3
            # on one side of the element and -1 on the other
            diff_nodes.data[0::4] *= 3
            diff_nodes.data[1::4] *= -1
            diff_nodes.data[2::4] *= -1
            diff_nodes.data[3::4] *= 3

            # The derivative of the edge (cell) basis functions are 4 and -4
            diff_edges = sps.block_diag(
                [np.array([[4], [-4]]) / vol for vol in sd.cell_volumes]
            )

            return sps.hstack((diff_nodes, diff_edges), format="csc")

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
        diff_nodes_0 = edge_nodes.copy().T
        diff_nodes_0.data[edge_nodes.data == -1] = -3
        diff_nodes_0.data[edge_nodes.data == 1] = -1

        diff_0 = sps.hstack((diff_nodes_0, 4 * sps.eye(num_edges)))

        # End of the edge
        # The nodal function associated with the start has derivative 1 here.
        # The other nodal function has derivative 3.
        diff_nodes_1 = edge_nodes.copy().T
        diff_nodes_1.data[edge_nodes.data == 1] = 3
        diff_nodes_1.data[edge_nodes.data == -1] = 1

        # Rescale due to design choices in Nedelec1
        diff_1 = second_dof_scaling * sps.hstack(
            (diff_nodes_1, -4 * sps.eye(num_edges))
        )

        # Combine
        return sps.vstack((diff_0, diff_1), format="csc")

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Construct the matrix for evaluating a P2 function at the
        cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_matrix: The matrix representing the projection at the cell centers.
        """
        val_at_cc = 1 / (sd.dim + 1)
        eval_nodes = sd.cell_nodes().T * val_at_cc * (2 * val_at_cc - 1)

        if sd.dim == 1:
            eval_edges = sps.eye(sd.num_cells)
        elif sd.dim == 2:
            eval_edges = np.abs(sd.cell_faces).T
        elif sd.dim == 3:
            eval_edges = np.abs(sd.cell_faces).T @ np.abs(sd.face_ridges).T
            eval_edges.data[:] = 1

        eval_edges = eval_edges * 4 * val_at_cc * val_at_cc

        return sps.hstack((eval_nodes, eval_edges), format="csc")

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a given function over the nodes of a grid.

        Args:
            sd (pg.Grid): The grid on which to interpolate the function.
            func (Callable[[np.ndarray], np.ndarray]): The function to be interpolated.

        Returns:
            np.ndarray: An array containing the interpolated values at each node of the grid.
        """
        if sd.dim == 0:
            edge_coords = []
        elif sd.dim == 1:
            edge_coords = sd.cell_centers
        elif sd.dim == 2:
            edge_coords = sd.face_centers
        elif sd.dim == 3:
            edge_coords = sd.nodes @ np.abs(sd.ridge_peaks) / 2

        coords = np.hstack((sd.nodes, edge_coords))

        return np.array([func(x) for x in coords.T])

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the 'natural' boundary condition
        (func, u)_Gamma with u a test function in Lagrange2

        Args:
            sd (pg.Grid): The grid object representing the computational domain
            func (Callable[[np.ndarray], np.ndarray]): The function used to evaluate
                the 'natural' boundary condition
            b_faces (np.ndarray): The array of boundary faces

        Returns:
            np.ndarray: The assembled 'natural' boundary condition values
        """
        # In 1D, we reuse the code from P1
        if sd.dim == 1:
            # NOTE we pass self so that ndof() is taken from P2, not P1
            return Lagrange1.assemble_nat_bc(self, sd, func, b_faces)

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
                edges = np.roll(edges, -1)[::-1]

                # Check whether each edge is opposite
                # the appropriate node in loc_n
                check = sd.face_nodes[:, [face] * 3].astype(bool) - edge_nodes[
                    :, edges
                ].astype(bool)
                assert np.all(loc_n[::-1] == check.indices)

            # Evaluate f at the nodes and edges
            f_vals = np.empty(sd.dim + len(edges))
            f_vals[: sd.dim] = [func(x) for x in sd.nodes[:, loc_n].T]

            for ind, edge in enumerate(edges):
                x0, x1 = sd.nodes[:, edge_nodes[:, edge].indices].T
                f_vals[sd.dim + ind] = func((x0 + x1) / 2)

            # Use the local mass matrix on the boundary
            vals_loc = M @ f_vals * sd.face_areas[face]

            vals[loc_n] += vals_loc[: sd.dim]
            vals[sd.num_nodes + edges] += vals_loc[sd.dim :]

        return vals

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
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
