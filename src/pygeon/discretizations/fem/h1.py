"""Module for the discretizations of the H1 space."""

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
        In this case, the number of nodes.

        Args:
            sd: Grid, or a subclass.

        Returns:
            ndof: The number of degrees of freedom.
        """
        return sd.num_nodes

    def assemble_grad_grad_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the (K grad u, grad v) matrix for the nodal finite elements. This
        corresponds to the output of assemble_stiff_matrix, except in 2D. In that case
        the diff operator is a rotated gradient, leading to a different output for
        tensor-valued K.

        The scalar (pg.WEIGHT) and tensor-valued (pg.SECOND_ORDER_TENSOR) entries in the
        data dictionary are used as weights in the inner product.

        Args:
            sd (pg.Grid): The grid.
            data (dict): A dictionary containing the weight for the inner product.

        Returns:
            sps.csc_array: The assembled stiffness matrix.
        """
        M = pg.VecPwConstants(self.keyword).assemble_mass_matrix(sd, data)
        grad = self.assemble_grad_to_p0(sd)

        return (grad.T @ M @ grad).tocsc()

    def assemble_grad_to_p0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix that computes the gradient as a piecewise constant vector.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The gradient matrix.
        """
        return self.assemble_broken_grad_matrix(sd)

    def assemble_adv_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles and returns the advection matrix for Lagrange1 finite
        elements, which is given by
        :math:`(\\boldsymbol{v} \\cdot \\nabla p, p)`.

        The trial and test functions :math:`p` are Lagrange1.
        :math:`\\boldsymbol{v}` is a given vector field, assumed constant per
        cell. If not provided, :math:`\\boldsymbol{v}` defaults to :math:`(0, 0, 0)`.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (dict | None): Optional data for scaling, in particular
            pg.VECTOR-FIELD (advection velocity field).

        Returns:
            sps.csc_array: The assembled advection matrix.
        """
        # Retrieve the vector field
        V = pg.get_cell_data(sd, data, self.keyword, pg.VECTOR_FIELD, pg.VECTOR)

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1D and 2D)
        _, _, _, R, dim, node_coords = pp.map_geometry.map_grid(sd)

        if not data or not data.get("is_tangential", False):
            # Rotate the vector field and delete last dimension
            if sd.dim < 3:
                V = V.copy()
                V = R @ V
                remove_dim = np.where(np.logical_not(dim))[0]
                V = np.delete(V, remove_dim, axis=0)

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

            # Compute the adv-H1 local matrix
            A = self.local_adv(
                V[0 : sd.dim, c],
                sd.cell_volumes[c],
                coord_loc,
                sd.dim,
            )

            # Save values for adv-H1 local matrix in the global structure
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
        match sd.dim:
            case 3:
                return sd.ridge_peaks.T.tocsc()
            case 2:
                return sd.face_ridges.T.tocsc()
            case 1:
                return sd.cell_faces.T.tocsc()
            case 0:
                return sps.csc_array((0, 0))
            case _:
                raise ValueError("Dimension must be 0, 1, 2, or 3.")

    def local_adv(
        self, V: np.ndarray, c_volume: np.ndarray, coord: np.ndarray, dim: int
    ) -> np.ndarray:
        """
        Compute the local advection matrix for P1.

        Args:
            V (np.ndarray): vector field over the cell of (dim, dim) shape.
            c_volume (np.ndarray): scalar cell volume.
            coord (np.ndarray): coordinates of the cell vertices of (dim+1, dim) shape.
            dim (int): dimension of the problem.

        Returns:
            np.ndarray: local advection matrix of (dim+1, dim+1) shape.
        """
        phi = np.full((dim + 1,), (1 / (dim + 1)))

        dphi = self.local_grads(coord, dim)

        return c_volume * np.outer(phi, V @ dphi)

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
        match dim:
            case 3:
                return pg.Nedelec0
            case 2:
                return pg.RT0
            case 1:
                return pg.PwConstants
            case _:
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
        In this case, the number of nodes plus the number of edges,
        where edges are one-dimensional mesh entities.

        Args:
            sd: Grid, or a subclass.

        Returns:
            ndof: The number of degrees of freedom.
        """
        return sd.num_nodes + sd.num_edges

    def get_edge_dof_indices(
        self, sd: pg.Grid, cell: int, faces: np.ndarray
    ) -> np.ndarray:
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
        match sd.dim:
            case 1:
                # The only edge in 1D is the cell
                edges = np.array([cell])
            case 2:
                # The edges (0, 1), (0, 2), and (1, 2)
                # are the faces opposite nodes 2, 1, and 0, respectively.
                edges = faces[::-1]
            case 3:
                # We first find the edges adjacent to the local faces
                cell_edges = abs(sd.face_ridges[:, faces]) @ np.ones((4, 1))
                edge_inds = np.where(cell_edges)[0]

                # Experimentally, we always find the following numbering
                edges = edge_inds[[5, 4, 2, 3, 1, 0]]

        # The edge dofs come after the nodal dofs
        return edges + sd.num_nodes

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the differential matrix based on the dimension of the grid.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The differential matrix.
        """
        match sd.dim:
            case 0:
                # In a point, the differential is the trivial map
                return sps.csc_array((0, 1))
            case 1:
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
            case 2:
                edge_nodes = sd.face_ridges
                num_edges = sd.num_faces
                # The second degree of freedom on an edge
                # is oriented in the same way as the first
                second_dof_scaling = 1
            case 3:
                edge_nodes = sd.ridge_peaks
                num_edges = sd.num_ridges
                # By design of Nedelec1, we orient the second dof
                # on an edge opposite to the first in 3D
                second_dof_scaling = -1
            case _:
                raise ValueError("Dimension must be 0, 1, 2, or 3.")

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
        p2 = pg.PwQuadratics()
        n_edges = p2.num_edges_per_cell(sd.dim)
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
        match sd.dim:
            case 0:
                # In a point, there are no edges, so we only evaluate at the node
                edge_coords = np.empty((pg.AMBIENT_DIM, 0))
            case 1:
                # In 1D, the edge coordinate is the cell center
                edge_coords = sd.cell_centers
            case 2:
                # In 2D, the edge coordinate is the face center
                edge_coords = sd.face_centers
            case 3:
                # In 3D, the edge coordinate is the midpoint of the two nodes opposite
                # to the ridge
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

            match sd.dim:
                case 2:
                    edges = np.array([face])
                case 3:
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
        match dim:
            case 3:
                return pg.Nedelec1
            case 2:
                return pg.BDM1
            case 1:
                return pg.PwLinears
            case _:
                raise NotImplementedError("There's no zero discretization in PyGeoN")
