"""Module for the discretizations of the H1 space."""

from functools import cache
from typing import Callable, Type

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class Lagrange(pg.Discretization):
    r"""
    Abstract class collecting the features shared by the nodal Lagrange finite element
    discretizations :math:`\mathbb{L}_k(\Omega) \subset H^1(\Omega)`, for a generic
    domain :math:`\Omega \subset \mathbb{R}^d`.
    """

    tensor_order = pg.SCALAR
    """Scalar-valued discretization"""

    def assemble_grad_grad_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles the :math:`(K \nabla u, \nabla v)_\Omega`, for
        :math:`u,v \in \mathbb{L}_k(\Omega)` matrix for the nodal finite elements.

        This corresponds to the output of assemble_stiff_matrix, except in 2D.
        In that case the diff operator is a rotated gradient, leading to a different
        output for tensor-valued K.

        The scalar (pg.WEIGHT) and tensor-valued (pg.SECOND_ORDER_TENSOR) entries in the
        data dictionary are used as weights in the inner product.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): A dictionary containing the weights for the inner
                product.

        Returns:
            sps.csc_array: The assembled stiffness matrix.
        """
        grad_space = self.get_broken_grad_discr()(self.keyword)
        M = grad_space.assemble_mass_matrix(sd, data)
        grad = self.assemble_broken_grad_matrix(sd)

        return (grad.T @ M @ grad).tocsc()

    def postprocess_darcy_flux(
        self, primary_var: np.ndarray, sd: pg.Grid, data: dict | None
    ) -> np.ndarray:
        r"""
        Computes the Darcy flux :math:`-K \nabla u` as a vector piecewise polynomial of
        degree :math:`k - 1`, with :math:`K` the tensor (pg.SECOND_ORDER_TENSOR) and
        the weight (pg.WEIGHT) in the data dictionary.

        Args:
            primary_var (np.ndarray): The degrees of freedom of the primary variable.
            sd (pg.Grid): The grid.
            data (dict | None): The data dictionary.

        Returns:
            np.ndarray: The degrees of freedom of the Darcy flux in the vector piecewise
            polynomial space of degree :math:`k - 1`.
        """
        grad = (self.assemble_broken_grad_matrix(sd) @ primary_var).reshape(sd.dim, -1)

        tensor = pg.get_cell_data(
            sd, data, self.keyword, pg.SECOND_ORDER_TENSOR, pg.MATRIX
        )
        weight = pg.get_cell_data(sd, data, self.keyword, pg.WEIGHT)
        R = sd.rotation_matrix
        K = np.tensordot(R.T, np.tensordot(R, tensor.values, (1, 0)), (0, 1))
        K *= weight

        # The local dofs of each polynomial component are ordered as (local, cell)
        cells = np.tile(np.arange(sd.num_cells), grad.shape[1] // sd.num_cells)

        return -np.einsum("ijn,jn->in", K[:, :, cells], grad).ravel()


class Lagrange1(Lagrange):
    r"""
    Class implementing the finite element discretization Lagrange1
    :math:`\mathbb{L}_1(\Omega) \subset H^1(\Omega)`, for a generic domain
    :math:`\Omega \subset \mathbb{R}^d`.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    def ndof_per_entity(self, dim: int) -> np.ndarray:
        """
        Returns the number of degrees of freedom per geometric entity, ordered by
        the dimension of the entity as [0, 1, 2, 3].
        In this case, one degree of freedom per node. A point grid has no nodes,
        since its only entity is the cell, hence no degrees of freedom.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: The number of degrees of freedom per entity.
        """
        return np.array([int(dim > 0), 0, 0, 0])  # nodes

    def assemble_adv_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles and returns the advection matrix for Lagrange1 finite
        elements, which is given by
        :math:`(\boldsymbol{\beta} \cdot \nabla u, v)_\Omega`, for
        :math:`u,v \in \mathbb{L}_1(\Omega)`.

        The data dictionary contains the vector field :math:`\boldsymbol{\beta}`
        accessible via pg.VECTOR_FIELD. It is given as a vector field, assumed
        constant per cell (:class:`~pygeon.VecPwConstants`). If not provided, it
        defaults to :math:`(0, 0, 0)`. On grids of dimension lower than 3, the field is
        rotated to the plane or the line of the grid, unless the entry "is_tangential"
        of data is True.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Optional data for scaling, in particular
                pg.VECTOR_FIELD (advection velocity field).

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
        r"""
        Assembles the differential matrix based on the dimension of the grid.

        The differential corresponds to the (co-)gradient operator :math:`d`,
        mapping from :class:`~pygeon.Lagrange1` to the appropriate range space:

        - 3D: :class:`~pygeon.NedelecR0`, H(curl)
        - 2D: :class:`~pygeon.RT0`, H(div)
        - 1D: :class:`~pygeon.PwConstants`, L2

        Args:
            sd (pg.Grid): The grid.

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
        Compute the local advection matrix for linear elements.

        Args:
            V (np.ndarray): The vector field in the cell, of shape (dim,).
            c_volume (np.ndarray): The cell volume.
            coord (np.ndarray): The coordinates of the cell vertices, of shape
                (dim, dim + 1).
            dim (int): The dimension of the problem.

        Returns:
            np.ndarray: The local advection matrix, of shape (dim + 1, dim + 1).
        """
        phi = np.full((dim + 1,), (1 / (dim + 1)))

        dphi = self.local_grads(coord, dim)

        return c_volume * np.outer(phi, V @ dphi)

    @staticmethod
    def local_grads(coord: np.ndarray, dim: int) -> np.ndarray:
        """
        Calculate the local gradients of the linear finite element basis functions.

        Args:
            coord (np.ndarray): The coordinates of the nodes in the element, of shape
                (dim, dim + 1).
            dim (int): The dimension of the element.

        Returns:
            np.ndarray: The gradients of the basis functions in the columns, of shape
            (dim, dim + 1).
        """
        Q = np.hstack((np.ones((dim + 1, 1)), coord.T))
        invQ = np.linalg.inv(Q)
        return invQ[1:, :]

    @cache
    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Construct the matrix for projecting a Lagrangian function to a piecewise linear
        function. The projection operator :math:`\Pi` takes a function from
        :class:`~pygeon.Lagrange1` and maps it to a piecewise linear function in
        :class:`~pygeon.PwLinears`.

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
            sps.csc_array: The evaluation matrix.
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
        return self.eval_func_at_coords(func, sd.nodes)

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        r"""
        Assembles the 'natural' boundary condition
        :math:`(v, g)_{\partial\Omega}` with :math:`v` a test function in
        :class:`~pygeon.Lagrange1` and :math:`g` the prescribed datum, evaluated at the
        face centers and distributed equally to the nodes of each face.

        Args:
            sd (pg.Grid): The grid.
            func (Callable[[np.ndarray], np.ndarray]): The datum :math:`g`.
            b_faces (np.ndarray): The boundary faces, as a boolean mask or as indices.

        Returns:
            np.ndarray: The assembled 'natural' boundary condition values.
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        # This will be sd.dim on simplicial grids, but we generalize to handle VEM
        num_nodes_per_face = (
            sd.face_nodes.indptr[b_faces + 1] - sd.face_nodes.indptr[b_faces]
        )
        func_vals = self.eval_func_at_coords(func, sd.face_centers[:, b_faces])

        face_vals = np.zeros(sd.num_faces)
        face_vals[b_faces] = func_vals * sd.face_areas[b_faces] / num_nodes_per_face

        return sd.face_nodes @ face_vals

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the appropriate range discretization class based on the dimension.

        Args:
            dim (int): The dimension of the problem.

        Returns:
            Type[pg.Discretization]: The range discretization class.

        Raises:
            NotImplementedError: In dimension 0, there is no zero discretization in
                PyGeoN.
        """
        match dim:
            case 3:
                return pg.NedelecR0
            case 2:
                return pg.RT0
            case 1:
                return pg.PwConstants
            case _:
                raise NotImplementedError("There's no zero discretization in PyGeoN")


class Lagrange2(Lagrange):
    r"""
    Class implementing the finite element discretization Lagrange2
    :math:`\mathbb{L}_2(\Omega) \subset H^1(\Omega)`, for a generic domain
    :math:`\Omega \subset \mathbb{R}^d`.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

    def ndof_per_entity(self, dim: int) -> np.ndarray:
        """
        Returns the number of degrees of freedom per geometric entity, ordered by
        the dimension of the entity as [0, 1, 2, 3].
        In this case, one degree of freedom per node and per edge. A point grid has
        no nodes, since its only entity is the cell, hence no degrees of freedom.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: The number of degrees of freedom per entity.
        """
        return np.array([int(dim > 0), 1, 0, 0])  # nodes and edges

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Assembles the differential matrix based on the dimension of the grid.

        The differential corresponds to the (co-)gradient operator :math:`d`,
        mapping from :class:`~pygeon.Lagrange2` to the appropriate range space:

        - 3D: :class:`~pygeon.NedelecF1`, H(curl)
        - 2D: :class:`~pygeon.BDM1`, H(div)
        - 1D: :class:`~pygeon.PwLinears`, L2

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The differential matrix.
        """
        match sd.dim:
            case 0:
                # In a point, the differential is the trivial map
                return sps.csc_array((0, 0))
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
                # By design of NedelecF1, we orient the second dof
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

        # Rescale due to design choices in NedelecF1
        diff_1 = second_dof_scaling * sps.hstack(
            (diff_nodes_1_csr, -4 * sps.eye_array(num_edges))
        )

        # Combine
        return sps.vstack((diff_0, diff_1)).tocsc()

    @cache
    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Construct the matrix for projecting a quadratic Lagrangian function to a
        piecewise quadratic function. The projection operator :math:`\Pi` takes a
        function from :class:`~pygeon.Lagrange2` and maps it to a piecewise quadratic
        function in :class:`~pygeon.PwQuadratics`.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        # Data allocation for the nodes mapping
        rows_I = np.arange(sd.num_cells * (sd.dim + 1))
        rows_I = rows_I.reshape((-1, sd.num_cells)).ravel(order="F")
        cols_J = sd.cell_nodes().indices
        data_IJ = np.ones_like(rows_I, dtype=float)
        proj_nodes = sps.csc_array((data_IJ, (rows_I, cols_J)))

        # Data allocation for the edges mapping
        p2 = pg.PwQuadratics()
        n_edges = p2.num_edges_per_cell(sd.dim)
        size = n_edges * sd.num_cells
        rows_I = np.arange(size)
        rows_I = rows_I.reshape((-1, sd.num_cells)).ravel(order="F")

        match sd.dim:
            case 1:
                # The only edge in 1D is the cell
                edges = np.arange(sd.num_cells)
            case 2:
                # In 2D, the edges are the faces
                edges = sd.cell_faces.indices
            case 3:
                # We find the cell-edge connectivity
                cell_edges = abs(sd.face_ridges) @ abs(sd.cell_faces)
                cell_edges.sort_indices()
                edges = cell_edges.indices
        cols_J = edges.ravel()

        data_IJ = np.ones(size)
        proj_edges = sps.csc_array((data_IJ, (rows_I, cols_J)))

        return sps.block_diag((proj_nodes, proj_edges)).tocsc()

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a given function at the nodes and at the midpoints of the edges
        of a grid.

        Args:
            sd (pg.Grid): The grid on which to interpolate the function.
            func (Callable[[np.ndarray], np.ndarray]): The function to be interpolated.

        Returns:
            np.ndarray: The values at the nodes followed by the values at the edge
            midpoints.
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

        return self.eval_func_at_coords(func, coords)

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        r"""
        Assembles the 'natural' boundary condition
        :math:`(v, g)_{\partial\Omega}` with :math:`v` a test function in
        :class:`~pygeon.Lagrange2` and :math:`g` the prescribed datum.

        Args:
            sd (pg.Grid): The grid.
            func (Callable[[np.ndarray], np.ndarray]): The datum :math:`g`, evaluated
                as in interpolate at the nodes and at the edge midpoints.
            b_faces (np.ndarray): The boundary faces, as a boolean mask or as indices.

        Returns:
            np.ndarray: The assembled 'natural' boundary condition values.
        """
        vals = np.zeros(self.ndof(sd))

        # In 1D, we reuse the code from P1
        if sd.dim == 1:
            vals[: sd.num_nodes] = Lagrange1().assemble_nat_bc(sd, func, b_faces)
            return vals

        # 2D and 3D
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        if b_faces.size == 0:
            return vals

        M = pg.PwQuadratics().assemble_local_mass(sd.dim - 1)

        # Nodes of each boundary face, shape (num_b_faces, dim)
        face_nodes = sd.face_nodes.tocsc()
        face_nodes.sort_indices()
        nodes = face_nodes.indices.reshape(sd.num_faces, sd.dim)[b_faces]

        match sd.dim:
            case 2:
                # the only edge of a face is the face itself
                edges = b_faces[:, None]
            case 3:
                # The ids of the edges of each face and of their nodes
                face_ridges = sd.face_ridges.tocsc()
                ridge_peaks = sd.ridge_peaks.tocsc()
                ridge_peaks.sort_indices()
                ridges = face_ridges.indices.reshape(sd.num_faces, -1)[b_faces]
                ridge_nodes = ridge_peaks.indices.reshape(-1, 2)[ridges]

                # In M the local edges are (0, 1), (0, 2), (1, 2), so the local edge k
                # is opposite the local node 2 - k. The node opposite each edge, shape
                # (num_b_faces, 3), is the one of the face not in the edge, i.e. the
                # difference of the sums of their ids
                opposite = nodes.sum(axis=1, keepdims=True) - ridge_nodes.sum(axis=2)

                # Its local index: is_opposite[f, e, j] tells whether the local node j
                # of the face f is opposite its edge e, comparing the nodes of each
                # face with the opposite node of each edge
                face_nodes_row = nodes.reshape(b_faces.size, 1, 3)
                opposite_col = opposite.reshape(b_faces.size, 3, 1)
                is_opposite = face_nodes_row == opposite_col
                loc_opposite = np.argmax(is_opposite, axis=2)

                edges = np.empty_like(ridges)
                np.put_along_axis(edges, 2 - loc_opposite, ridges, axis=1)

        # The local dofs of each boundary face, in the ordering of M, where func is
        # interpolated: at the nodes and at the midpoints of the edges
        dofs = np.hstack((nodes, sd.num_nodes + edges))
        f_vals = self.interpolate(sd, func)[dofs]

        # Use the local mass matrix on each boundary face, each row scaled by the area
        # of its face
        vals_loc = (f_vals @ M.T) * sd.face_areas[b_faces, None]

        # Sum the local contributions into the global dofs: a node or an edge shared
        # by several boundary faces appears more than once in dofs, so the unbuffered
        # np.add.at is needed, vals[dofs] += vals_loc would keep only one of them
        np.add.at(vals, dofs.ravel(), vals_loc.ravel())

        return vals

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the appropriate range discretization class based on the dimension.

        Args:
            dim (int): The dimension of the problem.

        Returns:
            Type[pg.Discretization]: The range discretization class.

        Raises:
            NotImplementedError: In dimension 0, there is no zero discretization in
                PyGeoN.
        """
        match dim:
            case 3:
                return pg.NedelecF1
            case 2:
                return pg.BDM1
            case 1:
                return pg.PwLinears
            case _:
                raise NotImplementedError("There's no zero discretization in PyGeoN")
