"""Module for the discretizations of the H1 space."""

from typing import Callable, Type

import numpy as np
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

    def assemble_adv_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles and returns the advection matrix for the Lagrange finite elements,
        which is given by :math:`(\boldsymbol{\beta} \cdot \nabla u, v)_\Omega`, for
        :math:`u,v \in \mathbb{L}_k(\Omega)`.

        The data dictionary contains the vector field :math:`\boldsymbol{\beta}`
        accessible via pg.VECTOR_FIELD. It is given as a vector field, assumed
        constant per cell (:class:`pg.VecPwConstants`). If not provided, it
        defaults to :math:`(0, 0, 0)`. On grids of dimension lower than 3, the field is
        rotated to the plane or the line of the grid with ``sd.rotation_matrix``,
        unless the entry "is_tangential" of data is True: then its first dim
        components are taken as the ones in those coordinates.

        Since the field is constant on each cell,
        :math:`\boldsymbol{\beta} \cdot \nabla u` is a piecewise polynomial of order
        k - 1, and the matrix is the composition

        .. math::

            A = P^\top M_k I_{k-1 \to k} D_{\boldsymbol{\beta}},

        with :math:`D_{\boldsymbol{\beta}}` the directional derivative of
        :meth:`pg.Discretization.assemble_broken_directional_derivative`, to the
        piecewise polynomials of order k - 1, :math:`I_{k-1 \to k}` their inclusion in
        the order k, :math:`M_k` the mass matrix of the latter, and P the projection of
        :math:`\mathbb{L}_k(\Omega)` to the piecewise polynomials of order k.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Optional data for scaling, in particular
                pg.VECTOR_FIELD (advection velocity field).

        Returns:
            sps.csc_array: The assembled advection matrix.
        """
        # Retrieve the vector field, in the ambient coordinates
        V = pg.get_cell_data(sd, data, self.keyword, pg.VECTOR_FIELD, pg.VECTOR)
        # TODO: remove is_tangential, see #340
        if data and data.get("is_tangential", False):
            V = sd.rotation_matrix.T @ V[: sd.dim]

        # The directional derivative of u along the field, in the piecewise
        # polynomials of order k - 1
        dir_der = self.assemble_broken_directional_derivative(sd, V)

        # Its inclusion in the piecewise polynomials of order k, where it is tested
        # with v through their mass matrix
        low = pg.get_PwPolynomials(self.poly_order - 1, pg.SCALAR)()
        high = pg.get_PwPolynomials(self.poly_order, pg.SCALAR)()
        incl = low.proj_to_higher_PwPolynomials(sd)
        mass = high.assemble_mass_matrix(sd)
        proj = self.proj_to_PwPolynomials(sd)
        adv = proj.T @ mass @ incl @ dir_der

        return pg.drop_small_entries(sps.csc_array(adv))

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

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Assembles the matrix corresponding to the differential operator.

        The differential corresponds to the (co-)gradient operator :math:`d`,
        mapping from :class:`pg.Lagrange1` to the appropriate range space:

        - 3D: :class:`pg.NedelecR0`, H(curl)
        - 2D: :class:`pg.RT0`, H(div)
        - 1D: :class:`pg.PwConstants`, L2

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The differential matrix.
        """
        match sd.dim:
            case 1 | 2 | 3:
                # The difference of the values at the two nodes of each edge
                return sd.edge_nodes.T.tocsc()
            case 0:
                return sps.csc_array((0, 0))
            case _:
                raise ValueError("Dimension must be 0, 1, 2, or 3.")

    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Construct the matrix for projecting a Lagrangian function to a piecewise linear
        function. The projection operator :math:`\Pi` takes a function from
        :class:`pg.Lagrange1` and maps it to a piecewise linear function in
        :class:`pg.PwLinears`.

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
        Assembles the matrix for evaluating the discretization at the cell centers.

        Args:
            sd (pg.Grid): The grid.

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
        :class:`pg.Lagrange1` and :math:`g` the prescribed datum, evaluated at the
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
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

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
        Assembles the matrix corresponding to the differential operator.

        The differential corresponds to the (co-)gradient operator :math:`d`,
        mapping from :class:`pg.Lagrange2` to the appropriate range space:

        - 3D: :class:`pg.NedelecF1`, H(curl)
        - 2D: :class:`pg.BDM1`, H(div)
        - 1D: :class:`pg.PwLinears`, L2

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
                # The edges are the cells, and the dofs of PwLinears are the values of
                # the derivative at the nodes of the cell in increasing order
                dof_nodes = sd.cell_nodes()
                scaling = 1 / sd.cell_volumes
                second_dof_scaling = 1
            case 2:
                # The edges are the faces, and the dofs of BDM1 are the moments at the
                # nodes of each face, in the order of face_nodes
                dof_nodes = sd.face_nodes
                scaling = np.ones(sd.num_faces)
                second_dof_scaling = 1
            case 3:
                # The edges are the ridges, and the dofs of NedelecF1 are the moments
                # at the nodes of each ridge, in the order of ridge_peaks, with the
                # second dof on an edge oriented opposite to the first
                dof_nodes = sd.ridge_peaks
                scaling = np.ones(sd.num_ridges)
                second_dof_scaling = -1
            case _:
                raise ValueError("Dimension must be 0, 1, 2, or 3.")

        # The nodes of the two dofs of each edge, shape (num_edges, 2), and the
        # orientation of the edge at each of them: -1 at its start, 1 at its end
        edges = np.arange(sd.num_edges)
        nodes = dof_nodes.indices.reshape((sd.num_edges, 2))
        signs = np.column_stack([sd.edge_nodes[nodes[:, i], edges] for i in range(2)])
        signs = signs * scaling.reshape((-1, 1)) * np.array([1, second_dof_scaling])

        # At the node a of an edge with other node b and midpoint dof m, the derivative
        # along the edge is s (3 u_a + u_b - 4 u_m), with s the orientation at a
        rows_I = np.tile(edges.reshape((-1, 1)) + sd.num_edges * np.arange(2), 3)
        mid = np.tile((sd.num_nodes + edges).reshape((-1, 1)), 2)
        cols_J = np.hstack((nodes, nodes[:, ::-1], mid))
        data_IJ = np.hstack((3 * signs, signs, -4 * signs))
        shape = (2 * sd.num_edges, sd.num_nodes + sd.num_edges)
        return sps.csc_array(
            (data_IJ.ravel(), (rows_I.ravel(), cols_J.ravel())), shape=shape
        )

    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Construct the matrix for projecting a quadratic Lagrangian function to a
        piecewise quadratic function. The projection operator :math:`\Pi` takes a
        function from :class:`pg.Lagrange2` and maps it to a piecewise quadratic
        function in :class:`pg.PwQuadratics`.

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
                # In 2D, the edges are the faces. The local edges of PwQuadratics are
                # (0, 1), (0, 2), (1, 2) of the nodes in increasing order, i.e. the
                # faces opposite the nodes in decreasing order
                opposite_nodes = sd.compute_opposite_nodes()
                faces = opposite_nodes.indices.reshape((sd.num_cells, -1))
                nodes = opposite_nodes.data.reshape((sd.num_cells, -1))
                edges = np.take_along_axis(faces, np.argsort(-nodes, axis=1), axis=1)
            case 3:
                # We find the cell-edge connectivity. The ridges are numbered by their
                # nodes in lexicographic order, so that sorting the ridges of a cell
                # gives the local edges (0, 1), (0, 2), ..., (2, 3) of PwQuadratics
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
                # In 3D, the edge coordinate is the midpoint of the two nodes of the
                # ridge
                edge_coords = sd.nodes @ abs(sd.ridge_peaks) / 2

        coords = np.hstack((sd.nodes, edge_coords))

        return self.eval_func_at_coords(func, coords)

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        r"""
        Assembles the 'natural' boundary condition
        :math:`(v, g)_{\partial\Omega}` with :math:`v` a test function in
        :class:`pg.Lagrange2` and :math:`g` the prescribed datum.

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
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

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
