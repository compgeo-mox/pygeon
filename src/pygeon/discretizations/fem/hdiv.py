"""Module for the discretizations of the H(div) space."""

from functools import cache
from typing import Callable, Tuple, Type

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class RT0(pg.Discretization):
    """
    Discretization class for Raviart-Thomas of lowest order.
    Each degree of freedom is the integral over a mesh face.

    The implementation of this class is inspired by the RT0 class in PorePy.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    tensor_order = pg.VECTOR
    """Vector-valued discretization"""

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of faces.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            int: The number of degrees of freedom.
        """
        return sd.num_faces

    def assemble_adv_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles and returns the advection matrix for mixed finite elements
        (RT0-P0), which is given by
        :math:`(D^{-1}\\boldsymbol{v} \\cdot \\boldsymbol{q}, p)`.

        The trial functions :math:`\\boldsymbol{q}` are lowest-order
        Raviart-Thomas (RT0) and test functions :math:`p` are piecewise
        constant (P0). :math:`\\boldsymbol{v}` is a given vector field and
        :math:`D^{-1}` is a given second-order tensor, both assumed constant
        per cell. If not provided, :math:`\\boldsymbol{v}` defaults to
        :math:`(0, 0, 0)`, and :math:`D^{-1}` defaults to the identity tensor.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict | None): Optional data for scaling, in particular
                pg.SECOND_ORDER_TENSOR (inverse diffusion or permeability
                tensor) and pg.VECTOR_FIELD (advection velocity field).

        Returns:
            sps.csc_array: The advection matrix obtained from the discretization.
        """
        # Retrieve the second order tensor
        D_inv = pg.get_cell_data(
            sd, data, self.keyword, pg.SECOND_ORDER_TENSOR, pg.MATRIX
        )
        # Retrieve the vector field
        V = pg.get_cell_data(sd, data, self.keyword, pg.VECTOR_FIELD, pg.VECTOR)

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        c_centers, f_normals, f_centers, R, dim, nodes = pp.map_geometry.map_grid(sd)
        nodes = nodes[: sd.dim, :]

        if not data or not data.get("is_tangential", False):
            # Rotate the inverse of the permeability tensor and vector field,
            # and delete last dimension
            if sd.dim < 3:
                D_inv = D_inv.copy()
                D_inv.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                D_inv.values = np.delete(D_inv.values, (remove_dim), axis=0)
                D_inv.values = np.delete(D_inv.values, (remove_dim), axis=1)

                V = V.copy()
                V = R @ V
                remove_dim = np.where(np.logical_not(dim))[0]
                V = np.delete(V, remove_dim, axis=0)

        # Allocate the data to store matrix A entries
        size = (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Compute the opposite nodes for each face
        opposite_nodes = sd.compute_opposite_nodes()

        for c in range(sd.num_cells):
            # For the current cell retrieve its faces
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            opposites_loc = opposite_nodes.data[loc]

            # get the opposite node id for each face
            coord_loc = nodes[:, opposites_loc]

            # Compute the flux reconstruction matrix
            psi_cell = pp.RT0.faces_to_cell(
                c_centers[:, c],
                coord_loc,
                f_centers[:, faces_loc],
                f_normals[:, faces_loc],
                dim,
                R,
            )[: sd.dim, :]

            # Compute the local weight for the local advection matrix
            weight = D_inv.values[..., c] @ V[:, c]

            # Compute the H_div-advection local matrix
            A = weight @ psi_cell

            # Save values for local matrix in the global structure
            cols = faces_loc
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = c
            cols_J[loc_idx] = faces_loc
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the lumped mass matrix L such that B^T L^{-1} B is a TPFA method.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict | None): Optional dictionary with physical parameters for
                scaling. In particular the pg.SECOND_ORDER_TENSOR that is the inverse of
                the diffusion tensor (permeability for porous media).

        Returns:
            sps.csc_array: The lumped mass matrix.
        """
        inv_K = pg.get_cell_data(
            sd, data, self.keyword, pg.SECOND_ORDER_TENSOR, pg.MATRIX
        )

        h_perp = np.zeros(sd.num_faces)
        for face, cell in zip(*sd.cell_faces.nonzero()):
            dist = sd.face_centers[:, face] - sd.cell_centers[:, cell]
            h_perp_loc = dist.T @ inv_K.values[:, :, cell] @ dist
            norm_dist = np.linalg.norm(dist)
            h_perp[face] += h_perp_loc / norm_dist if norm_dist else 0

        return sps.diags_array(h_perp / sd.face_areas).tocsc()

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator, the divergence
        in this case.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        return sps.csc_array(sd.cell_faces.T)

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable[[np.ndarray], np.ndarray]): A function that returns the
                function values at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """
        vals = [
            np.inner(func(x).flatten(), normal)
            for (x, normal) in zip(sd.face_centers.T, sd.face_normals.T)
        ]
        return np.array(vals)

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (n dot q, func)_Gamma

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function that defines
                the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        vals = np.zeros(self.ndof(sd))

        for dof in b_faces:
            vals[dof] = (
                func(sd.face_centers[:, dof]) * sd.cell_faces.tocsr()[dof, :].sum()
            )

        return vals

    def get_range_discr_class(self, _dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.PwConstants

    @cache
    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Constructs the projection matrix to the VecPwLinears space. This function is
        cached to speed up repetitive calls for the same grid.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: A sparse array in CSC format representing the projection from
            the current space to VecPwLinears.
        """
        bdm1 = pg.BDM1(self.keyword)
        proj_to_bdm1 = bdm1.proj_from_RT0(sd)
        proj_to_poly = bdm1.proj_to_PwPolynomials(sd)
        return proj_to_poly @ proj_to_bdm1


class BDM1(pg.Discretization):
    """
    BDM1 is a class that represents the BDM1 (Brezzi-Douglas-Marini) finite element
    method. It provides methods for assembling matrices, projecting to and from the RT0
    space, evaluating the solution at cell centers, interpolating a given function onto
    the grid, assembling the natural boundary condition term, and more.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    tensor_order = pg.VECTOR
    """Vector-valued discretization"""

    def ndof(self, sd: pg.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case the number of faces times the dimension.

        Args:
            sd (pp.Grid): Grid object or a subclass.

        Returns:
            int: The number of degrees of freedom.

        Raises:
            ValueError: If the input grid is not an instance of pp.Grid.
        """
        return sd.face_nodes.nnz

    @staticmethod
    def local_inner_product(dim: int) -> np.ndarray:
        """
        Compute the local inner product matrix for the given dimension.

        Args:
            dim (int): The dimension of the matrix.

        Returns:
            np.ndarray: The computed local inner product matrix.
        """
        M_loc = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        M_loc /= (dim + 1) * (dim + 2)

        return np.kron(M_loc, np.eye(3))

    def proj_to_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the function space to the lowest order Raviart-Thomas (RT0) space.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.

        Returns:
            sps.csc_array: The projection matrix to the RT0 space.
        """
        proj = sps.hstack([sps.eye_array(sd.num_faces)] * sd.dim) / sd.dim
        return proj.tocsc()

    def proj_from_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the RT0 finite element space onto the faces of the given grid.

        Args:
            sd (pg.Grid): The grid on which the projection is performed.

        Returns:
            sps.csc_array: The projection matrix.
        """
        return sps.vstack([sps.eye_array(sd.num_faces)] * sd.dim).tocsc()

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        rt0 = pg.RT0(self.keyword)
        RT0_diff = rt0.assemble_diff_matrix(sd)

        proj_to_rt0 = self.proj_to_RT0(sd)
        return RT0_diff @ proj_to_rt0

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a given function onto the grid.

        Args:
            sd (pg.Grid): The grid on which to interpolate the function.
            func (Callable[[np.ndarray], np.ndarray]): The function to be interpolated.

        Returns:
            np.ndarray: The interpolated values on the grid.
        """
        vals = np.zeros(self.ndof(sd))

        for face in np.arange(sd.num_faces):
            func_loc = np.array(
                [func(sd.nodes[:, node]) for node in sd.face_nodes[:, [face]].indices]
            ).T
            vals_loc = sd.face_normals[:, face] @ func_loc
            vals[face + np.arange(sd.dim) * sd.num_faces] = vals_loc

        return vals

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (n dot q, func)_Gamma

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function that defines
                the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        p1 = pg.PwLinears(self.keyword)
        local_mass = p1.assemble_local_mass(sd.dim - 1)

        vals = np.zeros(self.ndof(sd))
        signs = sd.cell_faces @ np.ones(sd.num_cells)
        fn = sd.face_nodes

        for face in b_faces:
            loc_vals = np.array(
                [
                    func(sd.nodes[:, node])
                    for node in fn.indices[fn.indptr[face] : fn.indptr[face + 1]]
                ]
            ).ravel()

            vals[face + np.arange(sd.dim) * sd.num_faces] = (
                signs[face] * local_mass @ loc_vals
            )

        return vals

    def get_range_discr_class(self, _dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.PwConstants

    @cache
    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Constructs the projection matrix from the current finite element space to the
        VecPwLinears space.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: A sparse array in CSC format representing the projection from
            the current space to VecPwLinears.
        """
        # Each contribution to the matrix corresponds to a (cell, face, node) triplet.
        # To avoid for-loops, we generate arrays with the relevant cell/face/node
        # indices.

        # We first extract the connected cell-face pairs.
        faces, cells, orien = sps.find(sd.cell_faces)

        # The first index array contains the node indices, ordered as:
        # [nodes of face_1, nodes of face_2, ...]
        fn = np.reshape(sd.face_nodes.indices, (sd.num_faces, -1))
        nodes = fn[faces].ravel()

        # The corresponding BDM1 dof indices form the column indices.
        dofs_at_face = np.reshape(np.arange(self.ndof(sd)), (sd.num_faces, -1), "F")
        cols_J = dofs_at_face[faces].ravel()
        cols_J = np.tile(cols_J, sd.dim)

        # Each cell-face pair appears once per node of the face, which means dim times.
        faces = np.repeat(faces, sd.dim)
        cells = np.repeat(cells, sd.dim)
        orien = np.repeat(orien, sd.dim)

        # To compute the values of the basis functions, we need to know the opposite
        # node index.
        opposite_nodes = sd.compute_opposite_nodes()
        oppos = opposite_nodes[faces, cells]

        # If the mesh is tilted, then the coordinates of the nodes need to be mapped
        coords = sd.rotation_matrix @ sd.nodes

        # We avoid inner products by using the identity:
        # tangent @ normal = dim * cell_volume * orientation
        tangents = coords[:, nodes] - coords[:, oppos]
        vals = tangents / (orien * sd.cell_volumes[cells] * sd.dim)
        data_IJ = vals.ravel()

        # Finally, we find the corresponding dof in p1 by generating a lookup matrix
        # that satisfies p1_lookup[node, cell] = dof_index at (node, cell)
        p1_lookup = pg.PwLinears.dof_lookup(sd)

        # The vector-valued analogue has sd.dim rows
        p1_dofs = p1_lookup[nodes, cells] + p1_lookup.nnz * np.arange(sd.dim)[:, None]
        rows_I = p1_dofs.ravel()

        return sps.csc_array((data_IJ, (rows_I, cols_J)))


class RT1(pg.Discretization):
    """
    RT1 Discretization class for H(div) finite element method.

    This class implements the Raviart-Thomas elements of order 1 (RT1) for
    discretizing vector fields in H(div) space. It provides methods for
    assembling mass matrices, differential matrices, evaluating basis functions,
    and interpolating functions onto the finite element space.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

    tensor_order = pg.VECTOR
    """Vector-valued discretization"""

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            int: The number of degrees of freedom.
        """
        return sd.dim * (sd.num_faces + sd.num_cells)

    def local_dofs_of_cell(self, sd: pg.Grid, faces_loc: np.ndarray, c: int):
        """
        Compute the local degrees of freedom (DOFs) indices for a cell.

        Args:
            sd (pp.Grid): Grid object or a subclass.
            faces_loc (np.ndarray):  Array of local face indices for the cell.
            c (int): Cell index.

        Returns:
            np.ndarray: Array of local DOF indices associated with the cell.
        """
        loc_face = np.hstack([faces_loc] * sd.dim)
        loc_face += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces
        loc_cell = sd.dim * sd.num_faces + sd.num_cells * np.arange(sd.dim) + c

        return np.hstack((loc_face, loc_cell))

    def reorder_faces(
        self, cell_faces: sps.csc_array, opposite_nodes: sps.csc_array, cell: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reorders the local nodes, faces, and corresponding cell-face orientations

        Args:
            cell_faces (sps.csc_array): Cell_face connectivity of the grid.
            opposite_nodes (sps.csc_array): Opposite nodes for each face.
            cell (int): Cell index.

        Returns:
            np.ndarray: The reordered local node indices
            np.ndarray: The reordered local face indices
            np.ndarray: The reordered cell-face orientation signs
        """
        # For the current cell retrieve its faces
        loc = slice(cell_faces.indptr[cell], cell_faces.indptr[cell + 1])
        faces_loc = cell_faces.indices[loc]
        signs_loc = cell_faces.data[loc]
        opposites_loc = opposite_nodes.data[loc]

        # Sort the nodes in ascending order
        nodes_loc = np.sort(opposites_loc)

        # Reorder the faces so that
        # - face_0 is (0, 1) and opposite node 2
        # - face_1 is (0, 2) and opposite node 1
        # - face_2 is (1, 2) and opposite node 0
        # I.e. the faces are reordered so that
        # the opposite node indices are descending
        sorter = np.argsort(opposites_loc)[::-1]
        faces_loc = faces_loc[sorter]
        signs_loc = signs_loc[sorter]

        return nodes_loc, faces_loc, signs_loc

    def eval_basis_functions(
        self, sd: pg.Grid, nodes_loc: np.ndarray, signs_loc: np.ndarray, volume: float
    ) -> np.ndarray:
        """
        Evaluates the basis functions at the nodes and edges of a cell.

        Args:
            sd (pg.Grid): The grid.
            nodes_loc (np.ndarray): Nodes of the cell.
            signs_loc (np.ndarray): Cell-face orientation signs.
            volume (float): Cell volume.

        Returns:
            np.ndarray: An array Psi in which [i, 3j : 3(j + 1)] contains the values of
            basis function phi_i at evaluation point j
        """
        dim = sd.dim

        # We assign each basis function to a node opposite a face (opp_node)
        # and the node at which the dof is located (loc_node)
        opp_nodes = np.tile(np.arange(dim + 1), dim)[::-1]
        loc_nodes = np.repeat(np.arange(dim + 1), dim)
        signs = np.tile(signs_loc, dim)

        # Compute the tangent in physical space by taking local indices
        tangent = lambda i, j: sd.nodes[:, nodes_loc[j]] - sd.nodes[:, nodes_loc[i]]

        # Helper functions psi_k as outlined in the notes in docs/RT1.md
        edge_nodes = pg.Lagrange2().get_local_edge_nodes(dim)
        n_edges = edge_nodes.shape[0]

        node_edges = np.array([np.nonzero(edge_nodes == n)[0] for n in range(dim + 1)])

        psi_nodes = np.zeros((dim + 1, 3 * (dim + 1)))
        psi_edges = np.zeros((dim + 1, 3 * n_edges))

        for edge, (i, j) in enumerate(edge_nodes):
            psi_edges[i, 3 * edge : 3 * (edge + 1)] = tangent(i, j) / 4
            psi_edges[j, 3 * edge : 3 * (edge + 1)] = tangent(j, i) / 4

        psi_k = np.hstack((psi_nodes, psi_edges))

        # Preallocation
        Psi = np.zeros((dim * (dim + 2), 3 * (dim + 1 + n_edges)))

        # Evaluate the basis functions of the face-dofs
        for dof, (i, j) in enumerate(zip(loc_nodes, opp_nodes)):
            # Face-dofs are one at their respective nodes
            Psi[dof, 3 * i : 3 * (i + 1)] = tangent(j, i)
            # Face-dofs are a half at the adjacent edges
            for edge in node_edges[i]:
                Psi[dof, 3 * (dim + 1 + edge) : 3 * (dim + 1 + edge + 1)] = (
                    0.5 * tangent(j, i)
                )
            # See docs/RT1.md
            Psi[dof] -= psi_k[j] - psi_k[i]
            Psi[dof] *= signs[dof]

        # Evaluate the basis functions of the cell-dofs
        Psi[-dim:] = psi_k[:dim]

        return Psi / (dim * volume)

    def eval_basis_functions_at_center(
        self, sd: pg.Grid, nodes_loc: np.ndarray, volume: float
    ) -> np.ndarray:
        """
        Evaluates the basis functions at the center of a cell.

        Args:
            sd (pg.Grid): The grid.
            nodes_loc (np.ndarray): Nodes of the cell.
            volume (float): Cell volume.

        Returns:
            np.ndarray: A (3 x dim) array with the values of the cell-based basis
            functions at the cell center.
        """
        # Preallocation
        basis = np.empty((3, sd.dim))

        # As outlined in docs/RT1, the only nonzero basis function
        # at the center are the cell-based ones, given by
        # phi_k = sum_i lambda_k lambda_i tau_ki
        for ind in np.arange(sd.dim):
            basis[:, ind] = np.sum(
                sd.nodes[:, nodes_loc] - sd.nodes[:, nodes_loc[ind]][:, None],
                axis=1,
            ) / ((sd.dim + 1) ** 2)

        return basis / (sd.dim * volume)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix for evaluating the discretization at the cell centers.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
             sps.csc_array: The evaluation matrix.
        """
        # If a 0-d grid is given then we return an empty matrix
        if sd.dim == 0:
            return sps.csc_array((3, 0))

        # Allocate the data to store matrix P entries
        size = 3 * sd.dim * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Compute the opposite nodes for each face
        opposite_nodes = sd.compute_opposite_nodes()

        for c in range(sd.num_cells):
            # For the current cell retrieve its faces
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            nodes_loc = np.sort(opposite_nodes.data[loc])

            P = self.eval_basis_functions_at_center(sd, nodes_loc, sd.cell_volumes[c])

            cell_dofs = self.local_dofs_of_cell(sd, np.zeros(sd.dim + 1), c)[-sd.dim :]

            # Save values for projection P local matrix in the global structure
            loc_idx = slice(idx, idx + P.size)
            rows_I[loc_idx] = np.repeat(c + np.arange(3) * sd.num_cells, sd.dim)
            cols_J[loc_idx] = np.tile(cell_dofs, 3)
            data_IJ[loc_idx] = P.ravel()
            idx += P.size

        # Construct the global matrix
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator, the divergence
        in this case.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        # Allocate the data to store matrix A entries
        size = (sd.dim * (sd.dim + 2)) * (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Precompute the local divergence matrix
        loc_div = self.compute_local_div_matrix(sd.dim)
        opposite_nodes = sd.compute_opposite_nodes()

        range_disc = pg.PwLinears()

        for c in range(sd.num_cells):
            _, faces_loc, signs_loc = self.reorder_faces(
                sd.cell_faces, opposite_nodes, c
            )

            # Change the sign of the face-dofs according to the cell-face orientation
            signs = np.ones(loc_div.shape[1])
            signs[: -sd.dim] = np.tile(signs_loc, sd.dim)

            div = loc_div * signs / (sd.dim * sd.cell_volumes[c])

            # Indices of the local degrees of freedom
            loc_dofs = self.local_dofs_of_cell(sd, faces_loc, c)
            div_dofs = np.tile(loc_dofs, sd.dim + 1)

            # Indices of the range degrees of freedom
            ran_dofs = range_disc.local_dofs_of_cell(sd, c)
            ran_dofs = np.repeat(ran_dofs, div.shape[1])

            # Save values of the local matrix in the global structure
            loc_idx = slice(idx, idx + div.size)
            rows_I[loc_idx] = ran_dofs
            cols_J[loc_idx] = div_dofs
            data_IJ[loc_idx] = div.ravel()
            idx += div.size

        # Construct the global matrices
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def compute_local_div_matrix(self, dim: int) -> np.ndarray:
        """
        Assembles the local divergence matrix using local node and face ordering

        Args:
            dim (int): Dimension of the grid.

        Returns:
            np.ndarray: The local divergence matrix
        """
        opp_node = np.tile(np.arange(dim + 1), dim)[::-1]
        loc_node = np.repeat(np.arange(dim + 1), dim)

        # The face basis function phi_i^j has divergence
        # 1 + (dim + 1) (lambda_i - lambda_j)
        face_div = np.ones((dim + 1, dim * (dim + 1)))
        face_div[loc_node, np.arange(loc_node.size)] += dim + 1
        face_div[opp_node, np.arange(loc_node.size)] -= dim + 1

        # The cell basis function phi_k has divergence
        # (dim + 1) lambda_k - 1
        cell_div = (dim + 1) * np.eye(dim + 1, dim)
        cell_div -= 1

        return np.hstack((face_div, cell_div))

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable[[np.ndarray], np.ndarray]): A function that returns the
                function values at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """
        # The face dofs are determined as in BDM1
        interp_faces = pg.BDM1().interpolate(sd, func)

        # The cell dofs are determined by solving (d x d) linear systems
        interp_cells = np.zeros(sd.dim * sd.num_cells)
        cell_nodes = sd.cell_nodes()

        for c in range(sd.num_cells):
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            basis_at_center = self.eval_basis_functions_at_center(
                sd, nodes_loc, sd.cell_volumes[c]
            )
            func_at_center = func(sd.cell_centers[:, c])

            # Compute the coefficients c_i such
            # that sum_i c_i phi_i = f at the cell center
            coefficients = np.linalg.solve(
                basis_at_center[: sd.dim, :], func_at_center[: sd.dim]
            )

            interp_cells[sd.num_cells * np.arange(sd.dim) + c] = coefficients

        return np.hstack((interp_faces, interp_cells))

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (n dot q, func)_Gamma

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function that defines
                the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        vals = np.zeros(self.ndof(sd))
        bdm1 = pg.BDM1()
        vals[: bdm1.ndof(sd)] = bdm1.assemble_nat_bc(sd, func, b_faces)

        return vals

    def get_range_discr_class(self, _dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.PwLinears

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the lumped matrix for the given grid,
        using the integration rule from Egger & Radu (2020)

        Args:
            sd (pg.Grid): The grid object.
            data (dict | None): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped matrix.
        """
        # If a 0-d grid is given then we return an empty matrix
        if sd.dim == 0:
            return sps.csc_array((0, 0))

        bdm1 = pg.BDM1(self.keyword)
        bdm1_lumped = bdm1.assemble_lumped_matrix(sd, data) / (sd.dim + 2)

        inv_K = pg.get_cell_data(
            sd, data, self.keyword, pg.SECOND_ORDER_TENSOR, pg.MATRIX
        )

        # Allocate the data to store matrix P entries
        size = sd.dim * sd.dim * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Compute the opposite nodes for each face
        opposite_nodes = sd.compute_opposite_nodes()

        for c in range(sd.num_cells):
            # For the current cell retrieve its faces
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            nodes_loc = np.sort(opposite_nodes.data[loc])

            P = self.eval_basis_functions_at_center(sd, nodes_loc, sd.cell_volumes[c])
            weight = inv_K.values[:, :, c]

            A = P.T @ weight @ P * sd.cell_volumes[c] * (sd.dim + 1) / (sd.dim + 2)

            loc_cell_dofs = sd.num_cells * np.arange(sd.dim) + c

            # Save values for projection P local matrix in the global structure
            cols = np.tile(loc_cell_dofs, (loc_cell_dofs.size, 1))
            loc_idx = slice(idx, idx + A.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += A.size

        # Construct the global matrix
        cell_dof_lumped = sps.csc_array((data_IJ, (rows_I, cols_J)))

        return sps.csc_array(sps.block_diag((bdm1_lumped, cell_dof_lumped)))

    def proj_to_PwPolynomials(self, sd: pg.Grid):
        """
        Constructs the projection matrix from the current finite element space to the
        VecPwQuadratics space.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: A sparse array in CSC format representing the projection from
            the current space to VecPwQuadratics.
        """
        # overestimate the size of a local computation
        loc_size = (
            sd.dim * (3 * sd.dim + 2) * ((sd.dim * (sd.dim + 1)) // 2) + sd.dim**2
        )
        size = loc_size * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        opposite_nodes = sd.compute_opposite_nodes()

        range_disc = pg.VecPwQuadratics()

        n_dof_per_cell = [0, 9, 18, 30][sd.dim]
        rearrange = np.reshape(np.arange(n_dof_per_cell), (3, -1)).ravel(order="F")

        for c in range(sd.num_cells):
            nodes_loc, faces_loc, signs_loc = self.reorder_faces(
                sd.cell_faces, opposite_nodes, c
            )

            Psi = self.eval_basis_functions(
                sd, nodes_loc, signs_loc, sd.cell_volumes[c]
            )

            Psi_i, Psi_j = np.nonzero(Psi)
            Psi_v = Psi[Psi_i, Psi_j]

            # Extract indices of local dofs
            loc_dofs = self.local_dofs_of_cell(sd, faces_loc, c)

            # Extract local dofs of VecPwQuadratics
            ran_dofs = range_disc.local_dofs_of_cell(sd, c, 3)
            ran_dofs = ran_dofs[rearrange]

            # Save values of the local matrix in the global structure
            loc_idx = slice(idx, idx + Psi_v.size)
            rows_I[loc_idx] = ran_dofs[Psi_j]
            cols_J[loc_idx] = loc_dofs[Psi_i]
            data_IJ[loc_idx] = Psi_v
            idx += Psi_v.size

        # Construct the global matrices
        return sps.csc_array((data_IJ[:idx], (rows_I[:idx], cols_J[:idx])))
