"""Module for the virtual element discretizations of the H1 space."""

from typing import Type

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class VLagrange1(pg.VDiscretization, pg.Lagrange1):
    """
    Discretization class for the virtual Lagrange1 method, on polytopal grids with
    simplicial faces, following the Hitchhiker's guide to the virtual element method.
    """

    def proj_to_PwPolynomials(self, sd: pg.Grid):
        r"""
        Computes the projection onto the monomials. Returns the coefficients
        :math:`\{a_i\}` in the expansion

        .. math::

            a_0 + \frac{1}{d}[a_1, a_2] \cdot (x - c)

        for each VL1 basis function, with :math:`d` the diameter and :math:`c` the
        center of the cell.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            np.ndarray: The coefficients of the local projection onto the monomials.
        """

        G = self.assemble_proj_lhs(sd)
        B = self.assemble_proj_rhs(sd)

        return pg.block_diag_solver(G, B)

    def assemble_proj_lhs(self, sd: pg.Grid):
        """
        Returns the system G from the Hitchhiker's (3.9), which forms the left-hand side
        for the L2 projection problem.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            np.ndarray: The system matrix G.
        """

        # The matrix coincides with the broken stiffness matrix of the piecewise
        # monomials, except for the first rows, which we adjust afterward.
        pwp = self.get_PwPolynomials(self.poly_order, self.tensor_order)(self.keyword)
        G_tilde = pwp.assemble_broken_grad_grad_matrix(sd)

        # We need to introduce the projection onto the constants in the first rows.
        # Following the Hitchhiker's guide, we take the average over the nodal values.
        cell_nodes = sd.cell_nodes()
        coords = sd.nodes[:, cell_nodes.indices]
        avg_coords = np.add.reduceat(coords, cell_nodes.indptr[:-1], axis=1)
        avg_coords /= np.diff(cell_nodes.indptr)

        # Evaluate the average value of the monomials {(x - c)/d, (y - c)/d, (z - c)/d}
        # and insert the constants at the top.
        vals = sd.rotation_matrix @ (avg_coords - sd.cell_centers) / sd.cell_diameters()
        vals = np.vstack([np.ones(vals.shape[1]), vals])

        # Compute the averaging operator and pad it with zeros so it can be added to
        # G_tilde
        P_0 = sps.hstack([sps.diags_array(row) for row in vals])
        P_0 = sps.csc_array(P_0, shape=G_tilde.shape)

        return G_tilde + P_0

    def assemble_proj_rhs(self, sd: pg.Grid):
        """
        Returns the matrix B from the Hitchhiker's (3.14), which forms the right-hand
        side of the L2 projection problem.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            np.ndarray: The righthand side B.
        """

        # First the projection onto the constants
        cell_nodes = sd.cell_nodes()
        P0_phi = (cell_nodes / np.diff(cell_nodes.indptr)).T

        # Now the integrals against the monomials. We use the fact that
        # \nabla m_i = ê_i / diam
        # with ê_i the i-th basis vector of R^n. Using Stokes, we need to integrate over
        # the faces of the cell.The integral over a face is then given by dividing over
        # the number of nodes on each face.
        num_nodes_per_face = np.diff(sd.face_nodes.indptr)
        node_face_avg = (sd.face_nodes / num_nodes_per_face).T.tocsc()

        # Each connected node-face-cell pair gives a contribution to the matrix. To sum
        # up the relevant face contributions, we start with a matrix with n_faces rows
        # and one column for each node-cell pair. The entries are given by
        # cf_orient / num_nodes_per_face / diameter.
        # Using an entry-wise multiplication, we ensure that only the connected triplets
        # are counted.
        nodes, cells, _ = sps.find(cell_nodes)
        nc_pair_faces = (
            sd.cell_faces[:, cells]
            * node_face_avg[:, nodes]
            / sd.cell_diameters()[cells]
        )

        # The summation over the relevant faces is then done with a matrix product.
        vals = sd.rotation_matrix @ sd.face_normals @ nc_pair_faces

        # Each row i in vals corresponds to the monomial in terms of x_i.
        P1_phi = sps.vstack([sps.csc_array((row, (cells, nodes))) for row in vals])

        # Combine with the constant projection.
        return sps.vstack((P0_phi, P1_phi))

    def assemble_dofs_of_monomials(self, sd: pg.Grid):
        """
        Returns the matrix :math:`D` whose rows are the degrees of freedom applied to
        the local monomial basis functions, from the Hitchhiker's (3.17).

        Args:
            sd (pg.Grid): The grid.

        Returns:
            np.ndarray: The matrix D.
        """

        # We need to evaluate the monomials at the nodes. However, some cells share a
        # node and we must avoid that their contributions get summed. We therefore
        # generate a matrix that has a row for each connected cell-node pair.
        cell_nodes = sd.cell_nodes()
        nodes, cells, _ = sps.find(cell_nodes)

        # Evaluate the monomials and insert the constants.
        vals = (
            sd.rotation_matrix
            @ (sd.nodes[:, nodes] - sd.cell_centers[:, cells])
            / sd.cell_diameters()[cells]
        )
        vals = np.vstack((np.ones(vals.shape[1]), vals))

        # Assemble
        D = sps.hstack(
            [sps.csc_array((row, (np.arange(cells.size), cells))) for row in vals]
        )

        return D

    def assemble_stabilization(self, sd: pg.Grid, cell_weights: np.ndarray):
        """
        Returns the stabilization term :math:`S` that measures the discrepancy between
        the basis function and its projection onto the monomials. This is computed by
        evaluating the original degree of freedoms for the monomial projections.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            np.ndarray: The stabilization matrix S.
        """

        D = self.assemble_dofs_of_monomials(sd)
        proj = self.proj_to_PwPolynomials(sd)

        Pi = D @ proj

        cell_nodes = sd.cell_nodes()
        nodes, cells, _ = sps.find(cell_nodes)

        # The identity tensor maps the value from the node to the relevant cell-node
        # pairs. These can be compared with the output from Pi.
        Id = sps.csc_array((np.ones_like(nodes), (np.arange(nodes.size), nodes)))

        return (Id - Pi).T @ sps.diags_array(cell_weights[cells]) @ (Id - Pi)

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles and returns the VEM stiffness matrix
        :math:`(\nabla u, \nabla v)_\Omega` using the virtual element method.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Optional data for the assembly process.

        Returns:
            sps.csc_array: The stiffness matrix obtained from the discretization.
        """
        # The stabilization is scaled with the mean eigenvalue of the tensor on the
        # tangent space of the grid, times the weight
        tensor = pg.get_cell_data(
            sd, data, self.keyword, pg.SECOND_ORDER_TENSOR, pg.MATRIX
        )
        weight = pg.get_cell_data(sd, data, self.keyword, pg.WEIGHT)
        R = sd.rotation_matrix
        trace = np.einsum("ij,jkc,ik->c", R, tensor.values, R) / sd.dim

        A = self.assemble_broken_grad_grad_matrix(sd, data)
        A += self.assemble_stabilization(
            sd, cell_weights=weight * trace * sd.cell_diameters() ** (sd.dim - 2)
        )

        return A

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles and returns the mass matrix.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Optional data for the assembly process.

        Returns:
            sps.csc_array: The sparse mass matrix obtained from the discretization.
        """

        weight = pg.get_cell_data(sd, data, self.keyword, pg.WEIGHT)

        M = super().assemble_mass_matrix(sd, data)
        M += self.assemble_stabilization(sd, cell_weights=weight * sd.cell_volumes)

        return M

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the lumped mass matrix, with the measure of the region of each cell
        associated with each node, see :meth:`pg.Grid.compute_subvolumes`, scaled with
        the weight.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Optional data for the assembly process.

        Returns:
            sps.csc_array: The diagonal lumped mass matrix.
        """
        weight = pg.get_cell_data(sd, data, self.keyword, pg.WEIGHT)
        subvolumes = sd.compute_subvolumes()

        return sps.diags_array(subvolumes @ weight).tocsc()

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            Type[pg.Discretization]: The range discretization class.

        Raises:
            NotImplementedError: For dimensions other than 1 and 2.
        """
        match dim:
            case 3:
                raise NotImplementedError(
                    "There's no Virtual NedelecR0 discretization in PyGeoN"
                )
            case 2:
                return pg.VRT0
            case 1:
                return pg.PwMConstants
            case _:
                raise NotImplementedError("There's no zero discretization in PyGeoN")

    # ---------------- old functionalities used in other VEM classes ----------------
    def assemble_loc_mass_matrix(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Computes the local VEM mass matrix :math:`(u, v)_K` on a given cell
        according to the Hitchhiker's (6.5).

        NOTE: Deprecated since #350, will be removed in the near future.

        Args:
            sd (pg.Grid): The grid.
            cell (int): The index of the cell on which to compute the mass matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of nodes associated with the cell.

        Returns:
            np.ndarray: The computed local VEM mass matrix.
        """
        proj = self.assemble_loc_proj_to_mon(sd, cell, diam, nodes)
        H = self.assemble_loc_monomial_mass(sd, cell, diam)

        D = self.assemble_loc_dofs_of_monomials(sd, cell, diam, nodes)
        I_minus_Pi = np.eye(nodes.size) - D @ proj

        return proj.T @ H @ proj + sd.cell_volumes[cell] * I_minus_Pi.T @ I_minus_Pi

    def assemble_loc_proj_to_mon(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        r"""
        Computes the local projection onto the monomials. Returns the coefficients
        :math:`\{a_i\}` in the expansion

        .. math::

            a_0 + \frac{1}{d}[a_1, a_2] \cdot (x - c)

        for each VL1 basis function, with :math:`d` the diameter and :math:`c` the
        center of the cell.

        NOTE: Deprecated since #350, will be removed in the near future.

        Args:
            sd (pg.Grid): The grid.
            cell (int): The index of the cell in which the projection is computed.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices of the cell.

        Returns:
            np.ndarray: The coefficients of the local projection onto the monomials.
        """
        G = self.assemble_loc_L2proj_lhs(sd, cell, diam, nodes)
        B = self.assemble_loc_L2proj_rhs(sd, cell, diam, nodes)

        return np.linalg.solve(G, B)

    def assemble_loc_L2proj_lhs(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Returns the system G from the Hitchhiker's (3.9)

        NOTE: Deprecated since #350, will be removed in the near future.

        Args:
            sd (pg.Grid): The grid.
            cell (int): The index of the cell.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.

        Returns:
            np.ndarray: The system matrix G.
        """
        G = sd.cell_volumes[cell] / (diam**2) * np.eye(3)
        G[0, 0] = 1
        G[0, 1:] = (
            sd.nodes[: sd.dim, nodes].mean(1) - sd.cell_centers[: sd.dim, cell]
        ) / diam

        return G

    def assemble_loc_L2proj_rhs(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Returns the righthand side B from the Hitchhiker's (3.14)

        NOTE: Deprecated since #350, will be removed in the near future.

        Args:
            sd (pg.Grid): The grid.
            cell (int): The cell index.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.

        Returns:
            np.ndarray: The righthand side B.
        """
        normals = (
            sd.face_normals[: sd.dim] * sd.cell_faces[:, cell].toarray().ravel()
        ) @ sd.face_nodes[nodes, :].T

        B = np.empty((3, nodes.size))
        B[0, :] = 1.0 / nodes.size
        B[1:, :] = normals / diam / 2

        return B

    def assemble_loc_monomial_mass(
        self, sd: pg.Grid, cell: int, diam: float
    ) -> np.ndarray:
        r"""
        Computes the inner products of the monomials

        .. math::

            \left\{1, \frac{x - c}{d}, \frac{y - c}{d}\right\}

        Reference: Hitchhiker's (5.3)

        NOTE: Deprecated since #350, will be removed in the near future.

        Args:
            sd (pg.Grid): The grid.
            cell (int): The index of the cell.
            diam (float): The diameter of the cell.

        Returns:
            np.ndarray: The computed inner products matrix.
        """
        H = np.zeros((3, 3))
        H[0, 0] = sd.cell_volumes[cell]

        M = np.ones((2, 2)) + np.eye(2)

        cell_col = np.array([cell])
        for face in sd.cell_faces[:, cell_col].indices:
            sub_volume = (
                np.dot(
                    sd.face_centers[:, face] - sd.cell_centers[:, cell],
                    sd.face_normals[:, face] * sd.cell_faces[face, cell],
                )
                / 2
            )

            vals = (
                sd.nodes[:2, sd.face_nodes[:, [face]].indices]
                - sd.cell_centers[:2, [cell] * 2]
            ) / diam

            H[1:, 1:] += sub_volume * vals @ M @ vals.T / 12

        return H

    def assemble_loc_dofs_of_monomials(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Returns the matrix :math:`D` whose rows are the degrees of freedom applied to
        the local monomial basis functions, from the Hitchhiker's (3.17).

        NOTE: Deprecated since #350, will be removed in the near future.

        Args:
            sd (pg.Grid): The grid.
            cell (int): The index of the cell.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.

        Returns:
            np.ndarray: The matrix D.
        """
        D = np.empty((nodes.size, 3))
        D[:, 0] = 1.0
        D[:, 1:] = (
            sd.nodes[: sd.dim, nodes] - sd.cell_centers[: sd.dim, [cell] * nodes.size]
        ).T / diam

        return D
