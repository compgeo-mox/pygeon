"""Module for the discretizations of the H(curl) space."""

from typing import Callable, Type

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class Nedelec0(pg.Discretization):
    """
    Discretization class for the Nedelec of the first kind of lowest order. Each degree
    of freedom is the integral over a mesh edge in 3D.

    While intended for three-dimensional grids, the space is generalized to 2D, where it
    corresponds to a rotated RT0.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    tensor_order = pg.VECTOR
    """Vector-valued discretization"""

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case, it returns the number of ridges in the given grid.

        Args:
            sd (pg.Grid): The grid for which the number of degrees of
                freedom is calculated.

        Returns:
            int: The number of degrees of freedom.
        """
        return sd.num_edges

    def assemble_mass_matrix(
        self, sd: pg.Grid, _data: dict | None = None
    ) -> sps.csc_array:
        """
        Constructs the projection matrix to the VecPwLinears space via Nedelec1.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: A sparse array in CSC format representing the projection from
            the current space to VecPwLinears.
        """
        proj_to_Ne1 = self.proj_to_Ne1(sd)
        proj_to_pwp = Nedelec1(self.keyword).proj_to_PwPolynomials(sd)

        return proj_to_pwp @ proj_to_Ne1

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the lumped mass matrix given by the row sums on the diagonal.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict | None): Dictionary with physical parameters for scaling.

        Returns:
            sps.csc_array: The lumped mass matrix.
        """
        diag_mass = self.assemble_mass_matrix(sd, data).sum(axis=0)
        return sps.diags_array(np.asarray(diag_mass).flatten()).tocsc()

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the differential matrix for the given grid.

        Args:
            sd (pg.Grid): The grid for which the differential matrix is assembled.

        Returns:
            sps.csc_array: The assembled differential matrix.
        """
        match sd.dim:
            case 3:
                diff = sd.face_ridges.T
            case 2:
                diff = sd.cell_faces.T
            case _:
                diff = sps.csc_array((0, self.ndof(sd)))

        return diff.tocsc()

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition matrix for the given grid and function.

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            func (Callable): The function defining the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition matrix.
        """
        raise NotImplementedError

    def get_range_discr_class(self, _dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class for the given grid.
        """
        match dim:
            case 2:
                return pg.PwConstants
            case 3:
                return pg.RT0
            case _:
                raise NotImplementedError

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a given function onto the grid using the hcurl discretization.

        Args:
            sd (pg.Grid): The grid on which to interpolate the function.
            func (Callable): The function to be interpolated.

        Returns:
            np.ndarray: The interpolated values on the grid.
        """
        tangents = sd.edge_tangents
        midpoints = sd.nodes @ abs(sd.ridge_peaks) / 2
        vals = [
            np.inner(func(x).flatten(), t) for (x, t) in zip(midpoints.T, tangents.T)
        ]
        return np.array(vals)

    def proj_to_Ne1(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the solution to the Nedelec of the second kind.

        Args:
            sd (pg.Grid): The grid object representing the discretization.

        Returns:
            sps.csc_array: The projection matrix to the Nedelec of the second kind.
        """
        return (
            sps.vstack([sps.eye_array(sd.num_edges), -sps.eye_array(sd.num_edges)])
        ).tocsc()


class Nedelec1(pg.Discretization):
    """
    Discretization class for the Nedelec of the second kind of lowest order.
    Each degree of freedom is a first moment over a mesh edge in 3D.

    While intended for three-dimensional grids, the space is generalized to 2D, where it
    corresponds to a rotated BDM1.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    tensor_order = pg.VECTOR
    """Vector-valued discretization"""

    def ndof(self, sd: pg.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case, it returns twice the number of ridges in the given grid.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            int: The number of degrees of freedom.
        """
        return 2 * sd.num_edges

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
        # Each contribution to the matrix corresponds to a (cell, edge, node) triplet.
        # To avoid for-loops, we generate arrays with the relevant cell/edge/node
        # indices.

        # We first extract the connected cell-edge and edge-node pairs.
        match sd.dim:
            case 1:
                cell_edges = sps.eye_array(sd.num_cells, format="csc")
                edge_nodes = sd.cell_nodes()
            case 2:
                cell_edges = sd.cell_faces
                edge_nodes = sd.face_ridges
            case 3:
                cell_edges = sd.face_ridges.astype("bool") @ sd.cell_faces.astype(
                    "bool"
                )
                edge_nodes = sd.ridge_peaks

        edges, cells, _ = sps.find(cell_edges)

        # Each edge has two nodes. We keep track of the node itself and its partner.
        en = np.reshape(edge_nodes.indices, (sd.num_edges, -1))
        nodes = en[edges].ravel()
        partner_nodes = en[edges, ::-1].ravel()

        # The column indices are given by the Nedelec dof indices.
        dofs_at_edge = np.reshape(np.arange(self.ndof(sd)), (sd.num_edges, -1), "F")
        cols_J = dofs_at_edge[edges].ravel()
        cols_J = np.tile(cols_J, sd.dim)

        # Each cell-edge pair appears once per node, so twice in total.
        edges = np.repeat(edges, 2)
        cells = np.repeat(cells, 2)

        # We need to find the unique face that is next to the node but does not border
        # the edge. We do that by taking the face opposite the partner node.
        opposite_nodes = sd.compute_opposite_nodes().tocoo()
        opp_f = opposite_nodes.row
        opp_c = opposite_nodes.col
        opp_n = opposite_nodes.data

        opposite_faces = sps.csc_array((opp_f, (opp_n, opp_c)))

        faces = opposite_faces[partner_nodes, cells]
        orien = sd.cell_faces[faces, cells]

        # Rotate the normals if the mesh is tilted
        normals = sd.rotation_matrix @ sd.face_normals

        # We avoid inner products by using the identity:
        # tangent @ normal = dim * cell_volume * orientation
        vals = -normals[:, faces] / (orien * sd.cell_volumes[cells] * sd.dim)
        data_IJ = vals.ravel()

        # Finally, we find the corresponding dof in p1 by generating a lookþup matrix
        # that satisfies p1_lookup[node, cell] = dof_index at (node, cell)
        p1_lookup = sd.cell_nodes().astype("int")
        p1_ndof = p1_lookup.nnz
        p1_lookup.data = np.reshape(np.arange(p1_ndof), (sd.num_cells, -1), "F").ravel()

        # The vector-valued analogue has sd.dim rows
        p1_dofs = p1_lookup[nodes, cells] + p1_ndof * np.arange(sd.dim)[:, None]
        rows_I = p1_dofs.ravel()

        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def proj_to_Ne0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the solution to the Nedelec of the first kind.

        Args:
            sd (pg.Grid): The grid object representing the discretization.

        Returns:
            sps.csc_array: The projection matrix to the Nedelec of the first kind.
        """
        return (
            sps.hstack([sps.eye_array(sd.num_edges), -sps.eye_array(sd.num_edges)]) / 2
        ).tocsc()

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the differential matrix for the H(curl) finite element space.

        Args:
            sd (pg.Grid): The grid on which the finite element space is defined.

        Returns:
            sps.csc_array: The assembled differential matrix.
        """
        n0 = pg.Nedelec0(self.keyword)
        Ne0_diff = n0.assemble_diff_matrix(sd)

        proj_to_ne0 = self.proj_to_Ne0(sd)
        return Ne0_diff @ proj_to_ne0

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates the given function `func` over the specified grid `sd`.

        Args:
            sd (pg.Grid): The grid over which to interpolate the function.
            func (Callable): The function to be interpolated.

        Returns:
            np.ndarray: The interpolated values.
        """
        vals = np.zeros(self.ndof(sd))
        for r in np.arange(sd.num_edges):
            loc = slice(sd.ridge_peaks.indptr[r], sd.ridge_peaks.indptr[r + 1])
            peaks = sd.ridge_peaks.indices[loc]
            t = sd.edge_tangents[:, r]
            vals[r] = np.inner(func(sd.nodes[:, peaks[0]]).flatten(), t)
            vals[r + sd.num_edges] = np.inner(func(sd.nodes[:, peaks[1]]).flatten(), -t)

        return vals

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition for the given grid, function, and
            boundary faces.

        Args:
            sd (pg.Grid): The grid on which to assemble the natural boundary condition.
            func (Callable): The function defining the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition.
        """
        raise NotImplementedError

    def get_range_discr_class(self, _dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return Nedelec0().get_range_discr_class(dim)
