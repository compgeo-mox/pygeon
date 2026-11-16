"""Module for the discretizations of the L2 space."""

from typing import Any, Callable, cast

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class PwMConstants(pg.VDiscretization, pg.PwConstants):
    r"""
    Class implementing the monomial basis for the constants :math:`\mathbb{P}_0(\Omega)
    \subset L^2(\Omega)`, for a polytopal grid :math:`\Omega \subset \mathbb{R}^d`.

    Each degree of freedom is the integral over the cell. The space coincides with the
    FEM piecewise constants except that it relates to the monomial P1 instead of the
    barycentric P1 when projecting to a higher order.
    """

    def proj_to_higher_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Projects from P0 to P1 monomials. The projection operator :math:`\Pi` takes a
        function from :class:`pg.PwMConstants` and maps it to a piecewise linear
        function in :class:`pg.PwMLinears`.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The projection matrix.
        """
        inc = self.eval_at_cell_centers(sd)
        range_discr = self.get_PwPolynomials(self.poly_order + 1, self.tensor_order)()
        return sps.csc_array(inc, shape=(range_discr.ndof(sd), self.ndof(sd)))


class PwMLinears(pg.VDiscretization, pg.PwLinears):
    r"""
    Class implementing the monomial basis for the linears :math:`\mathbb{P}_1(\Omega)
    \subset L^2(\Omega)`, for a polytopal grid :math:`\Omega \subset \mathbb{R}^d`.
    """

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Computes the mass matrix :math:`(\sigma u, v)_\Omega` for
        :math:`u, v \in \mathbb{VP}_1(\Omega)` and :math:`\sigma` a scalar.

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (dict | None): Dictionary with possible scaling.

        Returns:
            sps.csc_array: The mass matrix, of shape (ndof, ndof).
        """
        weight = pg.get_cell_data(sd, data, self.keyword, pg.WEIGHT)
        diams = sd.cell_diameters()

        faces, cells, orient = sps.find(sd.cell_faces)

        assert np.allclose(np.diff(sd.face_nodes.indptr), sd.dim), (
            "All faces must be simplicial, for now."
        )

        # Compute the volume of the simplicial subtesselation. It is given by connecting
        # the cell center to the nodes of the simplicial face.
        sub_volumes = (
            np.einsum(
                "ij,ij->j",
                sd.face_centers[:, faces] - sd.cell_centers[:, cells],
                sd.face_normals[:, faces] * orient,
            )
            / sd.dim
        )

        # We make an array of the values of the monomials in all the nodes of the
        # subsimplices. vals is of shape:
        # (n_nodes_per_subsimplex, n_monomials_per_cell, n_subsimplices)
        vals = np.zeros((sd.dim + 1, self.ndof_per_element(sd.dim), faces.size))

        # The constants are one on every subsimplex node
        vals[:, 0, :] = 1

        # Loop over the remaining d nodes of the subsimplex
        for i in range(sd.dim):
            node_i = sd.face_nodes.indices[sd.face_nodes.indptr[faces] + i]
            vals[i + 1, 1:, :] = (
                sd.rotation_matrix
                @ (sd.nodes[:, node_i] - sd.cell_centers[:, cells])
                / diams[cells]
            )

        # Get the local mass matrix from the piecewise linears
        M_loc = pg.PwLinears().assemble_local_mass(sd.dim)

        # Integrate over each subsimplex
        sub_prods = np.einsum("iac,ij,jbc->abc", vals, M_loc, vals)
        sub_prods *= sub_volumes

        # Sum the resulting matrices for each cell and scale with the weight
        data_IJ = np.zeros((*sub_prods.shape[:-1], sd.num_cells))
        indices = cast(Any, (slice(None), slice(None), cells))  # For mypy
        np.add.at(data_IJ, indices, sub_prods)
        data_IJ *= weight

        # Index lists for the degrees of freedom per element
        rows_I = np.tile(np.arange(self.ndof(sd)), sd.dim + 1).reshape(data_IJ.shape)
        cols_J = np.transpose(rows_I, (1, 0, 2))

        M = sps.csc_array((data_IJ.ravel(), (rows_I.ravel(), cols_J.ravel())))

        return pg.drop_small_entries(M)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix for evaluating the discretization at the cell centers.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The evaluation matrix.
        """
        return sps.eye_array(sd.num_cells, self.ndof(sd)).tocsc()

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the monomials by interpolating the function onto
        the virtual nodal space and projecting.

        Args:
            sd (pg.Grid): The grid.
            func (Callable): A function that returns the function values at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """
        VL1 = pg.VLagrange1()
        interp = VL1.interpolate(sd, func)
        proj = VL1.proj_to_PwPolynomials(sd)

        return proj @ interp

    def proj_to_lower_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Construct the matrix for projecting a piecewise linear function to a piecewise
        constant function. The projection operator :math:`\Pi` takes a function from
        :class:`pg.PwLinears` and maps it to a piecewise polynomial function
        of lower order in :class:`pg.PwMConstants`.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """

        return sps.diags_array(
            sd.cell_volumes, offsets=0, shape=(sd.num_cells, self.ndof(sd))
        ).tocsc()

    def assemble_broken_grad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Assembles the broken (element-wise) gradient matrix for the given grid.
        This operator maps to the vector-valued piecewise constant monomials
        :class:`pg.VecPwMConstants` from :math:`u \in \mathbb{P}_1(\Omega)`
        and is used for computing :math:`\nabla_h u` element-wise.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The assembled broken gradient matrix.
        """
        diagonal = np.tile(sd.cell_volumes / sd.cell_diameters(), sd.dim)
        range_discr = self.get_broken_grad_discr()()
        return sps.diags_array(
            diagonal, offsets=sd.num_cells, shape=(range_discr.ndof(sd), self.ndof(sd))
        ).tocsc()
