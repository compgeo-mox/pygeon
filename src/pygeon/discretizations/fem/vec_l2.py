"""Module for the discretizations of the vector L2 space."""

from __future__ import annotations

from typing import Callable, Type

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class VecPwPolynomials(pg.VecDiscretization):
    """
    A class representing an abstract vector piecewise polynomial discretization.
    """

    poly_order: int
    """Polynomial degree of the basis functions"""

    tensor_order: int = pg.VECTOR
    """Vector-valued discretization"""

    base_discr: pg.PwPolynomials | pg.VecPwPolynomials

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the mass matrix, using the scalar and tensor weights in data.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict | None): Dictionary with physical parameters for scaling.

        Returns:
            sps.csc_array: The mass matrix.
        """
        return self._assemble_tensor_weighted_inner_product(
            sd, data, "assemble_mass_matrix"
        )

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the lumped mass matrix, using the scalar and tensor weights in data.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict | None): Dictionary with physical parameters for scaling.

        Returns:
            sps.csc_array: The mass matrix.
        """
        return self._assemble_tensor_weighted_inner_product(
            sd, data, "assemble_lumped_matrix"
        )

    def _assemble_tensor_weighted_inner_product(
        self, sd: pg.Grid, data: dict | None, inner_product_method: str
    ) -> sps.csc_array:
        """
        Assemble an inner product weighted by a tensor, given in data.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict | None): Dictionary with physical parameters for scaling.
            inner_product_method (str): Assembly method of parent class for the inner
            product.

        Returns:
            sps.csc_array: The inner product matrix.
        """

        # Retrieve the block-diagonal mass or lumped matrix. This one is weighted with
        # the scalar pg.WEIGHT from the data, if provided.
        M = getattr(super(), inner_product_method)(sd, data)

        # Retrieve the second-order tensor from the data and assemble the weighting
        # matrix.
        sot = pg.get_cell_data(
            sd, data, self.keyword, pg.SECOND_ORDER_TENSOR, pg.MATRIX
        )
        W = self.assemble_weighting_matrix(sd, sot)

        # Since the basis functions are discontinuous, we can assemble the weights using
        # a matrix product.
        return M @ W

    def assemble_weighting_matrix(
        self, sd: pg.Grid, sot: pp.SecondOrderTensor
    ) -> sps.csc_array:
        """
        Assembles the weighting matrix based on a second-order tensor.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            sot (pp.SecondOrderTensor): The physical scaling parameter. Usually the
            inverse of the permeability

        Returns:
            sps.csc_array: The weighting matrix.
        """
        # Retrieve the underlying numpy array of shape (3, 3, n_cells) and rotate it to
        # the reference plane or line. Code taken from pp.SecondOrderTensor.rotate().
        R = sd.rotation_matrix
        rotated_sot = np.tensordot(R.T, np.tensordot(R, sot.values, (1, 0)), (0, 1))

        # Due to our dof numbering convention, we loop through the grid ndof_per_cell
        # times.
        tiled_sot = np.tile(rotated_sot, self.base_discr.ndof_per_cell(sd))

        # Create a block-array of diagonal matrices containing the tensor entries.
        bmat = [
            [sps.diags_array(tiled_sot[i, j, :]) for j in range(sd.dim)]
            for i in range(sd.dim)
        ]

        return sps.block_array(bmat, format="csc")

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a vector-valued function onto the finite element space

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable): A function that returns the function values at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom
        """
        # If the mesh is tilted, then the 3-vector from func needs to be rotated.
        rotated_func = lambda x: sd.rotation_matrix @ func(x)
        return super().interpolate(sd, rotated_func)

    def local_dofs_of_cell(
        self, sd: pg.Grid, c: int, ambient_dim: int = -1
    ) -> np.ndarray:
        """
        Compute the local degrees of freedom (DOFs) of a cell in a vector-valued
        finite element discretization.

        Args:
            sd (pg.Grid): The grid object representing the discretization domain.
            c (int): The index of the cell for which the local DOFs are to be computed.
                ambient_dim (int, optional): The ambient dimension of the space. If not
                provided, it defaults to the dimension of the grid (`sd.dim`).

        Returns:
            np.ndarray: An array containing the local DOFs of the specified cell,
            adjusted for the vector-valued nature of the discretization.
        """
        if ambient_dim == -1:
            ambient_dim = sd.dim

        n_base = self.base_discr.ndof(sd)

        dof_base = self.base_discr.local_dofs_of_cell(sd, c)
        shift = np.repeat(n_base * np.arange(ambient_dim), dof_base.size)

        dof_base = np.tile(dof_base, ambient_dim)

        return dof_base + shift

    def ndof_per_cell(self, sd: pg.Grid) -> int:
        """
        Computes the number of degrees of freedom (DOF) per cell for the given grid.

        This method calculates the total number of DOFs per cell by multiplying
        the number of DOFs per cell from the base discretization by the spatial
        dimension of the grid.

        Args:
            sd (pg.Grid): The grid object representing the spatial discretization.

        Returns:
            int: The total number of degrees of freedom per cell.
        """
        return self.base_discr.ndof_per_cell(sd) * sd.dim

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class for the range of the differential.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The discretization class for the range of the
            differential.
        """
        return self.base_discr.get_range_discr_class(dim)

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

    def proj_to_higher_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Projects the discretization to +1 order discretization.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.base_discr.proj_to_higher_PwPolynomials(sd)
        return self.vectorize(sd.dim, proj)

    def proj_to_lower_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Projects the discretization to -1 order discretization.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.base_discr.proj_to_lower_PwPolynomials(sd)
        return self.vectorize(sd.dim, proj)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix for evaluating the discretization at the cell centers.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
             sps.csc_array: The evaluation matrix.
        """
        Pi = super().eval_at_cell_centers(sd)

        # We need to map back from reference to physical coordinates
        R = sps.kron(sd.rotation_matrix.T, sps.eye_array(Pi.shape[0] // sd.dim), "csc")

        return (R @ Pi).tocsc()

    def assemble_broken_grad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the broken (element-wise) gradient matrix for the given grid.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            sps.csc_array: The assembled broken gradient matrix.
        """
        grad = self.base_discr.assemble_broken_grad_matrix(sd)
        return self.vectorize(sd.dim, grad)


class VecPwConstants(VecPwPolynomials):
    """
    A class representing the discretization using vector piecewise constant functions.
    """

    poly_order = 0
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The base discretization class is pg.PwConstants.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.PwConstants(keyword)


class VecPwLinears(VecPwPolynomials):
    """
    A class representing the discretization using vector piecewise linear functions.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The base discretization class is pg.PwLinears.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.PwLinears(keyword)


class VecPwQuadratics(VecPwPolynomials):
    """
    A class representing the discretization using vector piecewise quadratic functions.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The base discretization class is pg.PwQuadratics.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.PwQuadratics(keyword)
