import abc
import warnings
from typing import Tuple, Type

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class FiniteVolumeDiscretization(abc.ABC):
    """
    Abstract class for PyGeoN finite volume discretization methods.
    """

    def __init__(self, keyword=pg.UNITARY_DATA) -> None:
        """
        Initialize the FiniteVolumeDiscretization object.

        Args:
            keyword (str): The keyword used to identify the discretization method.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        self.keyword = keyword
        self.bc_type: Type[pg.FiniteVolumeBC]

    def ndof(self, sd) -> int:
        """
        Returns the number of degrees of freedom on a given grid.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom.
        """
        return self.ndof_per_cell(sd) * sd.num_cells

    def assemble_system_matrix(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assemble the system matrix, using the material parameters in the data dictionary.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary.

        Returns:
            sps.csc_array: The discretization matrix.
        """
        # Assemble the zero'th order accumulation terms
        M = self.assemble_accumulation_terms(sd, data)

        # Assemble the second order terms, given by div(dual variables)
        A = self.div(sd) @ self.assemble_dual_var_map(sd, data)

        # Assemble the matrix
        return (M + A).tocsc()

    def div(self, sd) -> sps.csc_array:
        """
        Assembles the block-diagonal divergence operator acting on all the face-based
        dual variables.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The divergence operator
        """
        return sps.kron(np.eye(self.ndof_per_cell(sd)), pg.div(sd), format="csc")

    def face_area_scaling(self, sd) -> np.ndarray:
        """
        Assembles the scaling vector with the face areas, for the face-based dual
        variables.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            np.ndarray: The scaling vector
        """
        return np.tile(sd.face_areas, self.ndof_per_cell(sd))

    def check_nonnegative_weights(self, weight: np.ndarray) -> None:
        """
        Saves the cell-face connectivity of the grid, the unit normal vectors, and
        weighted distances as attributes of the Discretization object. This avoids
        unnecessary recomputations.

        Args:
            sd (pg.Grid): The grid object.
            weight (np.ndarray): The physical parameter weights.
        """

        # Check if any cell centers are placed outside the cell
        if np.any(weight < 0):
            warnings.warn(
                f"There are {(weight < 0).sum()} extra-cellular \
                    cell centers."
            )

    def compute_harmonic_avg(self, faces: np.ndarray, dists: np.ndarray) -> np.ndarray:
        """
        Compute the quantity, at each face

        Args:
            sd (pg.Grid): Grid, or a subclass.
            faces (np.ndarray): The extended array of faces
            dists (np.ndarray): The extended array of weighted distances
        """
        return np.array(1 / np.bincount(faces, weights=dists))

    def get_bcs_from_data(self, sd: pg.Grid, data: dict) -> pg.FiniteVolumeBC:
        """
        Extracts the FiniteVolumeBC object from the data dictionary, if it exists.
        Else, it creates a new one and inserts it in data.

        Args:
            sd (pg.Grid): The grid object.
            data (dict): The data dictionary

        Returns:
            pg.FiniteVolumeBC: The boundary condition object
        """
        if "bc" in data[pp.PARAMETERS][self.keyword]:
            bcs = data[pp.PARAMETERS][self.keyword]["bc"]
        else:
            bcs = self.bc_type(sd, data, self.keyword)
        return bcs

    def assemble_rhs_boundary_vector(self, sd: pg.Grid, data: dict) -> np.ndarray:
        """
        Assembles the right-hand side vector related to the boundary conditions.

        Args:
            sd (pg.Grid): The grid object.
            data (dict): The data dictionary

        Returns:
            np.ndarray: The right-hand side vector
        """
        A_rhs = self.assemble_bdry_dual_var_map(sd, data)

        bcs = self.get_bcs_from_data(sd, data)
        dual = bcs.dual_var / sd.face_areas
        prim = bcs.primary_var

        g = np.hstack((dual.ravel(), prim.ravel()))

        return -self.div(sd) @ A_rhs @ g

    @abc.abstractmethod
    def ndof_per_cell(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom per cell.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom per cell.
        """

    @abc.abstractmethod
    def assemble_dual_var_map(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """ """

    @abc.abstractmethod
    def assemble_accumulation_terms(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """ """

    @abc.abstractmethod
    def assemble_bdry_dual_var_map(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assembles the matrix that maps from the boundary condition values to the dual
        variables on the boundary faces.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: the matrix to be multiplied with the boundary data g
        """
