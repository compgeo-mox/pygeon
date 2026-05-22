import warnings
from typing import cast

import numpy as np
import porepy as pp

import pygeon as pg


class FiniteVolumeBC:
    """
    Parent class of boundary condition objects for finite volume methods in PyGeoN
    """

    dim_of_bc_vals: int
    """Dimension of the boundary values, typically 1 or the dimension of the domain"""

    def __init__(self, _: pg.Grid, data: dict, keyword: str) -> None:
        """
        Initializes the FiniteVolumeBC object and places itself in the data dictionary

        Args:
            sd (pg.Grid): The grid.
            data (dict): The data dictionary.
            keyword (str): The keyword of the relevant finite volume discretization

        Returns:
            None
        """
        data[pp.PARAMETERS][keyword].update({"bc": self})

        self.weighted_dists: np.ndarray
        self.primary_var = np.zeros_like(self.weighted_dists)
        self.dual_var = np.zeros_like(self.weighted_dists)

    def _set_bcs(
        self,
        indices: np.ndarray | None,
        input: np.ndarray | None,
        internal_var: np.ndarray,
        dist: np.ndarray | float,
    ) -> None:
        """
        Abstract function to set boundary condition values.

        Args:
            indices (np.ndarray): The global face indices on which the bc is imposed
            input (np.ndarray): input[indices] contains the bc values
            internal_var (np.ndarray): The internal variable of the BC object
            data (dict): The data dictionary.
            dist (np.ndarray): dist[indices] contains the weighted distances on the
                exterior of the domain. Zero for Dirichlet, infinite for Neumann.

        Returns:
            None
        """
        if input is None:
            input = np.zeros_like(self.weighted_dists)
        if indices is None:
            indices = np.zeros_like(self.weighted_dists, dtype=bool)

        if indices.ndim == 1 and self.dim_of_bc_vals > 1:
            indices = np.tile(indices, (self.dim_of_bc_vals, 1))

        assert input.shape == self.weighted_dists.shape, (
            f"Boundary values must be of shape {self.weighted_dists.shape}, "
            f"not {input.shape}."
        )

        internal_var[indices] = input[indices]

        if isinstance(dist, np.ScalarType):
            self.weighted_dists[indices] = dist
        else:
            self.weighted_dists[indices] = cast(np.ndarray, dist)[indices]

        if np.any(np.logical_and(self.primary_var, self.dual_var)):
            warnings.warn(
                "Boundary conditions imposed on both the primary and dual variables may"
                "lead to unexpected results"
            )


class ElasticityBC(FiniteVolumeBC):
    def __init__(self, sd: pg.Grid, data: dict, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initializes the ElasticityBC object and places itself in the data dictionary

        Args:
            sd (pg.Grid): The grid.
            data (dict): The data dictionary.
            keyword (str): The keyword of the relevant finite volume discretization

        Returns:
            None
        """
        self.dim_of_bc_vals = sd.dim
        self.weighted_dists = np.zeros((self.dim_of_bc_vals, sd.num_faces))
        super().__init__(sd, data, keyword)

    def set_displacement_bcs(
        self,
        indices: np.ndarray | None = None,
        u_0: np.ndarray | None = None,
    ) -> None:
        """
        Sets displacement boundary conditions.

        Args:
            indices (np.ndarray): The global face indices on which the bc is imposed.
                Defaults to none.
            u_0 (np.ndarray): shape [dim, n_faces] such that u_0[indices] contains
                the bc values. Defaults to zero.

        Returns:
            None
        """
        self._set_bcs(indices, u_0, self.primary_var, 0)

    def set_traction_bcs(
        self,
        indices: np.ndarray | None = None,
        sig_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, sig_0, self.dual_var, np.inf)
        """
        Sets traction boundary conditions.

        Args:
            indices (np.ndarray): The global face indices on which the bc is imposed.
                Defaults to none.
            sig_0 (np.ndarray): shape [dim, n_faces] such that sig_0[indices] contains 
                the bc values. Defaults to zero.

        Returns:
            None
        """

    def set_spring_bcs(
        self,
        dists: np.ndarray,
        indices: np.ndarray | None = None,
        u_0: np.ndarray | None = None,
    ) -> None:
        """
        Sets spring boundary conditions, cf. (A2.21)
        n dot sigma = 2 / dists (u_0 - u)

        Args:
            dists(np.ndarray): The weighted distance (an inverse spring constant)
            indices (np.ndarray): The global face indices on which the bc is imposed.
                Defaults to none.
            sig_0 (np.ndarray): shape [dim, n_faces] such that sig_0[indices] contains
                the bc values. Defaults to zero.

        Returns:
            None
        """
        self._set_bcs(indices, u_0, self.primary_var, dists)


class FlowBC(FiniteVolumeBC):
    def __init__(self, sd: pg.Grid, data: dict, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initializes the FlowBC object and places itself in the data dictionary

        Args:
            sd (pg.Grid): The grid.
            data (dict): The data dictionary.
            keyword (str): The keyword of the relevant finite volume discretization

        Returns:
            None
        """
        self.dim_of_bc_vals = 1
        self.weighted_dists = np.zeros(sd.num_faces)
        super().__init__(sd, data, keyword)

    def set_pressure_bcs(
        self,
        indices: np.ndarray | None = None,
        p_0: np.ndarray | None = None,
    ) -> None:
        """
        Sets pressure boundary conditions

        Args:
            indices (np.ndarray): The global face indices on which the bc is imposed.
                Defaults to none.
            p_0 (np.ndarray): shape [n_faces, ] such that p_0[indices] contains
                the bc values. Defaults to zero.

        Returns:
            None
        """
        self._set_bcs(indices, p_0, self.primary_var, 0)

    def set_flux_bcs(
        self,
        indices: np.ndarray | None = None,
        q_0: np.ndarray | None = None,
    ) -> None:
        """
        Sets flux boundary conditions

        Args:
            indices (np.ndarray): The global face indices on which the bc is imposed.
                Defaults to none.
            q_0 (np.ndarray): shape [n_faces, ] such that q_0[indices] contains
                the bc values. Defaults to zero.

        Returns:
            None
        """
        self._set_bcs(indices, q_0, self.dual_var, np.inf)

    def set_robin_bcs(
        self,
        dists: np.ndarray,
        indices: np.ndarray | None = None,
        p_0: np.ndarray | None = None,
    ) -> None:
        """
        Sets Robin boundary conditions
        n dot flux = (p - p_0) / dists

        Args:
            dists(np.ndarray): The weighted distance (an inverse permeability)
            indices (np.ndarray): The global face indices on which the bc is imposed.
                Defaults to none.
            sig_0 (np.ndarray): shape [dim, n_faces] such that sig_0[indices] contains
                the bc values. Defaults to zero.

        Returns:
            None
        """
        self._set_bcs(indices, p_0, self.primary_var, dists)
