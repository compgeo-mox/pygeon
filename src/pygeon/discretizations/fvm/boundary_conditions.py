"""Module for finite-volume boundary condition classes."""

import warnings
from typing import cast

import numpy as np
import porepy as pp

import pygeon as pg


class FiniteVolumeBC:
    """
    Parent class of boundary condition objects for finite volume methods in PyGeoN.
    """

    dim_of_bc_vals: int
    """Dimension of the boundary values, typically 1 or the dimension of the domain"""

    def __init__(self, _sd: pg.Grid, data: dict, keyword: str) -> None:
        """
        Initialize the FiniteVolumeBC object and store it in the data dictionary.

        Args:
            _sd (pg.Grid): The grid.
            data (dict): The data dictionary.
            keyword (str): The keyword of the relevant finite volume discretization.
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
        Set boundary condition values on selected faces.

        Args:
            indices (np.ndarray): The global face indices on which the BC is imposed.
            input (np.ndarray): Array of boundary values.
            internal_var (np.ndarray): Internal BC storage to update.
            dist (np.ndarray): dist[indices] contains the weighted distances on the
                exterior of the domain. Zero for Dirichlet, infinite for Neumann.
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
        Initialize the ElasticityBC object and store it in the data dictionary.

        Args:
            sd (pg.Grid): The grid.
            data (dict): The data dictionary.
            keyword (str): The keyword of the relevant finite volume discretization.
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
            indices (np.ndarray): The global face indices on which the BC is imposed.
                Defaults to None.
            u_0 (np.ndarray): shape [dim, n_faces] such that u_0[indices] contains
                the BC values. Defaults to zero.
        """
        self._set_bcs(indices, u_0, self.primary_var, 0)

    def set_traction_bcs(
        self,
        indices: np.ndarray | None = None,
        sig_0: np.ndarray | None = None,
    ) -> None:
        """
        Set traction boundary conditions.

        Args:
            indices (np.ndarray): The global face indices on which the BC is imposed.
                Defaults to None.
            sig_0 (np.ndarray): shape [dim, n_faces] such that sig_0[indices]
                contains the BC values. Defaults to zero.
        """
        self._set_bcs(indices, sig_0, self.dual_var, np.inf)

    def set_spring_bcs(
        self,
        dists: np.ndarray,
        indices: np.ndarray | None = None,
        u_0: np.ndarray | None = None,
    ) -> None:
        """
        Set spring boundary conditions, cf. Appendix A2.21 in Nordbotten and
        Keilegavlen (2025), https://doi.org/10.1016/j.camwa.2025.07.035:
        $n \cdot \sigma = 2 / \mathrm{dists} \,(u_0 - u)$.

        Args:
            dists (np.ndarray): Weighted distance (inverse spring constant).
            indices (np.ndarray): The global face indices on which the BC is imposed.
                Defaults to None.
            u_0 (np.ndarray): shape [dim, n_faces] such that u_0[indices] contains
                the BC values. Defaults to zero.
        """
        self._set_bcs(indices, u_0, self.primary_var, dists)


class FlowBC(FiniteVolumeBC):
    def __init__(self, sd: pg.Grid, data: dict, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the FlowBC object and store it in the data dictionary.

        Args:
            sd (pg.Grid): The grid.
            data (dict): The data dictionary.
            keyword (str): The keyword of the relevant finite volume discretization.
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
        Set pressure boundary conditions.

        Args:
            indices (np.ndarray): The global face indices on which the BC is imposed.
                Defaults to None.
            p_0 (np.ndarray): shape [n_faces, ] such that p_0[indices] contains
                the BC values. Defaults to zero.
        """
        self._set_bcs(indices, p_0, self.primary_var, 0)

    def set_flux_bcs(
        self,
        indices: np.ndarray | None = None,
        q_0: np.ndarray | None = None,
    ) -> None:
        """
        Set flux boundary conditions.

        Args:
            indices (np.ndarray): The global face indices on which the BC is imposed.
                Defaults to None.
            q_0 (np.ndarray): shape [n_faces, ] such that q_0[indices] contains
                the BC values. Defaults to zero.
        """
        self._set_bcs(indices, q_0, self.dual_var, np.inf)

    def set_robin_bcs(
        self,
        dists: np.ndarray,
        indices: np.ndarray | None = None,
        p_0: np.ndarray | None = None,
    ) -> None:
        """
        Set Robin boundary conditions:
        $n \cdot q = (p - p_0) / \mathrm{dists}$.

        Args:
            dists (np.ndarray): Weighted distance (inverse permeability).
            indices (np.ndarray): The global face indices on which the BC is imposed.
                Defaults to None.
            p_0 (np.ndarray): shape [n_faces, ] such that p_0[indices] contains
                the BC values. Defaults to zero.
        """
        self._set_bcs(indices, p_0, self.primary_var, dists)
