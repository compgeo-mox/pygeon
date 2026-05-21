import warnings
from typing import cast

import numpy as np
import porepy as pp

import pygeon as pg


class FiniteVolumeBC:
    dim_of_bc_vals: int

    def __init__(self, _: pg.Grid, data: dict, keyword: str) -> None:
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
        if input is None:
            input = np.zeros_like(self.weighted_dists)
        if indices is None:
            indices = np.zeros_like(self.weighted_dists, dtype=bool)

        if indices.ndim == 1 and self.dim_of_bc_vals > 1:
            indices = np.tile(indices, (self.dim_of_bc_vals, 1))

        assert input.shape == self.weighted_dists.shape, (
            f"Input must be of shape {self.weighted_dists.shape}"
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
    dim_of_bc_vals = pg.AMBIENT_DIM

    def __init__(self, sd: pg.Grid, data: dict, keyword: str = pg.UNITARY_DATA) -> None:
        self.weighted_dists = np.zeros((self.dim_of_bc_vals, sd.num_faces))
        super().__init__(sd, data, keyword)

    def set_displacement_bcs(
        self,
        indices: np.ndarray | None = None,
        u_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, u_0, self.primary_var, 0)

    def set_traction_bcs(
        self,
        indices: np.ndarray | None = None,
        sig_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, sig_0, self.dual_var, np.inf)

    def set_spring_bcs(
        self,
        dists: np.ndarray,
        indices: np.ndarray | None = None,
        u_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, u_0, self.primary_var, dists)


class FlowBC(FiniteVolumeBC):
    dim_of_bc_vals = 1

    def __init__(self, sd: pg.Grid, data: dict, keyword: str = pg.UNITARY_DATA) -> None:
        self.weighted_dists = np.zeros(sd.num_faces)
        super().__init__(sd, data, keyword)

    def set_pressure_bcs(
        self,
        indices: np.ndarray | None = None,
        p_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, p_0, self.primary_var, 0)

    def set_flux_bcs(
        self,
        indices: np.ndarray | None = None,
        q_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, q_0, self.dual_var, np.inf)

    def set_robin_bcs(
        self,
        dists: np.ndarray,
        indices: np.ndarray | None = None,
        p_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, p_0, self.primary_var, dists)
