import numpy as np
import porepy as pp

import pygeon as pg


class FVM_BC:
    dim_of_bc_vals: int

    def __init__(self, sd: pg.Grid, data: dict, keyword: str) -> None:
        self.weighted_dists = np.zeros((pg.AMBIENT_DIM, sd.num_faces))
        data[pp.PARAMETERS][keyword].update({"bcs": self})

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

        if indices.ndim == 1:
            indices = np.tile(indices, (self.dim_of_bc_vals, 1))

        assert input.shape == self.weighted_dists.shape, (
            f"Input must be of shape ({self.dim_of_bc_vals}, num_faces)"
        )

        internal_var[indices] = input[indices]

        if isinstance(dist, np.ScalarType):
            self.weighted_dists[indices] = dist
        else:
            self.weighted_dists[indices] = dist[indices]


class TPSA_BC(FVM_BC):
    dim_of_bc_vals = pg.AMBIENT_DIM

    def __init__(self, sd: pg.Grid, data: dict, keyword: str) -> None:
        super().__init__(sd, data, keyword)
        self.disp = np.zeros_like(self.weighted_dists)
        self.trac = np.zeros_like(self.weighted_dists)

    def set_displacement_bcs(
        self,
        indices: np.ndarray | None = None,
        u_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, u_0, self.disp, 0)

    def set_traction_bcs(
        self,
        indices: np.ndarray | None = None,
        sig_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, sig_0, self.trac, np.inf)

    def set_spring_bcs(
        self,
        dists: np.ndarray,
        indices: np.ndarray | None = None,
        u_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, u_0, self.disp, dists)


class TPFA_BC(FVM_BC):
    dim_of_bc_vals = 1

    def __init__(self, sd: pg.Grid, data: dict, keyword: str) -> None:
        super().__init__(sd, data, keyword)
        self.pres = np.zeros_like(self.weighted_dists)
        self.flux = np.zeros_like(self.weighted_dists)

    def set_pressure_bcs(
        self,
        indices: np.ndarray | None = None,
        p_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, p_0, self.pres, 0)

    def set_flux_bcs(
        self,
        indices: np.ndarray | None = None,
        q_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, q_0, self.trac, np.inf)

    def set_robin_bcs(
        self,
        dists: np.ndarray,
        indices: np.ndarray | None = None,
        p_0: np.ndarray | None = None,
    ) -> None:
        self._set_bcs(indices, p_0, self.disp, dists)
