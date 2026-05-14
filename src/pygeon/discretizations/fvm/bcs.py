import numpy as np
import porepy as pp

import pygeon as pg


class TPSA_BC:
    def __init__(self, sd: pg.Grid, data: dict, keyword: str):
        self.weighted_dists = np.zeros((pg.AMBIENT_DIM, sd.num_faces))
        self.disp = np.zeros_like(self.weighted_dists)
        self.trac = np.zeros_like(self.weighted_dists)

        data[pp.PARAMETERS][keyword].update({"bcs": self})

    def _set_bcs(
        self,
        indices: np.ndarray | None,
        input: np.ndarray | None,
        internal_var: np.ndarray,
        dist: np.ndarray | float,
    ):
        if input is None:
            input = np.zeros_like(self.weighted_dists)
        if indices is None:
            indices = np.zeros_like(self.weighted_dists, dtype=bool)

        if indices.ndim == 1:
            indices = np.tile(indices, (3, 1))

        assert input.shape == self.weighted_dists.shape, (
            "Input must be of shape (3, num_faces)"
        )

        internal_var[indices] = input[indices]

        if isinstance(dist, np.ScalarType):
            self.weighted_dists[indices] = dist
        else:
            self.weighted_dists[indices] = dist[indices]

    def set_displacement_bcs(
        self,
        indices: np.ndarray | None = None,
        u_0: np.ndarray | None = None,
    ):
        self._set_bcs(indices, u_0, self.disp, 0)

    def set_traction_bcs(
        self,
        indices: np.ndarray | None = None,
        sig_0: np.ndarray | None = None,
    ):
        self._set_bcs(indices, sig_0, self.trac, np.inf)

    def set_spring_bcs(
        self,
        dists: np.ndarray,
        indices: np.ndarray | None = None,
        u_0: np.ndarray | None = None,
    ):
        self._set_bcs(indices, u_0, self.disp, dists)
