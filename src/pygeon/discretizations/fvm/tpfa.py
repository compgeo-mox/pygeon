from typing import Callable, cast

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class TPFA(pg.FiniteVolumeDiscretization):
    def __init__(self, keyword=pg.UNITARY_DATA) -> None:
        super().__init__(keyword)
        self.bc_type = pg.FlowBC

        self.K_bar: dict[pg.Grid, np.ndarray] = {}

    def ndof_per_cell(self, _) -> int:
        return 1

    def interpolate(self, sd: pg.Grid, pressure: Callable) -> np.ndarray:
        interp = pg.PwConstants().interpolate(sd, pressure)
        return interp / sd.cell_volumes

    def assemble_flow_matrix(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        perm = pg.get_cell_data(sd, data, self.keyword, pg.SECOND_ORDER_TENSOR, 2)

        # Precomputations without boundary conditions
        self.fvm_precomputations(sd, perm.values)

        faces, deltas = self.extend_faces_and_distances(sd, data)

        self.compute_harmonic_avg(sd, faces, deltas)

        A = self.assemble_dual_var_map(sd)

        return self.div(sd) @ A

    def compute_weighted_dists(self, sd: pg.Grid, perm: np.ndarray) -> np.ndarray:
        """
        Compute delta_k^i / K_nn for every physical face-cell pair. Boundary conditions
        are handled later.
        """
        faces, cells, orient = self.find_cf[sd]
        normals = self.unit_normals[sd][:, faces]

        K_nn = np.einsum("ijk,ik,jk->k", perm[:, :, cells], normals, normals)

        delta = np.sum(
            (
                (sd.face_centers[:, faces] - sd.cell_centers[:, cells])
                * (orient * normals)
            ),
            axis=0,
        )

        return delta / K_nn

    def compute_harmonic_avg(self, sd, faces, dists) -> None:
        """
        Compute the harmonic average of K divided by delta_k, at each face
        """
        self.K_bar[sd] = np.array(1 / np.bincount(faces, weights=dists))

    def assemble_dual_var_map(self, sd: pg.Grid) -> sps.sparray:
        return (self.face_area_scaling(sd) * self.K_bar[sd])[:, None] * sd.cell_faces

    def assemble_rhs_bdry_terms(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        rhs = np.empty(2, dtype=sps.sparray)

        K_bar = self.K_bar[sd]
        rhs[0] = sps.diags_array((K_bar == 0).astype(float))

        Delta_B = -sd.cell_faces.sum(axis=1)
        rhs[1] = sps.diags_array(K_bar * Delta_B)

        bcs = self.extract_bcs(sd, data)
        bcs = cast(pg.FlowBC, bcs)

        pres = bcs.primary_var
        flux = bcs.dual_var / sd.face_areas

        g = np.concatenate((flux, pres))

        return -(self.div(sd) * sd.face_areas) @ sps.hstack(rhs) @ g

    def assemble_source(self, sd: pg.Grid, source: Callable) -> np.ndarray:
        return pg.PwConstants().interpolate(sd, source)
