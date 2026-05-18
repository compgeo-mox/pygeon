import numpy as np
import scipy.sparse as sps

import pygeon as pg


class TPFA:
    def __init__(self, keyword=pg.UNITARY_DATA) -> None:
        self.keyword = keyword

    def assemble_flow_matrix(self, sd: pg.Grid, data: dict, bcs):
        div_F = pg.div(sd) * sd.face_areas
        codiv = sd.cell_faces
        faces, dists = self.extend_faces_and_distances(sd, bcs)

        perm = pg.get_cell_data(sd, data, self.keyword, pg.SECOND_ORDER_TENSOR)

        self.compute_weighted_dists(sd, perm)
        K = self.compute_harmonic_avg_K(faces, dists)

        return div_F @ (K * codiv)

    def extend_faces_and_distances(self, sd: pg.Grid, bcs: pg.TPFA_BC):
        # Incorporate the bc by extending the face and distance vectors
        faces = sps.find(sd.cell_faces)[0]
        bdry_faces = sd.tags["domain_boundary_faces"]
        ext_faces = np.hstack((faces, np.flatnonzero(bdry_faces)))
        ext_dists = np.hstack((self.weighted_dists, bcs.weighted_dists[:, bdry_faces]))

        return ext_faces, ext_dists

    def compute_harmonic_avg_K(self, faces, dists):
        """
        Compute the harmonic average of K divided by delta_k, at each face
        """
        # Displacement bc are handled naturally as a subset of spring_bdry
        # with zero (inverse) spring constant.
        # Tractions are handled with infinite spring constant.
        self.mu_bar_over_delta = np.array(1 / np.bincount(faces, weights=dists))
