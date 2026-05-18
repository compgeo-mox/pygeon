import warnings

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class FiniteVolumeDiscretization:
    bc_type = pg.FiniteVolumeBC

    def __init__(self, keyword=pg.UNITARY_DATA) -> None:
        self.keyword = keyword
        self.find_cf = {}
        self.unit_normals = {}
        self.weighted_dists = {}

    def ndof(self, sd) -> int:
        return self.ndof_per_cell(sd) * sd.num_cells

    def div_F(self, sd) -> sps.csc_array:
        div_F = pg.div(sd) * sd.face_areas
        return sps.kron(np.eye(self.ndof_per_cell(sd)), div_F)

    def fvm_precomputations(self, sd, weight):
        self.find_cf[sd] = sps.find(sd.cell_faces)
        self.unit_normals[sd] = sd.face_normals / sd.face_areas
        self.weighted_dists[sd] = self.compute_weighted_dists(sd, weight)

        # Check if any cell centers are placed outside the cell
        if np.any(self.weighted_dists[sd] <= 0):
            warnings.warn(
                f"There are {(self.weighted_dists[sd] <= 0).sum()} extra-cellular \
                    cell centers."
            )

    def extract_bcs(self, sd: pg.Grid, data: dict) -> pg.FiniteVolumeBC:
        # See if there is already a BoundaryConditions object in the data dict
        if "bcs" in data[pp.PARAMETERS][self.keyword]:
            bcs = data[pp.PARAMETERS][self.keyword]["bcs"]
        else:  # We create a default one that places itself in the data dict
            bcs = self.bc_type(sd, data, self.keyword)
        return bcs

    def extend_faces_and_distances(self, sd: pg.Grid, bcs: pg.FiniteVolumeBC):
        # Incorporate the bc by extending the face and distance vectors
        faces, *_ = self.find_cf[sd]
        bdry_faces = sd.tags["domain_boundary_faces"]
        ext_faces = np.hstack((faces, np.flatnonzero(bdry_faces)))
        ext_dists = np.concatenate(
            (self.weighted_dists[sd], bcs.weighted_dists[bdry_faces])
        )

        return ext_faces, ext_dists
