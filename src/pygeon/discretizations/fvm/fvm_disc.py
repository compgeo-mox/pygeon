import abc
import warnings
from typing import Type

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class FiniteVolumeDiscretization(abc.ABC):
    def __init__(self, keyword=pg.UNITARY_DATA) -> None:
        self.keyword = keyword
        self.find_cf: dict[pg.Grid, np.ndarray] = {}
        self.unit_normals: dict[pg.Grid, np.ndarray] = {}
        self.weighted_dists: dict[pg.Grid, np.ndarray] = {}
        self.bc_type: Type[pg.FiniteVolumeBC]

    def ndof(self, sd) -> int:
        return self.ndof_per_cell(sd) * sd.num_cells

    def div(self, sd) -> sps.csc_array:
        return sps.kron(np.eye(self.ndof_per_cell(sd)), pg.div(sd), format="csc")

    def face_area_scaling(self, sd) -> np.ndarray:
        return np.tile(sd.face_areas, self.ndof_per_cell(sd))

    def fvm_precomputations(self, sd: pg.Grid, weight: np.ndarray) -> None:
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
        if "bc" in data[pp.PARAMETERS][self.keyword]:
            bcs = data[pp.PARAMETERS][self.keyword]["bc"]
        else:  # We create a default one that places itself in the data dict
            bcs = self.bc_type(sd, data, self.keyword)
        return bcs

    def extend_faces_and_distances(self, sd: pg.Grid, data: dict) -> tuple:
        # Incorporate the bc by extending the face and distance vectors
        bcs = self.extract_bcs(sd, data)

        faces, *_ = self.find_cf[sd]
        bdry_faces = sd.tags["domain_boundary_faces"]
        ext_faces = np.hstack((faces, np.flatnonzero(bdry_faces)))
        ext_dists = np.concatenate(
            (self.weighted_dists[sd], bcs.weighted_dists[bdry_faces])
        )

        return ext_faces, ext_dists

    @abc.abstractmethod
    def ndof_per_cell(self, sd: pg.Grid) -> int:
        """ """

    @abc.abstractmethod
    def compute_weighted_dists(self, sd: pg.Grid, weights: np.ndarray) -> np.ndarray:
        """ """
