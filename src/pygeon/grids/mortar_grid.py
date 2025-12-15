"""This module contains the MortarGrid class."""

from typing import Tuple

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg
from pygeon.utils.set_membership import match_coordinates

"""
Acknowledgments:
    The functionalities related to the ridge computations are modified from
    github.com/anabudisa/md_aux_precond developed by Ana Budiša and Wietse M. Boon.
"""


class MortarGrid(pp.MortarGrid):
    """
    A class representing a mortar grid, which is used for the discretization of
    interfaces between subdomains in a numerical simulation.
    """

    def assign_sd_pair(self, sd_pair: Tuple[pg.Grid, pg.Grid]) -> None:
        self.sd_pair = sd_pair

    def compute_geometry(self) -> None:
        """
        Computes the geometry of the MortarGrid.

        Args:
            sd_pair: The pair of subdomains.

        Returns:
            None
        """
        super().compute_geometry()

        self.assign_signed_mortar_to_primary()
        self.assign_cell_faces()

        if self.dim >= 1:
            self.compute_ridges()

    def compute_ridges(self) -> None:
        """
        Assign the face-ridge and ridge-peak connectivities to the mortar grid

        Args:
            sd_pair (Tuple[pp.Grid, pp.Grid]): Pair of adjacent subdomains.

        Returns:
            None
        """
        sd_up, sd_down = self.sd_pair

        # High-dim ridges matching to low-dim face
        face_ridges = sps.lil_array((sd_up.num_ridges, sd_down.num_faces), dtype=int)
        # High-dim peaks matching to low-dim ridge
        ridge_peaks = sps.lil_array((sd_up.num_peaks, sd_down.num_ridges), dtype=int)

        # Find information about the two-dimensional grid
        if self.dim == 1:
            R = pp.map_geometry.project_plane_matrix(sd_up.nodes)
            rot = np.dot(
                R.T,
                np.dot(
                    np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), R
                ),
            )
        else:  # self.dim == 2
            R = pp.map_geometry.project_plane_matrix(sd_down.nodes)
            normal_to_sd_down = np.dot(R.T, [0, 0, 1])

        for face_up, cell_down in zip(*sps.find(self.cell_faces)[:-1]):
            # Faces of cell in lower-dim grid
            cf_down = sd_down.cell_faces
            faces_down = cf_down.indices[
                cf_down.indptr[cell_down] : cf_down.indptr[cell_down + 1]
            ]

            # Ridges of face in higher-dim grid
            fr_up = sd_up.face_ridges
            ridges_up = fr_up.indices[fr_up.indptr[face_up] : fr_up.indptr[face_up + 1]]

            # Swap ridges around so they match with lower-dim faces
            if self.dim == 1:
                face_xyz = sd_down.face_centers[:, faces_down]
                ridge_xyz = sd_up.nodes[:, ridges_up]
            else:  # self.dim == 2
                face_xyz = sd_down.nodes @ abs(sd_down.face_ridges[:, faces_down]) / 2
                ridge_xyz = sd_up.nodes @ abs(sd_up.ridge_peaks[:, ridges_up]) / 2

            ridges_up = ridges_up[match_coordinates(face_xyz, ridge_xyz)]

            # Ridge-peak connectivity in 3D
            if self.dim == 2:
                # Ridges of cell in lower-dim grid
                cr_down = sd_down.cell_nodes()
                ridges_down = cr_down.indices[
                    cr_down.indptr[cell_down] : cr_down.indptr[cell_down + 1]
                ]
                ridge_xyz = sd_down.nodes[:, ridges_down]

                # Nodes of face in higher-dim grid
                fn_up = sd_up.face_nodes
                peaks_up = fn_up.indices[
                    fn_up.indptr[face_up] : fn_up.indptr[face_up + 1]
                ]
                peak_xyz = sd_up.nodes[:, peaks_up]

                # Swap peaks around so they match with lower-dim ridges
                peaks_up = peaks_up[match_coordinates(ridge_xyz, peak_xyz)]

            # Take care of orientations
            # NOTE:this computation is done here so that we have access to the normal
            # vector

            # Find the normal vector oriented outward wrt the higher-dim grid
            is_outward = sd_up.cell_faces.tocsr()[face_up, :].data[0]
            normal_up = sd_up.face_normals[:, face_up] * is_outward

            # Find the normal to the lower-dim face
            normal_down = sd_down.face_normals[:, faces_down]

            # Identify orientation
            if self.dim == 1:
                # we say that orientations align if the rotated mortar
                # normal corresponds to the normal of the
                # lower-dimensional face
                orientations_fr = np.dot(np.dot(rot, normal_up), normal_down)

            else:  # self.dim == 2
                # we say that orientations align if the cross product
                # between the ridge tangent and the mortar normal corresponds
                # to the normal of the lower-dimensional face
                tangents = sd_up.nodes @ sd_up.ridge_peaks[:, ridges_up]
                products = np.cross(tangents, normal_up, axisa=0, axisc=0)
                orientations_fr = [
                    np.dot(products[:, i], normal_down[:, i])
                    for i in np.arange(np.size(tangents, 1))
                ]

                # The (virtual) line connecting the low-dim ridge to
                # the high-dim is oriented according to the normal to the fracture plane
                orientations_rp = -np.dot(normal_up, normal_to_sd_down) * np.ones(
                    peaks_up.shape
                )
                ridge_peaks[peaks_up, ridges_down] += np.sign(orientations_rp)

            face_ridges[ridges_up, faces_down] += np.sign(orientations_fr)

        # Ensure that double indices are mapped to +-1
        # This step ensures that the jump maps to zero at tips.
        face_ridges_csc = sps.csc_array(face_ridges, dtype=int)
        ridge_peaks_csc = sps.csc_array(ridge_peaks, dtype=int)

        face_ridges_csc.data = np.sign(face_ridges_csc.data)
        ridge_peaks_csc.data = np.sign(ridge_peaks_csc.data)

        # Set face_ridges and ridge_peaks as properties of the mortar grid
        self.face_ridges = face_ridges_csc
        self.ridge_peaks = ridge_peaks_csc

    def assign_signed_mortar_to_primary(self) -> None:
        """
        Compute the mapping from mortar cells to the faces of the primary grid that
        respects orientation.

        Args:
            sd_pair (Tuple[pp.Grid, pp.Grid]): Pair of adjacent subdomains.

        Returns:
            sps.csc_array: A sparse matrix representing the mapping from mortar
            cells to primary grid faces. The matrix has dimensions num_primary_faces x
            num_mortar_cells.
        """
        sd_up = self.sd_pair[0]
        cells, faces, _ = sps.find(self.primary_to_mortar_int())
        cf_csr = sd_up.cell_faces.tocsr()
        signs = [cf_csr[face, :].data[0] for face in faces]

        self.signed_mortar_to_primary = sps.csc_array(
            (signs, (faces, cells)), (sd_up.num_faces, self.num_cells)
        )

    def assign_cell_faces(self) -> None:
        """
        Assign the connectivity between cells of the secondary grid and faces of the
        primary grid.

        This method calculates and assigns the connectivity between the cells of the
        secondary grid and the faces of the primary grid. It uses the signed mortar
        values and the secondary-to-mortar interface function to determine the
        connectivity.

        Args:
            None

        Returns:
            None
        """
        self.cell_faces = (
            -self.signed_mortar_to_primary @ self.secondary_to_mortar_int()
        )
