import numpy as np
import scipy.sparse as sps
import porepy as pp

from pygeon.utils.set_membership import match_coordinates

"""
Acknowledgments:
    The functionalities related to the ridge computations are modified from
    github.com/anabudisa/md_aux_precond developed by Ana Budiša and Wietse M. Boon.
"""


class MortarGrid(pp.MortarGrid):
    def __init__(self, *args, **kwargs):
        super(MortarGrid, self).__init__(*args, **kwargs)

    def compute_geometry(self, pair):
        super(MortarGrid, self).compute_geometry()

        self.assign_signed_mortar_to_primary(pair)
        self.assign_cell_faces()

        if self.dim >= 1:
            self.compute_ridges(pair)

    def compute_ridges(self, pair):
        """
        Assign the face-ridge and ridge-peak connectivities to the mortar grid corresponding to an edge of a grid bucket.

        Parameters:
            gb (pp.GridBucket): The grid bucket.
            e (Tuple[pp.Grid, pp.Grid]): An edge of gb.
        """

        # Find high-dim faces matching to low-dim cell
        cell_faces = self.mortar_to_primary_int() * self.secondary_to_mortar_int()
        g_down, g_up = pair

        # High-dim ridges matching to low-dim face
        face_ridges = sps.lil_matrix((g_up.num_ridges, g_down.num_faces), dtype=int)
        # High-dim peaks matching to low-dim ridge
        ridge_peaks = sps.lil_matrix((g_up.num_peaks, g_down.num_ridges), dtype=int)

        # Find information about the two-dimensional grid
        if self.dim == 1:
            R = pp.map_geometry.project_plane_matrix(g_up.nodes)
            rot = np.dot(
                R.T,
                np.dot(
                    np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), R
                ),
            )
        else:  # mg.dim == 2
            R = pp.map_geometry.project_plane_matrix(g_down.nodes)
            normal_to_g_down = np.dot(R.T, [0, 0, 1])

        for (face_up, cell_down) in zip(*sps.find(cell_faces)[:-1]):
            # Faces of cell in lower-dim grid
            cf_down = g_down.cell_faces
            faces_down = cf_down.indices[
                cf_down.indptr[cell_down] : cf_down.indptr[cell_down + 1]
            ]

            # Ridges of face in higher-dim grid
            fr_up = g_up.face_ridges
            ridges_up = fr_up.indices[fr_up.indptr[face_up] : fr_up.indptr[face_up + 1]]

            # Swap ridges around so they match with lower-dim faces
            if self.dim == 1:
                face_xyz = g_down.face_centers[:, faces_down]
                ridge_xyz = g_up.nodes[:, ridges_up]
            else:  # mg.dim == 2
                face_xyz = g_down.nodes * abs(g_down.face_ridges[:, faces_down]) / 2
                ridge_xyz = g_up.nodes * abs(g_up.ridge_peaks[:, ridges_up]) / 2

            ridges_up = ridges_up[match_coordinates(face_xyz, ridge_xyz)]

            # Ridge-peak connectivity in 3D
            if self.dim == 2:
                # Ridges of cell in lower-dim grid
                cr_down = g_down.cell_nodes()
                ridges_down = cr_down.indices[
                    cr_down.indptr[cell_down] : cr_down.indptr[cell_down + 1]
                ]
                ridge_xyz = g_down.nodes[:, ridges_down]

                # Nodes of face in higher-dim grid
                fn_up = g_up.face_nodes
                peaks_up = fn_up.indices[
                    fn_up.indptr[face_up] : fn_up.indptr[face_up + 1]
                ]
                peak_xyz = g_up.nodes[:, peaks_up]

                # Swap peaks around so they match with lower-dim ridges
                peaks_up = peaks_up[match_coordinates(ridge_xyz, peak_xyz)]

            # Take care of orientations
            # NOTE:this computation is done here so that we have access to the normal vector

            # Find the normal vector oriented outward wrt the higher-dim grid
            is_outward = g_up.cell_faces.tocsr()[face_up, :].data[0]
            normal_up = g_up.face_normals[:, face_up] * is_outward

            # Find the normal to the lower-dim face
            normal_down = g_down.face_normals[:, faces_down]

            # Identify orientation
            if self.dim == 1:
                # we say that orientations align if the rotated mortar
                # normal corresponds to the normal of the
                # lower-dimensional face
                orientations_fr = np.dot(np.dot(rot, normal_up), normal_down)

            else:  # mg.dim == 2
                # we say that orientations align if the cross product
                # between the ridge tangent and the mortar normal corresponds
                # to the normal of the lower-dimensional face
                tangents = g_up.nodes * g_up.ridge_peaks[:, ridges_up]
                products = np.cross(tangents, normal_up, axisa=0, axisc=0)
                orientations_fr = [
                    np.dot(products[:, i], normal_down[:, i])
                    for i in np.arange(np.size(tangents, 1))
                ]

                # The (virtual) line connecting the low-dim ridge to
                # the high-dim is oriented according to the normal to the fracture plane
                orientations_rp = -np.dot(normal_up, normal_to_g_down) * np.ones(
                    peaks_up.shape
                )
                ridge_peaks[peaks_up, ridges_down] += np.sign(orientations_rp)

            face_ridges[ridges_up, faces_down] += np.sign(orientations_fr)

        # Ensure that double indices are mapped to +-1
        # This step ensures that the jump maps to zero at tips.
        face_ridges = sps.csc_matrix(face_ridges, dtype=int)
        ridge_peaks = sps.csc_matrix(ridge_peaks, dtype=int)

        face_ridges.data = np.sign(face_ridges.data)
        ridge_peaks.data = np.sign(ridge_peaks.data)

        # Set face_ridges and ridge_peaks as properties of the mortar grid
        self.face_ridges = face_ridges
        self.ridge_peaks = ridge_peaks

    def assign_signed_mortar_to_primary(self, pair):
        """
        Compute the mapping from mortar cells to the faces of the primary grid that respects orientation.

        Parameters:
            mg (pp.MortarGrid): The mortar grid.
            g (pp.Grid): The primary grid.

        Returns:
            sps.csc_matrix, num_primary_faces x num_mortar_cells.
        """
        g = pair[1]
        cells, faces, _ = sps.find(self.primary_to_mortar_int())
        signs = [g.cell_faces.tocsr()[face, :].data[0] for face in faces]

        self.assign_signed_mortar_to_primary = sps.csc_matrix(
            (signs, (faces, cells)), (g.num_faces, self.num_cells)
        )

    def assign_cell_faces(self):
        """
        Assign the connectivity between cells of the secondary grid and faces of the primary grid
        for each mortar grid of a grid bucket.

        Parameters:
            gb (pp.GridBucket): The grid bucket.
        """

        self.cell_faces = (
            -self.assign_signed_mortar_to_primary * self.secondary_to_mortar_int()
        )
