import numpy as np
import porepy as pp
import scipy.sparse as sps

"""
Acknowledgments:
    The functionalities related to the ridge computations are modified from
    github.com/anabudisa/md_aux_precond developed by Ana Budiša and Wietse M. Boon.
"""


class Grid(pp.Grid):
    def __init__(self, *args, **kwargs):
        super(Grid, self).__init__(*args, **kwargs)

    def compute_geometry(self):
        """
        Defines grid entities of codim 2 and 3.

        The entities are referred to by their codimension:
        0: "cells"
        1: "faces"
        2: "ridges"
        3: "peaks"
        """

        super(Grid, self).compute_geometry()
        self.compute_ridges()
        self.correct_concave_elements_2d()

    def compute_ridges(self):
        """
        Assigns the following attributes to the grid

        num_ridges: number of ridges
        num_peaks: number of peaks
        face_ridges: connectivity between each face and ridge
        ridge_peaks: connectivity between each ridge and peak
        tags['tip_ridges'], tags['tip_peaks']: tags for entities at fracture tips
        """

        if self.dim == 3:
            self._compute_ridges_3d()
        elif self.dim == 2:
            self._compute_ridges_2d()
        else:  # The grid is of dimension 0 or 1.
            self._compute_ridges_01d()

        self.tag_ridges()

    def compute_subvolumes(self):
        """
        Assigns the following attributes to the grid

        subvolumes: a csc_matrix with each entry [face, cell] describing
                      the signed measure of the associated sub-volume
        """
        self.sub_volumes = self.cell_faces.copy().astype(float)

        if self.dim == 3:
            # self._compute_subvolumes_3d()
            pass
        elif self.dim == 2:
            self._compute_subvolumes_2d()

    def _compute_ridges_01d(self):
        """
        Assign the number of ridges, number of peaks, and connectivity matrices to a
        grid of dimension 2.
        """

        self.num_peaks = 0
        self.num_ridges = 0
        self.ridge_peaks = sps.csc_matrix((self.num_peaks, self.num_ridges), dtype=int)
        self.face_ridges = sps.csc_matrix((self.num_ridges, self.num_faces), dtype=int)

    def _compute_ridges_2d(self):
        """
        Assign the number of ridges, number of peaks, and connectivity matrices to a
        grid of dimension 2.
        """

        self.num_peaks = 0
        self.num_ridges = self.num_nodes
        self.ridge_peaks = sps.csc_matrix((self.num_peaks, self.num_ridges), dtype=int)

        # We compute the face tangential by mapping the face normal to a reference grid
        # in the xy-plane, rotating locally, and mapping back.
        R = pp.map_geometry.project_plane_matrix(self.nodes)
        loc_rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rot = R.T @ loc_rot @ R
        rotated_normal = rot.dot(self.face_normals)

        # The face-ridge orientation is determined by whether the rotated normal
        # coincides with the difference vector between the ridges.
        face_ridges = self.face_nodes.copy().astype(int)
        face_ridges.data[::2] *= -1
        face_tangents = self.nodes @ face_ridges

        orients = np.sign(np.sum(rotated_normal * face_tangents, axis=0))

        self.face_ridges = face_ridges * sps.diags(orients)

    def _compute_ridges_3d(self):
        """
        Assign the number of ridges, number of peaks, and connectivity matrices to a
        grid of dimension 3.
        """

        self.num_peaks = self.num_nodes

        # Pre-allocation
        ridges = np.ndarray((2, self.face_nodes.nnz), dtype=int)

        fr_indptr = np.zeros(self.num_faces + 1, dtype=int)
        for face in np.arange(self.num_faces):
            # find indices for nodes of this face
            loc = self.face_nodes.indices[
                self.face_nodes.indptr[face] : self.face_nodes.indptr[face + 1]
            ]
            fr_indptr[face + 1] = fr_indptr[face] + loc.size

            # Define ridges between each pair of nodes
            # assuming ordering in face_nodes is done
            # according to right-hand rule
            ridges[:, fr_indptr[face] : fr_indptr[face + 1]] = np.row_stack(
                (loc, np.roll(loc, -1))
            )

        # Save orientation of each ridge w.r.t. the face
        orientations = np.sign(ridges[1, :] - ridges[0, :])

        # Ridges are oriented from low to high node indices
        ridges.sort(axis=0)
        ridges, _, indices = pp.utils.setmembership.unique_columns_tol(ridges)
        self.num_ridges = np.size(ridges, 1)

        # Generate ridge-peak connectivity such that
        # ridge_peaks(i, j) = +/- 1:
        # ridge j points to/away from peak i
        indptr = np.arange(0, ridges.size + 1, 2)
        ind = np.ravel(ridges, order="F")
        data = -((-1) ** np.arange(ridges.size))
        self.ridge_peaks = sps.csc_matrix((data, ind, indptr))

        # Generate face_ridges such that
        # face_ridges(i, j) = +/- 1:
        # face j has ridge i with same/opposite orientation
        # with the orientation defined according to the right-hand rule
        self.face_ridges = sps.csc_matrix((orientations, indices, fr_indptr))

    def tag_ridges(self):
        """
        Tag the peaks and ridges of the grid located on fracture tips.
        """

        self.tags["tip_peaks"] = np.zeros(self.num_peaks, dtype=bool)
        fr_bool = self.face_ridges.astype("bool")

        if self.dim == 2:
            self.tags["tip_ridges"] = fr_bool * self.tags["tip_faces"]
        else:
            self.tags["tip_ridges"] = np.zeros(self.num_ridges, dtype=bool)

        bd_ridges = fr_bool * self.tags["domain_boundary_faces"]
        self.tags["domain_boundary_ridges"] = bd_ridges.astype(bool)

    def _compute_subvolumes_2d(self):
        faces, cells, orient = sps.find(self.cell_faces)

        tangents = self.nodes @ self.face_ridges[:, faces] * orient
        rays = (
            self.nodes @ (self.face_ridges[:, faces] < 0) - self.cell_centers[:, cells]
        )

        self.sub_volumes[faces, cells] = np.cross(rays, tangents, axis=0)[-1, :] / 2

    def correct_concave_elements_2d(self):
        """
        Corrects the cell_center, cell_volume, and cell_faces for concave cells in 2D
        """

        # We have to double check whether the orientations are consistent
        cr = self.face_ridges * self.cell_faces

        if cr.nnz == 0:  # Orientations are fine
            return

        # Else, we first map to the xy-plane
        R = pp.map_geometry.project_plane_matrix(self.nodes)
        self.nodes = np.dot(R, self.nodes)

        while cr.nnz != 0:
            ridges, cells, orient = sps.find(cr)

            start_node = ridges[orient == -2][0]
            bad_cell = cells[orient == -2][0]

            local_fr = self.face_ridges[:, self.cell_faces[:, bad_cell].indices]

            # Loop through the faces and nodes from
            # the start: where two faces are oriented away from a ridge (cr == -2)
            # to the finish: where two faces are oriented to the same ridge (cr == 2)
            loop_ind = 0
            while loop_ind < local_fr.shape[0]:
                next_face = np.argmax(local_fr[start_node, :] == -1)
                self.cell_faces[next_face, :] *= -1
                start_node = np.argmax(local_fr[:, next_face] == 1)

                loop_ind += 1
                if cr[start_node, bad_cell] == 2:
                    break
            else:
                raise TimeoutError(
                    "Could not create a node loop, something is wrong in the mesh orientation."
                )

            cr = self.face_ridges * self.cell_faces

        faces, cells, orient = sps.find(self.cell_faces)

        # Recompute the volumes and reorient cell_faces
        tangents = (self.nodes * self.face_ridges[:, faces]) * orient
        rays = (
            self.nodes * (self.face_ridges[:, faces] < 0) - self.cell_centers[:, cells]
        )

        subsimplex_volumes = np.cross(rays, tangents, axis=0)[-1, :] / 2
        signed_volumes = np.bincount(cells, subsimplex_volumes)
        loop_orientation = np.sign(signed_volumes)

        self.cell_volumes = np.abs(signed_volumes)
        self.cell_faces = self.cell_faces * sps.diags(loop_orientation)

        # Recompute the cell centers
        subcentroids = (
            2 * self.face_centers[:, faces] + self.cell_centers[:, cells]
        ) / 3

        for x_dim in range(2):  # Third dimension is zero since we mapped to the plane
            self.cell_centers[x_dim, :] = np.bincount(
                cells,
                subcentroids[x_dim, :] * subsimplex_volumes / self.cell_volumes[cells],
            )

        # Set the nodes back in their original position.
        self.nodes = np.dot(R.T, self.nodes)
