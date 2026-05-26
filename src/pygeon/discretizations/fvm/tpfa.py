from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class TPFA(pg.FiniteVolumeDiscretization):
    """
    A simple implementation of the two-point flux approximation method for Darcy flow.

    Our implementation differs from Porepy (v1.13) because we take the normal distance
    between face and cell centers, instead of the Euclidean distance, cf. Aavatsmark
    (2002) or the DuMuX documentation on tpfa.

    Degrees of freedom are given by the cell center values.
    """

    def __init__(self, keyword=pg.UNITARY_DATA) -> None:
        """
        Initialize the TPFA object.

        Args:
            keyword (str): The keyword used to identify the discretization method.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.bc_type = pg.FlowBC

    def ndof_per_cell(self, _sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom per cell, in this case one.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom.
        """
        return 1

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a scalar function onto the grid

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable): A function that returns the pressure values
                at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom
        """
        interp = pg.PwConstants().interpolate(sd, func)
        return interp / sd.cell_volumes

    def assemble_accumulation_terms(self, sd: pg.Grid, _data: dict) -> sps.csc_array:
        """
        Assembles the accumulation terms such as the storativity
        S_0 dp/dt

        For now, it is zero, but this can be overwritten by a child class

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary.

        Returns:
            sps.csc_array: The TPFA discretization matrix.
        """
        return sps.csc_array((self.ndof(sd), self.ndof(sd)))

    def precompute_arrays(self, sd: pg.Grid, data: dict | None = None) -> dict:
        """
        Precomputations on the grid for easy access later.

        This function is typically called twice, once for the left-hand side, and once
        for the right.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary.

        Returns:
            dict: The precomputed arrays
        """
        # Retrieve cell-face connectivity
        find_cell_faces = sps.find(sd.cell_faces)

        # Computed the weighted distances
        weighted_dists = self.compute_weighted_dists(sd, data, find_cell_faces)
        self.check_nonnegative_weights(weighted_dists)

        # Extend the face and distance arrays to incorporate boundary conditions
        faces, dists = self.extend_faces_and_distances(
            sd, data, find_cell_faces[0], weighted_dists
        )

        # With the extended face and distance arrays, we can compute the effective
        # permeability, also on the boundary
        perm_effective = self.compute_harmonic_avg(faces, dists)

        # Gather these vectors in a dictionary for easy access in the assembly
        # procedures
        cached_arrays = {
            "find_cell_faces": find_cell_faces,
            "weighted_dists": weighted_dists,
            "perm_effective": perm_effective,
        }

        return cached_arrays

    def compute_weighted_dists(
        self, sd: pg.Grid, data: dict | None, find_cell_faces: Tuple
    ) -> np.ndarray:
        """
        Compute delta_k^i / normal_perm for every physical face-cell pair. Boundary
        conditions are handled later.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            weights (np.ndarray): The material parameter weights, in this case the
                values of a second-order tensor.

        Returns:
            np.ndarray: The weighted distances
        """
        faces, cells, orient = find_cell_faces
        unit_normals = sd.face_normals / sd.face_areas
        unit_normals = unit_normals[:, faces]

        # Compute the normal distance between face and cell center
        delta = np.sum(
            (
                (sd.face_centers[:, faces] - sd.cell_centers[:, cells])
                * (orient * unit_normals)
            ),
            axis=0,
        )

        # Compute the normal permeability per cell-face pair
        perm = pg.get_cell_data(
            sd, data, self.keyword, pg.SECOND_ORDER_TENSOR, 2
        ).values
        perm_nn = np.einsum(
            "ijk,ik,jk->k", perm[:, :, cells], unit_normals, unit_normals
        )

        return delta / perm_nn

    def extend_faces_and_distances(
        self, sd: pg.Grid, data: dict, faces: np.ndarray, weighted_dists: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Incorporate the boundary conditions by extending the face and distance arrays.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary.
            faces (np.ndarray): The array of face indices
            weighted_dists (np.ndarray): The array of weighted distances

        Returns:
            np.ndarray: The extended array of face indices
            np.ndarray: The extended array of weighted distances
        """
        bcs = self.get_bcs_from_data(sd, data)

        # Incorporate the bcs by extending the vectors
        bdry_faces = sd.tags["domain_boundary_faces"]
        ext_faces = np.hstack((faces, np.flatnonzero(bdry_faces)))

        ext_dists = np.concatenate((weighted_dists, bcs.weighted_dists[bdry_faces]))

        return ext_faces, ext_dists

    def assemble_dual_var_map(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assemble the mapping from cell-based primary variables to face-based dual
        variables.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: The matrix mapping primary to dual variables
        """
        cached_arrays = self.precompute_arrays(sd, data)
        K_eff = cached_arrays["perm_effective"]

        A_csc = sd.cell_faces.astype(float).tocsc()
        A_csc.data *= (sd.face_areas * K_eff)[A_csc.indices]

        return A_csc

    def assemble_bdry_dual_var_map(self, sd: pg.Grid, data: dict | None = None):
        """
        Assembles the matrix that maps from the boundary condition values to the dual
        variables on the boundary faces. This implementation handles Dirichlet, Robin,
        and Neumann in a unified way. Inspired by the TPSA paper, Appendix A2.2.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: the matrix to be multiplied with the boundary data g
        """
        # Preallocation and precomputations
        A_rhs = np.empty(2, dtype=sps.csc_array)
        cached_arrays = self.precompute_arrays(sd, data)
        K_eff = cached_arrays["perm_effective"]

        # Find out the weighted distances on the exterior of the domain. The interior
        # deltas will be filtered out later.
        delta_bdry = np.zeros(sd.num_faces)
        delta_bdry[cached_arrays["find_cell_faces"][0]] = cached_arrays[
            "weighted_dists"
        ]

        # Compute the complementary averaging operator from the TPSA paper (2.6) and
        # filter over the boundary faces. Xi_tilde is one on flux boundaries and zero on
        # pressure boundaries.
        Xi_tilde_bdry = 1 - delta_bdry * K_eff
        Xi_tilde_bdry *= sd.tags["domain_boundary_faces"]

        # The flux variable gets multiplied by Xi_tilde and the face area
        A_rhs[0] = sps.diags_array(sd.face_areas * Xi_tilde_bdry, format="csc")

        # For the pressure boundary conditions, we only need to know the sign of the
        # normal and the effective permeability
        Delta_B = sd.cell_faces.sum(axis=1)
        A_rhs[1] = -sps.diags_array(sd.face_areas * K_eff * Delta_B, format="csc")

        return sps.hstack(A_rhs, format="csc")

    def assemble_source(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Assemble the right-hand side for a given source func(x,y,z)

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable): The source function.

        Returns:
            np.ndarray: the right-hand side vector
        """
        return pg.PwConstants().interpolate(sd, func)
