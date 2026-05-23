from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class TPFA(pg.FiniteVolumeDiscretization):
    """
    A simple implementation of the two-point flux approximation method for Darcy flow.
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

    def ndof_per_cell(self, _) -> int:
        """
        Returns the number of degrees of freedom per cell, in this case one.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom.
        """
        return 1

    def interpolate(self, sd: pg.Grid, pressure: Callable) -> np.ndarray:
        """
        Interpolates a scalar function onto the grid

        Args:
            sd (pg.Grid): Grid, or a subclass.
            pressure (Callable): A function that returns the pressure values
                at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom
        """
        interp = pg.PwConstants().interpolate(sd, pressure)
        return interp / sd.cell_volumes

    def assemble_accumulation_terms(self, sd: pg.Grid, _data: dict) -> sps.csc_array:
        """
        Assemble the TPFA matrix for Darcy flow, using the material parameters in the
        data dictionary.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary.

        Returns:
            sps.csc_array: The TPFA discretization matrix.
        """
        return sps.csc_array((self.ndof(sd), self.ndof(sd)))

    def precompute_arrays(self, sd: pg.Grid, data: dict) -> dict:

        find_cell_faces = sps.find(sd.cell_faces)
        weighted_dists = self.compute_weighted_dists(sd, data, find_cell_faces)
        self.check_nonnegative_weights(weighted_dists)

        faces, dists = self.extend_faces_and_distances(
            sd, data, find_cell_faces[0], weighted_dists
        )

        # With the extended face and distance arrays, we can compute the effective
        # permeability
        perm_effective = super().compute_harmonic_avg(faces, dists)

        # Gather these vectors in a dictionary for easy access in the assembly procedure
        cached_arrays = {
            "find_cell_faces": find_cell_faces,
            "weighted_dists": weighted_dists,
            "perm_effective": perm_effective,
        }

        return cached_arrays

    def compute_weighted_dists(
        self, sd: pg.Grid, data: dict, find_cell_faces: Tuple
    ) -> np.ndarray:
        """
        Compute delta_k^i / K_nn for every physical face-cell pair. Boundary conditions
        are handled later.

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

        delta = np.sum(
            (
                (sd.face_centers[:, faces] - sd.cell_centers[:, cells])
                * (orient * unit_normals)
            ),
            axis=0,
        )

        perm = pg.get_cell_data(
            sd, data, self.keyword, pg.SECOND_ORDER_TENSOR, 2
        ).values
        K_nn = np.einsum("ijk,ik,jk->k", perm[:, :, cells], unit_normals, unit_normals)

        return delta / K_nn

    def extend_faces_and_distances(
        self, sd: pg.Grid, data: dict, faces: np.ndarray, weighted_dists: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Incorporate the boundary conditions by extending the face and distance arrays.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary.

        Returns:
            faces (np.ndarray): The extended array of faces
            dists (np.ndarray): The extended array of weighted distances
        """
        bcs = self.get_bcs_from_data(sd, data)

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

        scaling = (self.face_area_scaling(sd) * K_eff)[:, None]
        return (scaling * sd.cell_faces).tocsc()

    def assemble_bdry_dual_var_map(self, sd: pg.Grid, data: dict):
        """
        Assembles the matrix that maps from the boundary condition values to the dual
        variables on the boundary faces. This implementation handles Dirichlet, Robin,
        and Neumann in a unified way. Inspired by the TPSA paper, Appendix A2.2.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: the matrix to be multiplied with the boundary data g
        """
        A_rhs = np.empty(2, dtype=sps.csc_array)

        cached_arrays = self.precompute_arrays(sd, data)
        K_eff = cached_arrays["perm_effective"]

        delta_bdry = np.zeros(sd.num_faces)
        delta_bdry[cached_arrays["find_cell_faces"][0]] = cached_arrays[
            "weighted_dists"
        ]

        Xi_tilde_bdry = 1 - delta_bdry * K_eff
        Xi_tilde_bdry *= sd.tags["domain_boundary_faces"]

        A_rhs[0] = sps.diags_array(sd.face_areas * Xi_tilde_bdry, format="csc")

        Delta_B = sd.cell_faces.sum(axis=1)
        A_rhs[1] = -sps.diags_array(sd.face_areas * K_eff * Delta_B, format="csc")

        return sps.hstack(A_rhs, format="csc")

    def assemble_source(self, sd: pg.Grid, source: Callable) -> np.ndarray:
        """
        Assemble the right-hand side for a given source func(x,y,z)

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable): The source function.

        Returns:
            np.ndarray: the right-hand side vector
        """
        return pg.PwConstants().interpolate(sd, source)
