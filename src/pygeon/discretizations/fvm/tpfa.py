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

        self.K_bar: dict[pg.Grid, np.ndarray] = {}

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

    def assemble_flow_matrix(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assemble the TPFA matrix for Darcy flow, using the material parameters in the
        data dictionary.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary.

        Returns:
            sps.csc_array: The TPFA discretization matrix.
        """
        perm = pg.get_cell_data(sd, data, self.keyword, pg.SECOND_ORDER_TENSOR, 2)

        # Precomputations without boundary conditions
        self.finite_volume_precomputations(sd, perm.values)

        faces, deltas = self.extend_faces_and_distances(sd, data)

        self.compute_harmonic_avg(sd, faces, deltas)

        A = self.assemble_dual_var_map(sd)

        return self.div(sd) @ A

    def compute_weighted_dists(self, sd: pg.Grid, perm: np.ndarray) -> np.ndarray:
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

    def extend_faces_and_distances(
        self, sd: pg.Grid, data: dict
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

        faces, *_ = self.find_cf[sd]
        bdry_faces = sd.tags["domain_boundary_faces"]
        ext_faces = np.hstack((faces, np.flatnonzero(bdry_faces)))
        ext_dists = np.concatenate(
            (self.weighted_dists[sd], bcs.weighted_dists[bdry_faces])
        )

        return ext_faces, ext_dists

    def compute_harmonic_avg(
        self, sd: pg.Grid, faces: np.ndarray, dists: np.ndarray
    ) -> None:
        """
        Compute the harmonic average of K divided by delta_k, at each face

        Args:
            sd (pg.Grid): Grid, or a subclass.
            faces (np.ndarray): The extended array of faces
            dists (np.ndarray): The extended array of weighted distances
        """
        self.K_bar[sd] = np.array(1 / np.bincount(faces, weights=dists))

    def assemble_dual_var_map(self, sd: pg.Grid) -> sps.sparray:
        """
        Assemble the mapping from cell-based primary variables to face-based dual
        variables.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: The matrix mapping primary to dual variables
        """
        scaling = (self.face_area_scaling(sd) * self.K_bar[sd])[:, None]
        return (scaling * sd.cell_faces).tocsc()

    def assemble_bdry_dual_var_map(self, sd: pg.Grid):
        """
        Assembles the matrix that maps from the boundary condition values to the dual
        variables on the boundary faces. This implementation handles Dirichlet, Robin,
        and Neumann in a unified way. Inspired by the TPSA paper, Appendix A2.2.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: the matrix to be multiplied with the boundary data g
        """
        A_rhs = np.empty(2, dtype=sps.sparray)

        K_bar = self.K_bar[sd]

        delta_bdry = np.zeros(sd.num_faces)
        delta_bdry[self.find_cf[sd][0]] = self.weighted_dists[sd]

        Xi_tilde_bdry = 1 - delta_bdry * K_bar
        Xi_tilde_bdry *= sd.tags["domain_boundary_faces"]

        A_rhs[0] = sps.diags_array(Xi_tilde_bdry)

        Delta_B = sd.cell_faces.sum(axis=1)
        A_rhs[1] = -sps.diags_array(K_bar * Delta_B)

        return sd.face_areas[:, None] * sps.hstack(A_rhs, format="csc")

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
