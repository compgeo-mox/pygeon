from typing import Callable

import numpy as np
import scipy.sparse as sps

import pygeon as pg


def rotation_dim(dim: int) -> int:
    return dim * (dim - 1) // 2


class TPSA(pg.FiniteVolumeDiscretization):
    bc_type = pg.ElasticityBC

    def ndof_components(self, sd: pg.Grid) -> np.ndarray:
        return sd.num_cells * np.array([sd.dim, rotation_dim(sd.dim), 1])

    def ndof_per_cell(self, sd) -> int:
        return sd.dim + rotation_dim(sd.dim) + 1

    def assemble_elasticity_matrix(self, sd: pg.Grid, data: dict) -> None:
        """
        Assemble the TPSA matrix, given material constants in the data dictionary.
        """

        # Extract the Lamé parameters
        lame_mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        lame_lambda = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        # Precomputations without boundary conditions
        self.fvm_precomputations(sd, lame_mu)

        #
        bcs = self.extract_bcs(sd, data)
        faces, deltas = self.extend_faces_and_distances(sd, bcs)

        self.compute_delta_mu_k(faces, deltas)
        self.compute_harmonic_avg(faces, deltas)

        # Assemble the remaining terms in (3.9)
        A = self.div_F(sd) @ self.assemble_dual_var_map(sd)

        # Generate the zero'th order terms in (3.9)
        M = self.assemble_mass_terms(sd, lame_mu, lame_lambda)

        # Assemble the matrix from (3.9)
        return sps.csc_array(A - M)

    def compute_weighted_dists(self, sd: pg.Grid, weights: np.ndarray) -> np.ndarray:
        """
        Compute delta_k^i / mu from (2.1) for every physical face-cell pair. Boundary
        conditions are handled later.
        """
        faces, cells, orient = self.find_cf[sd]
        unit_normals = self.unit_normals[sd]

        delta = np.sum(
            (
                (sd.face_centers[:, faces] - sd.cell_centers[:, cells])
                * (orient * unit_normals[:, faces])
            ),
            axis=0,
        )

        return delta / weights[cells]

    def assemble_mass_terms(
        self, sd: pg.Grid, lame_mu: np.ndarray, lame_lambda: np.ndarray
    ) -> sps.csc_array:
        """
        The first-order terms on the diagonal of (3.9)
        """
        M_u = np.zeros(sd.dim * sd.num_cells)
        M_r = np.tile(sd.cell_volumes / lame_mu, rotation_dim(sd.dim))
        M_p = sd.cell_volumes / lame_lambda

        diagonal = np.concatenate((M_u, M_r, M_p))

        return sps.diags_array(diagonal).tocsc()

    def assemble_dual_var_map(self, sd: pg.Grid) -> sps.sparray:
        """
        Assemble the second-order terms
        """
        # Preallocate the main matrix
        A = np.empty((3, 3), dtype=sps.sparray)

        # Assemble the blocks of (3.7) where
        # A_ij is the block coupling variable i and j.
        A_uu = [
            -2 * mu_bar[:, None] * sd.cell_faces for mu_bar in self.mu_bar_over_delta
        ]
        A[0, 0] = sps.block_diag(A_uu)

        # Assemble the boundary terms of (A2.25)
        A[1, 1] = self.assemble_lhs_bdry_terms(sd)

        # The blocks in the first column depend on the averaging operator Xi
        Xi = self.assemble_xi(sd)
        R_Xi, n_Xi = self.assemble_RXi_and_NXi(sd, Xi, map_from_u=True)
        A[1, 0] = -R_Xi
        A[2, 0] = n_Xi

        # The blocks in the first row depend on the complementary operator Xi_tilde
        Xi_tilde = self.convert_to_xi_tilde_inplace(Xi)
        R_Xi_t, n_Xi_t = self.assemble_RXi_and_NXi(sd, Xi_tilde, map_from_u=False)
        A[0, 1] = -R_Xi_t
        A[0, 2] = n_Xi_t

        # Stabilization for the pressure
        delta_min = np.min(self.delta_mu_k, axis=0)
        A[2, 2] = -delta_min[:, None] * sd.cell_faces

        # Assembly by blocks
        return sps.block_array(A).tocsc()

    def assemble_rot(self, sd):
        nx, ny, nz = [sps.diags_array(n_i) for n_i in self.unit_normals[sd]]

        return sps.block_array(
            [
                [None, -nz, ny],
                [nz, None, -nx],
                [-ny, nx, None],
            ]
        )

    def assemble_lhs_bdry_terms(self, sd):
        R = self.assemble_rot(sd)
        R_squared = R @ R

        bdry_deltas = self.delta_mu_k * sd.tags["domain_boundary_faces"]
        delta = bdry_deltas.flatten()

        codiv = sps.kron(sps.eye_array(3), sd.cell_faces)
        return -delta[:, None] * R_squared @ codiv

    def extend_faces_and_distances(self, sd: pg.Grid, bcs: pg.ElasticityBC):
        # Incorporate the Robin bc by extending the vectors
        faces, *_ = self.find_cf[sd]

        bdry_faces = sd.tags["domain_boundary_faces"]
        ext_faces = np.concatenate((faces, np.flatnonzero(bdry_faces)))

        # We allow for different types of boundary conditions in the x, y, z directions.
        # We therefore have three instances of the distances, one for each direction.
        tiled_dists = np.tile(self.weighted_dists[sd], (pg.AMBIENT_DIM, 1))
        ext_dists = np.hstack((tiled_dists, bcs.weighted_dists[:, bdry_faces]))

        return ext_faces, ext_dists

    def compute_delta_mu_k(self, faces, dists) -> None:
        """
        Compute 0.5 * ( mu_i delta_k^-i + mu_j delta_k^-j)^-1
        for each face k with cells (i,j)

        This is the delta^mu_k of (3.5)
        """
        # Compute the reciprocal
        inv_dists = np.empty_like(dists)
        zero_dist = dists == 0

        inv_dists[~zero_dist] = 1 / dists[~zero_dist]
        inv_dists[zero_dist] = np.inf

        # Displacement boundaries have infinite mu/delta
        # Traction bc are handled naturally because mu/delta = 0 there.
        output_list = [1 / np.bincount(faces, weights=row) for row in inv_dists]
        self.delta_mu_k = np.array(output_list) / 2

    def compute_harmonic_avg(self, faces, dists) -> np.ndarray:
        """
        Compute the harmonic average of mu from (3.5), divided by delta_k, at each face
        """
        # Displacement bc are handled naturally as a subset of spring_bdry
        # with zero (inverse) spring constant.
        # Tractions are handled with infinite spring constant.
        output_list = [1 / np.bincount(faces, weights=row) for row in dists]
        self.mu_bar_over_delta = np.array(output_list)

    def assemble_xi(self, sd) -> list:
        """
        Compute the averaging operator Xi from (2.5)
        """
        faces, cells, _ = sps.find(sd.cell_faces)
        Xi = [
            sps.csc_array((2 * delta[faces] / self.weighted_dists[sd], (faces, cells)))
            for delta in self.delta_mu_k
        ]

        # Displacement bc are handled by dk_mu = 0
        # Traction bc are handled since dk_mu * mu / delta = 1
        # Spring bc are handled because the spring constant is in dk_mu

        return Xi

    def convert_to_xi_tilde_inplace(self, Xi: list) -> sps.sparray:
        """
        Compute the converse averaging operator Xi_tilde from (2.6)
        This is an in-place operation to save memory
        """
        for Xi_i in Xi:
            Xi_i.data = 1 - Xi_i.data
        return Xi

    def assemble_RXi_and_NXi(
        self,
        sd: pg.Grid,
        Xi: sps.sparray,
        map_from_u: bool = True,
    ) -> tuple[sps.sparray, sps.sparray]:
        """
        Assemble the off-diagonal terms n times Xi and n cdot Xi in (3.7)
        These are computed together because their construction uses similar components.
        """
        nx, ny, nz = [
            ni[:, None] * Xi_i for (ni, Xi_i) in zip(self.unit_normals[sd], Xi)
        ]

        # Compute n times Xi
        if sd.dim == 3:
            R_Xi = sps.block_array(
                [
                    [None, -nz, ny],
                    [nz, None, -nx],
                    [-ny, nx, None],
                ]
            )
        else:  # 2D
            if map_from_u:  # Maps from u to r
                R_Xi = sps.hstack([-ny, nx])
            else:  # Maps from r to u
                R_Xi = sps.vstack([ny, -nx])

        # Compute n cdot Xi
        if map_from_u:  # Maps from u to p
            n_Xi = sps.hstack([nx, ny, nz][: sd.dim])
        else:  # Maps from p to u
            n_Xi = sps.vstack([nx, ny, nz][: sd.dim])

        return R_Xi, n_Xi

    def assemble_rhs_boundary_terms(self, sd: pg.Grid, data: dict):

        # Preallocation
        rhs = np.empty((3, 2), dtype=sps.sparray)

        # Ingredients with the normal
        R = self.assemble_rot(sd)
        ndot = sps.hstack([sps.diags_array(n_i) for n_i in self.unit_normals[sd]])

        Delta_B = np.tile(-sd.cell_faces.sum(axis=1), sd.dim)

        Xi = self.assemble_xi(sd)
        Xi_B = 1 - np.hstack([Xi_i.sum(axis=1) for Xi_i in Xi])

        Xi_tilde = self.convert_to_xi_tilde_inplace(Xi)
        Xi_tilde_B = 1 - np.hstack([Xi_i.sum(axis=1) for Xi_i in Xi_tilde])

        mu_bar = self.mu_bar_over_delta.ravel()

        dmuk = self.delta_mu_k.ravel()[:, None]
        dmuk_min = np.min(self.delta_mu_k, axis=0)[:, None]

        # Traction terms
        rhs[0, 0] = sps.diags_array(Xi_tilde_B)
        rhs[1, 0] = dmuk * R * Delta_B
        rhs[2, 0] = -dmuk_min * ndot * Delta_B

        # Displacement terms
        rhs[0, 1] = -2 * sps.diags_array(mu_bar * Delta_B)
        rhs[1, 1] = -R * Xi_B
        rhs[2, 1] = ndot * Xi_B

        bcs = self.extract_bcs(sd, data)
        g = np.hstack((bcs.trac.ravel(), bcs.disp.ravel()))

        return -self.div_F(sd) @ sps.block_array(rhs) @ g

    #### Solving
    def assemble_fluid_pressure_source(
        self, sd: pg.Grid, data: dict, w: np.ndarray
    ) -> np.ndarray:
        """
        Assemble the right-hand side for a given isotropic stress field w,
        like a fluid pressure
        """
        lame_lambda = pg.get_cell_data(sd, data, pg.LAME_LAMBDA)
        alpha = pg.get_cell_data(sd, data, pg.BIOT_ALPHA)

        rhs = np.zeros(self.ndof(sd))
        rhs[-self.ndof_components[-1] :] = (
            self.sd.cell_volumes * alpha / lame_lambda * w
        )

        return rhs

    def assemble_body_force(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Assemble the right-hand side for a given body force f(x,y,z)
        """
        rhs_u = pg.PwConstants().interpolate(sd, func)
        rhs_r = np.zeros(self.ndof_components(sd)[1])
        rhs_p = np.zeros(self.ndof_components(sd)[2])

        return np.concatenate((rhs_u, rhs_r, rhs_p))
