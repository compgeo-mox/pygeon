from typing import Callable, cast

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class TPSA(pg.FiniteVolumeDiscretization):
    """
    An implementation of the two-point stress approximation method of Nordbotten and
    Keilegavlen (2025)

    Equation numbers refer to the manuscript:
    https://doi.org/10.1016/j.camwa.2025.07.035
    """

    def __init__(self, keyword=pg.UNITARY_DATA) -> None:
        super().__init__(keyword)
        self.bc_type = pg.ElasticityBC

        self.delta_mu_k: dict[pg.Grid, np.ndarray] = {}
        self.mu_bar: dict[pg.Grid, np.ndarray] = {}

    def ndof_per_cell(self, sd: pg.Grid) -> int:
        return sd.dim + rotation_dim(sd.dim) + 1

    def interpolate(
        self,
        sd: pg.Grid,
        displacement: Callable,
        rotation: Callable,
        solid_pressure: Callable,
    ) -> np.ndarray:
        u = pg.VecPwConstants().interpolate(sd, displacement)
        r = pg.get_PwPolynomials(0, sd.dim - 2)().interpolate(sd, rotation)
        p = pg.PwConstants().interpolate(sd, solid_pressure)

        interp = np.hstack((u, r, p))

        return interp / np.tile(sd.cell_volumes, self.ndof_per_cell(sd))

    def assemble_elasticity_matrix(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assemble the TPSA matrix, given material constants in the data dictionary.
        """

        # Extract the Lamé parameters
        lame_mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        lame_lambda = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        # Precomputations without boundary conditions
        self.fvm_precomputations(sd, lame_mu)

        #
        faces, deltas = self.extend_faces_and_distances(sd, data)

        self.compute_delta_mu_k(sd, faces, deltas)
        self.compute_harmonic_avg(sd, faces, deltas)

        # Assemble the remaining terms in (3.9)
        A = self.div(sd) @ self.assemble_dual_var_map(sd)

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

    def assemble_dual_var_map(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assemble the second-order terms
        """
        # Preallocate the main matrix
        A = np.empty((3, 3), dtype=sps.sparray)

        # Assemble the blocks of (3.7) where
        # A_ij is the block coupling variable i and j.
        A_uu = [-2 * mu_bar[:, None] * sd.cell_faces for mu_bar in self.mu_bar[sd]]
        A[0, 0] = sps.block_diag(A_uu)

        # Assemble the boundary terms of (A2.25)
        A[1, 1] = self.assemble_rot_rot_bdry_terms(sd)

        # The blocks in the first column depend on the averaging operator Xi
        Xi = self.assemble_xi(sd)
        R_Xi, n_Xi = self.assemble_first_column(sd, Xi)
        A[1, 0] = -R_Xi
        A[2, 0] = n_Xi

        # The blocks in the first row depend on the complementary operator Xi_tilde
        Xi_tilde = self.convert_to_xi_tilde_inplace(Xi)
        R_Xi_t, n_Xi_t = self.assemble_first_row(sd, Xi_tilde)
        A[0, 1] = -R_Xi_t
        A[0, 2] = n_Xi_t

        # Stabilization for the pressure
        delta_n = np.sum(
            self.unit_normals[sd][: sd.dim] ** 2 * self.delta_mu_k[sd], axis=0
        )
        A[2, 2] = -delta_n[:, None] * sd.cell_faces

        # Scaling by the face areas
        f_areas = self.face_area_scaling(sd)[:, None]

        # Assembly by blocks
        return f_areas * sps.block_array(A).tocsc()

    def assemble_rot(self, sd: pg.Grid) -> sps.csc_array:
        nx, ny, nz = [sps.diags_array(n_i) for n_i in self.unit_normals[sd]]

        match sd.dim:
            case 3:
                return sps.block_array(
                    [
                        [None, -nz, ny],
                        [nz, None, -nx],
                        [-ny, nx, None],
                    ]
                ).tocsc()
            case 2:
                return sps.hstack([-ny, nx])
            case _:
                raise ValueError("The dimension must be 2 or 3.")

    def assemble_ndot(self, sd: pg.Grid) -> sps.csc_array:
        return sps.hstack(
            [sps.diags_array(n_i) for n_i in self.unit_normals[sd][: sd.dim]]
        )

    def assemble_rot_rot_bdry_terms(self, sd: pg.Grid) -> sps.sparray:
        if sd.dim == 2:
            return None

        bdry_deltas = self.delta_mu_k[sd] * sd.tags["domain_boundary_faces"]
        delta = bdry_deltas.flatten()

        R = self.assemble_rot(sd)
        R_squared = (R * delta) @ R

        codiv = sps.kron(sps.eye_array(sd.dim), sd.cell_faces)
        return -R_squared @ codiv

    def extend_faces_and_distances(
        self,
        sd: pg.Grid,
        data: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        bcs = self.extract_bcs(sd, data)

        # Incorporate the Robin bc by extending the vectors
        faces, *_ = self.find_cf[sd]

        bdry_faces = sd.tags["domain_boundary_faces"]
        ext_faces = np.concatenate((faces, np.flatnonzero(bdry_faces)))

        # We allow for different types of boundary conditions in the x, y, z directions.
        # We therefore have three instances of the distances, one for each direction.
        tiled_dists = np.tile(self.weighted_dists[sd], (sd.dim, 1))
        ext_dists = np.hstack((tiled_dists, bcs.weighted_dists[: sd.dim, bdry_faces]))

        return ext_faces, ext_dists

    def compute_delta_mu_k(
        self, sd: pg.Grid, faces: np.ndarray, dists: np.ndarray
    ) -> None:
        """
        Compute 0.5 * ( mu_i delta_k^-i + mu_j delta_k^-j)^-1 for each face k with
        neighboring cells (i,j)

        This is the delta^mu_k of (3.5).
        """
        # Compute the reciprocal
        inv_dists = np.empty_like(dists)
        zero_dist = dists == 0

        inv_dists[~zero_dist] = 1 / dists[~zero_dist]
        inv_dists[zero_dist] = np.inf

        # Displacement boundaries have infinite mu/delta
        # Traction bc are handled naturally because mu/delta = 0 there.
        output_list = [1 / np.bincount(faces, weights=row) for row in inv_dists]
        self.delta_mu_k[sd] = np.array(output_list) / 2

    def compute_harmonic_avg(
        self, sd: pg.Grid, faces: np.ndarray, dists: np.ndarray
    ) -> None:
        """
        Compute the harmonic average of mu from (3.5), divided by delta_k, at each face
        """
        # Displacement bc are handled naturally as a subset of spring_bdry
        # with zero (inverse) spring constant.
        # Tractions are handled with infinite spring constant.
        output_list = [1 / np.bincount(faces, weights=row) for row in dists]
        self.mu_bar[sd] = np.array(output_list)

    def assemble_xi(self, sd: pg.Grid) -> list:
        """
        Compute the averaging operator Xi from (2.5)
        """
        faces, cells, _ = self.find_cf[sd]
        Xi = [
            sps.csc_array((2 * delta[faces] / self.weighted_dists[sd], (faces, cells)))
            for delta in self.delta_mu_k[sd]
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

    def assemble_first_column(
        self,
        sd: pg.Grid,
        Xi_list: list,
    ) -> tuple[sps.sparray, sps.sparray]:
        """
        Assemble the off-diagonal terms in the first column of (3.7)
        """
        Xi = sps.block_diag(Xi_list)
        R_Xi = self.assemble_rot(sd) @ Xi
        n_Xi = self.assemble_ndot(sd) @ Xi

        return R_Xi, n_Xi

    def assemble_first_row(
        self,
        sd: pg.Grid,
        Xi_list: list,
    ) -> tuple[sps.sparray, sps.sparray]:
        """
        Assemble the off-diagonal terms in the first row of (3.7)
        These are computed together because their construction uses similar components.
        """
        nx, ny, nz = [ni[:, None] for ni in self.unit_normals[sd]]

        match sd.dim:
            case 3:
                Xx, Xy, Xz = Xi_list
                R_Xi = sps.block_array(
                    [
                        [None, -nz * Xx, ny * Xx],
                        [nz * Xy, None, -nx * Xy],
                        [-ny * Xz, nx * Xz, None],
                    ]
                )
                n_Xi = sps.vstack([nx * Xx, ny * Xy, nz * Xz])
            case 2:
                Xx, Xy = Xi_list
                R_Xi = sps.vstack([ny * Xx, -nx * Xy])
                n_Xi = sps.vstack([nx * Xx, ny * Xy])

        return R_Xi, n_Xi

    def assemble_rhs_boundary_terms(self, sd: pg.Grid, data: dict):

        assert sd in self.unit_normals, (
            "The elasticity matrix has to be assembled first."
        )

        # Preallocation
        rhs = np.empty((3, 2), dtype=sps.sparray)

        # Ingredients with the normal
        R = self.assemble_rot(sd)
        ndot = self.assemble_ndot(sd)

        Delta_B = np.tile(-sd.cell_faces.sum(axis=1), sd.dim)

        Xi = self.assemble_xi(sd)
        Xi_B = 1 - np.hstack([Xi_i.sum(axis=1) for Xi_i in Xi])

        Xi_tilde = self.convert_to_xi_tilde_inplace(Xi)
        Xi_tilde_B = 1 - np.hstack([Xi_i.sum(axis=1) for Xi_i in Xi_tilde])

        mu_bar = self.mu_bar[sd].ravel()
        dmuk = self.delta_mu_k[sd].ravel()

        # Traction terms
        rhs[0, 0] = sps.diags_array(Xi_tilde_B)
        rhs[1, 0] = R * dmuk * Delta_B
        rhs[2, 0] = -ndot * dmuk * Delta_B

        # Displacement terms
        rhs[0, 1] = -2 * sps.diags_array(mu_bar * Delta_B)
        rhs[1, 1] = -R * Xi_B
        rhs[2, 1] = ndot * Xi_B

        bcs = self.extract_bcs(sd, data)
        bcs = cast(pg.ElasticityBC, bcs)

        trac = bcs.dual_var[: sd.dim] / sd.face_areas
        disp = bcs.primary_var[: sd.dim]

        g = np.hstack((trac.ravel(), disp.ravel()))
        f_areas = self.face_area_scaling(sd)

        return -self.div(sd) * f_areas @ sps.block_array(rhs) @ g

    def assemble_body_force(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Assemble the right-hand side for a given body force f(x,y,z)
        """
        rhs = np.zeros(self.ndof(sd))
        rhs[: sd.dim * sd.num_cells] = -pg.VecPwConstants().interpolate(sd, func)

        return rhs

    def split_solution(self, sd: pg.Grid, sol: np.ndarray):
        ndofs = sd.num_cells * np.array([sd.dim, rotation_dim(sd.dim)])

        return np.split(sol, np.cumsum(ndofs))


def rotation_dim(dim: int) -> int:
    return dim * (dim - 1) // 2
