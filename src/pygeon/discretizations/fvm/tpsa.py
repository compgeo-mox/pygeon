import numpy as np
import scipy.sparse as sps
import pygeon as pg


def rotation_dim(dim: int) -> int:
    return dim * (dim - 1) // 2


class TPSA:
    def ndof(self, sd: pg.Grid):
        return sd.num_cells * sum([sd.dim, rotation_dim(sd.dim), 1])

    def assemble_elasticity_matrix(self, sd: pg.Grid, data: dict) -> None:
        """
        Assemble the TPSA matrix, given material constants in the data dictionary.
        Also sets the reference pressure, if available.
        """

        # Extract the spring constant
        self.delta_bdry_over_mu = data.get(
            "inv_spring_constant", np.zeros(sd.num_faces)
        )

        # We precompute delta_k^i / mu here
        self.delta_ki_over_mu = self.assemble_delta_ki_over_mu(sd, data[pg.LAME_MU])
        self.dk_mu = self.assemble_delta_mu_k()

        # Generate the first order terms in (3.9)
        M = self.assemble_first_order_terms(data, scale_factor=self.scaling)

        # Precompute the operator that multiplies with the area
        # and applies the divergence
        div_F = sps.csc_array(sd.cell_faces.T) * sd.face_areas

        # Assemble the remaining terms in (3.9)
        A = self.assemble_second_order_terms(sd, div_F, scale_factor=self.scaling)

        # Assemble the matrix from (3.9)
        self.system = sps.csc_array(A - M)

    def assemble_second_order_terms(
        self, sd: pg.Grid, div_F: sps.sparray, scale_factor: float = 1.0
    ) -> sps.sparray:
        """
        Assemble the second-order terms
        """

        # Preallocate the main matrix
        A = np.empty((3, 3), dtype=sps.sparray)

        # Assemble the blocks of (3.7) where
        # A_ij is the block coupling variable i and j.
        mu_bar = self.harmonic_avg()
        A_uu = -2 * div_F * (mu_bar / scale_factor) @ sd.cell_faces
        A[0, 0] = sps.block_diag([A_uu] * sd.dim)

        # The blocks in the first column depend on the averaging operator Xi
        Xi = self.assemble_xi()
        A[1, 0], A[2, 0] = self.assemble_off_diagonal_terms(Xi, div_F, True)

        # The blocks in the first row depend on the complementary operator Xi_tilde
        Xi_tilde = self.convert_to_xi_tilde(Xi)
        A[0, 1], A[0, 2] = self.assemble_off_diagonal_terms(Xi_tilde, div_F, False)

        A[2, 2] = -div_F * (0.5 * self.dk_mu * scale_factor) @ sd.cell_faces

        # Assembly by blocks
        return sps.block_array(A).tocsc()

    def assemble_delta_ki_over_mu(self, sd: pg.Grid, mu: np.ndarray) -> np.ndarray:
        """
        Compute delta_k^i / mu from (2.1) for every physical face-cell pair.
        Boundary conditions are handled later
        """

        faces, cells, orient = self.find_cf

        def compute_delta_ki(indices=slice(None)):
            return np.sum(
                (
                    (
                        sd.face_centers[:, faces[indices]]
                        - sd.cell_centers[:, cells[indices]]
                    )
                    * (orient[indices] * self.unit_normals[:, faces[indices]])
                ),
                axis=0,
            )

        delta_ki = compute_delta_ki()

        # Check if any cell centers are placed outside the cell
        if np.any(delta_ki <= 0):
            print(
                "Moving {} extra-cellular centers to the mean of the nodes".format(
                    np.sum(delta_ki <= 0)
                )
            )
            for cell in cells[delta_ki <= 0]:
                cf_pairs = cells == cell

                # Define a cell-center based on the mean of the nodes
                xyz = sd.xyz_from_active_index(cell)
                sd.cell_centers[:, cell] = np.mean(xyz, axis=1)

                # Recompute the deltas with the updated cell center
                delta_ki[cf_pairs] = compute_delta_ki(cf_pairs)

            if np.any(delta_ki <= 0):
                # Report on the first problematic cell for visual inspection
                first_cell = cells[np.argmax(delta_ki <= 0)]
                ijk = sd.ijk_from_active_index(first_cell)
                glob_ind = sd.global_index(*ijk)

                print(
                    "There are {} extra-cellular centers".format(np.sum(delta_ki <= 0))
                )
                print("Inspect cell with global index {}\n".format(glob_ind))
            else:
                print("Fixed all cell-centers\n")

        return delta_ki / mu[cells]

    def assemble_delta_mu_k(self) -> np.ndarray:
        """
        Compute ( mu_i delta_k^-i + mu_j delta_k^-j)^-1
        for each face k with cells (i,j)

        This is double the delta^mu_k of (3.5)
        """

        faces, _, _ = self.find_cf
        dk_mu = self.delta_ki_over_mu

        # Incorporate the spring bc by extending the vectors
        faces = np.concatenate((faces, np.flatnonzero(self.sd.tags["sprng_bdry"])))
        dk_mu = np.concatenate(
            (dk_mu, self.delta_bdry_over_mu[self.sd.tags["sprng_bdry"]])
        )

        # Compute the reciprocal
        mu_dk = np.empty_like(dk_mu)
        positive_del = dk_mu != 0

        mu_dk[positive_del] = 1 / dk_mu[positive_del]
        mu_dk[~positive_del] = np.inf

        delta_mu_k = 1 / np.bincount(faces, weights=mu_dk)

        # Displacement boundaries have infinite mu/delta
        # Traction bc are handled naturally because mu/delta = 0 there.

        return delta_mu_k

    def harmonic_avg(self) -> np.ndarray:
        """
        Compute the harmonic average of mu from (3.5), divided by delta_k, at each face
        """

        faces, _, _ = self.find_cf
        dk_mu = self.delta_ki_over_mu

        # Incorporate the spring bc by extending the vectors
        faces = np.concatenate((faces, np.flatnonzero(self.sd.tags["sprng_bdry"])))
        dk_mu = np.concatenate(
            (dk_mu, self.delta_bdry_over_mu[self.sd.tags["sprng_bdry"]])
        )

        mu_bar_over_dk = 1 / (np.bincount(faces, weights=dk_mu))

        # Traction bc
        mu_bar_over_dk[self.sd.tags["free_bdry"]] = 0

        # Displacement bc are handled naturally as a subset of spring_bdry
        # with zero (inverse) spring constant

        return mu_bar_over_dk

    def assemble_xi(self) -> sps.sparray:
        """
        Compute the averaging operator Xi from (2.5)
        """

        faces, cells, _ = self.find_cf
        Xi = sps.csc_array((self.dk_mu[faces] / self.delta_ki_over_mu, (faces, cells)))

        # Displacement bc are handled by dk_mu = 0
        # Traction bc are handled since dk_mu * mu / delta_ki = 1
        # Spring bc are handled because the spring constant is in dk_mu

        return Xi

    def convert_to_xi_tilde(self, Xi: sps.sparray) -> sps.sparray:
        """
        Compute the converse averaging operator Xi_tilde from (2.6)
        This is an in-place operation to save memory
        """

        Xi.data = 1 - Xi.data
        return Xi

    def assemble_off_diagonal_terms(
        self,
        sd: pg.Grid,
        Xi: sps.sparray,
        div_F: sps.sparray,
        map_from_u: bool = True,
    ) -> tuple[sps.sparray, sps.sparray]:
        nx, ny, nz = [(div_F * ni) @ Xi for ni in self.unit_normals]
        """
        Assemble the off-diagonal terms n times Xi and n cdot Xi in (3.7)
        These are computed together because their construction uses similar components.
        """

        # Compute n times Xi
        if sd.dim == 3:
            R_Xi = -sps.block_array(
                [
                    [None, -nz, ny],
                    [nz, None, -nx],
                    [-ny, nx, None],
                ]
            )
        else:  # 2D
            if map_from_u:  # Maps from u to r
                R_Xi = -sps.hstack([-ny, nx])
            else:  # Maps from r to u
                R_Xi = -sps.vstack([ny, -nx])

        # Compute n cdot Xi
        if map_from_u:  # Maps from u to p
            n_Xi = sps.hstack([nx, ny, nz][: sd.dim])
        else:  # Maps from p to u
            n_Xi = sps.vstack([nx, ny, nz][: sd.dim])

        return R_Xi, n_Xi

    def assemble_first_order_terms(
        self, data: dict, scale_factor: float = 1.0
    ) -> sps.csc_array:
        """
        The first-order terms on the diagonal of (3.9)
        """

        volumes = self.sd.cell_volumes
        M_u = np.zeros(self.ndofs[0])
        M_r = np.tile(scale_factor * volumes / data[pg.LAME_MU], self.dim_r)
        M_p = scale_factor * volumes / data[pg.LAME_LAMBDA]

        diagonal = np.concatenate((M_u, M_r, M_p))

        return sps.diags_array(diagonal).tocsc()

    def assemble_isotropic_stress_source(self, data: dict, w: np.ndarray) -> np.ndarray:
        """
        Assemble the right-hand side for a given isotropic stress field w,
        like a fluid pressure
        """

        rhs_u = np.zeros(self.ndofs[0])
        rhs_r = np.zeros(self.ndofs[1])
        rhs_p = self.sd.cell_volumes * data["alpha"] / data[pg.LAME_LAMBDA] * w

        return np.concatenate((rhs_u, rhs_r, rhs_p))

    def assemble_gravity_force(self, data: dict) -> np.ndarray:
        """
        Assemble a gravity force.

        This function is not used in the main simulations because we assume that the
        medium is in equilibrium at initial time
        """

        w = np.zeros(self.ndofs[0])
        indices_uz = np.arange(
            (self.sd.dim - 1) * self.sd.num_cells, self.sd.dim * self.sd.num_cells
        )
        w[indices_uz] = data["gravity"]

        return self.assemble_body_force(w)

    def assemble_body_force(self, f: np.ndarray) -> np.ndarray:
        """
        Assemble the right-hand side for a given body force f(x,y,z)
        We assume that these are numbered as
            f[0]   = f_x(x_0)
            f[1]   = f_x(x_1)
            ...
            f[n_c] = f_y(x_0)
            ...
        """
        rhs_u = np.tile(self.sd.cell_volumes, self.sd.dim) * f
        rhs_r = np.zeros(self.ndofs[1])
        rhs_p = np.zeros(self.ndofs[2])

        return np.concatenate((rhs_u, rhs_r, rhs_p))

    def solve(
        self,
        data: dict,
        pressure_source: np.ndarray,
        solver,
        body_force=None,
    ) -> tuple[np.ndarray]:
        """
        Solve the system, using the scaling given in self.scaling
        """
        diff_pressure = pressure_source - self.ref_pressure
        rhs = self.assemble_isotropic_stress_source(data, diff_pressure)

        if body_force is not None:
            rhs += self.assemble_body_force(body_force)

        scale_scalar = data.get("scaling", 1.0)
        scale_vector = np.concatenate(
            (
                np.full(self.ndofs[0], 1 / np.sqrt(scale_scalar)),
                np.full(self.ndofs[1] + self.ndofs[2], np.sqrt(scale_scalar)),
            )
        )
        rhs *= scale_vector

        sol, info = solver.solve(rhs)
        assert info == 0, "Solver was unsuccessful"

        sol *= scale_vector
        u, r, p = np.split(sol, np.cumsum(self.ndofs)[:-1])

        return u, r, p

    def solve_body_force(
        self,
        data: dict,
        body_force: np.ndarray,
        solver,
    ) -> tuple[np.ndarray]:
        """
        Solve the system, using the scaling given in self.scaling
        """

        rhs = self.assemble_body_force(body_force)

        scale_scalar = data.get("scaling", 1.0)
        scale_vector = np.concatenate(
            (
                np.full(self.ndofs[0], 1 / np.sqrt(scale_scalar)),
                np.full(self.ndofs[1] + self.ndofs[2], np.sqrt(scale_scalar)),
            )
        )
        rhs *= scale_vector

        sol, info = solver.solve(rhs)
        assert info == 0, "Solver was unsuccessful"

        sol *= scale_vector
        u, r, p = np.split(sol, np.cumsum(self.ndofs)[:-1])

        return u, r, p

    def recover_volumetric_change(
        self, solid_p: np.ndarray, fluid_p: np.ndarray, data: dict
    ) -> np.ndarray:
        """
        Post-process the volumetric change fromt he solid and fluid pressures
        """

        return (solid_p + data["alpha"] * (fluid_p - self.ref_pressure)) / data[
            pg.LAME_LAMBDA
        ]

    def assemble_dual_var_map(self, scale_factor: float = 1.0) -> sps.sparray:
        """
        Assemble the matrix from (3.7) that maps primary to dual variables
        This function is only used for post-processing the stress
        """

        # Precompute the operator that multiplies with the area
        areas = sps.diags_array(self.sd.face_areas)

        return self.assemble_second_order_terms(areas, scale_factor)
