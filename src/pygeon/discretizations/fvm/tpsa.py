import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


def rotation_dim(dim: int) -> int:
    return dim * (dim - 1) // 2


class TPSA:
    def __init__(self, keyword=pg.UNITARY_DATA) -> None:
        self.keyword = keyword

    def ndofs(self, sd: pg.Grid) -> np.ndarray:
        ndof_per_cell = np.array([sd.dim, rotation_dim(sd.dim), 1])
        return sd.num_cells * ndof_per_cell

    def ndof(self, sd) -> int:
        return self.ndofs(sd).sum()

    def assemble_elasticity_matrix(self, sd: pg.Grid, data: dict) -> None:
        """
        Assemble the TPSA matrix, given material constants in the data dictionary.
        """

        # See if there is already a BoundaryConditions object in the data dict
        if "bcs" in data[pp.PARAMETERS][self.keyword]:
            self.bcs = data[pp.PARAMETERS][self.keyword]["bcs"]
        else:  # We create a default one
            self.bcs = pg.TPSA_BC(sd, data, self.keyword)

        # Precomputations of the geometry
        self.find_cf = sps.find(sd.cell_faces)
        self.unit_normals = sd.face_normals / sd.face_areas

        # Extract the Lamé parameters
        lame_mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        lame_lambda = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        # Generate the zero'th order terms in (3.9)
        M = self.assemble_mass_terms(sd, lame_mu, lame_lambda)

        # Precomputations that use grid and parameter values
        self.compute_weighted_dists(sd, lame_mu)

        faces, deltas = self.extend_faces_and_distances(sd)
        self.compute_delta_mu_k(faces, deltas)
        self.compute_harmonic_avg_mu(faces, deltas)

        # Precompute the operator that multiplies with the area
        # and applies the divergence
        div_F = pg.div(sd) * sd.face_areas
        div_F = sps.kron(np.eye(7), div_F)

        # Assemble the remaining terms in (3.9)
        A = div_F @ self.assemble_dual_var_map(sd)

        # Assemble the matrix from (3.9)
        self.system = sps.csc_array(A - M)

        return self.system

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
        A_uu = [-2 * mu_bar[:, None] * sd.cell_faces for mu_bar in self.mu_bar]
        A[0, 0] = sps.block_diag(A_uu)

        # Assemble the boundary terms of (A2.25)
        A[1, 1] = self.assemble_boundary_terms(sd)

        # The blocks in the first column depend on the averaging operator Xi
        Xi = self.assemble_xi()
        A[1, 0], A[2, 0] = self.assemble_off_diagonal_terms(sd, Xi, True)

        # The blocks in the first row depend on the complementary operator Xi_tilde
        Xi_tilde = self.convert_to_xi_tilde(Xi)
        A[0, 1], A[0, 2] = self.assemble_off_diagonal_terms(sd, Xi_tilde, False)

        delta_diff = 0.5 * np.min(self.delta_mu_k, axis=0)
        A[2, 2] = -delta_diff[:, None] * sd.cell_faces

        # Assembly by blocks
        return sps.block_array(A).tocsc()

    def assemble_boundary_terms(self, sd):

        nx, ny, nz = [sps.diags_array(n_i) for n_i in self.unit_normals]

        R = sps.block_array(
            [
                [None, -nz, ny],
                [nz, None, -nx],
                [-ny, nx, None],
            ]
        )
        R_squared = R @ R

        bdry_deltas = 0.5 * self.delta_mu_k * sd.tags["domain_boundary_faces"]
        delta = bdry_deltas.flatten()

        codiv = sps.kron(sps.eye_array(3), sd.cell_faces)
        return -delta[:, None] * R_squared @ codiv

    def compute_weighted_dists(self, sd: pg.Grid, lame_mu: np.ndarray) -> np.ndarray:
        """
        Compute delta_k^i / mu from (2.1) for every physical face-cell pair.
        Boundary conditions are handled later.
        """
        faces, cells, orient = self.find_cf

        def compute_delta(indices=slice(None)):
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

        delta = compute_delta()

        # Check if any cell centers are placed outside the cell
        if np.any(delta <= 0):
            print(
                "Moving {} extra-cellular centers to the mean of the nodes".format(
                    np.sum(delta <= 0)
                )
            )
            cell_nodes = sd.cell_nodes()
            cf_pairs = np.zeros_like(cells, dtype=bool)

            for cell in cells[delta <= 0]:
                cf_pairs |= cells == cell

                # Define a cell-center based on the mean of the nodes
                nodes = cell_nodes.indices[
                    cell_nodes.indptr[cell] : cell_nodes.indptr[cell + 1]
                ]
                sd.cell_centers[:, cell] = np.mean(sd.nodes[:, nodes], axis=1)

            # Recompute the deltas with the updated cell center
            delta[cf_pairs] = compute_delta(cf_pairs)

            if np.any(delta <= 0):
                # Report on the first problematic cell for visual inspection
                glob_ind = cells[delta <= 0]

                print("There are {} extra-cellular centers".format(np.sum(delta <= 0)))
                print("Inspect cells with index {}".format(glob_ind))
            else:
                print("Fixed all cell-centers")

        self.weighted_dists = delta / lame_mu[cells]

    def extend_faces_and_distances(self, sd):
        # Incorporate the spring bc by extending the vectors
        faces = self.find_cf[0]

        bdry_faces = sd.tags["domain_boundary_faces"]
        ext_faces = np.concatenate((faces, np.flatnonzero(bdry_faces)))

        # We allow for different types of boundary conditions in the x, y, z directions.
        # We therefore have three instances of the distances, one for each direction.
        tiled_dists = np.tile(self.weighted_dists, (pg.AMBIENT_DIM, 1))
        ext_dists = np.hstack((tiled_dists, self.bcs.weighted_dists[:, bdry_faces]))

        return ext_faces, ext_dists

    def compute_delta_mu_k(self, faces, dists) -> None:
        """
        Compute ( mu_i delta_k^-i + mu_j delta_k^-j)^-1
        for each face k with cells (i,j)

        This is double the delta^mu_k of (3.5)
        """
        # Compute the reciprocal
        inv_dists = np.empty_like(dists)
        positive_dist = dists != 0

        inv_dists[positive_dist] = 1 / dists[positive_dist]
        inv_dists[~positive_dist] = np.inf

        # Displacement boundaries have infinite mu/delta
        # Traction bc are handled naturally because mu/delta = 0 there.
        output_list = [1 / np.bincount(faces, weights=row) for row in inv_dists]
        self.delta_mu_k = np.array(output_list)

    def compute_harmonic_avg_mu(self, faces, dists) -> np.ndarray:
        """
        Compute the harmonic average of mu from (3.5), divided by delta_k, at each face
        """
        # Displacement bc are handled naturally as a subset of spring_bdry
        # with zero (inverse) spring constant.
        # Tractions are handled with infinite spring constant.
        output_list = [1 / np.bincount(faces, weights=row) for row in dists]
        self.mu_bar = np.array(output_list)

    def assemble_xi(self) -> list:
        """
        Compute the averaging operator Xi from (2.5)
        """
        faces, cells, _ = self.find_cf
        Xi = [
            sps.csc_array((delta[faces] / self.weighted_dists, (faces, cells)))
            for delta in self.delta_mu_k
        ]

        # Displacement bc are handled by dk_mu = 0
        # Traction bc are handled since dk_mu * mu / delta = 1
        # Spring bc are handled because the spring constant is in dk_mu

        return Xi

    def convert_to_xi_tilde(self, Xi: list) -> sps.sparray:
        """
        Compute the converse averaging operator Xi_tilde from (2.6)
        This is an in-place operation to save memory
        """
        for Xi_i in Xi:
            Xi_i.data = 1 - Xi_i.data
        return Xi

    def assemble_off_diagonal_terms(
        self,
        sd: pg.Grid,
        Xi: sps.sparray,
        map_from_u: bool = True,
    ) -> tuple[sps.sparray, sps.sparray]:
        """
        Assemble the off-diagonal terms n times Xi and n cdot Xi in (3.7)
        These are computed together because their construction uses similar components.
        """
        nx, ny, nz = [ni[:, None] * Xi_i for (ni, Xi_i) in zip(self.unit_normals, Xi)]

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

    #### Solving
    def assemble_fluid_pressure_source(
        self, sd: pg.Grid, data: dict, w: np.ndarray
    ) -> np.ndarray:
        """
        Assemble the right-hand side for a given isotropic stress field w,
        like a fluid pressure
        """

        rhs = np.zeros(self.ndof(sd))
        rhs[-self.ndofs[-1] :] = (
            self.sd.cell_volumes * data["alpha"] / data[pg.LAME_LAMBDA] * w
        )

        return rhs

    def assemble_gravity_force(self, sd: pg.Grid, data: dict) -> np.ndarray:
        """
        Assemble a gravity force.

        This function is not used in the main simulations because we assume that the
        medium is in equilibrium at initial time
        """

        w = np.zeros(self.ndofs(sd)[0])
        indices_uz = np.arange(
            (self.sd.dim - 1) * self.sd.num_cells, self.sd.dim * self.sd.num_cells
        )
        w[indices_uz] = data["gravity"]

        return self.assemble_body_force(w)

    def assemble_body_force(self, sd: pg.Grid, f: np.ndarray) -> np.ndarray:
        """
        Assemble the right-hand side for a given body force f(x,y,z)
        We assume that these are numbered as
            f[0]   = f_x(x_0)
            f[1]   = f_x(x_1)
            ...
            f[n_c] = f_y(x_0)
            ...
        """
        rhs_u = np.tile(sd.cell_volumes, sd.dim) * f
        rhs_r = np.zeros(self.ndofs(sd)[1])
        rhs_p = np.zeros(self.ndofs(sd)[2])

        return np.concatenate((rhs_u, rhs_r, rhs_p))

    def solve(
        self,
        sd: pg.Grid,
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
                np.full(self.ndofs(sd)[0], 1 / np.sqrt(scale_scalar)),
                np.full(self.ndofs(sd)[1] + self.ndofs(sd)[2], np.sqrt(scale_scalar)),
            )
        )
        rhs *= scale_vector

        sol, info = solver.solve(rhs)
        assert info == 0, "Solver was unsuccessful"

        sol *= scale_vector
        u, r, p = np.split(sol, np.cumsum(self.ndofs(sd))[:-1])

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

    # def assemble_dual_var_map(self) -> sps.sparray:
    #     """
    #     Assemble the matrix from (3.7) that maps primary to dual variables
    #     This function is only used for post-processing the stress
    #     """

    #     # Precompute the operator that multiplies with the area
    #     areas = sps.diags_array(self.sd.face_areas)

    #     return self.dual_var_map(areas)


class TPSA_BC:
    def __init__(self, sd: pg.Grid, data: dict, keyword: str):
        self.weighted_dists = np.zeros((pg.AMBIENT_DIM, sd.num_faces))
        self.disp = np.zeros_like(self.weighted_dists)
        self.trac = np.zeros_like(self.weighted_dists)

        data[pp.PARAMETERS][keyword].update({"bcs": self})

    def _set_bcs(
        self,
        indices: np.ndarray | None,
        input: np.ndarray | None,
        internal_var: np.ndarray,
        dist: np.ndarray | float,
    ):
        if input is None:
            input = np.zeros_like(self.weighted_dists)
        if indices is None:
            indices = np.zeros_like(self.weighted_dists, dtype=bool)

        assert input.shape == self.weighted_dists.shape, (
            "Input must be of shape (3, num_faces)"
        )

        internal_var[indices] = input[indices]

        if isinstance(dist, float):
            self.weighted_dists[indices] = dist
        else:
            self.weighted_dists[indices] = dist[indices]

    def set_displacement_bcs(
        self,
        indices: np.ndarray | None = None,
        u_0: np.ndarray | None = None,
    ):
        self._set_bcs(indices, u_0, self.disp, 0)

    def set_traction_bcs(
        self,
        indices: np.ndarray | None = None,
        sig_0: np.ndarray | None = None,
    ):
        self._set_bcs(indices, sig_0, self.trac, np.inf)

    def set_spring_bcs(
        self,
        dists: np.ndarray,
        indices: np.ndarray | None = None,
        u_0: np.ndarray | None = None,
    ):
        self._set_bcs(indices, u_0, self.disp, dists)
