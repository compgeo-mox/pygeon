"""Module for two-point stress approximation discretization."""

from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class TPSA(pg.FiniteVolumeDiscretization):
    """
    A vectorized implementation of the two-point stress approximation method for
    elasticity of Nordbotten and Keilegavlen (2025).

    Degrees of freedom are given by the cell center values and the variables are ordered
    as [ux, uy, uz, rx, ry, rz, p], with the cell index varying fastest.

    Our implementation differs from Porepy (v1.13) in the imposition of boundary
    conditions. In particular, we made the following changes:
        - The order of R and Xi_tilde in the [0, 1] block of (A.2.25)
        - The order of n and Xi_tilde in the [0, 2] block of (A.2.25)
        - Changed "delta R^2" to "R delta R" in the [1, 1] block of (A.2.25)
        - Used the delta^mu_k in the normal direction in the [2, 2] block of (A.2.25)

    These adaptations are necessary for consistency with rolling boundary conditions.
    We moreover used signed distances between face and cell centers.

    Equation numbers in the comments and docstrings refer to the manuscript:
    https://doi.org/10.1016/j.camwa.2025.07.035
    """

    def __init__(self, keyword=pg.UNITARY_DATA) -> None:
        """
        Initialize the TPSA object.

        Args:
            keyword (str): The keyword used to identify the discretization method.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.bc_type = pg.ElasticityBC

    def ndof_per_cell(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom per cell.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            int: The number of degrees of freedom per cell.
        """
        return sd.dim + rotation_dim(sd.dim) + 1

    def interpolate(
        self,
        sd: pg.Grid,
        displacement: Callable[[np.ndarray], np.ndarray],
        rotation: Callable[[np.ndarray], np.ndarray],
        solid_pressure: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Interpolate a triplet of functions onto the finite volume space.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            displacement (Callable): A function that returns the displacement values
                at coordinates.
            rotation (Callable): A function that returns the rotation values at
                coordinates.
            solid_pressure (Callable): A function that returns the solid pressure values
                at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom
        """
        u = pg.VecPwConstants().interpolate(sd, displacement)
        r = pg.get_PwPolynomials(0, sd.dim - 2)().interpolate(sd, rotation)
        p = pg.PwConstants().interpolate(sd, solid_pressure)

        interp = np.hstack((u, r, p))

        return interp / np.tile(sd.cell_volumes, self.ndof_per_cell(sd))

    def assemble_accumulation_terms(
        self, sd: pg.Grid, data: dict | None
    ) -> sps.csc_array:
        """
        Assemble the zeroth-order terms on the diagonal of (3.9).

        Args:
            sd (pg.Grid): Grid, or a subclass.
            lame_mu (np.ndarray): The Lamé parameter mu
            lame_lambda (np.ndarray): The Lamé parameter lambda

        Returns:
            sps.csc_array: The diagonal mass matrix
        """
        # Extract the Lamé parameters from the data dictionary
        lame_mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        lame_lambda = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        # Define the diagonal entries
        M_u = np.zeros(sd.dim * sd.num_cells)
        M_r = np.tile(sd.cell_volumes / lame_mu, rotation_dim(sd.dim))
        M_p = sd.cell_volumes / lame_lambda

        diagonal = np.concatenate((M_u, M_r, M_p))

        # Negate, cf. (3.9), and return
        return -sps.diags_array(diagonal).tocsc()

    def precompute_arrays(self, sd: pg.Grid, data: dict | None = None) -> dict:
        """
        Precomputations on the grid for easy access later.

        This function is typically called twice, once for the left-hand side, and once
        for the right.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary.

        Returns:
            dict: The precomputed arrays.
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

        # With the extended face and distance arrays, we can compute two more quantities
        delta_mu_k = self.compute_delta_mu_k(faces, dists)
        mu_effective = self.compute_harmonic_avg(faces, dists)

        # Gather these vectors in a dictionary for easy access in the assembly
        # procedures
        cached_arrays = {
            "find_cell_faces": find_cell_faces,
            "weighted_dists": weighted_dists,
            "delta_mu_k": delta_mu_k,
            "mu_effective": mu_effective,
        }

        return cached_arrays

    def compute_weighted_dists(
        self, sd: pg.Grid, data: dict | None, find_cell_faces: Tuple
    ) -> np.ndarray:
        """
        Computes delta_k^i / mu_i from (2.1) for every physical face-cell pair (k, i).
        Boundary conditions are handled later.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary.
            find_cell_faces (Tuple): Output of scipy.sparse.find on sd.cell_faces.

        Returns:
            np.ndarray: The weighted distances.
        """
        faces, cells, orient = find_cell_faces
        unit_normals = sd.face_normals / sd.face_areas

        # Compute the signed normal distance between face and cell center
        delta = np.sum(
            (
                (sd.face_centers[:, faces] - sd.cell_centers[:, cells])
                * (orient * unit_normals[:, faces])
            ),
            axis=0,
        )

        # Extract the lamé parameter mu
        lame_mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)

        return delta / lame_mu[cells]

    def extend_faces_and_distances(
        self,
        sd: pg.Grid,
        data: dict | None,
        faces: np.ndarray,
        weighted_dists: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Incorporate the boundary conditions by extending the face and distance arrays.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary.
            faces (np.ndarray): The array of face indices
            weighted_dists (np.ndarray): The array of weighted distances

        Returns:
            np.ndarray: The extended array of face indices.
            np.ndarray: The extended array of weighted distances.
        """
        bcs = self.get_bcs_from_data(sd, data)

        # Incorporate the bcs by extending the vectors
        bdry_faces = sd.tags["domain_boundary_faces"]
        ext_faces = np.concatenate((faces, np.flatnonzero(bdry_faces)))

        # We allow for different types of boundary conditions in the x, y, z directions.
        # We therefore have three instances of the distances, one for each direction.
        tiled_dists = np.tile(weighted_dists, (sd.dim, 1))
        ext_dists = np.hstack((tiled_dists, bcs.weighted_dists[: sd.dim, bdry_faces]))

        return ext_faces, ext_dists

    def compute_delta_mu_k(self, faces: np.ndarray, dists: np.ndarray) -> np.ndarray:
        """
        Compute the delta^mu_k of (3.5) given by
        0.5 * ( mu_i delta_k^-i + mu_j delta_k^-j)^-1
        for each face k with neighboring cells (i,j).

        Args:
            faces (np.ndarray): The extended array of faces.
            dists (np.ndarray): The extended array of weighted distances.

        Returns:
            np.ndarray: The array of $\delta_k^\mu$.
        """
        # Compute the reciprocal
        inv_dists = np.empty_like(dists)
        zero_dist = dists == 0

        # Traction bc are handled naturally because mu/delta = 0 there.
        inv_dists[~zero_dist] = 1 / dists[~zero_dist]

        # Displacement boundaries have zero delta, so infinite mu/delta
        inv_dists[zero_dist] = np.inf

        # Do a reciprocal on the bincount, this results in nonnegative, bounded values
        output_list = [1 / np.bincount(faces, weights=row) for row in inv_dists]

        return np.array(output_list) / 2

    def compute_harmonic_avg(self, faces: np.ndarray, dists: np.ndarray) -> np.ndarray:
        """
        Compute the harmonic average of mu from (3.5), divided by delta_k, at each face:
        mu_effective = ( delta_k^i / mu_i + delta_k^j / mu_j)^-1

        Args:
            faces (np.ndarray): The extended array of faces.
            dists (np.ndarray): The extended array of weighted distances.

        Returns:
            np.ndarray: The face-wise harmonic average of $\mu$.
        """
        output_list = [1 / np.bincount(faces, weights=row) for row in dists]
        return np.array(output_list)

    def assemble_dual_var_map(self, sd: pg.Grid, data: dict | None) -> sps.csc_array:
        """
        Assemble the mapping from cell-based primary variables to face-based dual
        variables.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: The matrix mapping primary to dual variables
        """
        # Preallocate the block matrix
        A = np.empty((3, 3), dtype=sps.csc_array)
        cached_arrays = self.precompute_arrays(sd, data)

        # Assemble the blocks of (3.7) where A_ij is the block coupling variable i and
        # j. The canonical order of the variables is [u, r, p]
        mu_effective = cached_arrays["mu_effective"]
        A_uu = [-2 * mu[:, None] * sd.cell_faces for mu in mu_effective]
        A[0, 0] = sps.block_diag(A_uu, format="csc")

        # Assemble the boundary terms of (A2.25)
        A[1, 1] = self.assemble_rot_rot_bdry_terms(sd, cached_arrays)

        # The blocks in the first column depend on the averaging operator Xi
        Xi = self.assemble_Xi(cached_arrays)
        R_Xi, n_Xi = self.assemble_first_column(sd, Xi)
        A[1, 0] = -R_Xi
        A[2, 0] = n_Xi

        # The blocks in the first row depend on the complementary operator Xi_tilde
        Xi_tilde = self.convert_to_xi_tilde_inplace(Xi)
        R_Xi_t, n_Xi_t = self.assemble_first_row(sd, Xi_tilde)
        A[0, 1] = -R_Xi_t
        A[0, 2] = n_Xi_t

        # Stabilization for the solid pressure
        unit_normals = sd.face_normals[: sd.dim] / sd.face_areas
        delta_n = np.sum(unit_normals**2 * cached_arrays["delta_mu_k"], axis=0)

        # Scale the codivergence (cell-faces) with -delta_n
        A_pp = sd.cell_faces.astype(float).tocsc()
        A_pp.data *= -delta_n[A_pp.indices]
        A[2, 2] = A_pp

        A_csc = sps.block_array(A, format="csc")

        # Scaling by the face areas
        f_areas = self.face_area_scaling(sd)
        A_csc.data *= f_areas[A_csc.indices]

        return A_csc

    def assemble_rot(self, sd: pg.Grid) -> sps.csc_array:
        """
        The operator R^n that performs a cross product with the normal vector n.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: The R^n matrix
        """
        unit_normals = sd.face_normals / sd.face_areas
        nx, ny, nz = [sps.diags_array(n_i) for n_i in unit_normals]

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
                raise ValueError("The grid dimension must be 2 or 3.")

    def assemble_ndot(self, sd: pg.Grid) -> sps.csc_array:
        """
        The operator that performs a dot product with the normal vector n.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: The n cdot matrix
        """
        unit_normals = sd.face_normals / sd.face_areas
        return sps.hstack(
            [sps.diags_array(n_i) for n_i in unit_normals[: sd.dim]], format="csc"
        )

    def assemble_rot_rot_bdry_terms(
        self, sd: pg.Grid, cached_arrays: dict
    ) -> sps.csc_array:
        """
        The operator R^n \delta R^n that is on the [1, 1] block of (A2.25).

        There is a slight discrepancy with the paper, because a simpler class of
        boundary conditions are assumed there. This implementation is the generalization
        to more involved boundary conditions, such as rollers.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            cached_arrays (dict): The output of self.precompute_arrays

        Returns:
            sps.csc_array: The double rotation matrix
        """
        delta_mu_k = cached_arrays["delta_mu_k"]

        # Extract the delta^mu_k on the boundaries
        bdry_deltas = delta_mu_k * sd.tags["domain_boundary_faces"]
        delta = bdry_deltas.flatten()

        R = self.assemble_rot(sd)
        minus_R_squared = (R * delta) @ R.T

        codiv = sps.kron(sps.eye_array(rotation_dim(sd.dim)), sd.cell_faces)
        return (minus_R_squared @ codiv).tocsc()

    def assemble_Xi(self, cached_arrays: dict) -> list:
        """
        Compute the averaging operator Xi from (2.5).

        Displacement bc are handled by delta_mu_k = 0. Traction bc are handled since 2 *
        delta_mu_k * mu / delta = 1. Spring bc are handled because the spring constant
        is contained in delta_mu_k.

        Args:
            cached_arrays (dict): The output of self.precompute_arrays.

        Returns:
            list: The averaging operators in the coordinate directions
        """
        faces, cells, _ = cached_arrays["find_cell_faces"]
        weighted_dists = cached_arrays["weighted_dists"]
        delta_mu_k = cached_arrays["delta_mu_k"]

        Xi = [
            sps.csc_array((2 * delta[faces] / weighted_dists, (faces, cells)))
            for delta in delta_mu_k
        ]

        return Xi

    def convert_to_xi_tilde_inplace(self, Xi: list) -> list:
        """
        Compute the complementary averaging operator Xi_tilde from (2.6).
        NOTE: This is an in-place operation.

        Args:
            Xi (list): The averaging operators in the coordinate directions.

        Returns:
            list: The tilde averaging operators in the coordinate directions
        """
        for Xi_i in Xi:
            Xi_i.data = 1 - Xi_i.data
        return Xi

    def assemble_first_column(
        self,
        sd: pg.Grid,
        Xi_list: list,
    ) -> Tuple[sps.csc_array, sps.csc_array]:
        """
        Assemble the off-diagonal terms in the first column of (3.7). These are computed
        together because their construction uses similar components.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            Xi_list (list): The averaging operators in the coordinate directions

        Returns:
            R_xi (sps.csc_array): The operator that averages and then crosses with n
            n_xi (sps.csc_array): The operator that averages and then dots with n
        """
        Xi = sps.block_diag(Xi_list, format="csc")
        R_Xi = self.assemble_rot(sd) @ Xi
        n_Xi = self.assemble_ndot(sd) @ Xi

        return R_Xi, n_Xi

    def assemble_first_row(
        self,
        sd: pg.Grid,
        Xi_list: list,
    ) -> Tuple[sps.csc_array, sps.csc_array]:
        """
        Assemble the off-diagonal terms in the first row of (3.7). These are computed
        together because their construction uses similar components.

        This is a generalization compared to the paper to handle more involved boundary
        conditions. In particular, we have to change the order of the operators

        Args:
            sd (pg.Grid): Grid, or a subclass.
            Xi_list (list): The tilde averaging operators in the coordinate directions

        Returns:
            R_xi (sps.csc_array): The operator that crosses with n and then averages
            n_xi (sps.csc_array): The operator that multiplies with n and then averages
        """
        unit_normals = sd.face_normals / sd.face_areas
        nx, ny, nz = [ni[:, None] for ni in unit_normals]

        match sd.dim:
            case 3:
                Xx, Xy, Xz = [Xi.tocsr() for Xi in Xi_list]
                R_Xi = sps.block_array(
                    [
                        [None, -nz * Xx, ny * Xx],
                        [nz * Xy, None, -nx * Xy],
                        [-ny * Xz, nx * Xz, None],
                    ],
                    format="csc",
                )
                n_Xi = sps.vstack([nx * Xx, ny * Xy, nz * Xz], format="csc")
            case 2:
                Xx, Xy = [Xi.tocsr() for Xi in Xi_list]
                R_Xi = sps.vstack([ny * Xx, -nx * Xy], format="csc")
                n_Xi = sps.vstack([nx * Xx, ny * Xy], format="csc")

        return R_Xi, n_Xi

    def assemble_bdry_dual_var_map(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assemble the second matrix on the right-hand side of (A2.25).

        Slight generalization: the [1, 0] and [2, 0] blocks first weigh with delta and
        then rot/dot with n.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary

        Returns:
            sps.csc_array: the matrix to be multiplied with the boundary data g
        """
        # Preallocation and precomputation
        A_rhs = np.empty((3, 2), dtype=sps.csc_array)
        cached_arrays = self.precompute_arrays(sd, data)
        delta_mu_k = cached_arrays["delta_mu_k"].ravel()
        mu_effective = cached_arrays["mu_effective"].ravel()

        # Ingredients with the normal
        R = self.assemble_rot(sd)
        ndot = self.assemble_ndot(sd)

        # Extract the sign of the normal on the faces
        Delta_bdry = np.tile(-sd.cell_faces.sum(axis=1), sd.dim)

        # Compute the Xi and Xi_tilde averaging operators for the exterior
        Xi = self.assemble_Xi(cached_arrays)
        Xi_bdry = 1 - np.hstack([Xi_i.sum(axis=1) for Xi_i in Xi])

        Xi_tilde = self.convert_to_xi_tilde_inplace(Xi)
        Xi_tilde_bdry = 1 - np.hstack([Xi_i.sum(axis=1) for Xi_i in Xi_tilde])

        # Traction terms
        A_rhs[0, 0] = sps.diags_array(Xi_tilde_bdry)
        A_rhs[1, 0] = R * delta_mu_k * Delta_bdry
        A_rhs[2, 0] = -ndot * delta_mu_k * Delta_bdry

        # Displacement terms
        A_rhs[0, 1] = -2 * sps.diags_array(mu_effective * Delta_bdry)
        A_rhs[1, 1] = -R * Xi_bdry
        A_rhs[2, 1] = ndot * Xi_bdry

        A_csc = sps.block_array(A_rhs, format="csc")

        # Efficient row scaling with the face areas
        f_areas = self.face_area_scaling(sd)
        A_csc.data *= f_areas[A_csc.indices]

        return A_csc

    def assemble_body_force(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Assemble the right-hand side for a given body-force function.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable): The body force function.

        Returns:
            np.ndarray: The right-hand side vector.
        """
        rhs = np.zeros(self.ndof(sd))
        rhs[: sd.dim * sd.num_cells] = -pg.VecPwConstants().interpolate(sd, func)

        return rhs

    def split_solution(self, sd: pg.Grid, sol: np.ndarray) -> list:
        """
        Split a given TPSA solution into its displacement, rotation, and solid pressure
        components

        Args:
            sd (pg.Grid): Grid, or a subclass.
            sol (np.ndarray): The solution to be split

        Returns:
            list: The solution components.
        """
        ndofs = sd.num_cells * np.array([sd.dim, rotation_dim(sd.dim)])

        return np.split(sol, np.cumsum(ndofs))


def rotation_dim(dim: int) -> int:
    """
    Determine the dimension of the rotation space.

    Args:
        dim (int): Dimension of the problem.

    Returns:
        int: Dimension of the rotation space.
    """
    return dim * (dim - 1) // 2
