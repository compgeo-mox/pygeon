from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class TPSA(pg.FiniteVolumeDiscretization):
    """
    A vectorized implementation of the two-point stress approximation method for
    elasticity of Nordbotten and Keilegavlen (2025).

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
        displacement: Callable,
        rotation: Callable,
        solid_pressure: Callable,
    ) -> np.ndarray:
        """
        Interpolates a triplet of functions onto the finite volume space

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

    def assemble_elasticity_matrix(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assemble the TPSA matrix, using the material parameters in the data dictionary.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            data (dict): The data dictionary.

        Returns:
            sps.csc_array: The TPSA discretization matrix.
        """
        # Assemble the second order terms in (3.9)
        A = self.div(sd) @ self.assemble_dual_var_map(sd, data)

        # Generate the accumulation terms in (3.9)
        M = self.assemble_mass_terms(sd, data)

        # Assemble the matrix from (3.9)
        return (A - M).tocsc()

    def compute_weighted_dists(self, sd: pg.Grid, data: dict) -> np.ndarray:
        """
        Computes delta_k^i / mu_i from (2.1) for every physical face-cell pair (k, i).
        Boundary conditions are handled later

        Args:
            sd (pg.Grid): Grid, or a subclass.
            weights (np.ndarray): The material parameter weights, in this case mu.

        Returns:
            np.ndarray: The weighted distances
        """
        faces, cells, orient = sps.find(sd.cell_faces)
        unit_normals = sd.face_normals / sd.face_areas

        delta = np.sum(
            (
                (sd.face_centers[:, faces] - sd.cell_centers[:, cells])
                * (orient * unit_normals[:, faces])
            ),
            axis=0,
        )

        lame_mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        return delta / lame_mu[cells]

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
        weighted_dists = self.compute_weighted_dists(sd, data)

        # Incorporate the bcs by extending the vectors
        faces, *_ = sps.find(sd.cell_faces)
        bdry_faces = sd.tags["domain_boundary_faces"]
        ext_faces = np.concatenate((faces, np.flatnonzero(bdry_faces)))

        # We allow for different types of boundary conditions in the x, y, z directions.
        # We therefore have three instances of the distances, one for each direction.
        tiled_dists = np.tile(weighted_dists, (sd.dim, 1))
        ext_dists = np.hstack((tiled_dists, bcs.weighted_dists[: sd.dim, bdry_faces]))

        return ext_faces, ext_dists

    def compute_delta_mu_k(self, sd: pg.Grid, data: dict) -> np.ndarray:
        """
        Compute the delta^mu_k of (3.5) given by
        0.5 * ( mu_i delta_k^-i + mu_j delta_k^-j)^-1
        for each face k with neighboring cells (i,j).

        Args:
            sd (pg.Grid): Grid, or a subclass.
            faces (np.ndarray): The extended array of faces
            dists (np.ndarray): The extended array of weighted distances
        """
        faces, dists = self.extend_faces_and_distances(sd, data)

        # Compute the reciprocal
        inv_dists = np.empty_like(dists)
        zero_dist = dists == 0

        inv_dists[~zero_dist] = 1 / dists[~zero_dist]

        # Displacement boundaries have infinite mu/delta
        inv_dists[zero_dist] = np.inf

        # Traction bc are handled naturally because mu/delta = 0 there.
        output_list = [1 / np.bincount(faces, weights=row) for row in inv_dists]
        return np.array(output_list) / 2

    def compute_harmonic_avg(self, sd: pg.Grid, data: dict) -> np.ndarray:
        """
        Compute the harmonic average of mu from (3.5), divided by delta_k, at each face

        Args:
            sd (pg.Grid): Grid, or a subclass.
            faces (np.ndarray): The extended array of faces
            dists (np.ndarray): The extended array of weighted distances
        """
        faces, dists = self.extend_faces_and_distances(sd, data)
        output_list = [1 / np.bincount(faces, weights=row) for row in dists]
        return np.array(output_list)

    def assemble_dual_var_map(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assemble the mapping from cell-based primary variables to face-based dual
        variables.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: The matrix mapping primary to dual variables
        """
        # Preallocate the block matrix
        A = np.empty((3, 3), dtype=sps.sparray)

        # Assemble the blocks of (3.7) where A_ij is the block coupling variable i and
        # j. The canonical order of the variables is [u, r, p]
        mu_bar = self.compute_harmonic_avg(sd, data)
        A_uu = [-2 * mu[:, None] * sd.cell_faces for mu in mu_bar]
        A[0, 0] = sps.block_diag(A_uu)

        # Assemble the boundary terms of (A2.25)
        A[1, 1] = self.assemble_rot_rot_bdry_terms(sd, data)

        # The blocks in the first column depend on the averaging operator Xi
        Xi = self.assemble_xi(sd, data)
        R_Xi, n_Xi = self.assemble_first_column(sd, Xi)
        A[1, 0] = -R_Xi
        A[2, 0] = n_Xi

        # The blocks in the first row depend on the complementary operator Xi_tilde
        Xi_tilde = self.convert_to_xi_tilde_inplace(Xi)
        R_Xi_t, n_Xi_t = self.assemble_first_row(sd, Xi_tilde)
        A[0, 1] = -R_Xi_t
        A[0, 2] = n_Xi_t

        # Stabilization for the solid pressure
        delta_mu_k = self.compute_delta_mu_k(sd, data)
        unit_normals = sd.face_normals[: sd.dim] / sd.face_areas
        delta_n = np.sum(unit_normals**2 * delta_mu_k, axis=0)
        A[2, 2] = -delta_n[:, None] * sd.cell_faces

        # Scaling by the face areas
        f_areas = self.face_area_scaling(sd)[:, None]

        return (f_areas * sps.block_array(A)).tocsc()

    def assemble_mass_terms(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        The first-order terms on the diagonal of (3.9). This is a diagonal matrix.

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

        M_u = np.zeros(sd.dim * sd.num_cells)
        M_r = np.tile(sd.cell_volumes / lame_mu, rotation_dim(sd.dim))
        M_p = sd.cell_volumes / lame_lambda

        diagonal = np.concatenate((M_u, M_r, M_p))

        return sps.diags_array(diagonal).tocsc()

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
        return sps.hstack([sps.diags_array(n_i) for n_i in unit_normals[: sd.dim]])

    def assemble_rot_rot_bdry_terms(self, sd: pg.Grid, data: dict) -> sps.sparray:
        """
        The operator R^n \delta R^n that is on the [1, 1] block of (A2.25).

        There is a slight discrepancy with the paper, because simplified boundary
        conditions are assumed there. This implementation is the generalization to more
        involved boundary conditions, such as rollers.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: The double rotation matrix
        """
        delta_mu_k = self.compute_delta_mu_k(sd, data)

        bdry_deltas = delta_mu_k * sd.tags["domain_boundary_faces"]
        delta = bdry_deltas.flatten()

        R = self.assemble_rot(sd)
        minus_R_squared = (R * delta) @ R.T

        codiv = sps.kron(sps.eye_array(rotation_dim(sd.dim)), sd.cell_faces)
        return minus_R_squared @ codiv

    def assemble_xi(self, sd: pg.Grid, data: dict) -> list:
        """
        Compute the averaging operator Xi from (2.5)

        Displacement bc are handled by delta_mu_k = 0. Traction bc are handled since 2 *
        delta_mu_k * mu / delta = 1. Spring bc are handled because the spring constant
        is contained in delta_mu_k.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            list: The averaging operators in the coordinate directions
        """
        faces, cells, _ = sps.find(sd.cell_faces)
        weighted_dists = self.compute_weighted_dists(sd, data)
        delta_mu_k = self.compute_delta_mu_k(sd, data)

        Xi = [
            sps.csc_array((2 * delta[faces] / weighted_dists, (faces, cells)))
            for delta in delta_mu_k
        ]

        return Xi

    def convert_to_xi_tilde_inplace(self, Xi: list) -> sps.sparray:
        """
        Compute the complementary averaging operator Xi_tilde from (2.6).
        NOTE: This is an in-place operation.

        Args:
            sd (pg.Grid): Grid, or a subclass.

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
    ) -> Tuple[sps.sparray, sps.sparray]:
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
        Xi = sps.block_diag(Xi_list)
        R_Xi = self.assemble_rot(sd) @ Xi
        n_Xi = self.assemble_ndot(sd) @ Xi

        return R_Xi, n_Xi

    def assemble_first_row(
        self,
        sd: pg.Grid,
        Xi_list: list,
    ) -> Tuple[sps.sparray, sps.sparray]:
        """
        Assemble the off-diagonal terms in the first row of (3.7). These are computed
        together because their construction uses similar components.

        This is a generalization compared to the paper to handle more involved boundary
        conditions.

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

    def assemble_bdry_dual_var_map(self, sd: pg.Grid, data: dict) -> sps.csc_array:
        """
        Assemble the second matrix on the right-hand side of (A2.25).

        Slight generalization: the [1, 0] and [2, 0] blocks first weigh with delta and
        then rot/dot with n.

        Args:
            sd (pg.Grid): Grid, or a subclass.

        Returns:
            sps.csc_array: the matrix to be multiplied with the boundary data g
        """
        # Preallocation
        A_rhs = np.empty((3, 2), dtype=sps.sparray)

        # Ingredients with the normal
        R = self.assemble_rot(sd)
        ndot = self.assemble_ndot(sd)

        Delta_bdry = np.tile(-sd.cell_faces.sum(axis=1), sd.dim)

        Xi = self.assemble_xi(sd, data)
        Xi_bdry = 1 - np.hstack([Xi_i.sum(axis=1) for Xi_i in Xi])

        Xi_tilde = self.convert_to_xi_tilde_inplace(Xi)
        Xi_tilde_bdry = 1 - np.hstack([Xi_i.sum(axis=1) for Xi_i in Xi_tilde])

        mu_bar = self.compute_harmonic_avg(sd, data).ravel()
        delta_mu_k = self.compute_delta_mu_k(sd, data)
        dmuk = delta_mu_k.ravel()

        # Traction terms
        A_rhs[0, 0] = sps.diags_array(Xi_tilde_bdry)
        A_rhs[1, 0] = R * dmuk * Delta_bdry
        A_rhs[2, 0] = -ndot * dmuk * Delta_bdry

        # Displacement terms
        A_rhs[0, 1] = -2 * sps.diags_array(mu_bar * Delta_bdry)
        A_rhs[1, 1] = -R * Xi_bdry
        A_rhs[2, 1] = ndot * Xi_bdry

        f_areas = self.face_area_scaling(sd)[:, None]

        return f_areas * sps.block_array(A_rhs, format="csc")

    def assemble_body_force(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Assemble the right-hand side for a given body force func(x,y,z)

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable): The body force function.

        Returns:
            np.ndarray: the right-hand side vector
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
            list: The solution components
        """
        ndofs = sd.num_cells * np.array([sd.dim, rotation_dim(sd.dim)])

        return np.split(sol, np.cumsum(ndofs))


def rotation_dim(dim: int) -> int:
    """
    Helper function to determine the dimension of the rotation space

    Args:
        dim (int): dimension of the problem

    Returns:
        int: dimension of the rotation space
    """
    return dim * (dim - 1) // 2
