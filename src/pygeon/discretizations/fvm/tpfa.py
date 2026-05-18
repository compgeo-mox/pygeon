import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class TPFA(pg.FiniteVolumeDiscretization):
    bc_type = pg.FlowBC

    def ndof_per_cell(self, *_) -> int:
        return 1

    def assemble_flow_matrix(self, sd: pg.Grid, data: dict):
        perm = pg.get_cell_data(sd, data, self.keyword, pg.SECOND_ORDER_TENSOR, 2)

        # Precomputations without boundary conditions
        self.fvm_precomputations(sd, perm.values)

        bcs = self.extract_bcs(sd, data)
        faces, deltas = self.extend_faces_and_distances(sd, bcs)

        self.compute_harmonic_avg(faces, deltas)

        codiv = sd.cell_faces

        return self.div_F(sd) @ (self.K_bar_over_delta[:, None] * codiv)

    def compute_weighted_dists(self, sd: pg.Grid, perm: np.ndarray) -> np.ndarray:
        """
        Compute delta_k^i / K_nn for every physical face-cell pair. Boundary conditions
        are handled later.
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

    def compute_harmonic_avg(self, faces, dists):
        """
        Compute the harmonic average of K divided by delta_k, at each face
        """
        self.K_bar_over_delta = np.array(1 / np.bincount(faces, weights=dists))

    def assemble_rhs_boundary_terms(self, sd: pg.Grid, data: dict):
        rhs = np.empty(2, dtype=sps.sparray)

        K_bar = self.K_bar_over_delta

        rhs[0] = sps.diags_array((K_bar == 0).astype(float))

        Delta_B = -sd.cell_faces.sum(axis=1)
        rhs[1] = sps.diags_array(K_bar * Delta_B)

        bcs = self.extract_bcs(sd, data)
        g = np.hstack((bcs.flux, bcs.pres))

        return -pg.div(sd) * sd.face_areas @ sps.hstack(rhs) @ g


if __name__ == "__main__":
    dims = {
        "xmin": 0,
        "xmax": 1,
        "ymin": 0,
        "ymax": 1,
        "zmin": -1,
        "zmax": 0,
    }
    sd = pp.CartGrid([5, 5, 5], dims)
    sd = pg.convert_from_pp(sd)
    sd.compute_geometry()

    K_vals = np.tile(np.array([1, 2, 3]), (sd.num_cells, 1)).T
    K = pp.SecondOrderTensor(*K_vals)

    tpfa = TPFA("test")

    data = pp.initialize_data({}, "test")
    bcs = pg.FlowBC(sd, data, "test")

    p_0 = sd.face_centers[-1]

    bdry_faces = sd.tags["domain_boundary_faces"]
    bottom = np.isclose(sd.face_centers[-1], 0)
    bcs.set_pressure_bcs(bottom, np.ones_like(p_0))

    flux_faces = np.logical_xor(bottom, bdry_faces)
    bcs.set_flux_bcs(flux_faces)

    M = tpfa.assemble_flow_matrix(sd, data)
    rhs = tpfa.assemble_rhs_boundary_terms(sd, data)

    sol = sps.linalg.spsolve(M, rhs)

    pass
