import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class CosseratTestMixed(unittest.TestCase):
    def run_cosserat_2d(self):
        sd = pp.StructuredTriangleGrid([16] * 2, [1] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        key = "elasticity"
        vec_rt0 = pg.VecRT0(key)
        rt0 = pg.RT0(key)
        vec_p0 = pg.VecPwConstants(key)
        p0 = pg.PwConstants(key)

        data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5}}}
        Ms = vec_rt0.assemble_mass_matrix(sd, data)
        Mw = rt0.assemble_mass_matrix(sd, data)
        Mu = vec_p0.assemble_mass_matrix(sd)
        Mr = p0.assemble_mass_matrix(sd)

        div_s = Mu @ vec_rt0.assemble_diff_matrix(sd)
        asym = Mr @ vec_rt0.assemble_asym_matrix(sd)
        div_w = Mr @ rt0.assemble_diff_matrix(sd)

        # fmt: off
        spp = sps.bmat([[     Ms, None, div_s.T, asym.T],
                        [   None,   Mw, None, div_w.T],
                        [ -div_s,  None, None,   None],
                        [-asym,  -div_w, None,   None]], format = "csc")
        # fmt: on

        b_faces = sd.tags["domain_boundary_faces"]
        # bc = vec_rt0.assemble_nat_bc(sd, u_boundary, b_faces)

        rhs = np.zeros(spp.shape[0])
        force = lambda x: np.array([0, -1])
        force_p0 = vec_p0.interpolate(sd, force)
        force_rhs = Mu @ force_p0

        split_idx = np.cumsum([vec_rt0.ndof(sd), rt0.ndof(sd), vec_p0.ndof(sd)])

        rhs[split_idx[1] : split_idx[2]] += force_rhs
        # rhs[: vec_rt0.ndof(sd)] = bc

        x = sps.linalg.spsolve(spp, rhs)

        sigma, w, u, r = np.split(x, split_idx)

        cell_sigma = vec_rt0.eval_at_cell_centers(sd) @ sigma
        cell_w = rt0.eval_at_cell_centers(sd) @ w
        cell_u = vec_p0.eval_at_cell_centers(sd) @ u
        cell_r = p0.eval_at_cell_centers(sd) @ r

        # we need to add the z component for the exporting
        cell_u = np.hstack((cell_u, np.zeros(sd.num_cells)))
        cell_u = cell_u.reshape((3, -1))

        save = pp.Exporter(sd, "sol_cosserat")
        save.write_vtu(
            [
                ("cell_u", cell_u),
            ]
        )

    # def test_elasticity_rbm_2d(self):
    #     N = 3
    #     u_boundary = lambda x: np.array([-0.5 - x[1], -0.5 + x[0], 0])
    #     cell_sigma, cell_u, cell_r, sd = self.run_elasticity_2d(u_boundary, N)

    #     key = "elasticity"
    #     vec_p0 = pg.VecPwConstants(key)
    #     interp = vec_p0.interpolate(sd, u_boundary)
    #     u_known = vec_p0.eval_at_cell_centers(sd) @ interp

    #     self.assertTrue(np.allclose(cell_sigma, 0))
    #     self.assertTrue(np.allclose(cell_u, u_known))
    #     self.assertTrue(np.allclose(cell_r, -1))

    # def test_elasticity_2d(self):
    #     N = 3
    #     u_boundary = lambda x: np.array([x[0], x[1], 0])
    #     cell_sigma, cell_u, cell_r, sd = self.run_elasticity_2d(u_boundary, N)

    #     key = "elasticity"
    #     vec_p0 = pg.VecPwConstants(key)
    #     interp = vec_p0.interpolate(sd, u_boundary)
    #     u_known = vec_p0.eval_at_cell_centers(sd) @ interp

    #     cell_sigma = cell_sigma.reshape((6, -1))

    #     self.assertTrue(np.allclose(cell_sigma[0], 2))
    #     self.assertTrue(np.allclose(cell_sigma[1], 0))
    #     self.assertTrue(np.allclose(cell_sigma[2], 0))
    #     self.assertTrue(np.allclose(cell_sigma[3], 0))
    #     self.assertTrue(np.allclose(cell_sigma[4], 2))
    #     self.assertTrue(np.allclose(cell_sigma[5], 0))
    #     self.assertTrue(np.allclose(cell_u, u_known))
    #     self.assertTrue(np.allclose(cell_r, 0))

    # def run_elasticity_3d(self, u_boundary, N):
    #     sd = pp.StructuredTetrahedralGrid([N] * 3, [1] * 3)
    #     pg.convert_from_pp(sd)
    #     sd.compute_geometry()

    #     key = "elasticity"
    #     vec_bdm1 = pg.VecBDM1(key)
    #     vec_p0 = pg.VecPwConstants(key)

    #     data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5}}}
    #     Ms = vec_bdm1.assemble_mass_matrix(sd, data)
    #     Mu = vec_p0.assemble_mass_matrix(sd)
    #     Mr = Mu

    #     div = Mu @ vec_bdm1.assemble_diff_matrix(sd)
    #     asym = Mr @ vec_bdm1.assemble_asym_matrix(sd)

    #     # fmt: off
    #     spp = sps.bmat([[   Ms, div.T, asym.T],
    #                     [ -div,  None,   None],
    #                     [-asym,  None,   None]], format = "csc")
    #     # fmt: on

    #     b_faces = sd.tags["domain_boundary_faces"]
    #     bc = vec_bdm1.assemble_nat_bc(sd, u_boundary, b_faces)

    #     rhs = np.zeros(spp.shape[0])
    #     rhs[: vec_bdm1.ndof(sd)] = bc

    #     x = sps.linalg.spsolve(spp, rhs)

    #     split_idx = np.cumsum([vec_bdm1.ndof(sd), vec_p0.ndof(sd)])
    #     sigma, u, r = np.split(x, split_idx)

    #     cell_sigma = vec_bdm1.eval_at_cell_centers(sd) @ sigma
    #     cell_u = vec_p0.eval_at_cell_centers(sd) @ u
    #     cell_r = vec_p0.eval_at_cell_centers(sd) @ r

    #     return cell_sigma, cell_u, cell_r, sd

    # def test_elasticity_rbm_3d(self):
    #     N = 3
    #     u_boundary = lambda x: np.array([-0.5 - x[1], -0.5 + x[0] - x[2], -0.5 + x[1]])
    #     cell_sigma, cell_u, cell_r, sd = self.run_elasticity_3d(u_boundary, N)

    #     key = "elasticity"
    #     vec_p0 = pg.VecPwConstants(key)
    #     interp = vec_p0.interpolate(sd, u_boundary)
    #     u_known = vec_p0.eval_at_cell_centers(sd) @ interp

    #     self.assertTrue(np.allclose(cell_sigma, 0))
    #     self.assertTrue(np.allclose(cell_u, u_known))

    #     cell_r = cell_r.reshape((3, -1))

    #     self.assertTrue(np.allclose(cell_r[0], 1))
    #     self.assertTrue(np.allclose(cell_r[1], 0))
    #     self.assertTrue(np.allclose(cell_r[2], 1))

    # def test_elasticity_3d(self):
    #     N = 3
    #     u_boundary = lambda x: np.array([x[0], x[1], x[2]])
    #     cell_sigma, cell_u, cell_r, sd = self.run_elasticity_3d(u_boundary, N)

    #     key = "elasticity"
    #     vec_p0 = pg.VecPwConstants(key)
    #     interp = vec_p0.interpolate(sd, u_boundary)
    #     u_known = vec_p0.eval_at_cell_centers(sd) @ interp

    #     cell_sigma = cell_sigma.reshape((9, -1))

    #     self.assertTrue(np.allclose(cell_sigma[0], 2.5))
    #     self.assertTrue(np.allclose(cell_sigma[1], 0))
    #     self.assertTrue(np.allclose(cell_sigma[2], 0))
    #     self.assertTrue(np.allclose(cell_sigma[3], 0))
    #     self.assertTrue(np.allclose(cell_sigma[4], 2.5))
    #     self.assertTrue(np.allclose(cell_sigma[5], 0))
    #     self.assertTrue(np.allclose(cell_sigma[6], 0))
    #     self.assertTrue(np.allclose(cell_sigma[7], 0))
    #     self.assertTrue(np.allclose(cell_sigma[8], 2.5))

    #     self.assertTrue(np.allclose(cell_u, u_known))
    #     self.assertTrue(np.allclose(cell_r, 0))


if __name__ == "__main__":
    CosseratTestMixed().run_cosserat_2d()
