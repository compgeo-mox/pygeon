""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class SpanningTreeTest(unittest.TestCase):
    def sptr(self, mdg):
        return [pg.SpanningTree(mdg), pg.SpanningWeightedTrees(mdg, [0.25, 0.5, 0.25])]

    def check_flux(self, mdg, sptr):
        """
        Check whether the constructed flux balances the given mass-source
        """
        f = np.arange(mdg.num_subdomain_cells())
        q_f = sptr.solve(f)

        self.assertTrue(np.allclose(pg.cell_mass(mdg) @ pg.div(mdg) @ q_f, f))

    def check_pressure(self, mdg, sptr):
        """
        Check whether the post-processing of the pressure is correct
        """
        div = pg.cell_mass(mdg) @ pg.div(mdg)
        face_mass = pg.face_mass(mdg)
        system = sps.bmat([[face_mass, -div.T], [div, None]], "csc")

        f = np.ones(div.shape[0])
        rhs = np.hstack([np.zeros(div.shape[1]), f])

        x = sps.linalg.spsolve(system, rhs)
        q = x[: div.shape[1]]
        p = x[div.shape[1] :]

        p_sptr = sptr.solve_transpose(face_mass @ q)

        self.assertTrue(np.allclose(p, p_sptr))

    def test_cart_grid(self):
        N = 3
        for dim in np.arange(1, 4):
            sd = pp.CartGrid([N] * dim, [1] * dim)
            mdg = pg.as_mdg(sd)
            pg.convert_from_pp(mdg)
            mdg.compute_geometry()

            [self.check_flux(mdg, s) for s in self.sptr(mdg)]

    def test_structured_triangle(self):
        N, dim = 3, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        for s in self.sptr(mdg):
            self.check_flux(mdg, s)
            self.check_pressure(mdg, s)

    def test_unstructured_triangle(self):
        sd = pg.unit_grid(2, 0.25, as_mdg=False)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        for s in self.sptr(mdg):
            self.check_flux(mdg, s)
            self.check_pressure(mdg, s)

    def test_structured_tetra(self):
        N, dim = 3, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        for s in self.sptr(mdg):
            self.check_flux(mdg, s)
            self.check_pressure(mdg, s)

    def test_2d_mdg(self):
        mesh_args = {"cell_size": 0.25, "cell_size_fracture": 0.125}
        grids = [
            pp.mdg_library.square_with_orthogonal_fractures("simplex", mesh_args, [1]),
            pp.mdg_library.square_with_orthogonal_fractures("simplex", mesh_args, [0]),
            pp.mdg_library.square_with_orthogonal_fractures(
                "simplex", mesh_args, [0, 1]
            ),
        ]

        for g in grids:
            mdg, _ = g
            pg.convert_from_pp(mdg)
            mdg.compute_geometry()

            for s in self.sptr(mdg):
                self.check_flux(mdg, s)
                self.check_pressure(mdg, s)

    def test_3d_mdg(self):
        mesh_args = {"cell_size": 0.5, "cell_size_fracture": 0.5}
        mdg, _ = pp.mdg_library.cube_with_orthogonal_fractures(
            "simplex", mesh_args, [0, 1, 2]
        )
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        for s in self.sptr(mdg):
            self.check_flux(mdg, s)
            self.check_pressure(mdg, s)

    def test_elasticity_tria_grid(self):
        sd = pg.unit_grid(2, 0.125)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        sptr = pg.SpanningTreeElasticity(mdg)
        sd = mdg.subdomains(dim=mdg.dim_max())[0]

        key = "tree"
        vec_bdm1 = pg.VecBDM1(key)
        vec_p0 = pg.VecPwConstants(key)
        p0 = pg.PwConstants(key)

        M_div = vec_p0.assemble_mass_matrix(sd)
        M_asym = p0.assemble_mass_matrix(sd)

        div = M_div @ vec_bdm1.assemble_diff_matrix(sd)
        asym = M_asym @ vec_bdm1.assemble_asym_matrix(sd)

        B = sps.vstack((-div, -asym))

        f = np.random.rand(B.shape[0])
        s_f = sptr.solve(f)

        self.assertTrue(np.allclose(B @ s_f, f))


if __name__ == "__main__":
    unittest.main()
