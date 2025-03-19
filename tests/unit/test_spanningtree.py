"""Module contains Spanning Tree tests."""

import unittest
import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class SpanningTreeTest(unittest.TestCase):
    def sptr(self, mdg):
        sd = mdg.subdomains(dim=mdg.dim_max())[0]
        bottom = np.isclose(sd.face_centers[1, :], sd.face_centers[1, :].min())

        return [
            pg.SpanningTree(mdg),
            pg.SpanningTree(mdg, "all_bdry"),
            pg.SpanningTree(mdg, bottom),
            pg.SpanningWeightedTrees(mdg, pg.SpanningTree, [0.25, 0.5, 0.25]),
        ]

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
        system = sps.block_array([[face_mass, -div.T], [div, None]], format="csc")

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

    def test_assemble_SI(self):
        N, dim = 3, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        for s in self.sptr(mdg):
            SI = s.assemble_SI()
            B = pg.cell_mass(mdg) @ pg.div(mdg)
            check = sps.eye_array(B.shape[0]) - B @ SI

            self.assertTrue(np.allclose(check.data, 0))

    def test_for_errors(self):
        sd = pg.unit_grid(2, 0.125)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        self.assertRaises(KeyError, pg.SpanningTree, mdg, "error_str")


class SpanningTreeElasticityTest(unittest.TestCase):
    def sptr(self, mdg):
        sd = mdg.subdomains(dim=mdg.dim_max())[0]
        bottom = np.isclose(sd.face_centers[1, :], sd.face_centers[1, :].min())

        return [
            pg.SpanningTreeElasticity(mdg),
            pg.SpanningTreeElasticity(mdg, "all_bdry"),
            pg.SpanningTreeElasticity(mdg, bottom),
            pg.SpanningWeightedTrees(mdg, pg.SpanningTreeElasticity, [0.25, 0.5, 0.25]),
        ]

    def assemble_B(self, mdg):
        sd = mdg.subdomains(dim=mdg.dim_max())[0]

        key = "tree"
        vec_bdm1 = pg.VecBDM1(key)
        vec_p0 = pg.VecPwConstants(key)

        M_div = vec_p0.assemble_mass_matrix(sd)
        if sd.dim == 2:
            p0 = pg.PwConstants(key)
            M_asym = p0.assemble_mass_matrix(sd)
        else:
            M_asym = M_div

        div = M_div @ vec_bdm1.assemble_diff_matrix(sd)
        asym = M_asym @ vec_bdm1.assemble_asym_matrix(sd)

        return sps.vstack((-div, -asym))

    def test_elasticity_tria_grid(self):
        sd = pg.unit_grid(2, 0.125)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        B = self.assemble_B(mdg)
        f = np.random.rand(B.shape[0])

        for sptr in self.sptr(mdg):
            s_f = sptr.solve(f)
            self.assertTrue(np.allclose(B @ s_f, f))

    def test_elasticity_struct_tet_grid(self):
        sd = pp.StructuredTetrahedralGrid([1] * 3)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        B = self.assemble_B(mdg)
        f = np.random.rand(B.shape[0])

        for sptr in self.sptr(mdg):
            s_f = sptr.solve(f)
            self.assertTrue(np.allclose(B @ s_f, f))

    def test_elasticity_unstruct_tet_grid(self):
        sd = pg.unit_grid(3, 1.0)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        B = self.assemble_B(mdg)
        f = np.random.rand(B.shape[0])

        for sptr in self.sptr(mdg):
            s_f = sptr.solve(f)
            self.assertTrue(np.allclose(B @ s_f, f))

    def test_assemble_SI(self):
        N, dim = 3, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        for s in self.sptr(mdg):
            SI = s.assemble_SI()
            B = self.assemble_B(mdg)
            check = sps.eye_array(B.shape[0]) - B @ SI

            self.assertTrue(np.allclose(check.data, 0))

    def test_for_errors(self):
        sd = pp.CartGrid(1, 1)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        self.assertRaises(NotImplementedError, pg.SpanningTreeElasticity, mdg)


class SpanningTreeCosseratTest(unittest.TestCase):
    def check(self, sd):
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        B = self.assemble_B(mdg)
        f = np.random.rand(B.shape[0])

        sptr = pg.SpanningTreeCosserat(mdg)

        s_f = sptr.solve(f)
        self.assertTrue(np.allclose(B @ s_f, f))

    def assemble_B(self, mdg):
        sd = mdg.subdomains(dim=mdg.dim_max())[0]

        key = "tree"
        vec_rt0 = pg.VecRT0(key)
        vec_p0 = pg.VecPwConstants(key)

        M = vec_p0.assemble_mass_matrix(sd)

        div = M @ vec_rt0.assemble_diff_matrix(sd)
        asym = M @ vec_rt0.assemble_asym_matrix(sd)

        return sps.block_array([[-div, None], [-asym, -div]], format="csc")

    def test_elasticity_struct_tet_grid(self):
        sd = pp.StructuredTetrahedralGrid([1] * 3)
        self.check(sd)

    def test_elasticity_unstruct_tet_grid(self):
        sd = pg.unit_grid(3, 1.0)
        self.check(sd)

    def test_assemble_SI(self):
        N, dim = 2, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        sptr = pg.SpanningTreeCosserat(mdg)

        SI = sptr.assemble_SI()
        B = self.assemble_B(mdg)
        check = sps.eye_array(B.shape[0]) - B @ SI

        self.assertTrue(np.allclose(check.data, 0))


if __name__ == "__main__":
    unittest.main()
